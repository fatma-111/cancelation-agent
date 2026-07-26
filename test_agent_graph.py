"""
Tests for the rewritten LLM-tool-calling graph.

Since this architecture makes EVERY decision through the LLM, testing it
end-to-end requires scripting what the LLM would decide at each step of
a realistic conversation - so these tests replace graph._llm_with_tools
with a small FakeLLM that returns a pre-scripted sequence of AIMessages
(some with tool_calls, some without), while every actual tool call still
runs for real against a MOCKED api.py (never real network). This
verifies the agent<->tools loop, InjectedState wiring, and MemorySaver
checkpointing all work together correctly - not just that the code
imports.

Run with:
    python3 test_agent_graph.py
"""

from unittest.mock import patch

from langchain_core.messages import AIMessage

import graph
import main as agent


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


class FakeLLM:
    """Pops one scripted AIMessage per .invoke() call, in order."""

    def __init__(self, responses):
        self._responses = list(responses)

    def invoke(self, messages):
        if not self._responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self._responses.pop(0)

    def remaining(self):
        return len(self._responses)


def _tool_call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def test_reference_cancel_full_conversation():
    section("Reference cancel: ask method -> lookup -> confirm -> check_status -> cancel -> success")

    booking = {
        "id": "GUID-1", "bookingRefNum": "GBN-2026-06-20-151", "statusName": "New", "status": 1,
        "doctorName": "Dr. Omar", "branchName": "Downtown", "bookingTimeFrom": "2026-08-20T13:00:00",
        "mobileNumber": "+201001255864",
    }

    fake = FakeLLM([
        # Turn 1: user says "I want to cancel my appointment" -> ask which method
        AIMessage(content="Would you like to cancel using your booking reference or phone number?"),
        # Turn 2: user gives the reference -> LLM calls lookup_appointment
        _tool_call("lookup_appointment", {"ref_number": "GBN-2026-06-20-151", "phone": ""}, "call_1"),
        # ...then, seeing found_one, presents it and asks to confirm
        AIMessage(content="I found booking GBN-2026-06-20-151 with Dr. Omar at Downtown on 20/08/2026 1:00 PM. Cancel it?"),
        # Turn 3: user says "yes" -> LLM calls check_booking_status
        _tool_call("check_booking_status", {"ref_number": "GBN-2026-06-20-151"}, "call_2"),
        # ...sees "active" -> calls cancel_appointment with the id
        _tool_call("cancel_appointment", {"booking_id": "GUID-1"}, "call_3"),
        # ...sees "success" -> final natural-language confirmation
        AIMessage(content="Your appointment has been cancelled successfully."),
    ])

    graph._llm_with_tools = fake

    with patch("api.get_bookings_by_ref", return_value={"success": True, "status_code": 200, "data": {"items": [booking]}, "error": None}) as mock_lookup, \
         patch("api.cancel_booking_by_guid", return_value={"success": True, "status_code": 200, "data": {"isSuccess": True}, "error": None}) as mock_cancel:

        r1 = agent.send_message("Dar El Oyoun-demo", "sess-ref-1", "I want to cancel my appointment")
        print("Turn 1:", r1)
        assert "reference" in r1.lower() or "phone" in r1.lower()

        r2 = agent.send_message("Dar El Oyoun-demo", "sess-ref-1", "GBN-2026-06-20-151")
        print("Turn 2:", r2)
        assert "GBN-2026-06-20-151" in r2
        assert mock_lookup.call_count == 1

        r3 = agent.send_message("Dar El Oyoun-demo", "sess-ref-1", "yes")
        print("Turn 3:", r3)
        assert "cancel" in r3.lower()
        assert mock_lookup.call_count == 2, "check_booking_status re-fetches by ref, calling get_bookings_by_ref again"
        assert mock_cancel.call_count == 1
        assert mock_cancel.call_args.args[1] == "GUID-1" or mock_cancel.call_args.kwargs.get("booking_guid") == "GUID-1"

    assert fake.remaining() == 0, "every scripted LLM response should have been consumed exactly once"
    print("PASSED")


def test_checkpointer_persists_chat_history_across_turns():
    section("Checkpointer: chat history accumulates across separate send_message() calls")

    fake = FakeLLM([
        AIMessage(content="Hi! How can I help?"),
        AIMessage(content="Sure, go ahead."),
    ])
    graph._llm_with_tools = fake

    agent.send_message("Dar El Oyoun-demo", "sess-persist-1", "hello")
    agent.send_message("Dar El Oyoun-demo", "sess-persist-1", "I want to cancel")

    snapshot = graph.graph.get_state(agent._config_for("sess-persist-1"))
    history = snapshot.values["messages"]
    print("Accumulated messages:", [(m.type, m.content) for m in history])

    # 2 human + 2 AI = 4 messages accumulated in one thread's history
    assert len(history) == 4
    assert history[0].type == "human" and history[0].content == "hello"
    assert history[2].type == "human" and history[2].content == "I want to cancel"

    print("PASSED")


def test_load_config_runs_once_per_thread():
    section("load_config only loads CSVs once per thread, not every turn")

    fake = FakeLLM([AIMessage(content="ok"), AIMessage(content="ok again")])
    graph._llm_with_tools = fake

    with patch("config.get_messages", wraps=__import__("config").get_messages) as spy:
        agent.send_message("Dar El Oyoun-demo", "sess-cfg-1", "hi")
        agent.send_message("Dar El Oyoun-demo", "sess-cfg-1", "hi again")
        print("config.get_messages call count across 2 turns:", spy.call_count)
        assert spy.call_count == 1, "templates/system_prompt must be cached after the first turn"

    print("PASSED")


def test_injected_state_hides_base_url_from_llm_schema():
    section("Tool schemas never expose base_url/state to the LLM (prevents URL hallucination)")

    import tools
    for t in tools.ALL_TOOLS:
        assert "state" not in t.args, f"{t.name} leaks 'state' into its LLM-visible schema"
        assert "base_url" not in t.args, f"{t.name} leaks 'base_url' into its LLM-visible schema"

    print("PASSED")


if __name__ == "__main__":
    test_reference_cancel_full_conversation()
    test_checkpointer_persists_chat_history_across_turns()
    test_load_config_runs_once_per_thread()
    test_injected_state_hides_base_url_from_llm_schema()
    print("\nALL AGENT GRAPH TESTS PASSED\n")
