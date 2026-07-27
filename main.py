"""
Agent entrypoint / orchestration layer.

REWRITTEN (see main.py.pre_rewrite_backup for the old version). The old
functions (start_cancellation_by_reference, start_cancellation_by_phone,
resume_with_value, pending_interrupt, etc.) existed to drive a fixed
sequence of graph interrupts. There is no such fixed sequence anymore -
the LLM decides everything - so this file collapses to the one thing
that's actually still needed: append a HumanMessage to a thread's chat
history, invoke the graph, and return whatever the LLM's final message
of that turn is. That's it. This is intentionally much simpler than
before, which is a direct consequence of moving all conversation logic
into the LLM.

The CLI (`python main.py`) still behaves the same way from the user's
point of view - free-text back and forth - even though its internal
implementation necessarily changed along with the graph shape.
"""

import logging
import time
from typing import Dict

from langchain_core.messages import HumanMessage

from config import SESSION_TIMEOUT_SECONDS, THREAD_ID_PREFIX, configure_logging
from graph import graph

configure_logging()
logger = logging.getLogger(__name__)


# ==========================================================
# Inactivity-based reset
# ==========================================================
#
# Tracks, per session_id, when it last sent a message and which
# "generation" it's currently on. If more than SESSION_TIMEOUT_SECONDS
# has passed since the last message, the next one bumps the generation
# counter - which changes the actual thread_id passed to the graph/
# checkpointer, so MemorySaver treats it as a brand new, empty
# conversation. The caller's own session_id never changes; this is
# entirely internal bookkeeping. Consistent with MemorySaver itself,
# this is in-process only (resets on server restart too - see README).

_last_active: Dict[str, float] = {}
_generation: Dict[str, int] = {}


_now = time.time  # dedicated reference so tests can patch main._now in
                  # isolation, without affecting time.time() globally
                  # (which LangGraph's own internals also call)


def _config_for(session_id: str) -> dict:
    now = _now()
    last = _last_active.get(session_id)

    if last is not None and (now - last) > SESSION_TIMEOUT_SECONDS:
        _generation[session_id] = _generation.get(session_id, 0) + 1
        logger.info(
            "session_id=%s: %.0fs since last message (> %ss timeout) - starting a fresh conversation (generation %s)",
            session_id, now - last, SESSION_TIMEOUT_SECONDS, _generation[session_id],
        )

    _last_active[session_id] = now

    generation = _generation.get(session_id, 0)
    thread_id = f"{THREAD_ID_PREFIX}:{session_id}:{generation}" if generation else f"{THREAD_ID_PREFIX}:{session_id}"

    return {"configurable": {"thread_id": thread_id}}


def _cancellation_just_succeeded(messages: list) -> bool:
    """Detect whether this turn's messages include a successful
    cancel_appointment tool call. Used to reset the session's memory
    right after a cancellation completes - see send_message()."""

    for msg in messages:
        if getattr(msg, "name", None) == "cancel_appointment":
            content = str(getattr(msg, "content", ""))
            if '"status": "success"' in content or "'status': 'success'" in content:
                return True

    return False


def send_message(client_id: str, session_id: str, message: str, channel_phone: str = None) -> str:
    """
    Send one user message for `session_id` and return the agent's reply
    text for this turn.

    This is the ONLY public function this file needs now: whether it's
    the first message of a brand new conversation or a follow-up to an
    earlier question, the call is identical - the checkpointer
    (MemorySaver, preserved) already knows the full prior chat history
    for this thread_id, so client_id/channel_phone only need to be
    supplied again in case they weren't set yet (load_config is a no-op
    once templates are already cached for this thread).

    RESET BEHAVIOR: memory resets automatically in two cases -
      1. After SESSION_TIMEOUT_SECONDS of inactivity (see _config_for).
      2. Immediately after a cancellation completes successfully (below) -
         a completed cancellation is a natural conversation boundary;
         without this, a long-lived thread that already finished one
         cancellation can carry stale context (e.g. a dialect used much
         earlier in an unrelated test) into what the user perceives as a
         brand new conversation, and skips the opening greeting since the
         graph doesn't consider it "new".
    """

    state = {
        "client_id": client_id,
        "session_id": session_id,
        "channel_phone": channel_phone,
        "messages": [HumanMessage(content=message)],
    }

    logger.info("session_id=%s: sending message", session_id)

    result = graph.invoke(state, config=_config_for(session_id))

    reply = result["messages"][-1].content
    logger.info("session_id=%s: reply=%r", session_id, reply)

    if _cancellation_just_succeeded(result["messages"]):
        _generation[session_id] = _generation.get(session_id, 0) + 1
        logger.info(
            "session_id=%s: cancellation succeeded this turn - resetting memory for the next message (generation %s)",
            session_id, _generation[session_id],
        )

    return reply


# ==========================================================
# CLI
# ==========================================================

def _run_cli() -> None:
    print("=== Guest Booking Cancellation Agent (CLI) ===")

    client_id = input("Client id [Dar El Oyoun-demo]: ").strip() or "Dar El Oyoun-demo"
    session_id = input("Session id [demo-session]: ").strip() or "demo-session"
    channel_phone = input("Channel/WhatsApp sender number (optional, press Enter to skip): ").strip() or None

    print("\nType your message below (e.g. 'I want to cancel my appointment'). Ctrl+C to quit.\n")

    message = input("You: ").strip()

    while True:
        try:
            reply = send_message(client_id, session_id, message, channel_phone=channel_phone)
        except Exception as exc:  # pragma: no cover - CLI convenience only
            print(f"\n[error] {exc}\n")
            break

        print(f"\nAssistant: {reply}\n")
        message = input("You: ").strip()


if __name__ == "__main__":
    _run_cli()
