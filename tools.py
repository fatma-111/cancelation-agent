"""
LangChain tools for the LLM-tool-calling Guest Booking Cancellation Agent.

REWRITTEN (see tools.py.pre_rewrite_backup for the old version). Every
tool now returns STRUCTURED DATA ONLY - no formatted sentences, no
message-template lookups, no natural language of any kind. The LLM
(driven by prompts.AGENT_SYSTEM_PROMPT_TEMPLATE) is solely responsible
for turning these status codes/data into user-facing replies. This is
the literal architecture change requested: tools never speak to the
user.

What did NOT change: api.py (all raw HTTP calls), config.py (client
config / base_url resolution), the timezone conversion math, the
active-booking filter, and the OTP dummy-provider mechanics. Those are
"Company APIs" / "booking logic" / "OTP logic" and were explicitly
required to stay untouched - only the OUTPUT SHAPE of the functions that
wrap them changed, from "already-formatted text" to "plain status/data".

Removed entirely (superseded by the LLM's own reasoning, since "the LLM
should decide" replaces every heuristic classifier):
  detect_message, extract_input_details, resolve_selection,
  parse_confirmation, detect_step_back, format_message,
  format_booking_card, format_booking_list, format_time_12h, format_date,
  find_matching_appointment (replaced by check_booking_status's ref-based
  re-lookup, simpler and equally safe since ref numbers are unique).
"""

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Annotated, Dict, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

import api
from config import (
    BOOKING_TIME_UTC_OFFSET_HOURS,
    CANCELLED_STATUS_CODE,
    CANCELLED_STATUS_NAME,
    DEFAULT_COUNTRY_CODE,
    OTP_PROVIDER,
    OTP_TTL_SECONDS,
    TEST_OTP,
)
from state import AgentState

logger = logging.getLogger(__name__)


# ==========================================================
# Pure data helpers (unchanged in spirit from the old tools.py - these
# are data transforms, not user-facing text, so they stay)
# ==========================================================

def normalize_phone_number(phone: Optional[str]) -> Optional[str]:
    """Normalize a phone number to E.164 (e.g. "+201001255864")."""

    if not phone:
        return phone

    cleaned = re.sub(r"[\s\-().]", "", phone.strip())

    if cleaned.startswith("+"):
        return cleaned
    if cleaned.startswith("00"):
        return "+" + cleaned[2:]
    if cleaned.startswith(DEFAULT_COUNTRY_CODE):
        return "+" + cleaned
    if cleaned.startswith("0"):
        return "+" + DEFAULT_COUNTRY_CODE + cleaned[1:]

    return "+" + DEFAULT_COUNTRY_CODE + cleaned


def _is_valid_phone_format(phone: Optional[str]) -> bool:
    if not phone:
        return False
    return bool(re.match(r"^\+\d{7,15}$", phone.strip()))


def to_riyadh(utc_string: Optional[str]) -> Optional[str]:
    """UTC ISO string -> Asia/Riyadh (+3h) ISO string."""

    if not utc_string:
        return None

    cleaned = utc_string.replace("Z", "")

    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(cleaned, fmt)
            break
        except ValueError:
            continue
    else:
        try:
            dt = datetime.fromisoformat(cleaned)
        except ValueError:
            return utc_string

    riyadh = dt + timedelta(hours=BOOKING_TIME_UTC_OFFSET_HOURS)
    return riyadh.isoformat() + "+03:00"


def _display_time_12h(iso_string: Optional[str]) -> str:
    """12-hour AM/PM display string - DATA, not a sentence, so tools may
    still compute it (an LLM doing manual date arithmetic is unreliable;
    this is exactly why the hard rule in prompts.py tells it to use this
    field instead of formatting timestamps itself)."""

    if not iso_string:
        return "-"
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "").split("+")[0])
    except ValueError:
        return iso_string
    return dt.strftime("%I:%M %p").lstrip("0") or dt.strftime("%I:%M %p")


def _display_date(iso_string: Optional[str]) -> str:
    if not iso_string:
        return "-"
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "").split("+")[0])
    except ValueError:
        return iso_string
    return dt.strftime("%d/%m/%Y")


_FIELD_MAP = (
    ("ref", ("bookingRefNum",)),
    ("servicePrice", ("servicePrice",)),
    ("patientFullName", ("patientFullName",)),
    ("mobileNumber", ("mobileNumber",)),
    ("email", ("email",)),
    ("statusName", ("statusName",)),
    ("branchName", ("branchName",)),
    ("doctorName", ("doctorName",)),
    ("serviceName", ("serviceName",)),
    ("specialtyName", ("specialtyName",)),
)


def _shape_appointment(item: dict) -> dict:
    """Flatten one raw API booking item into plain data fields - no
    sentences, just values, for the LLM to reference directly."""

    shaped = {}
    for name, keys in _FIELD_MAP:
        for key in keys:
            if key in item:
                shaped[name] = item[key]
                break

    riyadh_from = to_riyadh(item.get("bookingTimeFrom"))
    riyadh_to = to_riyadh(item.get("bookingTimeTo"))

    shaped["bookingTimeFrom"] = riyadh_from
    shaped["bookingTimeTo"] = riyadh_to
    shaped["date_display"] = _display_date(riyadh_from)
    shaped["time_display"] = _display_time_12h(riyadh_from)
    shaped["id"] = item.get("id")
    shaped["status"] = item.get("status")

    return shaped


def _filter_active(items: list) -> list:
    """Phone-path-only filter (preserved exactly from the old tools.py):
    excludes already-cancelled and past bookings. The reference-number
    path does NOT apply this filter - that asymmetry is intentional,
    carried over unchanged from the original business logic."""

    now = datetime.utcnow()
    active = []
    for item in items:
        if item.get("status") == CANCELLED_STATUS_CODE:
            continue
        raw_from = item.get("bookingTimeFrom")
        if not raw_from:
            continue
        try:
            dt = datetime.fromisoformat(raw_from.replace("Z", ""))
        except ValueError:
            continue
        if dt > now:
            active.append(item)
    return active


def _base_url(state: AgentState) -> str:
    return state.get("templates", {}).get("_base_url") or "https://demo.catalystsystems.io:1102"


# ==========================================================
# Tools - each returns STATUS + DATA ONLY, never a sentence
# ==========================================================

@tool
def validate_phone_format(phone: str) -> dict:
    """Validate that a phone number is in international format
    (starts with + and a country code). Returns {"status": "valid",
    "normalized": "+201234567890"} or {"status": "invalid"}."""

    if not _is_valid_phone_format(phone):
        return {"status": "invalid"}

    return {"status": "valid", "normalized": normalize_phone_number(phone)}


@tool
def compare_phone(provided_phone: str, channel_phone: str = "") -> dict:
    """Compare a user-provided phone number against the verified channel
    identity phone number (if any). Returns {"status": "match"} or
    {"status": "no_match"}. Never decide this yourself - always call
    this tool."""

    a = normalize_phone_number(provided_phone)
    b = normalize_phone_number(channel_phone) if channel_phone else None

    if a and b and a == b:
        return {"status": "match"}

    return {"status": "no_match"}


@tool
def lookup_appointment(
    state: Annotated[AgentState, InjectedState],
    ref_number: str = "",
    phone: str = "",
) -> dict:
    """Look up bookings by reference number OR phone number (whichever is
    given). Returns one of:
    {"status": "not_found"}
    {"status": "found_one", "appointment": {...}}
    {"status": "found_many", "appointments": [...]}
    Appointment fields: ref, doctorName, branchName, serviceName,
    specialtyName, statusName, date_display, time_display, patientFullName,
    mobileNumber, email, id."""

    base_url = _base_url(state)

    if ref_number:
        result = api.get_bookings_by_ref(base_url, ref_number)
    elif phone:
        result = api.get_bookings_by_phone(base_url, normalize_phone_number(phone))
    else:
        return {"status": "not_found"}

    if not result["success"]:
        return {"status": "not_found"}

    items = (result["data"] or {}).get("items", [])

    if not items:
        return {"status": "not_found"}

    if phone:
        # Phone path applies the active-only filter; ref path does not -
        # exact same asymmetry as the original business logic.
        items = _filter_active(items)
        if not items:
            return {"status": "not_found"}

    shaped = [_shape_appointment(i) for i in items]

    if len(shaped) == 1:
        return {"status": "found_one", "appointment": shaped[0]}

    return {"status": "found_many", "appointments": shaped}


@tool
def check_booking_status(
    state: Annotated[AgentState, InjectedState],
    ref_number: str,
) -> dict:
    """Re-fetch a booking by its reference number IMMEDIATELY before
    cancelling it - never trust anything earlier in the conversation as
    still current. Returns:
    {"status": "active", "appointment": {...}}
    {"status": "already_cancelled", "appointment": {...}}
    {"status": "not_found"}"""

    base_url = _base_url(state)
    result = api.get_bookings_by_ref(base_url, ref_number)

    if not result["success"]:
        return {"status": "not_found"}

    items = (result["data"] or {}).get("items", [])
    if not items:
        return {"status": "not_found"}

    appt = _shape_appointment(items[0])

    if appt.get("statusName") == CANCELLED_STATUS_NAME:
        return {"status": "already_cancelled", "appointment": appt}

    return {"status": "active", "appointment": appt}


@tool
def cancel_appointment(
    state: Annotated[AgentState, InjectedState],
    booking_id: str,
) -> dict:
    """Cancel a booking by its internal id (from a previous tool's
    "appointment"/"id" field - NEVER the human-readable reference
    number). Always call check_booking_status on the same booking
    immediately before this. Returns {"status": "success"} or
    {"status": "error"}."""

    base_url = _base_url(state)
    result = api.cancel_booking_by_guid(base_url, booking_id)

    if result["success"]:
        return {"status": "success"}

    return {"status": "error"}


# ==========================================================
# OTP (dummy provider by default, Authentica when configured) - internal
# mechanics unchanged from the old tools.py, only the return shape changed
# ==========================================================

_otp_storage: Dict[str, dict] = {}


@tool
def send_otp(phone: str) -> dict:
    """Send an OTP code to the given phone number (the number ON FILE
    for the booking, not necessarily what the user typed). Returns
    {"status": "otp_sent"}."""

    normalized = normalize_phone_number(phone)

    if OTP_PROVIDER == "authentica":
        api.authentica_send_otp(normalized)
        return {"status": "otp_sent"}

    _otp_storage[normalized] = {"otp": TEST_OTP, "created_at": time.time()}
    logger.info("OTP sent for %s (test otp=%s)", normalized, TEST_OTP)
    return {"status": "otp_sent"}


@tool
def verify_otp(phone: str, otp: str) -> dict:
    """Verify a user-entered OTP code against the one sent to `phone`.
    Returns {"status": "otp_valid"} or {"status": "otp_invalid"}."""

    normalized = normalize_phone_number(phone)

    if OTP_PROVIDER == "authentica":
        result = api.authentica_verify_otp(normalized, otp)
        return {"status": "otp_valid" if result["success"] else "otp_invalid"}

    record = _otp_storage.get(normalized)

    if not record:
        return {"status": "otp_invalid"}

    if time.time() - record["created_at"] > OTP_TTL_SECONDS:
        return {"status": "otp_invalid"}

    if str(otp).strip() == str(record["otp"]):
        return {"status": "otp_valid"}

    return {"status": "otp_invalid"}


ALL_TOOLS = [
    validate_phone_format,
    compare_phone,
    lookup_appointment,
    check_booking_status,
    cancel_appointment,
    send_otp,
    verify_otp,
]
