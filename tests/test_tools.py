import json

import pytest

from clinic_voice_agent.knowledge import load_retriever
from clinic_voice_agent.tools import ToolExecutor, build_tool_specs

DOB = "1985-03-02"


@pytest.fixture
def ex(cfg, calendar):
    return ToolExecutor(cfg, calendar, load_retriever())


def booking(**overrides):
    args = {
        "patient_name": "Test Patient",
        "date_of_birth": DOB,
        "phone": "+49 30 0000 1111",
        "appointment_type": "general",
        "start": "2026-10-13T10:00",
        "provider": None,
        "caller_confirmed": True,
    }
    return {**args, **overrides}


def test_specs_are_strict_and_complete(cfg):
    specs = build_tool_specs(cfg)
    names = {s["name"] for s in specs}
    assert {"search_faq", "book_appointment", "cancel_appointment", "transfer_to_human"} <= names
    for spec in specs:
        schema = spec["input_schema"]
        assert spec["strict"] is True
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
    json.dumps(specs)  # must be serialisable for the API


def test_every_spec_has_a_handler(cfg, ex):
    for spec in build_tool_specs(cfg):
        assert hasattr(ex, f"_tool_{spec['name']}")


def test_booking_requires_confirmation(ex, calendar):
    result = ex.run("book_appointment", booking(caller_confirmed=False))
    assert not result["ok"] and "confirm" in result["error"]
    assert calendar.appointments == {}
    result = ex.run("book_appointment", booking())
    assert result["ok"]
    assert result["appointment"]["spoken"] == "Tuesday 13 October at 10:00"


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"date_of_birth": "02/03/1985"}, "YYYY-MM-DD"),
        ({"date_of_birth": "1985-02-30"}, "real date"),
        ({"start": "tomorrow at ten"}, "YYYY-MM-DDTHH:MM"),
        ({"patient_name": "  "}, "patient_name is required"),
        ({"start": "2026-10-17T10:00"}, "closed on Saturdays"),
    ],
)
def test_bad_inputs_become_readable_errors(ex, override, error):
    result = ex.run("book_appointment", booking(**override))
    assert result["ok"] is False
    assert error in result["error"]


def test_find_slots_and_faq(ex):
    result = ex.run(
        "find_available_slots",
        {"appointment_type": "vaccination", "date": "2026-10-13", "provider": None,
         "earliest_time": "11:00"},
    )
    assert result["ok"] and result["slots"][0]["start"] == "2026-10-13T11:00"
    assert result["slots"][0]["provider_name"] == "Nurse Mei Lin"
    faq = ex.run("search_faq", {"question": "where can I park"})
    assert faq["passages"][0]["title"] == "Parking"


def test_earliest_time_without_date_rejected(ex):
    result = ex.run(
        "find_available_slots",
        {"appointment_type": "general", "date": None, "provider": None, "earliest_time": "11:00"},
    )
    assert not result["ok"]


def test_find_reschedule_cancel_flow(ex):
    appt_id = ex.run("book_appointment", booking())["appointment"]["appointment_id"]
    found = ex.run("find_appointment", {"patient_name": "test patient", "date_of_birth": DOB})
    assert [a["appointment_id"] for a in found["appointments"]] == [appt_id]
    moved = ex.run(
        "reschedule_appointment",
        {"appointment_id": appt_id, "date_of_birth": DOB, "new_start": "2026-10-15T09:00",
         "caller_confirmed": True},
    )
    assert moved["appointment"]["start"] == "2026-10-15T09:00"
    wrong = ex.run(
        "cancel_appointment",
        {"appointment_id": appt_id, "date_of_birth": "1990-01-01", "caller_confirmed": True},
    )
    assert not wrong["ok"]
    done = ex.run(
        "cancel_appointment",
        {"appointment_id": appt_id, "date_of_birth": DOB, "caller_confirmed": True},
    )
    assert done["appointment"]["status"] == "cancelled"


def test_message_transfer_and_unknown_tool(ex, calendar):
    msg = ex.run(
        "take_message",
        {"caller_name": "Test Patient", "phone": "+49 30 0000 1111",
         "message": "Please call back about my referral", "for_staff": "dr_weber",
         "urgent": False},
    )
    assert msg["ok"] and calendar.messages[0].for_staff == "dr_weber"
    assert ex.run("transfer_to_human", {"reason": "caller asked for a person"})["ok"]
    assert ex.transfers == ["caller asked for a person"]
    assert ex.run("delete_database", {}) == {"ok": False, "error": "unknown tool 'delete_database'"}
    assert len(ex.log) == 3
