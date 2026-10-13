from datetime import date, datetime

import pytest

from clinic_voice_agent.scheduling import BookingError, Calendar, seed_demo, speak_datetime

TUE_10 = datetime(2026, 10, 13, 10, 0)
DOB = "1990-05-17"


def book(calendar, start=TUE_10, type_id="general", provider=None, name="Test Patient"):
    return calendar.book(name, DOB, "+49 30 0000 1234", type_id, start, provider)


def test_book_assigns_a_free_provider(calendar):
    first = book(calendar)
    second = book(calendar, name="Other Patient")
    assert first.id == "A1001" and second.id == "A1002"
    assert {first.provider, second.provider} == {"dr_weber", "dr_okafor"}
    with pytest.raises(BookingError, match="already booked"):
        book(calendar, name="Third Patient")


@pytest.mark.parametrize(
    ("start", "type_id", "reason"),
    [
        (datetime(2026, 10, 12, 8, 30), "general", "past"),
        (datetime(2026, 10, 17, 10, 0), "general", "closed on Saturdays"),
        (datetime(2026, 10, 14, 13, 0), "general", "from 08:00 to 13:00"),
        (datetime(2026, 10, 13, 12, 30), "general", "lunch break"),
        (datetime(2026, 10, 13, 12, 15), "long", "lunch break"),  # 30 min runs into lunch
        (datetime(2026, 10, 13, 10, 10), "general", "every 15 minutes"),
        (datetime(2026, 10, 13, 11, 0), "blood_test", "no later than 10:00"),
        (datetime(2026, 12, 1, 10, 0), "general", "28 days ahead"),
        (datetime(2026, 10, 16, 14, 45), "long", "from 08:00 to 15:00"),
    ],
)
def test_rules_are_enforced(calendar, start, type_id, reason):
    with pytest.raises(BookingError, match=reason):
        book(calendar, start=start, type_id=type_id)


def test_provider_must_offer_the_appointment_type(calendar):
    with pytest.raises(BookingError, match="does not do vaccination"):
        book(calendar, type_id="vaccination", provider="dr_weber")
    assert book(calendar, type_id="vaccination").provider == "nurse_lin"


def test_find_slots_skips_booked_and_past_times(calendar):
    slots = calendar.find_slots("blood_test", limit=3)
    # Monday 09:00 is "now", so the first bookable slot is 09:15.
    assert [s.start for s in slots] == [
        datetime(2026, 10, 12, 9, 15),
        datetime(2026, 10, 12, 9, 30),
        datetime(2026, 10, 12, 9, 45),
    ]
    book(calendar, start=datetime(2026, 10, 12, 9, 15), type_id="blood_test")
    assert calendar.find_slots("blood_test", limit=1)[0].start == datetime(2026, 10, 12, 9, 30)


def test_find_slots_on_a_closed_day_is_empty(calendar):
    assert calendar.find_slots("general", day=date(2026, 10, 18)) == []


def test_find_slots_with_provider_and_after(calendar):
    slots = calendar.find_slots(
        "general", provider="dr_okafor", after=datetime(2026, 10, 13, 16, 30), limit=2
    )
    assert [(s.start.hour, s.start.minute) for s in slots] == [(16, 30), (16, 45)]
    assert all(s.provider == "dr_okafor" for s in slots)


def test_reschedule_and_cancel_need_matching_date_of_birth(calendar):
    appt = book(calendar)
    with pytest.raises(BookingError, match="couldn't find"):
        calendar.cancel(appt.id, "2000-01-01")
    moved = calendar.reschedule(appt.id.lower(), DOB, datetime(2026, 10, 15, 9, 0))
    assert moved.start == "2026-10-15T09:00"
    # Rescheduling into its own old slot is fine (the appointment ignores itself).
    calendar.reschedule(appt.id, DOB, datetime(2026, 10, 15, 9, 15))
    assert calendar.cancel(appt.id, DOB).status == "cancelled"
    with pytest.raises(BookingError):
        calendar.cancel(appt.id, DOB)
    # The freed slot can be booked again.
    assert book(calendar, start=datetime(2026, 10, 15, 9, 15), provider=appt.provider)


def test_find_appointments_matches_name_case_insensitively(calendar):
    book(calendar, name="Lena  Fischer")
    assert len(calendar.find_appointments("lena fischer", DOB)) == 1
    assert calendar.find_appointments("lena fischer", "1999-01-01") == []


def test_persistence_round_trip(cfg, tmp_path):
    path = tmp_path / "calendar.json"
    now = lambda: datetime(2026, 10, 12, 9, 0)  # noqa: E731
    cal = Calendar(cfg, path, now=now)
    appt = book(cal)
    cal.take_message("Test Patient", "+49 30 0000 1234", "Please call back", "dr_weber", False)
    reloaded = Calendar(cfg, path, now=now)
    assert reloaded.appointments[appt.id].start == appt.start
    assert reloaded.messages[0].id == "M1002"
    assert book(reloaded, start=datetime(2026, 10, 13, 11, 0)).id == "A1003"


def test_seed_demo_is_deterministic_and_valid(cfg):
    a = Calendar(cfg, now=lambda: datetime(2026, 10, 12, 9, 0))
    b = Calendar(cfg, now=lambda: datetime(2026, 10, 12, 9, 0))
    assert seed_demo(a, n=40) == 40
    seed_demo(b, n=40)
    assert [x.start for x in a.appointments.values()] == [x.start for x in b.appointments.values()]


def test_speak_datetime():
    assert speak_datetime(datetime(2026, 10, 13, 9, 5)) == "Tuesday 13 October at 9:05"
