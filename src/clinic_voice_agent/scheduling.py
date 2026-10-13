"""Mock appointment calendar: the system of record the agent's tools act on.

All times are naive datetimes in the clinic's local timezone (the config's `timezone`). Every
rule a real practice-management system would enforce is checked here, not in the LLM prompt:
opening hours, lunch break, slot grid, provider skills, double booking, booking horizon and
identity checks before changes. The LLM can only *ask* for an action; the calendar decides.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from clinic_voice_agent.config import WEEKDAYS, ClinicConfig


class BookingError(ValueError):
    """A request the calendar refuses. The message is safe to read back to the caller."""


@dataclass
class Appointment:
    id: str
    patient_name: str
    date_of_birth: str  # ISO date
    phone: str
    type: str
    provider: str
    start: str  # ISO local datetime, minutes precision
    minutes: int
    status: str = "booked"  # booked | cancelled
    history: list[str] = field(default_factory=list)

    @property
    def start_dt(self) -> datetime:
        return datetime.fromisoformat(self.start)

    @property
    def end_dt(self) -> datetime:
        return self.start_dt + timedelta(minutes=self.minutes)


@dataclass
class Message:
    id: str
    caller_name: str
    phone: str
    text: str
    for_staff: str | None
    urgent: bool
    created: str


@dataclass(frozen=True)
class Slot:
    start: datetime
    provider: str


def _norm_name(name: str) -> str:
    return " ".join(name.lower().split())


def speak_datetime(dt: datetime) -> str:
    """Format a time the way a receptionist says it: 'Tuesday 14 October at 9:30'."""
    return f"{dt:%A} {dt.day} {dt:%B} at {dt.hour}:{dt.minute:02d}"


class Calendar:
    def __init__(
        self,
        cfg: ClinicConfig,
        path: Path | str | None = None,
        now: Callable[[], datetime] | None = None,
    ):
        self.cfg = cfg
        self.path = Path(path) if path else None
        tz = ZoneInfo(cfg.timezone)
        self._now = now or (lambda: datetime.now(tz).replace(tzinfo=None, second=0, microsecond=0))
        self.appointments: dict[str, Appointment] = {}
        self.messages: list[Message] = []
        self._next_id = 1001
        if self.path and self.path.exists():
            self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.appointments = {a["id"]: Appointment(**a) for a in data["appointments"]}
        self.messages = [Message(**m) for m in data["messages"]]
        self._next_id = data["next_id"]

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "appointments": [asdict(a) for a in self.appointments.values()],
            "messages": [asdict(m) for m in self.messages],
            "next_id": self._next_id,
        }
        self.path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    def _new_id(self, prefix: str) -> str:
        ident = f"{prefix}{self._next_id}"
        self._next_id += 1
        return ident

    # ------------------------------------------------------------------ rules
    def now(self) -> datetime:
        return self._now()

    def _check_slot(self, type_id: str, start: datetime, provider: str, ignore: str | None = None):
        cfg = self.cfg
        if type_id not in cfg.appointment_types:
            raise BookingError(f"Unknown appointment type '{type_id}'.")
        appt_type = cfg.appointment_types[type_id]
        if provider not in appt_type.providers:
            who = cfg.providers[provider].name if provider in cfg.providers else provider
            raise BookingError(f"{who} does not do {appt_type.label.lower()} appointments.")
        now = self.now()
        if start <= now:
            raise BookingError("That time is in the past.")
        if start.date() > (now + timedelta(days=cfg.booking_horizon_days)).date():
            raise BookingError(
                f"We only book up to {cfg.booking_horizon_days} days ahead."
            )
        if start.minute % cfg.slot_minutes or start.second:
            raise BookingError(f"Appointments start every {cfg.slot_minutes} minutes.")
        day = WEEKDAYS[start.weekday()]
        if day not in cfg.hours:
            raise BookingError(f"The clinic is closed on {day.capitalize()}s.")
        opens, closes = cfg.hours[day]
        end = start + timedelta(minutes=appt_type.minutes)
        if start.time() < opens or end.time() > closes or end.date() != start.date():
            raise BookingError(
                f"On {day.capitalize()}s we see patients from {opens:%H:%M} to {closes:%H:%M}."
            )
        if cfg.lunch_break:
            lunch_start = datetime.combine(start.date(), cfg.lunch_break[0])
            lunch_end = datetime.combine(start.date(), cfg.lunch_break[1])
            if start < lunch_end and end > lunch_start:
                raise BookingError("That time falls in the lunch break.")
        if appt_type.latest_start and start.time() > appt_type.latest_start:
            raise BookingError(
                f"{appt_type.label} appointments start no later than "
                f"{appt_type.latest_start:%H:%M}."
            )
        for other in self.appointments.values():
            if other.status != "booked" or other.id == ignore or other.provider != provider:
                continue
            if start < other.end_dt and end > other.start_dt:
                raise BookingError(
                    f"{cfg.provider_name(provider)} is already booked at that time."
                )

    def is_free(self, type_id: str, start: datetime, provider: str) -> bool:
        try:
            self._check_slot(type_id, start, provider)
        except BookingError:
            return False
        return True

    # ------------------------------------------------------------------ queries
    def find_slots(
        self,
        type_id: str,
        day: date | None = None,
        provider: str | None = None,
        after: datetime | None = None,
        limit: int = 5,
    ) -> list[Slot]:
        """Earliest free slots, optionally on one day and/or with one provider."""
        if type_id not in self.cfg.appointment_types:
            raise BookingError(f"Unknown appointment type '{type_id}'.")
        candidates = self.cfg.appointment_types[type_id].providers
        if provider is not None:
            if provider not in self.cfg.providers:
                raise BookingError(f"Unknown provider '{provider}'.")
            candidates = (provider,)
        now = self.now()
        start_day = day or now.date()
        last_day = day or (now + timedelta(days=self.cfg.booking_horizon_days)).date()
        step = timedelta(minutes=self.cfg.slot_minutes)
        slots: list[Slot] = []
        current = start_day
        while current <= last_day and len(slots) < limit:
            hours = self.cfg.hours.get(WEEKDAYS[current.weekday()])
            if hours:
                t = datetime.combine(current, hours[0])
                close = datetime.combine(current, hours[1])
                while t < close and len(slots) < limit:
                    if after is None or t >= after:
                        for prov in candidates:
                            if self.is_free(type_id, t, prov):
                                slots.append(Slot(t, prov))
                                break
                    t += step
            current += timedelta(days=1)
        return slots

    def find_appointments(self, patient_name: str, date_of_birth: str) -> list[Appointment]:
        """Upcoming booked appointments for a patient (name + date of birth must both match)."""
        now = self.now()
        return sorted(
            (
                a
                for a in self.appointments.values()
                if a.status == "booked"
                and _norm_name(a.patient_name) == _norm_name(patient_name)
                and a.date_of_birth == date_of_birth
                and a.start_dt > now
            ),
            key=lambda a: a.start,
        )

    # ------------------------------------------------------------------ changes
    def book(
        self,
        patient_name: str,
        date_of_birth: str,
        phone: str,
        type_id: str,
        start: datetime,
        provider: str | None = None,
    ) -> Appointment:
        if not patient_name.strip():
            raise BookingError("I need the patient's name to book.")
        if type_id not in self.cfg.appointment_types:
            raise BookingError(f"Unknown appointment type '{type_id}'.")
        if provider is None:
            candidates = self.cfg.appointment_types[type_id].providers
            free = [p for p in candidates if self.is_free(type_id, start, p)]
            # No one free: fall through with the first provider so the error says why.
            provider = free[0] if free else candidates[0]
        self._check_slot(type_id, start, provider)
        appt = Appointment(
            id=self._new_id("A"),
            patient_name=" ".join(patient_name.split()),
            date_of_birth=date_of_birth,
            phone=phone,
            type=type_id,
            provider=provider,
            start=start.isoformat(timespec="minutes"),
            minutes=self.cfg.appointment_types[type_id].minutes,
        )
        appt.history.append(f"booked {appt.start}")
        self.appointments[appt.id] = appt
        self.save()
        return appt

    def _verified(self, appointment_id: str, date_of_birth: str) -> Appointment:
        appt = self.appointments.get(appointment_id.strip().upper())
        # Same message for "no such id" and "wrong date of birth": don't leak which one failed.
        if appt is None or appt.date_of_birth != date_of_birth or appt.status != "booked":
            raise BookingError("I couldn't find a booked appointment with those details.")
        return appt

    def reschedule(self, appointment_id: str, date_of_birth: str, new_start: datetime):
        appt = self._verified(appointment_id, date_of_birth)
        self._check_slot(appt.type, new_start, appt.provider, ignore=appt.id)
        appt.history.append(f"moved {appt.start} -> {new_start.isoformat(timespec='minutes')}")
        appt.start = new_start.isoformat(timespec="minutes")
        self.save()
        return appt

    def cancel(self, appointment_id: str, date_of_birth: str) -> Appointment:
        appt = self._verified(appointment_id, date_of_birth)
        appt.status = "cancelled"
        appt.history.append("cancelled")
        self.save()
        return appt

    def take_message(
        self, caller_name: str, phone: str, text: str, for_staff: str | None, urgent: bool
    ) -> Message:
        if not text.strip():
            raise BookingError("The message is empty.")
        msg = Message(
            id=self._new_id("M"),
            caller_name=caller_name,
            phone=phone,
            text=text.strip(),
            for_staff=for_staff,
            urgent=urgent,
            created=self.now().isoformat(timespec="minutes"),
        )
        self.messages.append(msg)
        self.save()
        return msg


FAKE_NAMES = (
    "Lena Fischer", "Jonas Becker", "Sara Novak", "Ali Demir", "Emma Schulz", "Noah Wagner",
    "Mia Hoffmann", "Ben Richter", "Hana Sato", "Luca Romano", "Ines Costa", "Tom Klein",
)


def seed_demo(calendar: Calendar, n: int = 60, seed: int = 7) -> int:
    """Fill the calendar with synthetic bookings so availability looks like a real week."""
    rng = random.Random(seed)
    types = list(calendar.cfg.appointment_types)
    booked = 0
    for _ in range(n * 10):
        if booked >= n:
            break
        day = calendar.now().date() + timedelta(days=rng.randint(0, 13))
        hours = calendar.cfg.hours.get(WEEKDAYS[day.weekday()])
        if not hours:
            continue
        open_min = hours[0].hour * 60 + hours[0].minute
        close_min = hours[1].hour * 60 + hours[1].minute
        minute = rng.randrange(open_min, close_min, calendar.cfg.slot_minutes)
        start = datetime.combine(day, datetime.min.time()) + timedelta(minutes=minute)
        type_id = rng.choice(types)
        provider = rng.choice(calendar.cfg.appointment_types[type_id].providers)
        if not calendar.is_free(type_id, start, provider):
            continue
        name = rng.choice(FAKE_NAMES)
        dob = date(rng.randint(1940, 2015), rng.randint(1, 12), rng.randint(1, 28)).isoformat()
        calendar.book(name, dob, f"+49 30 0000 {rng.randint(1000, 9999)}", type_id, start, provider)
        booked += 1
    return booked
