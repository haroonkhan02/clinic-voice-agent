"""Tools the LLM can call, plus a validating executor.

`TOOL_SPECS` are Claude tool definitions (`name`, `description`, `input_schema`) with
`strict: true`, so the model's arguments always match the schema. Every property is listed
in `required`; optional values are nullable instead of omitted, which keeps strict mode simple.

`ToolExecutor.run()` never raises: it returns a JSON-able dict with `ok` plus either a result or
an `error` the model can read back to the caller. Arguments are validated again here
(dates, times, ids) because the executor is the trust boundary, whatever produced the call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from clinic_voice_agent.config import ClinicConfig
from clinic_voice_agent.knowledge import BM25Retriever
from clinic_voice_agent.scheduling import BookingError, Calendar, speak_datetime

NULLABLE_STR = {"type": ["string", "null"]}
DATE_DESC = "Date as YYYY-MM-DD."
START_DESC = "Local start time as YYYY-MM-DDTHH:MM (24 h)."


def _tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


def build_tool_specs(cfg: ClinicConfig) -> list[dict[str, Any]]:
    """Claude tool definitions. Enums come from the clinic config so they never drift."""
    types = sorted(cfg.appointment_types)
    providers = sorted(cfg.providers)
    type_desc = ", ".join(f"{t} = {cfg.appointment_types[t].label}" for t in types)
    provider_desc = ", ".join(f"{p} = {cfg.providers[p].name}" for p in providers)
    appt_type = {"type": "string", "enum": types, "description": type_desc}
    provider = {
        "type": ["string", "null"],
        "enum": [*providers, None],
        "description": f"Optional provider: {provider_desc}. null = anyone available.",
    }
    return [
        _tool(
            "search_faq",
            "Look up clinic information (hours, address, parking, preparation, policies). "
            "Answer only from the returned passages; if none fits, offer to take a message.",
            {"question": {"type": "string", "description": "The caller's question."}},
        ),
        _tool(
            "find_available_slots",
            "List the earliest free appointment slots. Use before offering times to the caller.",
            {
                "appointment_type": appt_type,
                "date": {**NULLABLE_STR, "description": DATE_DESC + " null = from today."},
                "provider": provider,
                "earliest_time": {
                    **NULLABLE_STR,
                    "description": "Only slots at or after this HH:MM on the given date.",
                },
            },
        ),
        _tool(
            "book_appointment",
            "Book an appointment. Only call after reading the details back to the caller "
            "and hearing a clear yes; set caller_confirmed accordingly.",
            {
                "patient_name": {"type": "string"},
                "date_of_birth": {"type": "string", "description": DATE_DESC},
                "phone": {"type": "string", "description": "Callback number."},
                "appointment_type": appt_type,
                "start": {"type": "string", "description": START_DESC},
                "provider": provider,
                "caller_confirmed": {
                    "type": "boolean",
                    "description": "True only if the caller confirmed these exact details.",
                },
            },
        ),
        _tool(
            "find_appointment",
            "Find a patient's upcoming appointments by full name and date of birth.",
            {
                "patient_name": {"type": "string"},
                "date_of_birth": {"type": "string", "description": DATE_DESC},
            },
        ),
        _tool(
            "reschedule_appointment",
            "Move an existing appointment to a new free time (after the caller confirms).",
            {
                "appointment_id": {"type": "string", "description": "For example A1004."},
                "date_of_birth": {"type": "string", "description": DATE_DESC},
                "new_start": {"type": "string", "description": START_DESC},
                "caller_confirmed": {"type": "boolean"},
            },
        ),
        _tool(
            "cancel_appointment",
            "Cancel an existing appointment (after the caller confirms).",
            {
                "appointment_id": {"type": "string"},
                "date_of_birth": {"type": "string", "description": DATE_DESC},
                "caller_confirmed": {"type": "boolean"},
            },
        ),
        _tool(
            "take_message",
            "Leave a message for the clinic team or a specific staff member.",
            {
                "caller_name": {"type": "string"},
                "phone": {"type": "string"},
                "message": {"type": "string", "description": "Short summary, no diagnosis."},
                "for_staff": provider,
                "urgent": {"type": "boolean"},
            },
        ),
        _tool(
            "transfer_to_human",
            "Hand the call to a member of staff (caller asks for a person, is upset, or the "
            "request is outside what you can do).",
            {"reason": {"type": "string"}},
        ),
    ]


_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME = re.compile(r"^\d{2}:\d{2}$")


class ToolInputError(ValueError):
    pass


def _parse_date(value: Any, field_name: str) -> date:
    if not isinstance(value, str) or not _DATE.match(value):
        raise ToolInputError(f"{field_name} must be YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ToolInputError(f"{field_name} is not a real date") from exc


def _parse_start(value: Any, field_name: str = "start") -> datetime:
    if not isinstance(value, str) or not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$", value):
        raise ToolInputError(f"{field_name} must be YYYY-MM-DDTHH:MM")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ToolInputError(f"{field_name} is not a real date/time") from exc


def _require_str(args: dict, name: str) -> str:
    value = args.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ToolInputError(f"{name} is required")
    return value.strip()


@dataclass
class ToolExecutor:
    cfg: ClinicConfig
    calendar: Calendar
    retriever: BM25Retriever
    faq_k: int = 3
    faq_min_score: float = 0.0
    transfers: list[str] = field(default_factory=list)
    log: list[dict[str, Any]] = field(default_factory=list)

    def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            result = {"ok": False, "error": f"unknown tool {name!r}"}
        else:
            try:
                result = {"ok": True, **handler(args or {})}
            except (ToolInputError, BookingError) as exc:
                result = {"ok": False, "error": str(exc)}
        self.log.append({"tool": name, "args": args, "result": result})
        return result

    # ------------------------------------------------------------------ handlers
    def _tool_search_faq(self, args):
        hits = self.retriever.search(
            _require_str(args, "question"), k=self.faq_k, min_score=self.faq_min_score
        )
        return {
            "passages": [
                {"title": h.passage.title, "text": h.passage.text, "score": round(h.score, 2)}
                for h in hits
            ]
        }

    def _tool_find_available_slots(self, args):
        day = _parse_date(args["date"], "date") if args.get("date") else None
        after = None
        if args.get("earliest_time"):
            if day is None or not _TIME.match(args["earliest_time"]):
                raise ToolInputError("earliest_time needs a date and HH:MM")
            after = datetime.fromisoformat(f"{day.isoformat()}T{args['earliest_time']}")
        slots = self.calendar.find_slots(
            args.get("appointment_type", ""), day=day, provider=args.get("provider"), after=after
        )
        return {
            "slots": [
                {
                    "start": s.start.isoformat(timespec="minutes"),
                    "spoken": speak_datetime(s.start),
                    "provider": s.provider,
                    "provider_name": self.cfg.provider_name(s.provider),
                }
                for s in slots
            ]
        }

    def _appointment_view(self, appt) -> dict[str, Any]:
        return {
            "appointment_id": appt.id,
            "type": self.cfg.appointment_types[appt.type].label,
            "start": appt.start,
            "spoken": speak_datetime(appt.start_dt),
            "provider_name": self.cfg.provider_name(appt.provider),
            "status": appt.status,
        }

    @staticmethod
    def _require_confirmation(args):
        if args.get("caller_confirmed") is not True:
            raise ToolInputError(
                "not confirmed: read the details back to the caller and ask them to confirm"
            )

    def _tool_book_appointment(self, args):
        self._require_confirmation(args)
        appt = self.calendar.book(
            patient_name=_require_str(args, "patient_name"),
            date_of_birth=_parse_date(args.get("date_of_birth"), "date_of_birth").isoformat(),
            phone=_require_str(args, "phone"),
            type_id=_require_str(args, "appointment_type"),
            start=_parse_start(args.get("start")),
            provider=args.get("provider"),
        )
        return {"appointment": self._appointment_view(appt)}

    def _tool_find_appointment(self, args):
        found = self.calendar.find_appointments(
            _require_str(args, "patient_name"),
            _parse_date(args.get("date_of_birth"), "date_of_birth").isoformat(),
        )
        return {"appointments": [self._appointment_view(a) for a in found]}

    def _tool_reschedule_appointment(self, args):
        self._require_confirmation(args)
        appt = self.calendar.reschedule(
            _require_str(args, "appointment_id"),
            _parse_date(args.get("date_of_birth"), "date_of_birth").isoformat(),
            _parse_start(args.get("new_start"), "new_start"),
        )
        return {"appointment": self._appointment_view(appt)}

    def _tool_cancel_appointment(self, args):
        self._require_confirmation(args)
        appt = self.calendar.cancel(
            _require_str(args, "appointment_id"),
            _parse_date(args.get("date_of_birth"), "date_of_birth").isoformat(),
        )
        return {"appointment": self._appointment_view(appt)}

    def _tool_take_message(self, args):
        msg = self.calendar.take_message(
            caller_name=_require_str(args, "caller_name"),
            phone=_require_str(args, "phone"),
            text=_require_str(args, "message"),
            for_staff=args.get("for_staff"),
            urgent=bool(args.get("urgent")),
        )
        return {"message_id": msg.id}

    def _tool_transfer_to_human(self, args):
        reason = _require_str(args, "reason")
        self.transfers.append(reason)
        return {"transferred": True}
