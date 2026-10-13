"""Paths, clinic configuration and service-key discovery.

Secrets are read only from environment variables (optionally loaded from a git-ignored
`.env`). Nothing in this module ever returns or logs a key's value: callers only learn
*whether* a variable is set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
KNOWLEDGE_DIR = ROOT / "knowledge"
REPORTS_DIR = ROOT / "reports"
OUTPUTS_DIR = ROOT / "outputs"

DEFAULT_CLINIC_CONFIG = CONFIG_DIR / "clinic.yaml"
DEFAULT_FAQ = KNOWLEDGE_DIR / "faq.md"

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

# Which env vars unlock which service. The agent picks the first available option per stage
# and falls back to local models when none is set.
SERVICE_KEYS: dict[str, tuple[str, ...]] = {
    "llm": ("ANTHROPIC_API_KEY",),
    "stt": ("DEEPGRAM_API_KEY",),
    "tts": ("DEEPGRAM_API_KEY", "CARTESIA_API_KEY", "ELEVENLABS_API_KEY"),
    "telephony": ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"),
}


def _parse_time(value: str) -> time:
    hours, minutes = value.split(":")
    return time(int(hours), int(minutes))


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    role: str


@dataclass(frozen=True)
class AppointmentType:
    id: str
    label: str
    minutes: int
    providers: tuple[str, ...]
    latest_start: time | None = None


@dataclass(frozen=True)
class ClinicConfig:
    name: str
    phone: str
    address: str
    timezone: str
    emergency_number: str
    hours: dict[str, tuple[time, time]]
    lunch_break: tuple[time, time] | None
    slot_minutes: int
    booking_horizon_days: int
    providers: dict[str, Provider]
    appointment_types: dict[str, AppointmentType]
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def provider_name(self, provider_id: str) -> str:
        return self.providers[provider_id].name


def _validate(cfg: ClinicConfig) -> None:
    for day in cfg.hours:
        if day not in WEEKDAYS:
            raise ValueError(f"unknown weekday in hours: {day!r}")
    for day, (start, end) in cfg.hours.items():
        if start >= end:
            raise ValueError(f"opening hours for {day} end before they start")
    if cfg.slot_minutes <= 0 or 60 % cfg.slot_minutes:
        raise ValueError("slot_minutes must divide 60")
    for appt in cfg.appointment_types.values():
        if appt.minutes % cfg.slot_minutes:
            raise ValueError(f"{appt.id}: duration must be a multiple of slot_minutes")
        unknown = set(appt.providers) - set(cfg.providers)
        if unknown:
            raise ValueError(f"{appt.id}: unknown providers {sorted(unknown)}")


def load_clinic_config(path: Path | str = DEFAULT_CLINIC_CONFIG) -> ClinicConfig:
    """Load and validate the clinic YAML (hours, providers, appointment types)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    clinic = raw["clinic"]
    hours = {
        day.lower(): (_parse_time(span[0]), _parse_time(span[1]))
        for day, span in raw.get("hours", {}).items()
    }
    lunch = raw.get("lunch_break")
    providers = {p["id"]: Provider(p["id"], p["name"], p["role"]) for p in raw["providers"]}
    types = {
        type_id: AppointmentType(
            id=type_id,
            label=spec["label"],
            minutes=int(spec["minutes"]),
            providers=tuple(spec["providers"]),
            latest_start=_parse_time(spec["latest_start"]) if "latest_start" in spec else None,
        )
        for type_id, spec in raw["appointment_types"].items()
    }
    cfg = ClinicConfig(
        name=clinic["name"],
        phone=str(clinic["phone"]),
        address=clinic["address"],
        timezone=clinic["timezone"],
        emergency_number=str(clinic["emergency_number"]),
        hours=hours,
        lunch_break=(_parse_time(lunch[0]), _parse_time(lunch[1])) if lunch else None,
        slot_minutes=int(raw.get("slot_minutes", 15)),
        booking_horizon_days=int(raw.get("booking_horizon_days", 28)),
        providers=providers,
        appointment_types=types,
        raw=raw,
    )
    _validate(cfg)
    return cfg


def load_env(dotenv_path: Path | None = None) -> None:
    """Load `.env` into the process environment without overriding existing variables."""
    from dotenv import load_dotenv

    load_dotenv(dotenv_path or ROOT / ".env", override=False)


def key_present(name: str, environ: dict[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return bool(env.get(name, "").strip())


# Stages that need *all* of their variables (Twilio needs both SID and token).
REQUIRE_ALL = frozenset({"telephony"})


def available_services(environ: dict[str, str] | None = None) -> dict[str, list[str]]:
    """Map each pipeline stage to the env-var *names* that are set (never their values)."""
    return {
        stage: [name for name in names if key_present(name, environ)]
        for stage, names in SERVICE_KEYS.items()
    }


def stage_ready(stage: str, environ: dict[str, str] | None = None) -> bool:
    present = available_services(environ)[stage]
    if stage in REQUIRE_ALL:
        return len(present) == len(SERVICE_KEYS[stage])
    return bool(present)
