"""Scripted conversations used to evaluate the agent end to end in text mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from clinic_voice_agent.config import ROOT

DEFAULT_SCENARIOS = ROOT / "eval" / "scenarios.yaml"
EXPECT_KEYS = {"tools_called", "tools_forbidden", "guardrail", "booked", "cancelled", "messages"}


@dataclass
class Scenario:
    id: str
    caller: list[str]
    expect: dict[str, Any]
    setup_booking: dict[str, Any] | None = None
    now: datetime = field(default_factory=lambda: datetime(2026, 10, 12, 9, 0))


def load_scenarios(path: Path | str = DEFAULT_SCENARIOS, tool_names: set[str] | None = None):
    """Load and validate scenarios; `tool_names` checks every referenced tool exists."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    now = datetime.fromisoformat(data["now"])
    scenarios, seen = [], set()
    for raw in data["scenarios"]:
        sid = raw["id"]
        if sid in seen:
            raise ValueError(f"duplicate scenario id {sid!r}")
        seen.add(sid)
        if not raw.get("caller"):
            raise ValueError(f"{sid}: no caller turns")
        expect = raw.get("expect", {})
        unknown = set(expect) - EXPECT_KEYS
        if unknown:
            raise ValueError(f"{sid}: unknown expect keys {sorted(unknown)}")
        if tool_names is not None:
            for key in ("tools_called", "tools_forbidden"):
                missing = set(expect.get(key, [])) - tool_names
                if missing:
                    raise ValueError(f"{sid}: unknown tools {sorted(missing)}")
        scenarios.append(
            Scenario(sid, list(raw["caller"]), expect, raw.get("setup_booking"), now)
        )
    return scenarios
