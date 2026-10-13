from datetime import datetime

import pytest

from clinic_voice_agent.cli import slots_main
from clinic_voice_agent.guardrails import screen
from clinic_voice_agent.scenarios import load_scenarios
from clinic_voice_agent.tools import build_tool_specs


def test_shipped_scenarios_are_valid(cfg):
    names = {s["name"] for s in build_tool_specs(cfg)}
    scenarios = load_scenarios(tool_names=names)
    assert len(scenarios) >= 10
    assert scenarios[0].now == datetime(2026, 10, 12, 9, 0)


def test_guardrail_scenarios_trigger_on_some_turn(cfg):
    for sc in load_scenarios():
        wanted = sc.expect.get("guardrail")
        if wanted:
            assert any(screen(turn).action == wanted for turn in sc.caller), sc.id


def test_unknown_tool_rejected(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(
        'now: "2026-10-12T09:00"\nscenarios:\n'
        "  - {id: x, caller: [hi], expect: {tools_called: [fly_to_moon]}}\n"
    )
    with pytest.raises(ValueError, match="unknown tools"):
        load_scenarios(path, tool_names={"search_faq"})


def test_slots_cli_runs(capsys):
    assert slots_main(["vaccination", "-n", "2", "--seed-bookings", "5"]) == 0
    assert "Nurse Mei Lin" in capsys.readouterr().out
