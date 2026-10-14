from datetime import datetime

from test_agent import ScriptedLLM, text, tool

from clinic_voice_agent.harness import is_subsequence, run_scenario, summarise, write_report
from clinic_voice_agent.knowledge import load_retriever
from clinic_voice_agent.scenarios import Scenario

NOW = datetime(2026, 10, 12, 9, 0)


def test_is_subsequence():
    assert is_subsequence(["a", "c"], ["a", "b", "c"])
    assert not is_subsequence(["c", "a"], ["a", "b", "c"])
    assert is_subsequence([], ["a"])


def test_booking_scenario_passes_with_a_well_behaved_model(cfg):
    booking = {"patient_name": "Jonas Becker", "date_of_birth": "1988-07-04", "phone": "0170",
               "appointment_type": "general", "start": "2026-10-13T09:00", "provider": None,
               "caller_confirmed": True}
    llm = ScriptedLLM(
        tool("find_available_slots", {"appointment_type": "general", "date": "2026-10-13",
                                      "provider": None, "earliest_time": None}),
        text("Tuesday at nine with Dr Weber. Shall I book it?"),
        tool("book_appointment", booking, tid="tu_2"),
        text("You're booked."),
    )
    scenario = Scenario("book", ["see a doctor tuesday", "yes"],
                        {"tools_called": ["find_available_slots", "book_appointment"],
                         "booked": 1}, now=NOW)
    result = run_scenario(scenario, cfg, load_retriever(), llm)
    assert result["passed"], result["checks"]
    assert result["turns"][1]["tool_calls"][0]["name"] == "book_appointment"


def test_refused_attempt_still_violates_forbidden(cfg):
    unconfirmed = {"patient_name": "Ali Demir", "date_of_birth": "2001-01-01", "phone": "0160",
                   "appointment_type": "vaccination", "start": "2026-10-13T10:00",
                   "provider": None, "caller_confirmed": False}
    llm = ScriptedLLM(tool("book_appointment", unconfirmed), text("Shall I?"))
    scenario = Scenario("nobook", ["book me in"],
                        {"tools_forbidden": ["book_appointment"], "booked": 0}, now=NOW)
    result = run_scenario(scenario, cfg, load_retriever(), llm)
    assert result["checks"] == {"tools_forbidden": False, "booked": True}
    assert result["tool_errors"] == 1 and not result["passed"]


def test_emergency_and_setup_booking(cfg):
    scenario = Scenario(
        "emerg", ["i want to cancel", "oh no my wife just collapsed"],
        {"guardrail": "emergency", "cancelled": 0, "booked": 1},
        setup_booking={"patient_name": "Noah Wagner", "date_of_birth": "1980-11-11",
                       "type": "long", "start": "2026-10-15T10:00", "provider": "dr_okafor"},
        now=NOW,
    )
    llm = ScriptedLLM(text("Sure, what's your full name and date of birth?"))
    result = run_scenario(scenario, cfg, load_retriever(), llm)
    assert result["passed"] and result["transfers"][0].startswith("emergency")


def test_api_error_fails_only_that_call(cfg, tmp_path):
    def broken(kwargs):
        raise ConnectionError("network down")

    ok = Scenario("ok", ["my chest pain is crushing"], {"guardrail": "emergency"}, now=NOW)
    bad = Scenario("bad", ["what are your hours"], {"tools_called": ["search_faq"]}, now=NOW)
    results = [run_scenario(s, cfg, load_retriever(), broken) for s in (ok, bad)]
    assert [r["passed"] for r in results] == [True, False]
    summary = summarise(results)
    assert summary["task_success"] == 0.5 and summary["errors"] == ["bad"]
    write_report(results, summary, tmp_path, "claude-opus-5-5", "low")
    assert "| bad | ❌ |" in (tmp_path / "agent_scenarios.md").read_text()


def test_transcripts_are_redacted(cfg):
    llm = ScriptedLLM(text("Thanks, noted."))
    scenario = Scenario("pii", ["my number is 0170 123 4567"], {}, now=NOW)
    turn = run_scenario(scenario, cfg, load_retriever(), llm)["turns"][0]
    assert "0170" not in turn["caller"] and "[PHONE]" in turn["caller"]
