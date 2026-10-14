"""Text-mode harness: play scripted calls through the agent and score what they leave behind.

Each scenario runs on a fresh calendar with its own clock, so scenarios are independent. The
checks look at *effects* (tool log, calendar state, messages, guardrail actions), not at the
wording of the replies, so they work with any model and stay deterministic to score.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from clinic_voice_agent.agent import ClinicAgent, LLMCall
from clinic_voice_agent.config import ClinicConfig
from clinic_voice_agent.guardrails import redact
from clinic_voice_agent.knowledge import BM25Retriever
from clinic_voice_agent.scenarios import Scenario
from clinic_voice_agent.scheduling import Calendar
from clinic_voice_agent.tools import ToolExecutor


def is_subsequence(expected: list[str], actual: list[str]) -> bool:
    """True if `expected` appears in `actual` in order (other items may be interleaved)."""
    it = iter(actual)
    return all(name in it for name in expected)


def build_agent(
    cfg: ClinicConfig, retriever: BM25Retriever, llm_call: LLMCall, now: datetime
) -> ClinicAgent:
    calendar = Calendar(cfg, now=lambda: now)
    return ClinicAgent(cfg, ToolExecutor(cfg, calendar, retriever), llm_call=llm_call)


def score(scenario: Scenario, agent: ClinicAgent) -> dict[str, bool]:
    expect, log = scenario.expect, agent.executor.log
    ok_calls = [entry["tool"] for entry in log if entry["result"]["ok"]]
    attempted = {entry["tool"] for entry in log}
    appts = agent.executor.calendar.appointments.values()
    checks: dict[str, bool] = {}
    if "tools_called" in expect:
        checks["tools_called"] = is_subsequence(expect["tools_called"], ok_calls)
    if "tools_forbidden" in expect:
        # Strict: even an attempt the executor refused counts as a violation.
        checks["tools_forbidden"] = not (set(expect["tools_forbidden"]) & attempted)
    if "guardrail" in expect:
        checks["guardrail"] = any(t.guardrail == expect["guardrail"] for t in agent.turns)
    if "booked" in expect:
        booked = sum(a.status == "booked" for a in appts)
        checks["booked"] = booked == expect["booked"]
    if "cancelled" in expect:
        cancelled = sum(a.status == "cancelled" for a in appts)
        checks["cancelled"] = cancelled == expect["cancelled"]
    if "messages" in expect:
        checks["messages"] = len(agent.executor.calendar.messages) == expect["messages"]
    return checks


def run_scenario(
    scenario: Scenario, cfg: ClinicConfig, retriever: BM25Retriever, llm_call: LLMCall
) -> dict[str, Any]:
    agent = build_agent(cfg, retriever, llm_call, scenario.now)
    if scenario.setup_booking:
        b = scenario.setup_booking
        # Setup bookings count like any other appointment in the end state.
        agent.executor.calendar.book(
            patient_name=b["patient_name"],
            date_of_birth=b["date_of_birth"],
            phone="+49 30 0000 9999",
            type_id=b["type"],
            start=datetime.fromisoformat(b["start"]),
            provider=b.get("provider"),
        )
    error = None
    try:
        for line in scenario.caller:
            agent.respond(line)
    except Exception as exc:  # an API error ends the call; the scenario fails, the run goes on
        error = f"{type(exc).__name__}: {exc}"
    checks = score(scenario, agent)
    return {
        "id": scenario.id,
        "passed": error is None and all(checks.values()),
        "checks": checks,
        "error": error,
        "turns": [
            {**asdict(t), "caller": redact(t.caller), "reply": redact(t.reply)}
            for t in agent.turns
        ],
        "tool_errors": sum(not e["result"]["ok"] for e in agent.executor.log),
        "transfers": list(agent.executor.transfers),
    }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))]


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    check_totals: dict[str, list[bool]] = {}
    for r in results:
        for name, ok in r["checks"].items():
            check_totals.setdefault(name, []).append(ok)
    llm_turns = [t for r in results for t in r["turns"] if t["llm_calls"]]
    ttft = [t["ttft_s"] for t in llm_turns if t["ttft_s"] is not None]
    return {
        "n_scenarios": len(results),
        "task_success": sum(r["passed"] for r in results) / max(1, len(results)),
        "checks": {k: {"pass_rate": sum(v) / len(v), "n": len(v)} for k, v in check_totals.items()},
        "llm_turns": len(llm_turns),
        "llm_calls_per_turn": statistics.mean(t["llm_calls"] for t in llm_turns)
        if llm_turns
        else None,
        "ttft_median_s": statistics.median(ttft) if ttft else None,
        "ttft_p90_s": _percentile(ttft, 0.9),
        "llm_time_per_turn_median_s": statistics.median(t["llm_total_s"] for t in llm_turns)
        if llm_turns
        else None,
        "tool_errors": sum(r["tool_errors"] for r in results),
        "errors": [r["id"] for r in results if r["error"]],
    }


def write_report(results, summary, out_dir: Path, model: str, effort: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "effort": effort, "summary": summary, "results": results}
    (out_dir / "agent_scenarios.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    def fmt(x):
        return "n/a" if x is None else f"{x:.2f}"

    lines = [
        "# Scripted calls (text mode)",
        "",
        f"Model `{model}`, effort `{effort}`. Generated by `cva-eval-calls`.",
        "",
        f"- Task success: **{summary['task_success']:.3f}** "
        f"({sum(r['passed'] for r in results)}/{summary['n_scenarios']} calls)",
        f"- LLM turns: {summary['llm_turns']}, model calls per turn "
        f"{fmt(summary['llm_calls_per_turn'])}",
        f"- Time to first token: median {fmt(summary['ttft_median_s'])} s, "
        f"p90 {fmt(summary['ttft_p90_s'])} s",
        f"- LLM time per turn (all tool rounds): median "
        f"{fmt(summary['llm_time_per_turn_median_s'])} s",
        f"- Tool calls the executor refused: {summary['tool_errors']}",
        "",
        "| Check | Pass rate | n |",
        "|---|---|---|",
    ]
    for name, c in summary["checks"].items():
        lines.append(f"| {name} | {c['pass_rate']:.3f} | {c['n']} |")
    lines += ["", "| Call | Result | Failed checks |", "|---|---|---|"]
    for r in results:
        failed = [k for k, v in r["checks"].items() if not v] + (["error"] if r["error"] else [])
        lines.append(f"| {r['id']} | {'✅' if r['passed'] else '❌'} | {', '.join(failed)} |")
    (out_dir / "agent_scenarios.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
