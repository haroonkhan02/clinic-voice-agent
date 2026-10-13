"""Command-line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from clinic_voice_agent.config import (
    REQUIRE_ALL,
    SERVICE_KEYS,
    available_services,
    load_env,
    stage_ready,
)

FALLBACKS = {
    "llm": "none (the text harness can still run with a mocked LLM)",
    "stt": "local faster-whisper",
    "tts": "local Piper/Kokoro voice",
    "telephony": "local microphone/speaker transport",
}


def check_env_main(argv: list[str] | None = None) -> int:
    """Report which services are configured. Prints variable *names* only, never values."""
    parser = argparse.ArgumentParser(description=check_env_main.__doc__)
    parser.parse_args(argv)
    load_env()
    present = available_services()
    print(f"{'stage':<10} {'status':<8} detail")
    for stage, names in SERVICE_KEYS.items():
        if stage_ready(stage):
            status, detail = "ready", "set: " + ", ".join(present[stage])
        else:
            missing = [n for n in names if n not in present[stage]]
            status = "missing"
            joiner = " and " if stage in REQUIRE_ALL else " or "
            detail = f"needs {joiner.join(missing)} -> fallback: {FALLBACKS[stage]}"
        print(f"{stage:<10} {status:<8} {detail}")
    return 0


def faq_main(argv: list[str] | None = None) -> int:
    """Search the clinic knowledge base the way the agent's FAQ tool does."""
    from clinic_voice_agent.knowledge import load_retriever

    parser = argparse.ArgumentParser(description=faq_main.__doc__)
    parser.add_argument("question", nargs="+")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args(argv)
    for hit in load_retriever().search(" ".join(args.question), k=args.k):
        print(f"{hit.score:6.2f}  {hit.passage.title}\n        {hit.passage.text[:160]}...")
    return 0


def faq_eval_main(argv: list[str] | None = None) -> int:
    """Score the FAQ retriever on knowledge/faq_eval.yaml and write reports/faq_retrieval.*"""
    from clinic_voice_agent.config import REPORTS_DIR
    from clinic_voice_agent.faq_eval import evaluate, load_questions, write_report
    from clinic_voice_agent.knowledge import load_retriever

    parser = argparse.ArgumentParser(description=faq_eval_main.__doc__)
    parser.add_argument("--out", default=str(REPORTS_DIR))
    args = parser.parse_args(argv)
    results, summary = evaluate(load_retriever(), load_questions())
    write_report(results, summary, Path(args.out))
    print(
        f"answerable={summary['n_answerable']} hit@1={summary['hit@1']:.3f} "
        f"hit@3={summary['hit@3']:.3f} mrr@3={summary['mrr@3']:.3f}"
    )
    for row in summary["thresholds"]:
        print(
            f"  min_score={row['min_score']:.1f} kept={row['answerable_correct_and_kept']:.3f} "
            f"rejected={row['unanswerable_rejected']:.3f}"
        )
    return 0


def guardrail_eval_main(argv: list[str] | None = None) -> int:
    """Score the guardrails on eval/guardrail_utterances.yaml and write reports/guardrails.*"""
    from clinic_voice_agent.config import REPORTS_DIR
    from clinic_voice_agent.guardrail_eval import evaluate, load_utterances, write_report

    parser = argparse.ArgumentParser(description=guardrail_eval_main.__doc__)
    parser.add_argument("--out", default=str(REPORTS_DIR))
    args = parser.parse_args(argv)
    summary = evaluate(load_utterances())
    write_report(summary, Path(args.out))
    print(f"n={summary['n']} accuracy={summary['accuracy']:.3f}")
    for label, m in summary["per_class"].items():
        print(f"  {label:<15} P={m['precision']:.3f} R={m['recall']:.3f} n={m['support']}")
    for e in summary["errors"]:
        print(f"  MISS [{e['gold']} -> {e['pred']}] {e['text']}")
    return 0


def slots_main(argv: list[str] | None = None) -> int:
    """Show the earliest free slots on a demo calendar seeded with synthetic bookings."""
    from datetime import date

    from clinic_voice_agent.config import load_clinic_config
    from clinic_voice_agent.scheduling import Calendar, seed_demo, speak_datetime

    cfg = load_clinic_config()
    parser = argparse.ArgumentParser(description=slots_main.__doc__)
    parser.add_argument("type", choices=sorted(cfg.appointment_types))
    parser.add_argument("--date", help="YYYY-MM-DD (default: from today)")
    parser.add_argument("--provider", choices=sorted(cfg.providers))
    parser.add_argument("-n", type=int, default=5)
    parser.add_argument("--seed-bookings", type=int, default=60)
    args = parser.parse_args(argv)
    calendar = Calendar(cfg)
    seed_demo(calendar, n=args.seed_bookings)
    day = date.fromisoformat(args.date) if args.date else None
    for slot in calendar.find_slots(args.type, day=day, provider=args.provider, limit=args.n):
        print(f"{speak_datetime(slot.start):<32} {cfg.provider_name(slot.provider)}")
    return 0
