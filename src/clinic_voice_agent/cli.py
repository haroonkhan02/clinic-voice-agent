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
    """Score the guardrails on a labelled set (dev or held-out) and write reports/guardrails*"""
    from clinic_voice_agent.config import REPORTS_DIR
    from clinic_voice_agent.guardrail_eval import SETS, evaluate, load_utterances, write_report

    parser = argparse.ArgumentParser(description=guardrail_eval_main.__doc__)
    parser.add_argument("--set", dest="which", choices=sorted(SETS), default="dev")
    parser.add_argument("--out", default=str(REPORTS_DIR))
    args = parser.parse_args(argv)
    summary = evaluate(load_utterances(SETS[args.which][0]))
    write_report(summary, Path(args.out), args.which)
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


def _require_llm_key() -> bool:
    load_env()
    if stage_ready("llm"):
        return True
    print("ANTHROPIC_API_KEY is not set (add it to .env). The agent needs it for this command.")
    return False


def chat_main(argv: list[str] | None = None) -> int:
    """Talk to the receptionist in the terminal (text mode, same agent as the phone line)."""
    from datetime import datetime

    from clinic_voice_agent.agent import EFFORT, MODEL, ClinicAgent
    from clinic_voice_agent.config import OUTPUTS_DIR, load_clinic_config
    from clinic_voice_agent.knowledge import load_retriever
    from clinic_voice_agent.scheduling import Calendar, seed_demo
    from clinic_voice_agent.tools import ToolExecutor

    parser = argparse.ArgumentParser(description=chat_main.__doc__)
    parser.add_argument("--now", help="Fake clinic clock, YYYY-MM-DDTHH:MM (default: real time)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--effort", default=EFFORT)
    parser.add_argument("--calendar", default=str(OUTPUTS_DIR / "calendar.json"))
    parser.add_argument("--show-tools", action="store_true")
    args = parser.parse_args(argv)
    if not _require_llm_key():
        return 2
    cfg = load_clinic_config()
    now = (lambda: datetime.fromisoformat(args.now)) if args.now else None
    calendar = Calendar(cfg, path=args.calendar, now=now)
    if not calendar.appointments:
        seed_demo(calendar)
    agent = ClinicAgent(
        cfg, ToolExecutor(cfg, calendar, load_retriever()), model=args.model, effort=args.effort
    )
    print(f"{cfg.name} receptionist (Ctrl-D to hang up)")
    while True:
        try:
            line = input("caller> ").strip()
        except EOFError:
            print()
            return 0
        if not line:
            continue
        turn = agent.respond(line)
        if args.show_tools:
            for call in turn.tool_calls:
                print(f"   [tool] {call['name']} ok={call['ok']} {call['args']}")
        ttft = f" ttft={turn.ttft_s:.2f}s" if turn.ttft_s else ""
        print(f"agent>  {turn.reply}   ({turn.guardrail}{ttft})")


def eval_calls_main(argv: list[str] | None = None) -> int:
    """Play eval/scenarios.yaml through the agent and write reports/agent_scenarios.*"""
    from clinic_voice_agent.agent import EFFORT, MODEL, anthropic_llm_call
    from clinic_voice_agent.config import REPORTS_DIR, load_clinic_config
    from clinic_voice_agent.harness import run_scenario, summarise, write_report
    from clinic_voice_agent.knowledge import load_retriever
    from clinic_voice_agent.scenarios import load_scenarios
    from clinic_voice_agent.tools import build_tool_specs

    parser = argparse.ArgumentParser(description=eval_calls_main.__doc__)
    parser.add_argument("--only", nargs="*", help="Scenario ids to run (default: all)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--effort", default=EFFORT)
    parser.add_argument("--out", default=str(REPORTS_DIR))
    args = parser.parse_args(argv)
    if not _require_llm_key():
        return 2
    cfg = load_clinic_config()
    scenarios = load_scenarios(tool_names={t["name"] for t in build_tool_specs(cfg)})
    if args.only:
        scenarios = [s for s in scenarios if s.id in set(args.only)]
    retriever, llm_call = load_retriever(), anthropic_llm_call()
    results = []
    for scenario in scenarios:

        def call(kwargs, _model=args.model, _effort=args.effort):
            return llm_call({**kwargs, "model": _model, "output_config": {"effort": _effort}})

        result = run_scenario(scenario, cfg, retriever, call)
        results.append(result)
        print(f"{'PASS' if result['passed'] else 'FAIL'}  {scenario.id}  {result['checks']}")
    summary = summarise(results)
    write_report(results, summary, Path(args.out), args.model, args.effort)
    print(f"task success {summary['task_success']:.3f} over {len(results)} calls")
    return 0


def speech_eval_main(argv: list[str] | None = None) -> int:
    """Deepgram TTS → phone μ-law → Nova-3 / Flux STT → guardrails; writes reports/speech_*"""
    from clinic_voice_agent.config import OUTPUTS_DIR, REPORTS_DIR
    from clinic_voice_agent.guardrail_eval import load_utterances
    from clinic_voice_agent.scenarios import load_scenarios
    from clinic_voice_agent.speech_eval import collect_items, run, summarise, write_report

    parser = argparse.ArgumentParser(description=speech_eval_main.__doc__)
    parser.add_argument("--limit", type=int, help="Only the first N lines (smoke test)")
    parser.add_argument("--out", default=str(REPORTS_DIR))
    args = parser.parse_args(argv)
    load_env()
    if not stage_ready("stt"):
        print("DEEPGRAM_API_KEY is not set (add it to .env).")
        return 2
    items = collect_items(load_utterances(), load_scenarios())[: args.limit]
    results = run(items, OUTPUTS_DIR / "speech_cache")
    summary = summarise(results)
    write_report(results, summary, Path(args.out))
    for name, m in summary["engines"].items():
        print(f"{name}: wer={m['wer']:.3f} guardrail_agreement={m['guardrail_agreement']:.3f}")
    print(f"flux end-of-turn latency: {summary['flux_eot_latency_s']}")
    return 0


def flux_sweep_main(argv: list[str] | None = None) -> int:
    """Replay cached phone audio through Flux at several eot_threshold values."""
    import asyncio

    from clinic_voice_agent.config import OUTPUTS_DIR, REPORTS_DIR
    from clinic_voice_agent.eot_sweep import DEFAULT_THRESHOLDS, sweep, write_report
    from clinic_voice_agent.guardrail_eval import load_utterances
    from clinic_voice_agent.scenarios import load_scenarios
    from clinic_voice_agent.speech_eval import collect_items

    parser = argparse.ArgumentParser(description=flux_sweep_main.__doc__)
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--limit", type=int, help="Only the first N lines (smoke test)")
    parser.add_argument("--out", default=str(REPORTS_DIR))
    args = parser.parse_args(argv)
    load_env()
    if not stage_ready("stt"):
        print("DEEPGRAM_API_KEY is not set (add it to .env).")
        return 2
    items = collect_items(load_utterances(), load_scenarios())[: args.limit]
    result = asyncio.run(sweep(items, OUTPUTS_DIR / "speech_cache", args.thresholds))
    write_report(result, Path(args.out))
    for t, block in result.items():
        s = block["summary"]
        print(f"thr={t} premature={s['premature_lines']}/{s['n']} eot={s['eot_latency_s']}")
    return 0


def local_speech_eval_main(argv: list[str] | None = None) -> int:
    """Offline speech: faster-whisper on the cached phone audio + a Kokoro → whisper round trip."""
    import asyncio

    from clinic_voice_agent.config import OUTPUTS_DIR, REPORTS_DIR, ROOT
    from clinic_voice_agent.guardrail_eval import load_utterances
    from clinic_voice_agent.local_speech import run, summarise, write_report
    from clinic_voice_agent.scenarios import load_scenarios
    from clinic_voice_agent.speech_eval import collect_items

    parser = argparse.ArgumentParser(description=local_speech_eval_main.__doc__)
    parser.add_argument("--limit", type=int, help="Only the first N lines (smoke test)")
    parser.add_argument("--out", default=str(REPORTS_DIR))
    args = parser.parse_args(argv)
    items = collect_items(load_utterances(), load_scenarios())[: args.limit]
    rows = asyncio.run(run(items, OUTPUTS_DIR / "speech_cache", ROOT / "models"))
    summary = summarise(rows)
    write_report(rows, summary, Path(args.out))
    for s in summary:
        print(f"{s['audio']}/{s['stt']}: wer={s['wer']:.3f} "
              f"agreement={s['guardrail_agreement']:.3f} stt={s['stt_s']}")
    return 0


def mic_main(argv: list[str] | None = None) -> int:
    """Talk to the receptionist through this computer's microphone and speakers."""
    import asyncio

    from clinic_voice_agent.config import load_clinic_config
    from clinic_voice_agent.knowledge import load_retriever
    from clinic_voice_agent.scheduling import Calendar, seed_demo
    from clinic_voice_agent.tools import ToolExecutor
    from clinic_voice_agent.voice import run_call, speech_backend

    argparse.ArgumentParser(description=mic_main.__doc__).parse_args(argv)
    load_env()
    try:
        from pipecat.transports.local.audio import (
            LocalAudioTransport,
            LocalAudioTransportParams,
        )
    except ImportError:
        print("Microphone mode needs PortAudio and pyaudio: brew install portaudio && "
              "uv pip install pyaudio")
        return 2
    cfg = load_clinic_config()
    calendar = Calendar(cfg)
    seed_demo(calendar)
    transport = LocalAudioTransport(LocalAudioTransportParams(
        audio_in_enabled=True, audio_out_enabled=True))
    print(f"Speech: {speech_backend()}. Speak after the greeting; Ctrl+C to hang up.")
    asyncio.run(run_call(transport, cfg, ToolExecutor(cfg, calendar, load_retriever()),
                         sample_rate=16000, handle_sigint=True, local=True))
    return 0


def plots_main(argv: list[str] | None = None) -> int:
    """Latency analysis + figures from the saved reports (no API calls)."""
    from clinic_voice_agent.config import REPORTS_DIR
    from clinic_voice_agent.plots import analyse, make_figures, write_analysis

    parser = argparse.ArgumentParser(description=plots_main.__doc__)
    parser.add_argument("--reports", default=str(REPORTS_DIR))
    args = parser.parse_args(argv)
    reports = Path(args.reports)
    result = analyse(reports)
    write_analysis(result, reports)
    for path in make_figures(result, reports):
        print(f"wrote {path}")
    return 0


def phone_sim_main(argv: list[str] | None = None) -> int:
    """Fake-Twilio calls into the real /media endpoint; writes reports/phone_sim.*"""
    import asyncio
    import os

    from clinic_voice_agent.config import OUTPUTS_DIR, REPORTS_DIR
    from clinic_voice_agent.guardrail_eval import load_utterances
    from clinic_voice_agent.phone_sim import run_sim, summarise, write_report

    parser = argparse.ArgumentParser(description=phone_sim_main.__doc__)
    parser.add_argument("--per-label", type=int, default=6)
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--speech", choices=("deepgram", "local"), default="deepgram",
                        help="Speech stages inside the server (the caller audio and the check of "
                             "the reply always use Deepgram)")
    parser.add_argument("--out", default=str(REPORTS_DIR))
    args = parser.parse_args(argv)
    load_env()
    os.environ["CVA_SPEECH"] = args.speech
    if not stage_ready("stt"):
        print("DEEPGRAM_API_KEY is not set (add it to .env).")
        return 2
    items, counts = [], {}
    for text, label in load_utterances():
        if label in ("emergency", "medical_advice") and counts.get(label, 0) < args.per_label:
            counts[label] = counts.get(label, 0) + 1
            items.append((text, label))
    rows = asyncio.run(run_sim(items, args.port, OUTPUTS_DIR / "speech_cache",
                               os.environ["DEEPGRAM_API_KEY"],
                               greeting_gap_s=1.2 if args.speech == "deepgram" else 3.0))
    summary = summarise(rows)
    write_report(rows, summary, Path(args.out), args.speech)
    print(summary)
    return 0
