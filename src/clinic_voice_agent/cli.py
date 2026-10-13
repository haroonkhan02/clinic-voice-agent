"""Command-line entry points."""

from __future__ import annotations

import argparse

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
