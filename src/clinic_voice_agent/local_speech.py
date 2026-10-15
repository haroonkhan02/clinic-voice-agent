"""Offline speech fallback: Kokoro TTS and faster-whisper STT, no API keys.

Two questions, both on the same caller lines as `cva-speech-eval`:

1. **Local STT on the same phone audio.** The cached Deepgram Aura-2 clips (8 kHz μ-law) are
   transcribed by faster-whisper on the CPU, so its WER and guardrail agreement can be compared
   line by line with Nova-3 and Flux.
2. **Fully local round trip.** Kokoro (82M parameters, ONNX) speaks each line, the audio is
   resampled to 8 kHz and encoded to μ-law like a phone call, then each faster-whisper model
   transcribes it.

Latency is measured per clip: Kokoro time to the first audio chunk and the whole clip, and
whisper time per clip (a segmented model only starts after the caller stops talking, so this
is the delay a caller would feel on top of the end-of-speech detection).

Model files are downloaded on first use into models/ (git-ignored).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from clinic_voice_agent.audio import (
    PHONE_RATE,
    mulaw_to_pcm16,
    pcm16_to_mulaw,
    word_errors,
)
from clinic_voice_agent.guardrails import screen
from clinic_voice_agent.speech_eval import Item, _stats

KOKORO_VOICE = "af_heart"
WHISPER_MODELS = ("base.en", "distil-medium.en")


def kokoro_paths(models_dir: Path) -> tuple[Path, Path]:
    return models_dir / "kokoro" / "kokoro-v1.0.onnx", models_dir / "kokoro" / "voices-v1.0.bin"


def load_kokoro(models_dir: Path):
    from kokoro_onnx import Kokoro
    from pipecat.services.kokoro.tts import _ensure_model_files

    model, voices = kokoro_paths(models_dir)
    model.parent.mkdir(parents=True, exist_ok=True)
    _ensure_model_files(model, voices)
    return Kokoro(str(model), str(voices))


async def kokoro_phone_audio(kokoro, text: str) -> tuple[bytes, float, float]:
    """(8 kHz μ-law audio, seconds to first chunk, seconds for the whole clip)."""
    import numpy as np
    import soxr

    t0, first, chunks, rate = time.perf_counter(), None, [], 24000
    async for samples, chunk_rate in kokoro.create_stream(text, voice=KOKORO_VOICE,
                                                          lang="en-us"):
        if first is None:
            first = time.perf_counter() - t0
        chunks.append(samples)
        rate = chunk_rate
    total = time.perf_counter() - t0
    audio = soxr.resample(np.concatenate(chunks), rate, PHONE_RATE)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    return pcm16_to_mulaw(pcm), first or total, total


def load_whisper(name: str, models_dir: Path):
    from faster_whisper import WhisperModel

    # int8 on the CPU: faster-whisper (CTranslate2) has no Apple GPU backend.
    return WhisperModel(name, device="cpu", compute_type="int8",
                        download_root=str(models_dir / "whisper"))


def whisper_transcribe(model, ulaw: bytes) -> tuple[str, float]:
    import numpy as np
    import soxr

    pcm = np.frombuffer(mulaw_to_pcm16(ulaw), dtype="<i2").astype(np.float32) / 32768
    audio = soxr.resample(pcm, PHONE_RATE, 16000)  # Whisper expects 16 kHz
    t0 = time.perf_counter()
    segments, _ = model.transcribe(audio, language="en", beam_size=1,
                                   condition_on_previous_text=False)
    text = " ".join(s.text.strip() for s in segments).strip()
    return text, time.perf_counter() - t0


@dataclass
class Row:
    text: str
    label: str | None
    audio: str  # "aura" (cached Deepgram clip) or "kokoro"
    stt: str
    transcript: str
    stt_s: float
    edits: int
    ref_words: int
    guard_text: str
    guard_transcript: str
    tts_first_s: float | None = None
    tts_total_s: float | None = None


def _row(item: Item, audio: str, stt: str, ulaw: bytes, model, **tts) -> Row:
    text, secs = whisper_transcribe(model, ulaw)
    edits, ref = word_errors(item.text, text)
    return Row(item.text, item.label, audio, stt, text, secs, edits, ref,
               screen(item.text).action, screen(text).action, **tts)


async def run(items: list[Item], cache_dir: Path, models_dir: Path,
              whisper_models=WHISPER_MODELS, progress=print) -> list[Row]:
    from clinic_voice_agent.eot_sweep import cached_audio

    whispers = {name: load_whisper(name, models_dir) for name in whisper_models}
    kokoro = load_kokoro(models_dir)
    rows: list[Row] = []
    for n, item in enumerate(items, 1):
        aura = cached_audio(item, cache_dir)
        ulaw, first, total = await kokoro_phone_audio(kokoro, item.text)
        for name, model in whispers.items():
            rows.append(_row(item, "aura", name, aura, model))
            rows.append(_row(item, "kokoro", name, ulaw, model,
                             tts_first_s=first, tts_total_s=total))
        progress(f"[{n}/{len(items)}] {rows[-1].transcript[:60]}")
    return rows


def summarise(rows: list[Row]) -> list[dict[str, Any]]:
    out = []
    for audio, stt in dict.fromkeys((r.audio, r.stt) for r in rows):
        sel = [r for r in rows if r.audio == audio and r.stt == stt]
        gold = {lab: [r for r in sel if r.label == lab] for lab in ("emergency", "medical_advice")}
        out.append({
            "audio": audio, "stt": stt, "n": len(sel),
            "wer": sum(r.edits for r in sel) / max(1, sum(r.ref_words for r in sel)),
            "exact": sum(r.edits == 0 for r in sel) / len(sel),
            "guardrail_agreement": sum(r.guard_text == r.guard_transcript for r in sel) / len(sel),
            "recall": {lab: sum(r.guard_transcript == lab for r in g) / len(g)
                       for lab, g in gold.items() if g},
            "stt_s": _stats([r.stt_s for r in sel]),
            "tts_first_s": _stats([r.tts_first_s for r in sel if r.tts_first_s is not None]),
            "tts_total_s": _stats([r.tts_total_s for r in sel if r.tts_total_s is not None]),
            "flips": [{"text": r.text, "transcript": r.transcript, "on_text": r.guard_text,
                       "on_transcript": r.guard_transcript}
                      for r in sel if r.guard_text != r.guard_transcript],
        })
    return out


def write_report(rows: list[Row], summary: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "local_speech.json").write_text(json.dumps(
        {"kokoro_voice": KOKORO_VOICE, "summary": summary, "items": [asdict(r) for r in rows]},
        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def f(x):
        return "n/a" if x is None else f"{x:.2f}"

    lines = [
        "# Offline speech: Kokoro TTS and faster-whisper STT on 8 kHz μ-law",
        "",
        "Generated by `cva-local-speech-eval` on the CPU (faster-whisper int8). `aura` = the "
        "cached Deepgram Aura-2 clips used by `cva-speech-eval`; `kokoro` = Kokoro "
        f"(`{KOKORO_VOICE}`) audio, resampled to 8 kHz μ-law.",
        "",
        "| Audio | STT | WER | Exact | Guardrail decision unchanged | Emergency recall | "
        "Medical-advice recall | STT median (s) | p90 (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        rec = s["recall"]
        lines.append(
            f"| {s['audio']} | {s['stt']} | {s['wer']:.3f} | {s['exact']:.3f} | "
            f"{s['guardrail_agreement']:.3f} | {f(rec.get('emergency'))} | "
            f"{f(rec.get('medical_advice'))} | {f(s['stt_s']['median'])} | "
            f"{f(s['stt_s']['p90'])} |")
    kok = next((s for s in summary if s["audio"] == "kokoro"), None)
    if kok:
        first = kok["tts_first_s"]
        lines += ["", f"Kokoro on the CPU: first audio chunk median {f(first['median'])}"
                  f" s (p90 {f(first['p90'])} s), whole clip median "
                  f"{f(kok['tts_total_s']['median'])} s."]
    for s in summary:
        if s["flips"]:
            lines += ["", f"**Guardrail flips ({s['audio']} → {s['stt']})**", ""]
            lines += [f"- `{x['on_text']}` → `{x['on_transcript']}`: *{x['text']}* → "
                      f"*{x['transcript']}*" for x in s["flips"]]
    (out_dir / "local_speech.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
