"""Speech round trip: do the guardrails still fire after text → phone audio → transcript?

For every caller line in the evaluation sets:

1. Deepgram TTS (Aura-2) speaks it straight into the phone codec: 8 kHz G.711 μ-law, the
   format Twilio Media Streams carries. Time to first byte and total time are recorded.
2. The μ-law audio is transcribed twice:
   - Deepgram Nova-3 (pre-recorded REST): a strong batch transcript and its latency.
   - Deepgram Flux (streaming WebSocket, the model the live pipeline uses): audio is decoded to
     16-bit PCM and sent in 80 ms chunks at real-time pace, followed by silence. The time from
     the last speech chunk to Flux's `EndOfTurn` is the end-of-turn latency a caller feels.
3. Word error rate is computed against the original text, and `guardrails.screen()` is run on
   both the original and the transcripts to see whether any decision flips.

The voice is synthetic and clean, so these word error rates are a lower bound for real callers
(accents, noise, mobile networks). Audio is cached under outputs/ so reruns don't re-buy TTS.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from clinic_voice_agent.audio import (
    PHONE_RATE,
    mulaw_to_pcm16,
    trim_trailing_silence,
    word_errors,
)
from clinic_voice_agent.guardrails import screen

API = "https://api.deepgram.com"
TTS_VOICE = "aura-2-thalia-en"
BATCH_MODEL = "nova-3"
FLUX_MODEL = "flux-general-en"
CHUNK_MS = 80


@dataclass
class Item:
    text: str
    label: str | None  # gold guardrail label when the line comes from the labelled set
    source: str


@dataclass
class ItemResult:
    text: str
    label: str | None
    source: str
    audio_s: float
    tts_ttfb_s: float | None
    tts_total_s: float | None
    batch_text: str
    batch_latency_s: float
    flux_text: str
    flux_eot_latency_s: float | None
    flux_premature_eots: int
    guard_text: str
    guard_batch: str
    guard_flux: str
    batch_edits: int
    flux_edits: int
    ref_words: int


def collect_items(utterances: list[tuple[str, str]], scenarios: list[Any]) -> list[Item]:
    items = [Item(t, label, "guardrail_set") for t, label in utterances]
    seen = {i.text for i in items}
    for s in scenarios:
        for line in s.caller:
            if line not in seen:
                seen.add(line)
                items.append(Item(line, None, f"scenario:{s.id}"))
    return items


def _key() -> str:
    key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set")
    return key


def synthesize(client, text: str, cache_dir: Path) -> tuple[bytes, float | None, float | None]:
    """μ-law 8 kHz audio for `text`; timings are None when the audio came from the cache."""
    name = hashlib.sha1(f"{TTS_VOICE}|{text}".encode()).hexdigest()[:16]
    path = cache_dir / f"{name}.ulaw"
    meta = path.with_suffix(".json")
    if path.exists():
        # Timings measured when the clip was first synthesised (None if they weren't stored).
        timing = json.loads(meta.read_text()) if meta.exists() else {}
        return path.read_bytes(), timing.get("ttfb"), timing.get("total")
    params = {"model": TTS_VOICE, "encoding": "mulaw", "sample_rate": PHONE_RATE,
              "container": "none"}
    t0, ttfb, audio = time.perf_counter(), None, bytearray()
    with client.stream("POST", f"{API}/v1/speak", params=params, json={"text": text}) as r:
        r.raise_for_status()
        for chunk in r.iter_bytes():
            if ttfb is None and chunk:
                ttfb = time.perf_counter() - t0
            audio += chunk
    total = time.perf_counter() - t0
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(audio))
    meta.write_text(json.dumps({"ttfb": ttfb, "total": total}))
    return bytes(audio), ttfb, total


def transcribe_batch(client, ulaw: bytes) -> tuple[str, float]:
    params = {"model": BATCH_MODEL, "encoding": "mulaw", "sample_rate": PHONE_RATE,
              "smart_format": "true"}
    t0 = time.perf_counter()
    r = client.post(f"{API}/v1/listen", params=params, content=ulaw,
                    headers={"Content-Type": "application/octet-stream"})
    r.raise_for_status()
    latency = time.perf_counter() - t0
    alt = r.json()["results"]["channels"][0]["alternatives"][0]
    return alt["transcript"], latency


async def transcribe_flux(pcm: bytes, key: str, max_silence_s: float = 6.0,
                          params: dict[str, Any] | None = None):
    """Stream PCM at real-time pace, like a phone line, and follow Flux's turns.

    Returns (transcript of all turns, seconds from end of speech to the final EndOfTurn,
    number of EndOfTurn events that arrived while the caller was still speaking). A premature
    EndOfTurn is what makes a live bot answer in the middle of a sentence. `params` are extra
    Flux query parameters, e.g. {"eot_threshold": 0.8}.
    """
    from urllib.parse import urlencode

    import websockets

    query = {"model": FLUX_MODEL, "encoding": "linear16", "sample_rate": PHONE_RATE,
             **(params or {})}
    url = f"wss://api.deepgram.com/v2/listen?{urlencode(query)}"
    chunk = PHONE_RATE * 2 * CHUNK_MS // 1000
    speech = trim_trailing_silence(pcm)
    silence = bytes(chunk)
    state: dict[str, Any] = {"speech_end": None, "eots": [], "texts": [], "premature": 0}

    def finished() -> bool:
        # Done once a turn has ended after the caller stopped talking.
        return bool(state["eots"]) and state["eots"][-1] >= state["speech_end"]

    async with websockets.connect(url, additional_headers={"Authorization": f"Token {key}"}) as ws:

        async def sender():
            start = time.perf_counter()
            for sent, i in enumerate(range(0, len(speech), chunk), 1):
                await ws.send(speech[i : i + chunk])
                await asyncio.sleep(max(0.0, start + sent * CHUNK_MS / 1000 - time.perf_counter()))
            state["speech_end"] = time.perf_counter()
            for _ in range(int(max_silence_s * 1000 / CHUNK_MS)):
                if finished():
                    break
                await ws.send(silence)
                await asyncio.sleep(CHUNK_MS / 1000)
            await ws.send(json.dumps({"type": "CloseStream"}))

        async def receiver():
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "TurnInfo" and msg.get("event") == "EndOfTurn":
                    state["eots"].append(time.perf_counter())
                    state["texts"].append(msg.get("transcript", ""))
                    if state["speech_end"] is None:
                        state["premature"] += 1
                elif msg.get("type") == "Error":
                    raise RuntimeError(f"Flux error: {msg.get('description', msg)}")

        recv = asyncio.create_task(receiver())
        await sender()
        try:
            await asyncio.wait_for(recv, timeout=5)
        except asyncio.TimeoutError:
            recv.cancel()
    latency = state["eots"][-1] - state["speech_end"] if finished() else None
    return " ".join(t for t in state["texts"] if t).strip(), latency, state["premature"]


def _retry(fn, *args, attempts: int = 3):
    """Retry transient network failures (an idle keep-alive connection can be dropped while a
    Flux stream is running). HTTP errors such as 401/400 are raised immediately."""
    import httpx

    for attempt in range(attempts):
        try:
            return fn(*args)
        except (httpx.TransportError, OSError):
            if attempt == attempts - 1:
                raise
            time.sleep(1.0 + attempt)


def _flux_sync(pcm: bytes, key: str):
    return asyncio.run(transcribe_flux(pcm, key))


def _print(line: str) -> None:
    print(line, flush=True)


def run(items: list[Item], cache_dir: Path, progress=_print) -> list[ItemResult]:
    import httpx

    key = _key()
    results = []
    with httpx.Client(headers={"Authorization": f"Token {key}"}, timeout=60) as client:
        for n, item in enumerate(items, 1):
            ulaw, ttfb, total = _retry(synthesize, client, item.text, cache_dir)
            batch_text, batch_latency = _retry(transcribe_batch, client, ulaw)
            flux_text, eot, premature = _retry(_flux_sync, mulaw_to_pcm16(ulaw), key)
            b_edits, ref_n = word_errors(item.text, batch_text)
            f_edits, _ = word_errors(item.text, flux_text)
            results.append(ItemResult(
                text=item.text, label=item.label, source=item.source,
                audio_s=len(ulaw) / PHONE_RATE, tts_ttfb_s=ttfb, tts_total_s=total,
                batch_text=batch_text, batch_latency_s=batch_latency,
                flux_text=flux_text, flux_eot_latency_s=eot, flux_premature_eots=premature,
                guard_text=screen(item.text).action, guard_batch=screen(batch_text).action,
                guard_flux=screen(flux_text).action,
                batch_edits=b_edits, flux_edits=f_edits, ref_words=ref_n,
            ))
            progress(f"[{n}/{len(items)}] flux_eot={eot if eot is None else round(eot, 2)} "
                     f"edits={b_edits}/{f_edits} {item.text[:50]}")
    return results


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p90": None, "n": 0}
    v = sorted(values)
    return {"median": statistics.median(v), "p90": v[min(len(v) - 1, round(0.9 * (len(v) - 1)))],
            "n": len(v)}


def summarise(results: list[ItemResult]) -> dict[str, Any]:
    ref = sum(r.ref_words for r in results) or 1
    labelled = [r for r in results if r.label]
    out: dict[str, Any] = {
        "n_items": len(results),
        "audio_seconds": round(sum(r.audio_s for r in results), 1),
        "tts_ttfb_s": _stats([r.tts_ttfb_s for r in results if r.tts_ttfb_s is not None]),
        "tts_total_s": _stats([r.tts_total_s for r in results if r.tts_total_s is not None]),
        "batch_latency_s": _stats([r.batch_latency_s for r in results]),
        "flux_eot_latency_s": _stats(
            [r.flux_eot_latency_s for r in results if r.flux_eot_latency_s is not None]
        ),
        "flux_no_eot": sum(r.flux_eot_latency_s is None for r in results),
        "flux_premature_lines": [r.text for r in results if r.flux_premature_eots],
        "engines": {},
    }
    for engine in ("batch", "flux"):
        edits = sum(getattr(r, f"{engine}_edits") for r in results)
        guard = [getattr(r, f"guard_{engine}") for r in results]
        flips = [r for r, g in zip(results, guard, strict=True) if g != r.guard_text]
        recall = {}
        for label in ("emergency", "medical_advice"):
            gold = [(r, g) for r, g in zip(results, guard, strict=True) if r.label == label]
            if gold:
                recall[label] = {
                    "on_text": sum(r.guard_text == label for r, _ in gold) / len(gold),
                    "on_transcript": sum(g == label for _, g in gold) / len(gold),
                    "n": len(gold),
                }
        out["engines"][engine] = {
            "wer": edits / ref,
            "exact": sum(getattr(r, f"{engine}_edits") == 0 for r in results) / len(results),
            "guardrail_agreement": 1 - len(flips) / len(results),
            "flips": [{"text": r.text, "transcript": getattr(r, f"{engine}_text"),
                       "on_text": r.guard_text, "on_transcript": getattr(r, f"guard_{engine}")}
                      for r in flips],
            "recall": recall,
            "labelled_accuracy": sum(
                getattr(r, f"guard_{engine}") == r.label for r in labelled
            ) / max(1, len(labelled)),
        }
    out["labelled_accuracy_on_text"] = sum(r.guard_text == r.label for r in labelled) / max(
        1, len(labelled)
    )
    return out


def write_report(results: list[ItemResult], summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"tts_voice": TTS_VOICE, "batch_model": BATCH_MODEL, "flux_model": FLUX_MODEL,
               "summary": summary, "items": [asdict(r) for r in results]}
    (out_dir / "speech_roundtrip.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def s(x):
        return "n/a" if x is None else f"{x:.2f}"

    e = summary["engines"]
    lines = [
        "# Speech round trip (Deepgram TTS → 8 kHz μ-law → STT → guardrails)",
        "",
        f"{summary['n_items']} caller lines, {summary['audio_seconds']} s of audio. "
        f"Voice `{TTS_VOICE}`; generated by `cva-speech-eval`.",
        "",
        "| Engine | WER | Exact transcripts | Guardrail decision unchanged | "
        "Emergency recall (text → transcript) | Medical-advice recall |",
        "|---|---|---|---|---|---|",
    ]
    for name, label in (("batch", f"Nova-3 batch (`{BATCH_MODEL}`)"),
                        ("flux", f"Flux streaming (`{FLUX_MODEL}`)")):
        m = e[name]
        em, md = m["recall"].get("emergency", {}), m["recall"].get("medical_advice", {})
        lines.append(
            f"| {label} | {m['wer']:.3f} | {m['exact']:.3f} | {m['guardrail_agreement']:.3f} | "
            f"{s(em.get('on_text'))} → {s(em.get('on_transcript'))} | "
            f"{s(md.get('on_text'))} → {s(md.get('on_transcript'))} |"
        )
    lines += [
        "",
        "| Latency | Median (s) | p90 (s) | n |",
        "|---|---|---|---|",
    ]
    for key, label in (("tts_ttfb_s", "TTS time to first byte (REST)"),
                       ("tts_total_s", "TTS full clip (REST)"),
                       ("batch_latency_s", "Nova-3 batch request"),
                       ("flux_eot_latency_s", "Flux: end of speech → EndOfTurn")):
        st = summary[key]
        lines.append(f"| {label} | {s(st['median'])} | {s(st['p90'])} | {st['n']} |")
    early = summary["flux_premature_lines"]
    lines += ["", f"Flux gave no final EndOfTurn for {summary['flux_no_eot']} lines. It ended the "
              f"turn while the caller was still speaking on {len(early)} lines (a live bot would "
              "start answering mid-sentence there):", ""]
    lines += [f"- *{t}*" for t in early] + [""]
    for name in ("batch", "flux"):
        if e[name]["flips"]:
            lines += [f"**Guardrail flips ({name})**", ""]
            lines += [f"- `{f['on_text']}` → `{f['on_transcript']}`: *{f['text']}* → "
                      f"*{f['transcript']}*" for f in e[name]["flips"]]
            lines.append("")
    (out_dir / "speech_roundtrip.md").write_text("\n".join(lines), encoding="utf-8")
