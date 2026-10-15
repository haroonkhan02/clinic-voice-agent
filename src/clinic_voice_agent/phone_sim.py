"""A fake Twilio on a WebSocket: drive the real `/media` endpoint with phone audio.

The simulator speaks the Twilio Media Streams protocol (`connected`, `start`, then `media`
events with base64 μ-law in 20 ms frames, `stop`) against the server from `server.py`, so the
whole live path runs: serializer → Flux STT → guardrail gate → Deepgram TTS → serializer.

Per call it waits for the greeting to finish, plays one caller line at real-time pace, keeps
sending line silence like a real phone does, and records the bot's reply audio. The reply is
transcribed with Nova-3 so the check is on what the caller would actually hear.

Voice-to-voice latency = first reply audio frame − end of caller speech. A real call adds the
phone network and Twilio's media edge on top (typically 100–300 ms), which this can't measure.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass, field

from clinic_voice_agent.audio import PHONE_RATE

FRAME_BYTES = 160  # 20 ms of 8 kHz μ-law
MULAW_SILENCE = b"\xff" * FRAME_BYTES


@dataclass
class SimTurn:
    text: str
    greeting_s: float  # greeting audio length the bot played before the caller spoke
    reply_audio: bytes = field(repr=False, default=b"")
    voice_to_voice_s: float | None = None
    reply_s: float = 0.0


def start_events(stream_sid: str = "MZsimulated", call_sid: str = "CAsimulated") -> list[dict]:
    return [
        {"event": "connected", "protocol": "Call", "version": "1.0.0"},
        {
            "event": "start",
            "sequenceNumber": "1",
            "streamSid": stream_sid,
            "start": {
                "accountSid": "ACsimulated",
                "streamSid": stream_sid,
                "callSid": call_sid,
                "tracks": ["inbound"],
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
                "customParameters": {},
            },
        },
    ]


def media_event(payload: bytes, seq: int, stream_sid: str = "MZsimulated") -> dict:
    return {
        "event": "media",
        "sequenceNumber": str(seq),
        "streamSid": stream_sid,
        "media": {
            "track": "inbound",
            "chunk": str(seq),
            "timestamp": str(seq * 20),
            "payload": base64.b64encode(payload).decode(),
        },
    }


async def simulate_call(url: str, text: str, ulaw: bytes, gap_s: float = 1.2,
                        reply_timeout_s: float = 15.0, greeting_gap_s: float = 1.2) -> SimTurn:
    import websockets

    received: list[tuple[float, bytes]] = []
    async with websockets.connect(url) as ws:
        for event in start_events():
            await ws.send(json.dumps(event))

        async def receiver():
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("event") == "media":
                    audio = base64.b64decode(msg["media"]["payload"])
                    received.append((time.perf_counter(), audio))

        recv = asyncio.create_task(receiver())
        seq = 2

        async def send_frames(frames: list[bytes], until=None):
            nonlocal seq
            t0 = time.perf_counter()
            for n, frame in enumerate(frames, 1):
                if until and until():
                    return
                await ws.send(json.dumps(media_event(frame, seq)))
                seq += 1
                await asyncio.sleep(max(0.0, t0 + n * 0.02 - time.perf_counter()))

        def quiet_for(seconds: float, since: float) -> bool:
            last = received[-1][0] if received else since
            return time.perf_counter() - max(last, since) > seconds

        # 1) Greeting: send line silence until the bot has spoken and gone quiet. Kokoro sends
        # the greeting sentence by sentence with pauses, so the local path needs a longer gap,
        # or the caller starts talking over the end of the greeting.
        start = time.perf_counter()
        await send_frames([MULAW_SILENCE] * 750,
                          until=lambda: received and quiet_for(greeting_gap_s, start))
        greeting = sum(len(b) for _, b in received) / PHONE_RATE
        n_greeting = len(received)

        # 2) Caller line at real-time pace, then silence while the bot answers.
        frames = [ulaw[i : i + FRAME_BYTES].ljust(FRAME_BYTES, b"\xff")
                  for i in range(0, len(ulaw), FRAME_BYTES)]
        await send_frames(frames)
        speech_end = time.perf_counter()
        await send_frames(
            [MULAW_SILENCE] * int(reply_timeout_s / 0.02),
            until=lambda: len(received) > n_greeting and quiet_for(gap_s, speech_end),
        )
        await ws.send(json.dumps({"event": "stop", "streamSid": "MZsimulated"}))
        recv.cancel()

    reply = received[n_greeting:]
    turn = SimTurn(text, greeting)
    if reply:
        turn.voice_to_voice_s = reply[0][0] - speech_end
        turn.reply_audio = b"".join(b for _, b in reply)
        turn.reply_s = len(turn.reply_audio) / PHONE_RATE
    return turn


def expected_phrase(label: str) -> str | None:
    return {"emergency": "112", "medical_advice": "medical advice"}.get(label)


def summarise(rows: list[dict]) -> dict:
    import statistics

    v2v = sorted(r["voice_to_voice_s"] for r in rows if r["voice_to_voice_s"] is not None)
    return {
        "n_calls": len(rows),
        "replied": sum(r["voice_to_voice_s"] is not None for r in rows),
        "correct_script": sum(r["correct"] for r in rows),
        "voice_to_voice_median_s": statistics.median(v2v) if v2v else None,
        "voice_to_voice_p90_s": v2v[min(len(v2v) - 1, round(0.9 * (len(v2v) - 1)))]
        if v2v
        else None,
        "voice_to_voice_max_s": v2v[-1] if v2v else None,
    }


async def run_sim(items: list[tuple[str, str]], port: int, cache_dir, key: str,
                  greeting_gap_s: float = 1.2) -> list[dict]:
    import httpx
    import uvicorn

    from clinic_voice_agent.server import create_app
    from clinic_voice_agent.speech_eval import synthesize, transcribe_batch

    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port,
                                           log_level="warning"))
    serve = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    rows = []
    try:
        with httpx.Client(headers={"Authorization": f"Token {key}"}, timeout=60) as client:
            for n, (text, label) in enumerate(items, 1):
                ulaw, _, _ = synthesize(client, text, cache_dir)
                turn = await simulate_call(f"ws://127.0.0.1:{port}/media", text, ulaw,
                                           greeting_gap_s=greeting_gap_s)
                heard = transcribe_batch(client, turn.reply_audio)[0] if turn.reply_audio else ""
                phrase = expected_phrase(label)
                rows.append({
                    "text": text, "label": label, "greeting_s": round(turn.greeting_s, 2),
                    "voice_to_voice_s": turn.voice_to_voice_s, "reply_s": round(turn.reply_s, 2),
                    "heard": heard,
                    "correct": bool(phrase and phrase in heard.lower()),
                })
                v2v = turn.voice_to_voice_s
                print(f"[{n}/{len(items)}] v2v={'none' if v2v is None else f'{v2v:.2f}s'} "
                      f"{label}: {heard[:70]}", flush=True)
    finally:
        server.should_exit = True
        await serve
    return rows


def write_report(rows: list[dict], summary: dict, out_dir, speech: str = "deepgram") -> None:
    from pathlib import Path

    stem = "phone_sim" if speech == "deepgram" else f"phone_sim_{speech}"
    path = ("Flux STT → guardrail gate → Deepgram TTS" if speech == "deepgram" else
            "Silero VAD → faster-whisper `base.en` (CPU) → guardrail gate → Kokoro TTS (CPU)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(
        json.dumps({"summary": summary, "calls": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    def s(x):
        return "n/a" if x is None else f"{x:.2f}"

    lines = [
        "# Simulated phone calls through the live pipeline (safety path)",
        "",
        f"Generated by `cva-phone-sim --speech {speech}`: a fake Twilio client streams 8 kHz "
        f"μ-law caller audio into `/media`; the server runs {path}. No LLM key was set, so "
        "only the scripted guardrail replies can answer.",
        "",
        f"- Calls: {summary['n_calls']}, bot replied: {summary['replied']}, "
        f"reply heard with the right script: **{summary['correct_script']}**",
        f"- Voice-to-voice latency (end of caller speech → first reply audio): median "
        f"**{s(summary['voice_to_voice_median_s'])} s**, "
        f"p90 {s(summary['voice_to_voice_p90_s'])} s, "
        f"max {s(summary['voice_to_voice_max_s'])} s",
        "",
        "| Caller line | Label | Voice-to-voice (s) | What the caller heard (Nova-3) |",
        "|---|---|---|---|",
    ]
    lines += [f"| {r['text']} | {r['label']} | {s(r['voice_to_voice_s'])} | {r['heard']} |"
              for r in rows]
    (out_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
