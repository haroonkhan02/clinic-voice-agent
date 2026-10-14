"""Twilio-facing web server.

1. Twilio calls `POST /voice` when the number rings. We answer with TwiML that opens a
   bidirectional Media Stream to `wss://<PUBLIC_HOST>/media`.
2. Twilio connects to `/media` and streams the caller's audio (8 kHz μ-law, base64 in JSON).
   Pipecat's `TwilioFrameSerializer` turns that into audio frames and back, and the voice
   pipeline runs for the length of the call.

`PUBLIC_HOST` is the public hostname of a tunnel (e.g. ngrok) that forwards to this server.
Optional `TWILIO_AUTH_TOKEN` enables signature checking of the webhook, and lets the serializer
hang up the call through Twilio's REST API when the pipeline ends.
"""


import os
from xml.sax.saxutils import quoteattr

TWIML = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url={url} />
  </Connect>
</Response>
"""


def twiml_for(public_host: str) -> str:
    host = public_host.removeprefix("https://").removeprefix("wss://").strip("/")
    if not host or any(c in host for c in " <>\"'"):
        raise ValueError("PUBLIC_HOST must be a bare hostname such as abc123.ngrok.app")
    return TWIML.format(url=quoteattr(f"wss://{host}/media"))


def valid_twilio_signature(auth_token: str, url: str, params: dict[str, str], signature: str):
    """Twilio's X-Twilio-Signature: base64(HMAC-SHA1(token, url + sorted key/value pairs))."""
    import base64
    import hashlib
    import hmac

    payload = url + "".join(k + params[k] for k in sorted(params))
    digest = hmac.new(auth_token.encode(), payload.encode(), hashlib.sha1).digest()
    return hmac.compare_digest(base64.b64encode(digest).decode(), signature or "")


def create_app():
    from fastapi import FastAPI, HTTPException, Request, WebSocket
    from fastapi.responses import Response

    from clinic_voice_agent.config import OUTPUTS_DIR, load_clinic_config, load_env
    from clinic_voice_agent.knowledge import load_retriever
    from clinic_voice_agent.scheduling import Calendar
    from clinic_voice_agent.tools import ToolExecutor

    load_env()
    cfg = load_clinic_config()
    retriever = load_retriever()
    app = FastAPI(title="clinic-voice-agent")

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/voice")
    async def voice(request: Request):
        host = os.environ.get("PUBLIC_HOST", "")
        token = os.environ.get("TWILIO_AUTH_TOKEN", "")
        if token:
            form = {k: str(v) for k, v in (await request.form()).items()}
            url = f"https://{host}{request.url.path}"
            if not valid_twilio_signature(
                token, url, form, request.headers.get("X-Twilio-Signature", "")
            ):
                raise HTTPException(status_code=403, detail="bad Twilio signature")
        try:
            body = twiml_for(host)
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return Response(content=body, media_type="application/xml")

    @app.websocket("/media")
    async def media(websocket: WebSocket):
        from pipecat.runner.utils import parse_telephony_websocket
        from pipecat.serializers.twilio import TwilioFrameSerializer
        from pipecat.transports.websocket.fastapi import (
            FastAPIWebsocketParams,
            FastAPIWebsocketTransport,
        )

        from clinic_voice_agent.voice import run_call

        await websocket.accept()
        transport_type, call_data = await parse_telephony_websocket(websocket)
        if transport_type != "twilio":
            await websocket.close()
            return
        sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
        token = os.environ.get("TWILIO_AUTH_TOKEN", "")
        serializer = TwilioFrameSerializer(
            stream_sid=call_data["stream_id"],
            call_sid=call_data["call_id"],
            account_sid=sid,
            auth_token=token,
            # Hanging up through Twilio's REST API needs both credentials.
            params=TwilioFrameSerializer.InputParams(auto_hang_up=bool(sid and token)),
        )
        params = FastAPIWebsocketParams(
            audio_in_enabled=True, audio_out_enabled=True, add_wav_header=False,
            serializer=serializer,
        )
        transport = FastAPIWebsocketTransport(websocket=websocket, params=params)
        # One shared calendar file, like the clinic's real booking system would be.
        calendar = Calendar(cfg, path=OUTPUTS_DIR / "calendar.json")
        await run_call(transport, cfg, ToolExecutor(cfg, calendar, retriever), sample_rate=8000)

    return app


def main(argv: list[str] | None = None) -> int:
    """Run the Twilio server: cva-serve --port 8765 (expose it with a tunnel)."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0
