import base64
import hashlib
import hmac
from datetime import datetime

import pytest

from clinic_voice_agent.server import twiml_for, valid_twilio_signature
from clinic_voice_agent.voice import guardrail_reply, voice_system_prompt


def test_twiml_points_the_stream_at_media():
    xml = twiml_for("abc123.ngrok.app")
    assert '<Stream url="wss://abc123.ngrok.app/media" />' in xml
    assert "wss://abc.example/media" in twiml_for("https://abc.example/")
    for bad in ("", "evil\" onload=\"x", "a b"):
        with pytest.raises(ValueError):
            twiml_for(bad)


def test_twilio_signature_matches_an_independent_hmac():
    token, url = "test-token", "https://abc.example/voice"
    params = {"CallSid": "CA123", "From": "+15550000000", "To": "+15551111111"}
    payload = url + "CallSidCA123From+15550000000To+15551111111"
    sig = base64.b64encode(hmac.new(token.encode(), payload.encode(), hashlib.sha1).digest())
    assert valid_twilio_signature(token, url, params, sig.decode())
    assert not valid_twilio_signature(token, url, {**params, "To": "+1"}, sig.decode())
    assert not valid_twilio_signature(token, url, params, "")


def test_guardrail_reply_matches_text_mode(cfg):
    action, reply = guardrail_reply("my wife collapsed", cfg)
    assert action == "emergency" and "112" in reply
    assert guardrail_reply("what dose of paracetamol", cfg)[0] == "medical_advice"
    assert guardrail_reply("can I book a flu jab", cfg) is None


def test_voice_prompt_has_clinic_and_clock(cfg):
    prompt = voice_system_prompt(cfg, datetime(2026, 10, 12, 9, 0))
    assert cfg.name in prompt and "2026-10-12" in prompt and "112" in prompt


def test_pipeline_builds_without_network(cfg, calendar):
    pytest.importorskip("pipecat")
    from clinic_voice_agent.knowledge import load_retriever
    from clinic_voice_agent.tools import ToolExecutor, build_tool_specs
    from clinic_voice_agent.voice import build_pipeline, function_schemas

    names = [f.name for f in function_schemas(cfg).standard_tools]
    assert names == [t["name"] for t in build_tool_specs(cfg)]

    class FakeTransport:
        def input(self):
            from pipecat.processors.frame_processor import FrameProcessor

            return FrameProcessor()

        output = input

    ex = ToolExecutor(cfg, calendar, load_retriever())
    env = {"DEEPGRAM_API_KEY": "dummy", "ANTHROPIC_API_KEY": "dummy"}
    pipeline, context, events = build_pipeline(FakeTransport(), cfg, ex, env=env)
    assert events == [] and len(context.tools.standard_tools) == len(names)


def test_server_answers_twilio_webhook(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from clinic_voice_agent.server import create_app

    monkeypatch.setenv("PUBLIC_HOST", "abc.example")
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    client = TestClient(create_app())
    r = client.post("/voice", data={"CallSid": "CA1"})
    assert r.status_code == 200 and "wss://abc.example/media" in r.text
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret")
    assert client.post("/voice", data={"CallSid": "CA1"}).status_code == 403
