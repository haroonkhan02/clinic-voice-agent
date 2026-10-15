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


def test_phone_sim_speaks_twilio_media_streams():
    import base64 as b64

    from clinic_voice_agent.phone_sim import FRAME_BYTES, media_event, start_events

    connected, start = start_events()
    assert connected["event"] == "connected"
    assert start["start"]["mediaFormat"] == {"encoding": "audio/x-mulaw", "sampleRate": 8000,
                                             "channels": 1}
    msg = media_event(b"\xff" * FRAME_BYTES, 7)
    assert msg["event"] == "media" and msg["media"]["timestamp"] == "140"
    assert b64.b64decode(msg["media"]["payload"]) == b"\xff" * 160


def test_pipecat_detects_the_simulated_stream_as_twilio():
    pytest.importorskip("pipecat")
    from pipecat.runner.utils import _detect_transport_type_from_message

    from clinic_voice_agent.phone_sim import start_events

    assert _detect_transport_type_from_message(start_events()[1]) == "twilio"


def test_phone_sim_summary():
    from clinic_voice_agent.phone_sim import expected_phrase, summarise

    rows = [{"voice_to_voice_s": 0.5, "correct": True}, {"voice_to_voice_s": 0.9, "correct": True},
            {"voice_to_voice_s": None, "correct": False}]
    s = summarise(rows)
    assert (s["n_calls"], s["replied"], s["correct_script"]) == (3, 2, 2)
    assert s["voice_to_voice_median_s"] == 0.7 and s["voice_to_voice_max_s"] == 0.9
    assert expected_phrase("emergency") == "112" and expected_phrase("ok") is None


def test_speech_backend_selection():
    from clinic_voice_agent.voice import speech_backend

    assert speech_backend({"DEEPGRAM_API_KEY": "k"}) == "deepgram"
    assert speech_backend({}) == "local"
    assert speech_backend({"DEEPGRAM_API_KEY": "k", "CVA_SPEECH": "local"}) == "local"
    with pytest.raises(RuntimeError):
        speech_backend({"CVA_SPEECH": "deepgram"})
