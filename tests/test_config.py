from datetime import time

import pytest

from clinic_voice_agent.config import (
    DEFAULT_CLINIC_CONFIG,
    available_services,
    load_clinic_config,
    stage_ready,
)


def test_default_clinic_config_loads():
    cfg = load_clinic_config()
    assert cfg.hours["monday"] == (time(8, 0), time(18, 0))
    assert "saturday" not in cfg.hours
    assert cfg.appointment_types["blood_test"].latest_start == time(10, 0)
    assert cfg.provider_name("nurse_lin") == "Nurse Mei Lin"


def test_invalid_duration_rejected(tmp_path):
    src = tmp_path / "clinic.yaml"
    src.write_text(DEFAULT_CLINIC_CONFIG.read_text().replace("minutes: 30", "minutes: 20"))
    with pytest.raises(ValueError, match="multiple of slot_minutes"):
        load_clinic_config(src)


def test_services_report_names_only():
    env = {"DEEPGRAM_API_KEY": "secret-value", "TWILIO_ACCOUNT_SID": "AC123"}
    found = available_services(env)
    assert found["stt"] == ["DEEPGRAM_API_KEY"]
    assert found["tts"] == ["DEEPGRAM_API_KEY"]
    assert found["llm"] == []
    assert "secret-value" not in repr(found)
    assert stage_ready("stt", env)
    # Twilio needs both the SID and the auth token.
    assert not stage_ready("telephony", env)
    assert stage_ready("telephony", {**env, "TWILIO_AUTH_TOKEN": "x"})


def test_blank_key_counts_as_missing():
    assert not stage_ready("llm", {"ANTHROPIC_API_KEY": "   "})
