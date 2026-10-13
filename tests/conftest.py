from datetime import datetime

import pytest

from clinic_voice_agent.config import load_clinic_config
from clinic_voice_agent.scheduling import Calendar

# Monday 12 October 2026, 09:00 clinic time.
NOW = datetime(2026, 10, 12, 9, 0)


@pytest.fixture
def cfg():
    return load_clinic_config()


@pytest.fixture
def calendar(cfg):
    return Calendar(cfg, now=lambda: NOW)
