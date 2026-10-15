import json

import pytest

from clinic_voice_agent.plots import analyse, pearson, write_analysis


def test_pearson_known_values():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_analysis_from_saved_reports(tmp_path):
    items = [
        {"text": "a", "label": "emergency", "flux_eot_latency_s": 1.0},
        {"text": "b", "label": "emergency", "flux_eot_latency_s": 1.4},
        {"text": "c", "label": "medical_advice", "flux_eot_latency_s": 0.4},
        {"text": "d", "label": None, "flux_eot_latency_s": 0.9},
    ]
    calls = [{"text": t, "label": lab, "voice_to_voice_s": v}
             for t, lab, v in (("a", "emergency", 1.1), ("b", "emergency", 1.5),
                               ("c", "medical_advice", 0.5))]
    (tmp_path / "speech_roundtrip.json").write_text(json.dumps({"items": items}))
    (tmp_path / "phone_sim.json").write_text(json.dumps({"calls": calls}))
    result = analyse(tmp_path)
    assert result["eot_by_label"]["emergency"] == {"median": 1.2, "n": 2}
    assert "ok" not in result["eot_by_label"]  # unlabelled lines are left out
    assert result["sim_vs_eot"]["n"] == 3
    assert result["sim_vs_eot"]["median_remainder_s"] == pytest.approx(0.1)
    write_analysis(result, tmp_path)
    assert "r = 1.00" in (tmp_path / "latency_analysis.md").read_text()
