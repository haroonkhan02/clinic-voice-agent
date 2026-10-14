import struct

from clinic_voice_agent.audio import (
    mulaw_to_pcm16,
    normalise_words,
    pcm16_duration,
    trim_trailing_silence,
    word_errors,
)
from clinic_voice_agent.speech_eval import Item, ItemResult, collect_items, summarise


def test_mulaw_known_values():
    # G.711 reference points: 0x00/0x80 are the extremes, 0x7F/0xFF are zero.
    pcm = struct.unpack("<4h", mulaw_to_pcm16(bytes([0x00, 0x80, 0x7F, 0xFF])))
    assert pcm == (-32124, 32124, 0, 0)
    assert pcm16_duration(mulaw_to_pcm16(bytes(8000))) == 1.0


def test_mulaw_is_monotonic_within_each_sign():
    pcm = struct.unpack("<256h", mulaw_to_pcm16(bytes(range(256))))
    negative, positive = pcm[:128], pcm[128:]
    assert list(negative) == sorted(negative)  # 0x00 most negative → 0x7F zero
    assert list(positive) == sorted(positive, reverse=True)


def test_trim_trailing_silence():
    loud = struct.pack("<160h", *([5000] * 160))  # 20 ms
    quiet = bytes(320 * 10)  # 200 ms of zeros
    assert trim_trailing_silence(loud + quiet) == loud
    assert trim_trailing_silence(quiet) == b""


def test_normalisation_and_word_errors():
    assert normalise_words("Born 4th of July, 0170!") == normalise_words(
        "born fourth of july zero one seven zero"
    )
    assert word_errors("I can't breathe", "i cant breathe.") == (0, 3)
    assert word_errors("chest pain", "chess pain") == (1, 2)
    assert word_errors("a b c", "") == (3, 3)


def _result(text, label, batch, flux, eot=0.5):
    from clinic_voice_agent.guardrails import screen

    return ItemResult(
        text=text, label=label, source="t", audio_s=1.0, tts_ttfb_s=0.3, tts_total_s=0.6,
        batch_text=batch, batch_latency_s=0.2, flux_text=flux, flux_eot_latency_s=eot,
        flux_premature_eots=0,
        guard_text=screen(text).action, guard_batch=screen(batch).action,
        guard_flux=screen(flux).action, batch_edits=word_errors(text, batch)[0],
        flux_edits=word_errors(text, flux)[0], ref_words=word_errors(text, batch)[1],
    )


def test_summary_counts_guardrail_flips():
    results = [
        _result("my chest pain is bad", "emergency", "my chest pain is bad",
                "my chess pain is bad"),
        _result("what time do you open", "ok", "what time do you open", "what time do you open",
                eot=None),
    ]
    s = summarise(results)
    assert s["engines"]["batch"]["guardrail_agreement"] == 1.0
    assert s["engines"]["flux"]["guardrail_agreement"] == 0.5
    assert s["engines"]["flux"]["recall"]["emergency"] == {"on_text": 1.0, "on_transcript": 0.0,
                                                           "n": 1}
    assert s["flux_no_eot"] == 1 and s["flux_eot_latency_s"]["n"] == 1
    assert abs(s["engines"]["flux"]["wer"] - 1 / 10) < 1e-9


def test_collect_items_dedupes_scenario_lines():
    class S:
        def __init__(self, i, caller):
            self.id, self.caller = i, caller

    items = collect_items([("hello", "ok")], [S("a", ["hello", "bye"]), S("b", ["bye"])])
    assert [(i.text, i.label) for i in items] == [("hello", "ok"), ("bye", None)]
    assert isinstance(items[1], Item) and items[1].source == "scenario:a"
