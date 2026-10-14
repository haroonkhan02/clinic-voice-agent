"""Small audio and text helpers for the phone line (pure Python, no numpy needed).

Twilio Media Streams carry G.711 μ-law at 8 kHz, one byte per sample. Deepgram Flux wants
16-bit linear PCM, so the speech evaluation decodes μ-law exactly like the phone path does.
"""

from __future__ import annotations

import re
import struct

PHONE_RATE = 8000


def _mulaw_to_linear(byte: int) -> int:
    # ITU-T G.711: invert, split into sign / exponent / mantissa, add the bias back.
    u = ~byte & 0xFF
    sign, exponent, mantissa = u & 0x80, (u >> 4) & 0x07, u & 0x0F
    sample = (((mantissa << 3) + 0x84) << exponent) - 0x84
    return -sample if sign else sample


_MULAW_TABLE = [_mulaw_to_linear(b) for b in range(256)]


def mulaw_to_pcm16(data: bytes) -> bytes:
    """Decode μ-law bytes to little-endian 16-bit PCM (two bytes per sample)."""
    return struct.pack(f"<{len(data)}h", *(_MULAW_TABLE[b] for b in data))


def pcm16_duration(pcm: bytes, rate: int = PHONE_RATE) -> float:
    return len(pcm) / 2 / rate


def trim_trailing_silence(
    pcm: bytes, rate: int = PHONE_RATE, threshold: int = 300, frame_ms: int = 20
) -> bytes:
    """Cut the quiet tail TTS adds, so end-of-turn latency is timed from the last speech."""
    frame = rate * frame_ms // 1000 * 2
    end = len(pcm) - len(pcm) % frame
    while end > 0:
        chunk = pcm[end - frame : end]
        samples = struct.unpack(f"<{len(chunk) // 2}h", chunk)
        if max(abs(s) for s in samples) > threshold:
            break
        end -= frame
    return pcm[:end]


_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
# Number words STT may write as digits (or the other way round). Both sides of the comparison
# go through the same mapping, so "fourth" and "4th" count as the same word.
_NUMBER_WORDS = {
    "ten": "10", "eleven": "11", "twelve": "12", "fifteen": "15", "twenty": "20",
    "thirty": "30", "first": "1", "second": "2", "third": "3", "fourth": "4",
}


def normalise_words(text: str) -> list[str]:
    """Lower-case, drop punctuation, spell digits one by one so '0170' == 'zero one seven zero'."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text.replace("'", ""))
    words = []
    for word in text.split():
        word = _NUMBER_WORDS.get(word.strip("'"), word.strip("'"))
        if word.isdigit():
            words.extend(_DIGITS[int(d)] for d in word)
        elif word:
            words.append(word)
    return words


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    """(word edits, reference length) after normalisation; WER = sum(edits) / sum(lengths)."""
    r, h = normalise_words(ref), normalise_words(hyp)
    return edit_distance(r, h), len(r)
