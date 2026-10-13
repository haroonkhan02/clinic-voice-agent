"""Clinic FAQ knowledge base and a small BM25 retriever.

The knowledge base is a Markdown file in which every `##` section is one passage. Phone answers
must be short and grounded, so the agent gets the top passages (title + text) and is told to
answer only from them. When nothing scores above a threshold, the agent offers to take a message
instead of guessing.

BM25 is implemented here (about 40 lines) instead of pulling in a search library. The corpus
is a few dozen passages, so lexical retrieval is fast (microseconds) and easy to inspect.
Speech-to-text also produces plain words, so there is no markup to strip.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from clinic_voice_agent.config import DEFAULT_FAQ

_STOPWORD_TEXT = """
    a an and are as at be but by can could do does for from have how i if in is it its me my
    of on or our please should so that the their them then there this to us was we what when
    where which who will with would you your yes hi hello okay ok um uh just want need like
    tell know"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())

# Spoken questions use everyday words; map a few onto the vocabulary of the knowledge base.
SYNONYMS = {
    "open": "hours",
    "opening": "hours",
    "close": "hours",
    "closed": "hours",
    "located": "address",
    "location": "address",
    "car": "parking",
    "park": "parking",
    "wheelchair": "accessible",
    "jab": "vaccination",
    "shot": "vaccination",
    "vaccine": "vaccination",
    "vaccines": "vaccination",
    "bloods": "blood",
    "fast": "fasting",
    "eat": "fasting",
    "cancel": "cancelling",
    "reschedule": "rescheduling",
    "move": "rescheduling",
    "results": "result",
    "prescription": "prescriptions",
    "refill": "prescriptions",
    "specialist": "referrals",
    "referral": "referrals",
    "online": "video",
    "pay": "payment",
    "cost": "payment",
    "weekend": "hours",
    "saturday": "hours",
    "sunday": "hours",
    "night": "out-of-hours",
    "evening": "out-of-hours",
}

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _stem(word: str) -> str:
    """Very light suffix stripping, enough to match 'vaccinations' with 'vaccination'."""
    if len(word) > 4 and word.endswith(("ses", "xes", "ches", "shes")):
        return word[:-2]  # buses -> bus, boxes -> box
    for suffix in ("ations", "ation", "ings", "ing", "ies", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix) and not word.endswith("ss"):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN.findall(text.lower()):
        if raw in STOPWORDS:
            continue
        # Look synonyms up on the surface form and on the stem ("shots" -> "shot" -> ...).
        word = SYNONYMS.get(raw) or SYNONYMS.get(_stem(raw), raw)
        tokens.append(_stem(word))
    return tokens


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


@dataclass(frozen=True)
class Passage:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float


def load_passages(path: Path | str = DEFAULT_FAQ) -> list[Passage]:
    """Split the Markdown knowledge base into one passage per `##` section."""
    passages: list[Passage] = []
    title: str | None = None
    lines: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if title is not None:
                passages.append(Passage(slugify(title), title, " ".join(lines).strip()))
            title, lines = line[3:].strip(), []
        elif title is not None and line.strip():
            lines.append(line.strip())
    if title is not None:
        passages.append(Passage(slugify(title), title, " ".join(lines).strip()))
    ids = [p.id for p in passages]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate section titles in knowledge base")
    return passages


class BM25Retriever:
    """Okapi BM25 over passage title + text (the title is counted twice: it is a strong signal)."""

    def __init__(self, passages: list[Passage], k1: float = 1.2, b: float = 0.75):
        if not passages:
            raise ValueError("no passages")
        self.passages = passages
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(f"{p.title} {p.title} {p.text}")) for p in passages]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / len(self.lengths)
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(doc.keys())
        n = len(passages)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: str) -> list[float]:
        terms = tokenize(query)
        scores = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
            s = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
            scores.append(s)
        return scores

    def search(self, query: str, k: int = 3, min_score: float = 0.0) -> list[Hit]:
        scores = self.score(query)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [
            Hit(self.passages[i], scores[i])
            for i in order[:k]
            if scores[i] > min_score
        ]


def load_retriever(path: Path | str = DEFAULT_FAQ) -> BM25Retriever:
    return BM25Retriever(load_passages(path))
