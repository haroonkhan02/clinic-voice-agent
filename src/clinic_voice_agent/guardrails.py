"""Deterministic guardrails that run on every caller turn *before* the LLM.

1. Emergencies: phrases that suggest a medical emergency skip the LLM entirely. The caller
   hears a fixed script (hang up, call 112 / 911) and the call is flagged for staff. A false
   alarm costs one unnecessary sentence; a missed emergency could cost a life, so the patterns
   lean towards recall and deliberately ignore negation ("no chest pain" still triggers).
2. Medical advice: questions about diagnoses, symptoms' meaning, doses or stopping medicines
   are routed to a clinician instead of being answered.
3. PII redaction for logs and transcripts: phone numbers, e-mail addresses and dates.

Rules are plain regular expressions so they are fast (well under a millisecond), auditable and
testable. The LLM's system prompt repeats the same rules as a second layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

EMERGENCY_PATTERNS: dict[str, str] = {
    "chest_pain": r"\bchest (?:pain|pains|tightness|pressure)\b"
    r"|\b(?:pain|pressure|tightness) in (?:my|his|her|the) chest\b"
    r"|\bheart attack\b",
    "breathing": r"\b(?:can(?:no|')?t|cannot|can not|struggling to|hard to|trouble) "
    r"breath(?:e|ing)\b"
    r"|\bnot breathing\b|\bstopped breathing\b|\bchoking\b|\bshort(?:ness)? of breath\b",
    "unconscious": r"\bunconscious\b|\bpassed out\b|\bcollapsed\b|\bunresponsive\b"
    r"|\bwon'?t wake up\b|\bnot waking up\b",
    "stroke": r"\bstroke\b|\bface (?:is )?drooping\b|\bslurred speech\b|\bslurring\b"
    r"|\bcan'?t (?:move|feel) (?:my|his|her|one) (?:left |right )?(?:arm|leg|side)\b",
    "bleeding": r"\b(?:heavy|heavily|severe|a lot of|lots of|won'?t stop|uncontrolled) bleed"
    r"|\bbleeding (?:heavily|a lot|badly|won'?t stop)\b"
    r"|\bbleeding (?:that )?(?:won'?t|doesn'?t) stop\b",
    "seizure": r"\bseizure\b|\bfitting\b|\bconvuls",
    "anaphylaxis": r"\banaphyla"
    r"|\b(?:throat|tongue|lips?) (?:is |are )?(?:swelling|swollen|closing)\b"
    r"|\bsevere allergic\b",
    "self_harm": r"\b(?:kill|hurt|harm) (?:my ?self|himself|herself)\b|\bsuicid"
    r"|\bend(?:ing)? (?:my|his|her) (?:own )?life\b|\bwant to die\b|\boverdos",
    "poisoning": r"\bswallowed (?:bleach|poison|pills|tablets|batter)|\bpoison(?:ed|ing)\b",
    "injury": r"\bhit (?:his|her|my) head\b.*\b(?:vomit|confus|unconscious)"
    r"|\bbroken (?:bone|neck)\b"
    r"|\bcar (?:crash|accident)\b",
}

MEDICAL_ADVICE_PATTERNS: dict[str, str] = {
    "dose": r"\bhow (?:much|many) (?:\w+ ){0,3}(?:should|can|do) (?:i|he|she|we) (?:take|give)\b"
    r"|\bdos(?:e|age)\b|\bmg\b|\bmilligrams?\b"
    r"|\b(?:can|should) i give (?:him|her|them|my \w+) (?:something|any|ibuprofen|paracetamol"
    r"|medicine|calpol)",
    "medication_change": r"\b(?:should|can|could) (?:i|he|she) (?:stop|start|double|skip|"
    r"increase|reduce|mix|combine) (?:taking )?(?:my |his |her )?(?:\w+ ){0,3}"
    r"(?:medic|tablet|pill|dose|antibiotic|insulin|ibuprofen|paracetamol)"
    r"|\bsafe to (?:take|mix|combine)\b|\bside effects?\b|\binteract",
    "diagnosis": r"\bwhat (?:do you think )?"
    r"(?:is wrong|could it be|might it be|does (?:it|this) mean)"
    r"|\b(?:is|if) (?:it|this|that|this \w+|it \w+) (?:is )?"
    r"(?:serious|cancer|dangerous|normal|contagious|infected)\b"
    r"|\bdo (?:i|you think i) have\b|\bdiagnos",
    "treatment": r"\bhow (?:do|should|can) i (?:treat|cure|get rid of)\b"
    r"|\bwhat (?:should|can) i (?:take|do) (?:for|about) (?:my|the|this|a|his|her)\b"
    r"|\bwhich (?:medicine|medication|antibiotic|cream|tablet)s?\b",
    "results_interpretation": r"\b(?:is|are) my (?:\w+ )?(?:results?|levels?|numbers?) "
    r"(?:ok|okay|normal|bad|good|high|low)\b|\bwhat do my results mean\b"
    r"|\bmy (?:\w+ )?(?:levels?|sugar|pressure|results?) (?:is |are )?too (?:high|low)\b",
}

_EMERGENCY = {k: re.compile(v, re.I) for k, v in EMERGENCY_PATTERNS.items()}
_MEDICAL = {k: re.compile(v, re.I) for k, v in MEDICAL_ADVICE_PATTERNS.items()}


@dataclass(frozen=True)
class GuardrailResult:
    action: str  # "emergency" | "medical_advice" | "ok"
    categories: tuple[str, ...] = ()

    @property
    def blocked(self) -> bool:
        return self.action != "ok"


def _normalise(text: str) -> str:
    # STT output may use curly apostrophes; patterns use straight ones.
    return text.replace("’", "'").replace("‘", "'")


def check_emergency(text: str) -> tuple[str, ...]:
    text = _normalise(text)
    return tuple(name for name, rx in _EMERGENCY.items() if rx.search(text))


def check_medical_advice(text: str) -> tuple[str, ...]:
    text = _normalise(text)
    return tuple(name for name, rx in _MEDICAL.items() if rx.search(text))


def screen(text: str) -> GuardrailResult:
    """Classify one caller utterance. Emergencies take priority over everything else."""
    if found := check_emergency(text):
        return GuardrailResult("emergency", found)
    if found := check_medical_advice(text):
        return GuardrailResult("medical_advice", found)
    return GuardrailResult("ok")


def emergency_script(emergency_number: str = "112") -> str:
    return (
        "This sounds like it could be an emergency. Please hang up now and call "
        f"{emergency_number} straight away, or 911 if you are in the United States. "
        "I am also alerting our clinic staff."
    )


def medical_advice_script() -> str:
    return (
        "I'm the clinic's booking assistant, so I can't give medical advice. "
        "A doctor or nurse can help with that. Would you like me to book an appointment "
        "or leave a message for them to call you back?"
    )


_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{6,}\d(?!\w)")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_DATE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)? (?:of )?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"[a-z]* \d{4}\b",
    re.I,
)


def redact(text: str) -> str:
    """Mask phone numbers, e-mail addresses and dates (e.g. dates of birth) for logging."""
    text = _EMAIL.sub("[EMAIL]", text)
    text = _DATE.sub("[DATE]", text)
    return _PHONE.sub("[PHONE]", text)
