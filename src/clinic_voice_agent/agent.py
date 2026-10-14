"""The receptionist agent: guardrails first, then Claude with tool use.

One caller turn goes through three steps:

1. `guardrails.screen()` runs on the transcript. Emergencies and medical-advice requests get a
   fixed script and never reach the LLM, so the safety path doesn't depend on the model.
2. Otherwise the turn is sent to Claude with the clinic tools. The loop runs tools until Claude
   answers in text (at most `max_tool_rounds` model calls per turn).
3. The reply is plain spoken text: it goes to TTS in voice mode or is printed in text mode.

The history is append-only (full `content` blocks, including thinking blocks, are sent back
unchanged), which is what prompt caching and preserved thinking need. The model call itself is
injected (`llm_call`), so tests drive the loop with scripted responses and never touch the API.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from clinic_voice_agent.config import ClinicConfig
from clinic_voice_agent.guardrails import emergency_script, medical_advice_script, screen
from clinic_voice_agent.tools import ToolExecutor, build_tool_specs

MODEL = "claude-opus-5-5"
# Opus 5.5 always thinks; effort is the only dial. "low" keeps time-to-first-token short,
# which matters more on a phone line than depth of reasoning for admin requests.
EFFORT = "low"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

FALLBACK_REPLY = (
    "Sorry, I'm having trouble with that. Let me put you through to a member of our team."
)

SYSTEM_PROMPT = """\
You are the phone receptionist of {name}, {address}. You speak with callers on a phone line, \
so everything you write is read aloud by a text-to-speech voice.

How to speak:
- Keep each reply to one to three short sentences. Ask one question at a time.
- Plain spoken English only: no lists, markdown, emojis or abbreviations a voice would misread.
- Say times the way people do ("Tuesday the 13th at half past nine"), never ISO dates.

What you can do: answer questions about the clinic from search_faq, find free appointment \
slots, book, reschedule and cancel appointments, take a message, and transfer the call to a \
person. Use the tools for every fact about the clinic or the calendar; never guess opening \
hours, prices, policies or free times.

Rules:
- You are not a clinician. Never give medical advice, diagnoses, doses, or interpret results. \
Offer an appointment or a message for a doctor or nurse instead.
- If the caller describes an emergency, tell them to hang up and call {emergency} (911 in the \
United States) and transfer the call.
- Before booking, rescheduling or cancelling, read the details back (name, date of birth, type, \
day and time, provider) and wait for a clear yes. Only then call the tool with \
caller_confirmed set to true. If the caller hesitates or says they need to check, don't book.
- To change or cancel an appointment, first look it up with find_appointment using the full \
name and date of birth.
- Spell back phone numbers and dates of birth only when confirming.
- If search_faq returns nothing that answers the question, say you don't know and offer to \
take a message. Don't answer questions about other businesses.
- Test results are never given over the phone; follow the clinic policy from search_faq.
- If a tool returns an error, explain it simply and offer an alternative.
"""


def build_system(cfg: ClinicConfig, now: datetime) -> list[dict[str, Any]]:
    """Stable instructions first (cached), the volatile clock last."""
    stable = SYSTEM_PROMPT.format(
        name=cfg.name, address=cfg.address, emergency=cfg.emergency_number
    )
    clock = f"Current local time at the clinic: {now:%A %d %B %Y, %H:%M} ({now:%Y-%m-%d})."
    return [
        {"type": "text", "text": stable, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": clock},
    ]


@dataclass
class LLMResult:
    """What one model call returns, independent of the SDK's classes."""

    message: Any  # needs .content (list of blocks) and .stop_reason
    ttft_s: float | None  # time to the first streamed text token, if any text was streamed
    total_s: float


LLMCall = Callable[[dict[str, Any]], LLMResult]


def anthropic_llm_call(client: Any = None) -> LLMCall:
    """Real model call: streams so time-to-first-token can be measured."""
    import anthropic

    client = client or anthropic.Anthropic()

    def call(kwargs: dict[str, Any]) -> LLMResult:
        t0 = time.perf_counter()
        ttft = None
        with client.beta.messages.stream(**kwargs) as stream:
            for event in stream:
                if (
                    ttft is None
                    and event.type == "content_block_delta"
                    and getattr(event.delta, "type", "") == "text_delta"
                ):
                    ttft = time.perf_counter() - t0
            message = stream.get_final_message()
        return LLMResult(message, ttft, time.perf_counter() - t0)

    return call


@dataclass
class Turn:
    caller: str
    reply: str
    guardrail: str  # "emergency" | "medical_advice" | "ok"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: int = 0
    ttft_s: float | None = None  # first model call of the turn
    llm_total_s: float = 0.0
    stop_reason: str | None = None


def _block_get(block: Any, name: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


class ClinicAgent:
    def __init__(
        self,
        cfg: ClinicConfig,
        executor: ToolExecutor,
        llm_call: LLMCall | None = None,
        now: Callable[[], datetime] | None = None,
        model: str = MODEL,
        effort: str = EFFORT,
        max_tokens: int = 4096,
        max_tool_rounds: int = 6,
    ):
        self.cfg = cfg
        self.executor = executor
        self._llm_call = llm_call
        self.now = now or executor.calendar.now
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_tool_rounds = max_tool_rounds
        self.tools = build_tool_specs(cfg)
        self.messages: list[dict[str, Any]] = []
        self.turns: list[Turn] = []
        self.escalated = False

    @property
    def llm_call(self) -> LLMCall:
        if self._llm_call is None:
            self._llm_call = anthropic_llm_call()
        return self._llm_call

    def request_kwargs(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": build_system(self.cfg, self.now()),
            "tools": self.tools,
            "messages": self.messages,
            "output_config": {"effort": self.effort},
            # On a safety decline the API retries on Anthropic's recommended fallback model.
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }

    def _scripted(self, caller_text: str, reply: str, action: str, categories) -> Turn:
        # Keep the history coherent: the model sees what was said on the next turn.
        self.messages.append({"role": "user", "content": caller_text})
        self.messages.append({"role": "assistant", "content": reply})
        if action == "emergency":
            self.escalated = True
            reason = "emergency: " + ", ".join(categories)
            self.executor.run("transfer_to_human", {"reason": reason})
        return Turn(caller_text, reply, action)

    def respond(self, caller_text: str) -> Turn:
        guard = screen(caller_text)
        if guard.action == "emergency":
            turn = self._scripted(
                caller_text, emergency_script(self.cfg.emergency_number), "emergency",
                guard.categories,
            )
        elif guard.action == "medical_advice":
            turn = self._scripted(caller_text, medical_advice_script(), "medical_advice", ())
        else:
            turn = self._llm_turn(caller_text)
        self.turns.append(turn)
        return turn

    def _llm_turn(self, caller_text: str) -> Turn:
        turn = Turn(caller_text, "", "ok")
        self.messages.append({"role": "user", "content": caller_text})
        for _ in range(self.max_tool_rounds):
            result = self.llm_call(self.request_kwargs())
            turn.llm_calls += 1
            turn.llm_total_s += result.total_s
            if turn.ttft_s is None:
                turn.ttft_s = result.ttft_s
            message = result.message
            turn.stop_reason = message.stop_reason
            if message.stop_reason == "refusal":
                # Don't append a refused turn; hand over to a person instead.
                turn.reply = FALLBACK_REPLY
                self.messages.append({"role": "assistant", "content": FALLBACK_REPLY})
                self.executor.run("transfer_to_human", {"reason": "model declined"})
                return turn
            self.messages.append({"role": "assistant", "content": message.content})
            blocks = message.content
            text = " ".join(
                _block_get(b, "text", "") for b in blocks if _block_get(b, "type") == "text"
            ).strip()
            tool_uses = [b for b in message.content if _block_get(b, "type") == "tool_use"]
            if message.stop_reason != "tool_use" or not tool_uses:
                turn.reply = text or FALLBACK_REPLY
                return turn
            results = []
            for block in tool_uses:
                name, args = _block_get(block, "name"), _block_get(block, "input") or {}
                output = self.executor.run(name, args)
                turn.tool_calls.append({"name": name, "args": args, "ok": output["ok"]})
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": _block_get(block, "id"),
                        "content": json.dumps(output, ensure_ascii=False),
                        "is_error": not output["ok"],
                    }
                )
            # All results of one assistant message go back in a single user message.
            self.messages.append({"role": "user", "content": results})
        turn.reply = FALLBACK_REPLY
        return turn
