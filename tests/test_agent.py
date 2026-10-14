import json
from types import SimpleNamespace

import pytest

from clinic_voice_agent.agent import FALLBACK_REPLY, ClinicAgent, LLMResult
from clinic_voice_agent.knowledge import load_retriever
from clinic_voice_agent.tools import ToolExecutor


def text(t):
    return SimpleNamespace(content=[{"type": "text", "text": t}], stop_reason="end_turn")


def tool(name, args, tid="tu_1", say=None):
    content = [{"type": "text", "text": say}] if say else []
    content.append({"type": "tool_use", "id": tid, "name": name, "input": args})
    return SimpleNamespace(content=content, stop_reason="tool_use")


class ScriptedLLM:
    """Stands in for Claude: returns canned messages and records each request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, kwargs):
        # Snapshot the messages: the agent keeps appending to the same list.
        self.requests.append({**kwargs, "messages": json.loads(json.dumps(kwargs["messages"]))})
        return LLMResult(self.responses.pop(0), ttft_s=0.1, total_s=0.2)


@pytest.fixture
def make_agent(cfg, calendar):
    def make(*responses):
        llm = ScriptedLLM(*responses)
        ex = ToolExecutor(cfg, calendar, load_retriever())
        return ClinicAgent(cfg, ex, llm_call=llm), llm

    return make


def test_emergency_skips_the_llm(make_agent):
    agent, llm = make_agent()
    turn = agent.respond("my husband collapsed and isn't breathing")
    assert turn.guardrail == "emergency"
    assert "112" in turn.reply and "911" in turn.reply
    assert llm.requests == [] and agent.escalated
    assert agent.executor.transfers and agent.executor.transfers[0].startswith("emergency")


def test_medical_advice_is_scripted_and_kept_in_history(make_agent):
    agent, llm = make_agent(text("Of course, who is the message for?"))
    turn = agent.respond("how much ibuprofen can I give my son?")
    assert turn.guardrail == "medical_advice" and llm.requests == []
    agent.respond("then I'd like to leave a message")
    sent = llm.requests[0]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert "medical advice" in sent[1]["content"]


def test_tool_loop_runs_tools_and_returns_results_in_one_message(make_agent):
    agent, llm = make_agent(
        tool("find_available_slots", {"appointment_type": "general", "date": "2026-10-13",
                                      "provider": None, "earliest_time": None}),
        text("I have Tuesday at eight. Would that work?"),
    )
    turn = agent.respond("I need to see a doctor tomorrow")
    assert turn.reply.startswith("I have Tuesday")
    assert turn.llm_calls == 2 and turn.tool_calls[0]["ok"]
    second = llm.requests[1]["messages"]
    results = second[-1]["content"]
    assert second[-1]["role"] == "user" and results[0]["type"] == "tool_result"
    assert results[0]["tool_use_id"] == "tu_1" and results[0]["is_error"] is False
    assert json.loads(results[0]["content"])["slots"]


def test_unconfirmed_booking_is_refused_by_the_executor(make_agent):
    args = {"patient_name": "Ali Demir", "date_of_birth": "2001-01-01", "phone": "0160",
            "appointment_type": "vaccination", "start": "2026-10-13T10:00", "provider": None,
            "caller_confirmed": False}
    agent, llm = make_agent(tool("book_appointment", args), text("Shall I book it?"))
    agent.respond("book me a vaccination tomorrow at ten")
    result = llm.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True and "not confirmed" in result["content"]
    assert agent.executor.calendar.appointments == {}


def test_refusal_hands_over_to_a_person(make_agent):
    agent, _ = make_agent(SimpleNamespace(content=[], stop_reason="refusal"))
    turn = agent.respond("something the model declines")
    assert turn.reply == FALLBACK_REPLY and agent.executor.transfers == ["model declined"]
    assert agent.messages[-1] == {"role": "assistant", "content": FALLBACK_REPLY}


def test_tool_round_limit(make_agent):
    q = {"question": "hours"}
    agent, llm = make_agent(*[tool("search_faq", q, tid=f"t{i}") for i in range(3)])
    agent.max_tool_rounds = 3
    assert agent.respond("when are you open").reply == FALLBACK_REPLY
    assert len(llm.requests) == 3


def test_request_shape(make_agent):
    agent, llm = make_agent(text("Hello"))
    agent.respond("hello")
    req = llm.requests[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["output_config"] == {"effort": "low"} and req["fallbacks"] == "default"
    assert "thinking" not in req  # Opus 5.5 always thinks; effort is the control
    stable, clock = req["system"]
    assert "cache_control" in stable and "2026-10-12" in clock["text"]
    assert all(t["strict"] for t in req["tools"])
