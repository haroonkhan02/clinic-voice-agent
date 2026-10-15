"""Real-time voice pipeline (Pipecat): phone audio in, phone audio out.

    transport.input → Deepgram Flux STT → GuardrailGate → user aggregator → Claude (+ tools)
        → Deepgram TTS → transport.output → assistant aggregator

Flux does the turn detection (end of turn from acoustics *and* words), and Silero VAD in the
user aggregator lets the caller interrupt the bot (barge-in). `GuardrailGate` screens every
final transcript before it reaches the context: emergencies and medical-advice requests are
answered with the fixed scripts and never reach Claude, exactly as in text mode.

Without a Deepgram key (or with CVA_SPEECH=local) the speech stages run on the machine:
Silero VAD segments the audio, faster-whisper (`base.en`, int8 on the CPU) transcribes each
segment, and Kokoro speaks the replies. Turn-taking is then plain VAD silence, not Flux's
semantic end-of-turn.

Tools use the same `ToolExecutor` as the text harness. Pipecat's portable tool schema has no
`strict` flag, so here the executor's own validation is the only argument check.

Pipecat is an optional dependency (`pip install -e ".[voice]"`); this module imports it lazily
so the rest of the package and the CI tests don't need it.
"""

from __future__ import annotations

import os
from typing import Any

from clinic_voice_agent.agent import EFFORT, MODEL, SYSTEM_PROMPT
from clinic_voice_agent.config import ClinicConfig
from clinic_voice_agent.guardrails import emergency_script, medical_advice_script, screen
from clinic_voice_agent.tools import ToolExecutor, build_tool_specs

TTS_VOICE = "aura-2-thalia-en"
FLUX_MODEL = "flux-general-en"
LOCAL_WHISPER = "base.en"
LOCAL_VOICE = "af_heart"
GREETING = "Hello, you've reached {name}. This is the automated receptionist. How can I help?"


def voice_system_prompt(cfg: ClinicConfig, now) -> str:
    base = SYSTEM_PROMPT.format(name=cfg.name, address=cfg.address, emergency=cfg.emergency_number)
    return base + f"\nCurrent local time at the clinic: {now:%A %d %B %Y, %H:%M} ({now:%Y-%m-%d})."


def speech_backend(env=os.environ) -> str:
    """'deepgram' or 'local'. CVA_SPEECH forces one; otherwise Deepgram when its key is set."""
    forced = env.get("CVA_SPEECH", "").strip().lower()
    if forced in ("deepgram", "local"):
        if forced == "deepgram" and not env.get("DEEPGRAM_API_KEY"):
            raise RuntimeError("CVA_SPEECH=deepgram but DEEPGRAM_API_KEY is not set")
        return forced
    return "deepgram" if env.get("DEEPGRAM_API_KEY") else "local"


def speech_services(env=os.environ):
    """(VAD processor or None, STT, TTS, aggregator VAD analyzer or None) for the backend."""
    from pipecat.audio.vad.silero import SileroVADAnalyzer

    if speech_backend(env) == "deepgram":
        from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
        from pipecat.services.deepgram.tts import DeepgramTTSService

        stt = DeepgramFluxSTTService(
            api_key=env["DEEPGRAM_API_KEY"],
            settings=DeepgramFluxSTTService.Settings(model=FLUX_MODEL),
        )
        tts = DeepgramTTSService(
            api_key=env["DEEPGRAM_API_KEY"], settings=DeepgramTTSService.Settings(voice=TTS_VOICE)
        )
        # Flux finds the end of turn itself; VAD in the aggregator only handles barge-in.
        return None, stt, tts, SileroVADAnalyzer()

    from pipecat.processors.audio.vad_processor import VADProcessor
    from pipecat.services.kokoro.tts import KokoroTTSService
    from pipecat.services.whisper.stt import WhisperSTTService

    from clinic_voice_agent.config import ROOT
    from clinic_voice_agent.local_speech import kokoro_paths

    model, voices = kokoro_paths(ROOT / "models")
    stt = WhisperSTTService(device="cpu", compute_type="int8",
                            settings=WhisperSTTService.Settings(model=LOCAL_WHISPER))
    tts = KokoroTTSService(model_path=str(model), voices_path=str(voices),
                           settings=KokoroTTSService.Settings(voice=LOCAL_VOICE))
    # Whisper transcribes whole segments, so VAD must run before it, not in the aggregator.
    return VADProcessor(vad_analyzer=SileroVADAnalyzer()), stt, tts, None


def guardrail_reply(text: str, cfg: ClinicConfig) -> tuple[str, str] | None:
    """(action, scripted reply) when a transcript must not reach the LLM, else None."""
    guard = screen(text)
    if guard.action == "emergency":
        return guard.action, emergency_script(cfg.emergency_number)
    if guard.action == "medical_advice":
        return guard.action, medical_advice_script()
    return None


def function_schemas(cfg: ClinicConfig):
    """Clinic tools as Pipecat FunctionSchemas (same names, descriptions and properties)."""
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.adapters.schemas.tools_schema import ToolsSchema

    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name=spec["name"],
                description=spec["description"],
                properties=spec["input_schema"]["properties"],
                required=spec["input_schema"]["required"],
            )
            for spec in build_tool_specs(cfg)
        ]
    )


def make_guardrail_gate(cfg: ClinicConfig, context, executor: ToolExecutor, events: list):
    """A FrameProcessor that answers blocked transcripts itself and swallows them."""
    from pipecat.frames.frames import TranscriptionFrame, TTSSpeakFrame
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

    class GuardrailGate(FrameProcessor):
        async def process_frame(self, frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame) and frame.text.strip():
                blocked = guardrail_reply(frame.text, cfg)
                if blocked:
                    action, reply = blocked
                    events.append({"action": action, "text": frame.text})
                    context.add_message({"role": "user", "content": frame.text})
                    if action == "emergency":
                        executor.run("transfer_to_human", {"reason": "emergency (voice)"})
                    # append_to_context: the assistant aggregator records what was said.
                    await self.push_frame(TTSSpeakFrame(reply), direction)
                    return
            await self.push_frame(frame, direction)

    return GuardrailGate()


def build_pipeline(transport, cfg: ClinicConfig, executor: ToolExecutor, env=os.environ):
    """Assemble the Pipecat pipeline for one call. Returns (pipeline, context, events)."""
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.services.anthropic.llm import AnthropicLLMService
    from pipecat.services.llm_service import FunctionCallParams

    vad, stt, tts, aggregator_vad = speech_services(env)
    # Without a key the pipeline still runs: greetings and the scripted guardrail replies work,
    # but every turn that needs Claude fails (that is how `cva-phone-sim` tests the safety path).
    llm = AnthropicLLMService(
        api_key=env.get("ANTHROPIC_API_KEY", ""),
        settings=AnthropicLLMService.Settings(
            model=MODEL,
            system_instruction=voice_system_prompt(cfg, executor.calendar.now()),
            enable_prompt_caching=True,
            extra={"output_config": {"effort": EFFORT}},
        ),
    )

    async def run_tool(params: FunctionCallParams):
        result = executor.run(params.function_name, dict(params.arguments or {}))
        await params.result_callback(result)

    llm.register_function(None, run_tool)  # one catch-all handler: the executor dispatches

    context = LLMContext(tools=function_schemas(cfg))
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=aggregator_vad)
    )
    events: list[dict[str, Any]] = []
    gate = make_guardrail_gate(cfg, context, executor, events)
    pipeline = Pipeline(
        [
            transport.input(),
            *([vad] if vad else []),
            stt,
            gate,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )
    return pipeline, context, events


async def run_call(transport, cfg: ClinicConfig, executor: ToolExecutor, sample_rate: int,
                   handle_sigint: bool = False, local: bool = False) -> list[dict[str, Any]]:
    """Run one call until the caller hangs up; returns the guardrail events.

    `local=True` is the microphone mode: there is no client to connect, so the greeting is
    queued straight away and the call ends with Ctrl+C.
    """
    from pipecat.frames.frames import TTSSpeakFrame
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.workers.runner import WorkerRunner

    pipeline, _context, events = build_pipeline(transport, cfg, executor)
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            # Whisper reads raw samples as 16 kHz, so the local path resamples phone audio up
            # (the Twilio serializer converts 8 kHz μ-law to the pipeline's input rate).
            audio_in_sample_rate=max(sample_rate, 16000) if speech_backend() == "local"
            else sample_rate,
            audio_out_sample_rate=sample_rate,
            enable_metrics=True,  # per-service TTFB / processing time in the logs
            enable_usage_metrics=True,
        ),
    )
    runner = WorkerRunner(handle_sigint=handle_sigint)
    await runner.add_workers(worker)

    greeting = TTSSpeakFrame(GREETING.format(name=cfg.name))
    if local:
        await worker.queue_frames([greeting])
    else:

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            await worker.queue_frames([greeting])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            await runner.cancel()

    await runner.run()
    return events
