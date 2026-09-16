"""Privacy-preserving per-turn telemetry and provider-rejection fallback.

No transcript text, raw audio, exception text, headers, credentials, prompts, or
provider response bodies enter the snapshot produced by this observer.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict, defaultdict
from typing import Awaitable, Callable

from api.errors.failure import (
    DograhFailure,
    ErrorSource,
    ErrorType,
    classify_exception,
    classify_message,
    failure_metadata_for_processor,
)
from api.services.pipecat.receptionist_runtime_policy import (
    ReceptionistRuntimePolicy,
)
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
)
from pipecat.metrics.metrics import TTFAMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed

PUBLIC_SAFE_PROVIDER_REJECTION_FALLBACK = (
    "I can't help with internal or restricted information. "
    "I can help with approved public information about Gen I Labs and AgentFlow."
)

_MAX_TURNS = 100
_MAX_OUTCOMES = 100
_MAX_FRAME_IDS = 4096
_PROVIDER_REJECTION_CODE_MARKERS = (
    "content-filter",
    "content-policy",
    "responsible-ai",
    "safety-policy",
)
_PROVIDER_REJECTION_MESSAGE_RE = re.compile(
    r"(?i)content[_ -]?filter|responsible ai|safety policy"
)


def _failure_for_frame(frame: ErrorFrame) -> DograhFailure:
    metadata = failure_metadata_for_processor(frame.processor)
    if frame.exception is not None:
        return classify_exception(
            frame.exception,
            source=metadata.source,
            provider=metadata.provider,
            error_owner=metadata.error_owner,
        )
    return classify_message(
        frame.error,
        source=metadata.source,
        provider=metadata.provider,
        error_owner=metadata.error_owner,
    )


def _is_provider_rejected_input(failure: DograhFailure, frame: ErrorFrame) -> bool:
    """Identify structured pre-response safety rejection without masking config faults."""

    if frame.fatal or failure.source != ErrorSource.LLM:
        return False
    provider_code = failure.provider_error_code or ""
    if any(marker in provider_code for marker in _PROVIDER_REJECTION_CODE_MARKERS):
        return True
    # Some OpenAI-compatible clients omit the structured code while retaining
    # the provider's safety category in the in-memory exception. This value is
    # inspected only for classification and is never persisted or returned.
    return failure.code.endswith("-400") and bool(
        _PROVIDER_REJECTION_MESSAGE_RE.search(failure.internal_message)
    )


def _failure_category(failure: DograhFailure, frame: ErrorFrame) -> str:
    if _is_provider_rejected_input(failure, frame):
        return "provider_rejected_input"
    if failure.type == ErrorType.CONFIG_ERROR:
        return "configuration_error"
    if failure.type == ErrorType.QUOTA_ERROR:
        return "quota_error"
    if failure.type == ErrorType.PROVIDER_ERROR:
        return "provider_unavailable"
    return "system_error"


class ReceptionistRuntimeObserver(BaseObserver):
    """Collect bounded categorical timing data and queue one safe TTS fallback."""

    def __init__(
        self,
        *,
        policy: ReceptionistRuntimePolicy,
        queue_fallback: Callable[[TTSSpeakFrame], Awaitable[None]],
    ) -> None:
        super().__init__()
        self._policy = policy
        self._queue_fallback = queue_fallback
        self._started = time.monotonic()
        self._frames_seen: OrderedDict[str, None] = OrderedDict()
        self._error_frames_seen: set[str] = set()
        self._turn = 0
        self._turns: dict[int, dict[str, object]] = {}
        self._stage_times: dict[int, dict[str, float]] = defaultdict(dict)
        self._outcomes: list[dict[str, object]] = []
        self._llm_text_seen = False
        self._fallback_queued_for_turn: set[int] = set()
        self._saw_final_transcription = False

    @property
    def saw_user_speech(self) -> bool:
        return self._saw_final_transcription

    def attach_turn_tracking_observer(self, observer) -> None:
        @observer.event_handler("on_turn_started")
        async def on_turn_started(_observer, turn_number: int):
            self._turn = turn_number
            self._turn_entry(turn_number)
            self._record_stage("turn_started")

        @observer.event_handler("on_turn_ended")
        async def on_turn_ended(
            _observer,
            turn_number: int,
            _duration: float,
            was_interrupted: bool,
        ):
            self._turn = turn_number
            self._record_stage("turn_ended")
            if was_interrupted:
                self._record_outcome("interrupted")

        @observer.event_handler("on_user_speech_started_for_turn")
        async def on_user_speech_started(_observer, turn_number: int, _data):
            self._turn = turn_number
            self._record_stage("stt_start")

        @observer.event_handler("on_user_speech_stopped_for_turn")
        async def on_user_speech_stopped(_observer, turn_number: int, _data):
            self._turn = turn_number
            self._record_stage("user_speech_complete")

        @observer.event_handler("on_bot_started_speaking")
        async def on_bot_started_speaking(_observer, turn_number: int, _data):
            self._turn = turn_number
            self._record_stage("playback_start")

        @observer.event_handler("on_bot_stopped_speaking")
        async def on_bot_stopped_speaking(_observer, turn_number: int, _data):
            self._turn = turn_number
            self._record_stage("playback_complete")

    def _elapsed_ms(self) -> int:
        return max(0, int(round((time.monotonic() - self._started) * 1000)))

    def _turn_entry(self, turn: int | None = None) -> dict[str, object]:
        turn_id = self._turn if turn is None else turn
        if turn_id not in self._turns:
            if len(self._turns) >= _MAX_TURNS:
                turn_id = -1
            self._turns.setdefault(
                turn_id,
                {"turn": turn_id, "events": [], "latencies_ms": {}},
            )
        return self._turns[turn_id]

    def _record_stage(self, stage: str, *, latency_ms: int | None = None) -> None:
        entry = self._turn_entry()
        turn_id = int(entry["turn"])
        stage_times = self._stage_times[turn_id]
        if stage in stage_times:
            return
        stage_times[stage] = time.monotonic()
        events = entry["events"]
        if isinstance(events, list):
            event: dict[str, object] = {
                "stage": stage,
                "elapsed_ms": self._elapsed_ms(),
            }
            if latency_ms is not None:
                event["latency_ms"] = max(0, latency_ms)
            events.append(event)

        pairs = {
            "stt_final": ("stt_start", "stt"),
            "llm_first_token": ("llm_request", "llm_first_token"),
            "llm_complete": ("llm_request", "llm_complete"),
            "tts_first_audio": ("tts_start", "tts_first_audio"),
            "playback_start": ("user_speech_complete", "end_to_first_playback"),
        }
        start_stage, metric_name = pairs.get(stage, (None, None))
        if start_stage and start_stage in stage_times:
            latencies = entry["latencies_ms"]
            if isinstance(latencies, dict):
                latencies[metric_name] = max(
                    0,
                    int(round((stage_times[stage] - stage_times[start_stage]) * 1000)),
                )

    def _record_outcome(self, category: str, **fields: object) -> None:
        if len(self._outcomes) >= _MAX_OUTCOMES:
            return
        outcome = {
            "turn": self._turn,
            "category": category,
            "elapsed_ms": self._elapsed_ms(),
        }
        outcome.update(fields)
        self._outcomes.append(outcome)

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame
        tracked_types = (
            TranscriptionFrame,
            LLMFullResponseStartFrame,
            LLMTextFrame,
            LLMFullResponseEndFrame,
            TTSStartedFrame,
            TTSAudioRawFrame,
            BotStartedSpeakingFrame,
            BotStoppedSpeakingFrame,
            InterruptionFrame,
            MetricsFrame,
            ErrorFrame,
        )
        if not isinstance(frame, tracked_types):
            return
        if frame.id in self._frames_seen:
            return
        self._frames_seen[frame.id] = None
        if len(self._frames_seen) > _MAX_FRAME_IDS:
            self._frames_seen.popitem(last=False)

        if isinstance(frame, TranscriptionFrame):
            self._saw_final_transcription = True
            self._record_stage("stt_final")
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._llm_text_seen = False
            self._record_stage("llm_request")
        elif isinstance(frame, LLMTextFrame):
            self._llm_text_seen = True
            self._record_stage("llm_first_token")
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._record_stage("llm_complete")
        elif isinstance(frame, TTSStartedFrame):
            self._record_stage("tts_start")
        elif isinstance(frame, TTSAudioRawFrame):
            self._record_stage("tts_first_audio")
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._record_stage("playback_start")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._record_stage("playback_complete")
        elif isinstance(frame, InterruptionFrame):
            self._record_outcome("interrupted")
        elif isinstance(frame, MetricsFrame):
            self._record_metrics(frame)
        elif isinstance(frame, ErrorFrame):
            await self._handle_error(frame)

    def _record_metrics(self, frame: MetricsFrame) -> None:
        for metric in frame.data:
            processor = (getattr(metric, "processor", "") or "").lower()
            if isinstance(metric, TTFBMetricsData):
                latency_ms = int(round(metric.value * 1000))
                if "llm" in processor:
                    self._record_stage("llm_first_token", latency_ms=latency_ms)
                elif "tts" in processor:
                    self._record_stage("tts_first_audio", latency_ms=latency_ms)
            elif isinstance(metric, TTFAMetricsData) and "tts" in processor:
                self._record_stage(
                    "tts_first_audio",
                    latency_ms=int(round(metric.ttfa * 1000)),
                )

    async def handle_provider_error(self, _processor, frame: ErrorFrame) -> None:
        """Classify then sanitize a provider error before it enters pipeline logs."""

        await self._handle_error(frame)
        frame.error = "Provider request could not be completed."
        frame.exception = None

    async def _handle_error(self, frame: ErrorFrame) -> None:
        if frame.id in self._error_frames_seen:
            return
        if len(self._error_frames_seen) < _MAX_OUTCOMES:
            self._error_frames_seen.add(frame.id)

        failure = _failure_for_frame(frame)
        category = _failure_category(failure, frame)
        turn_id = int(self._turn_entry()["turn"])
        fallback_queued = False
        if (
            category == "provider_rejected_input"
            and not self._llm_text_seen
            and turn_id not in self._fallback_queued_for_turn
        ):
            try:
                await self._queue_fallback(
                    TTSSpeakFrame(
                        text=PUBLIC_SAFE_PROVIDER_REJECTION_FALLBACK,
                        append_to_context=False,
                        persist_to_logs=False,
                    )
                )
                self._fallback_queued_for_turn.add(turn_id)
                fallback_queued = True
            except Exception:
                # The underlying queue failure remains a normal pipeline error;
                # never persist exception text from the fallback path.
                self._record_outcome("fallback_queue_failed")

        self._record_outcome(
            category,
            fatal=bool(frame.fatal),
            retryable=bool(failure.retryable),
            fallback_queued=fallback_queued,
        )

    def snapshot(self) -> dict[str, object]:
        turns = [self._turns[key] for key in sorted(self._turns)]
        return {
            "schema_version": "1.0",
            "policy_version": self._policy.policy_version,
            "privacy": {
                "raw_audio_persisted": False,
                "transcript_persisted": False,
                "raw_provider_error_persisted": False,
                "content_fields_persisted": False,
            },
            "ephemeral_test": self._policy.ephemeral_test,
            "turns": turns,
            "outcomes": list(self._outcomes),
            "counters": {
                "turns": len(turns),
                "provider_rejected_inputs": sum(
                    item.get("category") == "provider_rejected_input"
                    for item in self._outcomes
                ),
                "safe_fallbacks_queued": sum(
                    item.get("fallback_queued") is True for item in self._outcomes
                ),
            },
        }
