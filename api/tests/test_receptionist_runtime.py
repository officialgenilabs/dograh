from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import openai
import pytest
from pipecat.frames.frames import (
    ErrorFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.metrics.metrics import STTUsage, STTUsageMetricsData
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection

from api.errors.failure import ErrorOwner, ErrorSource, annotate_failure_metadata
from api.services.pipecat.event_handlers import register_event_handlers
from api.services.pipecat.in_memory_buffers import InMemoryLogsBuffer
from api.services.pipecat.pipeline_metrics_aggregator import PipelineMetricsAggregator
from api.services.pipecat.realtime_feedback_observer import RealtimeFeedbackObserver
from api.services.pipecat.receptionist_runtime_observer import (
    PUBLIC_SAFE_PROVIDER_REJECTION_FALLBACK,
    ReceptionistRuntimeObserver,
)
from api.services.pipecat.receptionist_runtime_policy import (
    ReceptionistRuntimePolicy,
    resolve_receptionist_runtime_policy,
    sanitize_receptionist_gathered_context,
)
from api.services.pipecat.termination_funnel_processor import (
    TerminationFunnelProcessor,
)


def _pushed(frame):
    return FramePushed(
        source=SimpleNamespace(),
        destination=SimpleNamespace(),
        frame=frame,
        direction=FrameDirection.UPSTREAM
        if isinstance(frame, ErrorFrame)
        else FrameDirection.DOWNSTREAM,
        timestamp=0,
    )


def _azure_llm_processor():
    processor = type("AzureLLMService", (), {})()
    annotate_failure_metadata(
        processor,
        source=ErrorSource.LLM,
        provider="azure",
        error_owner=ErrorOwner.USER,
    )
    return processor


def test_receptionist_policy_is_operator_scoped_and_ephemeral_is_bounded():
    disabled = resolve_receptionist_runtime_policy(
        5,
        {"_agentflow_voice_ephemeral_test": True},
        environ={},
    )
    assert disabled.enabled is False
    assert disabled.ephemeral_test is False

    enabled_without_ephemeral_authorization = resolve_receptionist_runtime_policy(
        5,
        {"_agentflow_voice_ephemeral_test": True},
        environ={"PUBLIC_RECEPTIONIST_WORKFLOW_IDS": "5, 8"},
    )
    assert enabled_without_ephemeral_authorization.enabled is True
    assert enabled_without_ephemeral_authorization.ephemeral_test is False

    enabled = resolve_receptionist_runtime_policy(
        5,
        {"_agentflow_voice_ephemeral_test": True},
        environ={
            "PUBLIC_RECEPTIONIST_WORKFLOW_IDS": "5, 8",
            "PUBLIC_RECEPTIONIST_EPHEMERAL_TEST_WORKFLOW_IDS": "5",
        },
    )
    assert enabled.enabled is True
    assert enabled.ephemeral_test is True
    assert enabled.persist_audio is False
    assert enabled.persist_transcript is False
    assert enabled.persist_realtime_feedback is False
    assert enabled.emit_completion_side_effects is False


def test_receptionist_context_sanitizer_drops_conversation_content():
    sanitized = sanitize_receptionist_gathered_context(
        {
            "call_tags": ["user_speech", "[SYNTHETIC_CLIENT_ALPHA]"],
            "end_task_reason": "user_hangup",
            "mapped_call_disposition": "[SYNTHETIC_CLIENT_ALPHA] private",
            "captured_name": "[SYNTHETIC_CLIENT_ALPHA]",
            "full_transcript": "restricted content",
        }
    )
    assert sanitized == {
        "call_tags": ["user_speech"],
        "end_task_reason": "user_hangup",
    }


@pytest.mark.asyncio
async def test_provider_content_filter_queues_exactly_one_public_safe_fallback():
    queued = []

    async def queue_fallback(frame):
        queued.append(frame)

    observer = ReceptionistRuntimeObserver(
        policy=ReceptionistRuntimePolicy(enabled=True),
        queue_fallback=queue_fallback,
    )
    processor = _azure_llm_processor()
    provider_error = type(
        "ProviderError",
        (Exception,),
        {"status_code": 400, "code": "content_filter"},
    )("raw provider body must not persist")

    await observer.on_push_frame(_pushed(LLMFullResponseStartFrame()))
    await observer.on_push_frame(
        _pushed(
            ErrorFrame(
                "raw provider body must not persist",
                processor=processor,
                exception=provider_error,
            )
        )
    )
    await observer.on_push_frame(
        _pushed(
            ErrorFrame(
                "second error in same turn",
                processor=processor,
                exception=provider_error,
            )
        )
    )

    assert len(queued) == 1
    assert queued[0].text == PUBLIC_SAFE_PROVIDER_REJECTION_FALLBACK
    assert queued[0].append_to_context is False
    assert queued[0].persist_to_logs is False
    assert len([part for part in queued[0].text.split(".") if part.strip()]) <= 3

    snapshot = observer.snapshot()
    assert snapshot["counters"]["provider_rejected_inputs"] == 2
    assert snapshot["counters"]["safe_fallbacks_queued"] == 1
    assert "raw provider body" not in str(snapshot)
    assert "content_filter" not in str(snapshot)


@pytest.mark.asyncio
async def test_generic_bad_configuration_does_not_mask_failure_with_fallback():
    queued = []

    async def queue_fallback(frame):
        queued.append(frame)

    observer = ReceptionistRuntimeObserver(
        policy=ReceptionistRuntimePolicy(enabled=True),
        queue_fallback=queue_fallback,
    )
    processor = _azure_llm_processor()
    provider_error = type(
        "ProviderError",
        (Exception,),
        {"status_code": 400, "code": "invalid_request"},
    )("invalid deployment configuration")

    await observer.on_push_frame(
        _pushed(
            ErrorFrame(
                "invalid deployment configuration",
                processor=processor,
                exception=provider_error,
            )
        )
    )

    assert queued == []
    assert observer.snapshot()["outcomes"][0]["category"] == "configuration_error"


@pytest.mark.asyncio
async def test_partial_llm_output_prevents_fallback():
    queued = []

    async def queue_fallback(frame):
        queued.append(frame)

    observer = ReceptionistRuntimeObserver(
        policy=ReceptionistRuntimePolicy(enabled=True),
        queue_fallback=queue_fallback,
    )
    processor = _azure_llm_processor()
    provider_error = type(
        "ProviderError",
        (Exception,),
        {"status_code": 400, "code": "content_filter"},
    )("content filtered after streaming")

    await observer.on_push_frame(_pushed(LLMFullResponseStartFrame()))
    await observer.on_push_frame(_pushed(LLMTextFrame("partial")))
    await observer.on_push_frame(
        _pushed(ErrorFrame("filtered", processor=processor, exception=provider_error))
    )

    assert queued == []


@pytest.mark.asyncio
async def test_privacy_profile_error_event_contains_no_provider_detail(monkeypatch):
    messages = []
    logs = InMemoryLogsBuffer(workflow_run_id=9)

    async def ws_sender(message):
        messages.append(message)

    monkeypatch.setattr(
        "api.services.pipecat.realtime_feedback_observer.log_failure",
        lambda *args, **kwargs: None,
    )
    processor = _azure_llm_processor()
    secret = "Bearer synthetic-secret-value"
    observer = RealtimeFeedbackObserver(
        ws_sender=ws_sender,
        logs_buffer=logs,
        persist_events=False,
    )

    await observer.on_push_frame(
        _pushed(
            ErrorFrame(
                f"provider body {secret}",
                processor=processor,
                exception=type(
                    "ProviderError",
                    (Exception,),
                    {"status_code": 400, "code": "content_filter"},
                )(f"provider body {secret}"),
            )
        )
    )

    assert len(messages) == 1
    assert messages[0]["payload"] == {
        "error": "The request could not be completed.",
        "fatal": False,
        "category": "config_error",
        "retryable": False,
    }
    assert secret not in str(messages)
    assert logs.is_empty


@pytest.mark.asyncio
async def test_observability_has_stage_timing_without_transcript_text():
    queued = []

    async def queue_fallback(frame):
        queued.append(frame)

    observer = ReceptionistRuntimeObserver(
        policy=ReceptionistRuntimePolicy(enabled=True),
        queue_fallback=queue_fallback,
    )
    await observer.on_push_frame(
        _pushed(
            TranscriptionFrame(
                "[SYNTHETIC_CLIENT_ALPHA] private words",
                user_id="synthetic-user",
                timestamp="2026-09-16T00:00:00Z",
            )
        )
    )
    await observer.on_push_frame(_pushed(LLMFullResponseStartFrame()))
    await observer.on_push_frame(_pushed(LLMTextFrame("synthetic response")))

    snapshot_text = str(observer.snapshot())
    assert "stt_final" in snapshot_text
    assert "llm_request" in snapshot_text
    assert "llm_first_token" in snapshot_text
    assert "SYNTHETIC_CLIENT_ALPHA" not in snapshot_text
    assert "private words" not in snapshot_text
    assert "synthetic response" not in snapshot_text


@pytest.mark.asyncio
async def test_pipeline_metrics_aggregates_stt_audio_seconds():
    aggregator = PipelineMetricsAggregator()
    await aggregator._handle_stt_usage_metrics(
        STTUsageMetricsData(
            processor="DeepgramSTTService#1",
            model="nova-3-general",
            value=STTUsage(audio_seconds=1.25),
        )
    )
    await aggregator._handle_stt_usage_metrics(
        STTUsageMetricsData(
            processor="DeepgramSTTService#1",
            model="nova-3-general",
            value=STTUsage(audio_seconds=0.75),
        )
    )

    assert aggregator.get_stt_usage_metrics() == {
        "DeepgramSTTService#1|||nova-3-general": 2.0
    }


@pytest.mark.asyncio
async def test_openai_content_filter_is_classified_before_error_is_sanitized():
    queued = []

    async def queue_fallback(frame):
        queued.append(frame)

    observer = ReceptionistRuntimeObserver(
        policy=ReceptionistRuntimePolicy(enabled=True),
        queue_fallback=queue_fallback,
    )
    processor = _azure_llm_processor()
    request = httpx.Request("POST", "https://synthetic.invalid/chat/completions")
    response = httpx.Response(400, request=request)
    provider_error = openai.BadRequestError(
        "raw provider content-filter detail",
        response=response,
        body={"code": "content_filter", "type": "invalid_request_error"},
    )
    frame = ErrorFrame(
        "raw provider content-filter detail",
        processor=processor,
        exception=provider_error,
    )

    await observer.handle_provider_error(processor, frame)

    assert [item.text for item in queued] == [PUBLIC_SAFE_PROVIDER_REJECTION_FALLBACK]
    assert frame.error == "Provider request could not be completed."
    assert frame.exception is None
    assert "raw provider" not in str(observer.snapshot())


class _EventSource:
    def __init__(self):
        self.handlers = {}

    def event_handler(self, name):
        def decorator(handler):
            self.handlers[name] = handler
            return handler

        return decorator


@pytest.mark.asyncio
async def test_ephemeral_receptionist_finish_persists_only_sanitized_operational_data(
    monkeypatch,
):
    upload = AsyncMock()
    update_run = AsyncMock()
    add_disposition = AsyncMock()
    enqueue = AsyncMock()
    capture = AsyncMock()
    monkeypatch.setattr(
        "api.services.pipecat.event_handlers.upload_workflow_run_artifacts", upload
    )
    monkeypatch.setattr(
        "api.services.pipecat.event_handlers.db_client.get_workflow_run_by_id",
        AsyncMock(return_value=SimpleNamespace(workflow_id=5, campaign_id=None)),
    )
    monkeypatch.setattr(
        "api.services.pipecat.event_handlers.db_client.update_workflow_run", update_run
    )
    monkeypatch.setattr(
        "api.services.pipecat.event_handlers.db_client.add_call_disposition_code",
        add_disposition,
    )
    monkeypatch.setattr("api.services.pipecat.event_handlers.enqueue_job", enqueue)
    monkeypatch.setattr(
        "api.services.pipecat.event_handlers._capture_call_event", capture
    )

    task = _EventSource()
    task.wait_for_observers = AsyncMock()
    task.turn_trace_observer = None
    transport = _EventSource()
    audio_buffer = SimpleNamespace(
        start_recording=AsyncMock(),
        stop_recording=AsyncMock(),
    )
    transcript = SimpleNamespace(flush=AsyncMock())
    generate_transcript = Mock(return_value="must not be read")
    logs_buffer = SimpleNamespace(
        contains_user_speech=Mock(side_effect=AssertionError("privacy observer required")),
        is_empty=False,
        get_events=Mock(side_effect=AssertionError("generic event log must not persist")),
        generate_transcript_text=generate_transcript,
    )
    engine = SimpleNamespace(
        end_call_with_reason=AsyncMock(),
        record_call_tags=Mock(),
        record_context=Mock(),
        get_gathered_context=AsyncMock(
            return_value={
                "call_tags": ["user_speech"],
                "end_task_reason": "user_hangup",
                "mapped_call_disposition": "answered",
                "captured_name": "[SYNTHETIC_CLIENT_ALPHA]",
                "full_transcript": "must not persist",
            }
        ),
        cleanup=AsyncMock(),
    )
    receptionist_observer = SimpleNamespace(
        saw_user_speech=True,
        snapshot=lambda: {
            "schema_version": "1.0",
            "privacy": {
                "raw_audio_persisted": False,
                "transcript_persisted": False,
                "raw_provider_error_persisted": False,
                "content_fields_persisted": False,
            },
            "turns": [],
            "outcomes": [],
            "counters": {"turns": 0},
        },
    )
    runtime_session = SimpleNamespace(
        name="synthetic-integration",
        on_call_finished=AsyncMock(),
    )

    register_event_handlers(
        task=task,
        transport=transport,
        workflow_run_id=88,
        engine=engine,
        audio_buffer=audio_buffer,
        in_memory_logs_buffer=logs_buffer,
        transcript_log_coordinator=transcript,
        pipeline_metrics_aggregator=SimpleNamespace(
            get_all_usage_metrics_serialized=lambda: {}
        ),
        termination_funnel=TerminationFunnelProcessor(),
        audio_config=SimpleNamespace(pipeline_sample_rate=16000),
        integration_runtime_sessions=[runtime_session],
        receptionist_policy=ReceptionistRuntimePolicy(
            enabled=True,
            ephemeral_test=True,
        ),
        receptionist_observer=receptionist_observer,
    )

    await task.handlers["on_pipeline_finished"](task, None)

    transcript.flush.assert_not_awaited()
    audio_buffer.stop_recording.assert_not_awaited()
    generate_transcript.assert_not_called()
    runtime_session.on_call_finished.assert_not_awaited()
    add_disposition.assert_not_awaited()
    enqueue.assert_not_awaited()
    capture.assert_not_awaited()
    upload.assert_not_awaited()

    first_update = update_run.await_args_list[0].kwargs
    assert first_update["gathered_context"] == {
        "call_tags": ["user_speech"],
        "end_task_reason": "user_hangup",
        "mapped_call_disposition": "answered",
    }
    assert "[SYNTHETIC_CLIENT_ALPHA]" not in str(update_run.await_args_list)
    assert "must not persist" not in str(update_run.await_args_list)
    assert update_run.await_args_list[1].kwargs["logs"] == {
        "receptionist_observability": receptionist_observer.snapshot()
    }
