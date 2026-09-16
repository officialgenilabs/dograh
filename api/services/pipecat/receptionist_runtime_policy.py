"""Narrow runtime policy for the controlled public-receptionist profile.

The profile is deliberately opt-in by workflow id. It does not alter workflow
prompts, model configuration, credentials, routes, or global retention behavior.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Mapping

from loguru import logger

_RECEPTIONIST_WORKFLOW_IDS_ENV = "PUBLIC_RECEPTIONIST_WORKFLOW_IDS"
_EPHEMERAL_WORKFLOW_IDS_ENV = "PUBLIC_RECEPTIONIST_EPHEMERAL_TEST_WORKFLOW_IDS"
_EPHEMERAL_CONTEXT_KEY = "_agentflow_voice_ephemeral_test"
_OPERATIONAL_CATEGORY_RE = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,63}")


@dataclass(frozen=True)
class ReceptionistRuntimePolicy:
    """Feature-scoped persistence and fallback behavior for one run."""

    enabled: bool = False
    ephemeral_test: bool = False
    policy_version: str = "afv-m1d3a-v1"

    @property
    def persist_transcript(self) -> bool:
        return not self.enabled

    @property
    def persist_audio(self) -> bool:
        return not self.enabled

    @property
    def persist_realtime_feedback(self) -> bool:
        return not self.enabled

    @property
    def emit_completion_side_effects(self) -> bool:
        return not self.ephemeral_test


def _configured_workflow_ids(
    environ: Mapping[str, str],
    variable: str = _RECEPTIONIST_WORKFLOW_IDS_ENV,
) -> frozenset[int]:
    raw = environ.get(variable, "")
    workflow_ids: set[int] = set()
    for item in raw.split(","):
        candidate = item.strip()
        if not candidate:
            continue
        try:
            workflow_ids.add(int(candidate))
        except ValueError:
            logger.warning(f"Ignoring invalid {variable} entry")
    return frozenset(workflow_ids)


def resolve_receptionist_runtime_policy(
    workflow_id: int,
    initial_context: Mapping[str, object] | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ReceptionistRuntimePolicy:
    """Resolve the profile without trusting caller context to enable it.

    Caller context may request the bounded ephemeral certification behavior only
    when the operator has separately allowlisted the workflow for ephemeral tests.
    The public marker alone is never authoritative.
    """

    environment = os.environ if environ is None else environ
    configured = _configured_workflow_ids(environment)
    ephemeral_configured = _configured_workflow_ids(
        environment,
        _EPHEMERAL_WORKFLOW_IDS_ENV,
    )
    enabled = workflow_id in configured
    ephemeral_test = (
        enabled
        and workflow_id in ephemeral_configured
        and (initial_context or {}).get(_EPHEMERAL_CONTEXT_KEY) is True
    )
    return ReceptionistRuntimePolicy(
        enabled=enabled,
        ephemeral_test=ephemeral_test,
    )


def sanitize_receptionist_gathered_context(context: Mapping[str, object]) -> dict:
    """Retain bounded operational categories, never extracted conversation data."""

    sanitized: dict[str, object] = {}
    call_tags = context.get("call_tags")
    if isinstance(call_tags, list) and "user_speech" in call_tags:
        sanitized["call_tags"] = ["user_speech"]

    for key in ("call_status", "end_task_reason", "mapped_call_disposition"):
        value = context.get(key)
        if isinstance(value, str) and _OPERATIONAL_CATEGORY_RE.fullmatch(value):
            sanitized[key] = value

    transferred = context.get("external_pbx_transferred")
    if isinstance(transferred, bool):
        sanitized["external_pbx_transferred"] = transferred
    return sanitized
