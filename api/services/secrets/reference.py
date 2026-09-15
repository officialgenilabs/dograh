"""Constrained, non-secret logical references for provider credentials."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_LOGICAL_ID_RE = re.compile(
    r"^provider/(?P<provider>[a-z0-9][a-z0-9_-]{0,47})/"
    r"(?P<slot>[a-z0-9][a-z0-9_-]{0,47})$"
)


class SecretReference(BaseModel):
    """A non-secret logical id; never a vault URL, URI, or secret value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Literal["azure_key_vault"] = "azure_key_vault"
    logical_id: str = Field(min_length=1, max_length=128)

    @field_validator("logical_id")
    @classmethod
    def validate_logical_id(cls, value: str) -> str:
        if not _LOGICAL_ID_RE.fullmatch(value):
            raise ValueError(
                "secret_ref.logical_id must match provider/<provider>/<slot>"
            )
        return value

    @property
    def provider_namespace(self) -> str:
        match = _LOGICAL_ID_RE.fullmatch(self.logical_id)
        if (
            match is None
        ):  # Defensive; model validation normally makes this unreachable.
            raise ValueError("invalid logical secret reference")
        return match.group("provider")
