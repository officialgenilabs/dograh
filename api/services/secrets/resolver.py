"""Provider-agnostic secret resolver contract and trusted reference policy."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from api.services.secrets.errors import SecretReferenceInvalidError
from api.services.secrets.reference import SecretReference

_AZURE_SECRET_NAME_RE = re.compile(r"^[0-9A-Za-z-]{1,127}$")


class SecretResolver(Protocol):
    async def resolve(
        self,
        reference: SecretReference,
        *,
        expected_provider: str,
        organization_id: int,
    ) -> str: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class SecretPolicyEntry:
    provider: str
    secret_name: str
    organization_ids: frozenset[int]


class SecretReferencePolicy:
    """Exact server-controlled mapping from logical ids to vault secret names."""

    def __init__(self, entries: dict[str, SecretPolicyEntry]):
        self._entries = dict(entries)

    @classmethod
    def from_json(cls, value: str) -> "SecretReferencePolicy":
        try:
            raw = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SecretReferenceInvalidError() from exc
        if not isinstance(raw, dict) or not raw:
            raise SecretReferenceInvalidError()

        entries: dict[str, SecretPolicyEntry] = {}
        for logical_id, item in raw.items():
            try:
                reference = SecretReference(
                    backend="azure_key_vault", logical_id=logical_id
                )
            except Exception as exc:
                raise SecretReferenceInvalidError() from exc
            if not isinstance(item, dict) or set(item) != {
                "provider",
                "secret_name",
                "organization_ids",
            }:
                raise SecretReferenceInvalidError(provider=reference.provider_namespace)
            provider = item.get("provider")
            secret_name = item.get("secret_name")
            organization_ids = item.get("organization_ids")
            if (
                not isinstance(provider, str)
                or provider != reference.provider_namespace
                or not isinstance(secret_name, str)
                or not _AZURE_SECRET_NAME_RE.fullmatch(secret_name)
                or not isinstance(organization_ids, list)
                or not organization_ids
                or any(
                    type(org_id) is not int or org_id <= 0
                    for org_id in organization_ids
                )
            ):
                raise SecretReferenceInvalidError(provider=provider)
            entries[logical_id] = SecretPolicyEntry(
                provider=provider,
                secret_name=secret_name,
                organization_ids=frozenset(organization_ids),
            )
        return cls(entries)

    def authorize(
        self,
        reference: SecretReference,
        *,
        expected_provider: str,
        organization_id: int,
    ) -> str:
        if reference.backend != "azure_key_vault":
            raise SecretReferenceInvalidError(provider=expected_provider)
        if reference.provider_namespace != expected_provider:
            raise SecretReferenceInvalidError(provider=expected_provider)
        entry = self._entries.get(reference.logical_id)
        if (
            entry is None
            or entry.provider != expected_provider
            or organization_id not in entry.organization_ids
        ):
            raise SecretReferenceInvalidError(provider=expected_provider)
        return entry.secret_name
