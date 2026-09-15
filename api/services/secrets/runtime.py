"""Late, in-memory-only materialization of externally held provider secrets."""

from __future__ import annotations

import os
from collections.abc import Iterable

from api.schemas.ai_model_configuration import EffectiveAIModelConfiguration
from api.services.secrets.azure_key_vault import AzureKeyVaultSecretResolver
from api.services.secrets.errors import SecretReferenceInvalidError
from api.services.secrets.resolver import SecretReferencePolicy, SecretResolver

_SERVICE_SECTIONS = ("llm", "stt", "tts", "embeddings", "realtime")


def _external_services(
    configuration: EffectiveAIModelConfiguration,
    sections: tuple[str, ...],
) -> Iterable[object]:
    for section in sections:
        service = getattr(configuration, section, None)
        if (
            service is not None
            and getattr(service, "credential_source", None) == "secret_ref"
        ):
            yield service


def build_secret_resolver_from_env() -> SecretResolver:
    backend = os.getenv("SECRET_BACKEND")
    if backend != "azure_key_vault":
        raise SecretReferenceInvalidError()
    policy_json = os.getenv("VOICE_SECRET_REFERENCE_MAP_JSON", "")
    return AzureKeyVaultSecretResolver(
        vault_url=os.getenv("VOICE_KEY_VAULT_URL", ""),
        managed_identity_client_id=os.getenv("VOICE_UAMI_CLIENT_ID", ""),
        policy=SecretReferencePolicy.from_json(policy_json),
    )


async def materialize_runtime_secrets(
    configuration: EffectiveAIModelConfiguration,
    *,
    organization_id: int,
    resolver: SecretResolver | None = None,
    sections: tuple[str, ...] = _SERVICE_SECTIONS,
) -> EffectiveAIModelConfiguration:
    """Return a runtime-only copy with private resolved credentials bound.

    Resolved values live only in Pydantic private attributes. They are excluded
    from model dumps, JSON, repr, and persistence payloads. There is no fallback
    to an inline key when a reference is selected.
    """

    if not sections or any(section not in _SERVICE_SECTIONS for section in sections):
        raise SecretReferenceInvalidError()
    services = list(_external_services(configuration, sections))
    if not services:
        return configuration
    if type(organization_id) is not int or organization_id <= 0:
        raise SecretReferenceInvalidError()

    # Re-validate from serializable fields rather than copying private state, so
    # a previously materialized object can never carry a stale runtime secret.
    runtime_configuration = EffectiveAIModelConfiguration.model_validate(
        configuration.model_dump(mode="python")
    )
    runtime_services = list(_external_services(runtime_configuration, sections))
    owned_resolver = resolver is None
    active_resolver = resolver or build_secret_resolver_from_env()
    per_operation_cache: dict[tuple[str, str], str] = {}
    try:
        for service in runtime_services:
            provider = str(
                service.provider.value
                if hasattr(service.provider, "value")
                else service.provider
            )
            reference = service.secret_ref
            cache_key = (provider, reference.logical_id)
            if cache_key not in per_operation_cache:
                per_operation_cache[cache_key] = await active_resolver.resolve(
                    reference,
                    expected_provider=provider,
                    organization_id=organization_id,
                )
            service.bind_resolved_api_key(per_operation_cache[cache_key])
        return runtime_configuration
    finally:
        # This cache is request/conversation-local only; erase references promptly.
        per_operation_cache.clear()
        if owned_resolver:
            await active_resolver.close()
