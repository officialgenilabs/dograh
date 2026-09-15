import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from api.schemas.ai_model_configuration import (
    OrganizationAIModelConfigurationV2,
    compile_ai_model_configuration_v2,
)
from api.services.configuration.ai_model_configuration import (
    mask_ai_model_configuration_v2,
    merge_ai_model_configuration_v2_secrets,
)
from api.services.configuration.masking import mask_user_config
from api.services.configuration.merge import merge_user_configurations
from api.services.configuration.registry import DeepgramSTTConfiguration
from api.services.pipecat.service_factory import create_stt_service
from api.services.secrets.azure_key_vault import (
    AzureKeyVaultSecretResolver,
    validate_vault_url,
)
from api.services.secrets.errors import (
    SecretAccessDeniedError,
    SecretBackendUnavailableError,
    SecretNotFoundError,
    SecretReferenceInvalidError,
)
from api.services.secrets.reference import SecretReference
from api.services.secrets.resolver import SecretReferencePolicy
from api.services.secrets.runtime import materialize_runtime_secrets

TEST_SECRET = "TEST_ONLY_NOT_A_REAL_SECRET_M1D1A_001"
LOGICAL_REF = "provider/deepgram/production"


def _external_pipeline_dict() -> dict:
    ref = {"backend": "azure_key_vault", "logical_id": LOGICAL_REF}
    return {
        "version": 2,
        "mode": "byok",
        "byok": {
            "mode": "pipeline",
            "pipeline": {
                "llm": {
                    "provider": "openai",
                    "api_key": "TEST_ONLY_INLINE_LLM",
                    "model": "gpt-4.1",
                },
                "stt": {
                    "provider": "deepgram",
                    "credential_source": "secret_ref",
                    "secret_ref": ref,
                    "model": "nova-3-general",
                    "language": "multi",
                },
                "tts": {
                    "provider": "deepgram",
                    "credential_source": "secret_ref",
                    "secret_ref": ref,
                    "voice": "aura-2-helena-en",
                },
            },
        },
    }


class FakeSecretResolver:
    def __init__(self, value=TEST_SECRET, error=None):
        self.value = value
        self.error = error
        self.calls = []
        self.closed = False

    async def resolve(self, reference, *, expected_provider, organization_id):
        self.calls.append((reference.logical_id, expected_provider, organization_id))
        if self.error:
            raise self.error
        return self.value

    async def close(self):
        self.closed = True


def _policy() -> SecretReferencePolicy:
    return SecretReferencePolicy.from_json(
        json.dumps(
            {
                LOGICAL_REF: {
                    "provider": "deepgram",
                    "secret_name": "agentflow-voice-deepgram-production",
                    "organization_ids": [42],
                }
            }
        )
    )


def test_legacy_inline_configuration_remains_accepted():
    config = DeepgramSTTConfiguration(
        provider="deepgram", api_key="TEST_ONLY_INLINE", model="nova-3-general"
    )
    assert config.credential_source == "inline"
    assert config.api_key == "TEST_ONLY_INLINE"


def test_inline_configuration_still_requires_api_key_when_provider_requires_it():
    with pytest.raises(ValidationError):
        DeepgramSTTConfiguration(provider="deepgram", model="nova-3-general")


def test_external_reference_configuration_is_accepted_without_inline_key():
    config = OrganizationAIModelConfigurationV2.model_validate(
        _external_pipeline_dict()
    )
    stt = config.byok.pipeline.stt
    assert stt.credential_source == "secret_ref"
    assert stt.secret_ref.logical_id == LOGICAL_REF
    with pytest.raises(SecretReferenceInvalidError, match="SECRET_REFERENCE_INVALID"):
        _ = stt.api_key


@pytest.mark.parametrize(
    "reference",
    [
        "",
        "../../secret",
        "https://attacker.example/",
        "https://another-vault.vault.azure.net/secrets/foo",
        "provider/deepgram/",
        "provider/deepgram/" + "x" * 129,
        "arbitrary/keyvault/secret",
    ],
)
def test_malformed_or_arbitrary_reference_is_rejected(reference):
    payload = _external_pipeline_dict()
    payload["byok"]["pipeline"]["stt"]["secret_ref"]["logical_id"] = reference
    with pytest.raises(ValidationError):
        OrganizationAIModelConfigurationV2.model_validate(payload)


def test_unsupported_backend_is_rejected():
    payload = _external_pipeline_dict()
    payload["byok"]["pipeline"]["stt"]["secret_ref"]["backend"] = "other"
    with pytest.raises(ValidationError):
        OrganizationAIModelConfigurationV2.model_validate(payload)


def test_cross_provider_reference_is_rejected():
    payload = _external_pipeline_dict()
    payload["byok"]["pipeline"]["stt"]["secret_ref"][
        "logical_id"
    ] = "provider/azure/production"
    with pytest.raises(ValidationError, match="provider namespace"):
        OrganizationAIModelConfigurationV2.model_validate(payload)


def test_ambiguous_inline_and_external_credentials_are_rejected():
    payload = _external_pipeline_dict()
    payload["byok"]["pipeline"]["stt"]["api_key"] = "TEST_ONLY_STALE_RAW"
    with pytest.raises(ValidationError, match="api_key must be absent"):
        OrganizationAIModelConfigurationV2.model_validate(payload)


def test_policy_rejects_unmapped_ref_cross_provider_and_other_org():
    policy = _policy()
    deepgram = SecretReference(logical_id=LOGICAL_REF)
    assert (
        policy.authorize(deepgram, expected_provider="deepgram", organization_id=42)
        == "agentflow-voice-deepgram-production"
    )
    with pytest.raises(SecretReferenceInvalidError):
        policy.authorize(deepgram, expected_provider="azure", organization_id=42)
    with pytest.raises(SecretReferenceInvalidError):
        policy.authorize(deepgram, expected_provider="deepgram", organization_id=99)
    with pytest.raises(SecretReferenceInvalidError):
        policy.authorize(
            SecretReference(logical_id="provider/deepgram/unapproved"),
            expected_provider="deepgram",
            organization_id=42,
        )


@pytest.mark.asyncio
async def test_fake_resolver_materializes_once_and_never_writes_back():
    persisted = OrganizationAIModelConfigurationV2.model_validate(
        _external_pipeline_dict()
    )
    persisted_before = persisted.model_dump(mode="json", exclude_none=True)
    effective = compile_ai_model_configuration_v2(persisted)
    resolver = FakeSecretResolver()

    runtime = await materialize_runtime_secrets(
        effective, organization_id=42, resolver=resolver
    )

    assert runtime is not effective
    assert runtime.stt.api_key == TEST_SECRET
    assert runtime.tts.api_key == TEST_SECRET
    assert resolver.calls == [(LOGICAL_REF, "deepgram", 42)]
    assert persisted.model_dump(mode="json", exclude_none=True) == persisted_before
    assert TEST_SECRET not in json.dumps(persisted.model_dump(mode="json"))
    assert TEST_SECRET not in json.dumps(runtime.model_dump(mode="json"))
    assert TEST_SECRET not in json.dumps(mask_user_config(runtime))
    assert TEST_SECRET not in repr(runtime)


@pytest.mark.asyncio
async def test_new_operation_re_resolves_reference_without_persistent_cache():
    persisted = OrganizationAIModelConfigurationV2.model_validate(
        _external_pipeline_dict()
    )
    effective = compile_ai_model_configuration_v2(persisted)
    effective_before = effective.model_dump(mode="json")

    first = await materialize_runtime_secrets(
        effective,
        organization_id=42,
        resolver=FakeSecretResolver(value="TEST_ONLY_ROTATION_OLD"),
    )
    second = await materialize_runtime_secrets(
        effective,
        organization_id=42,
        resolver=FakeSecretResolver(value="TEST_ONLY_ROTATION_NEW"),
    )

    assert first.stt.api_key == "TEST_ONLY_ROTATION_OLD"
    assert second.stt.api_key == "TEST_ONLY_ROTATION_NEW"
    assert effective.model_dump(mode="json") == effective_before


@pytest.mark.asyncio
async def test_materialization_resolves_only_explicit_runtime_sections():
    persisted = OrganizationAIModelConfigurationV2.model_validate(
        _external_pipeline_dict()
    )
    effective = compile_ai_model_configuration_v2(persisted)
    resolver = FakeSecretResolver()

    runtime = await materialize_runtime_secrets(
        effective,
        organization_id=42,
        resolver=resolver,
        sections=("llm",),
    )

    assert runtime is effective
    assert resolver.calls == []
    with pytest.raises(SecretReferenceInvalidError):
        _ = runtime.stt.api_key


@pytest.mark.asyncio
async def test_materialization_rejects_unknown_runtime_sections():
    persisted = OrganizationAIModelConfigurationV2.model_validate(
        _external_pipeline_dict()
    )
    effective = compile_ai_model_configuration_v2(persisted)
    with pytest.raises(SecretReferenceInvalidError):
        await materialize_runtime_secrets(
            effective,
            organization_id=42,
            resolver=FakeSecretResolver(),
            sections=("arbitrary",),
        )


@pytest.mark.asyncio
async def test_missing_server_side_backend_configuration_fails_closed(monkeypatch):
    persisted = OrganizationAIModelConfigurationV2.model_validate(
        _external_pipeline_dict()
    )
    effective = compile_ai_model_configuration_v2(persisted)
    monkeypatch.delenv("SECRET_BACKEND", raising=False)
    monkeypatch.delenv("VOICE_SECRET_REFERENCE_MAP_JSON", raising=False)
    monkeypatch.delenv("VOICE_KEY_VAULT_URL", raising=False)
    monkeypatch.delenv("VOICE_UAMI_CLIENT_ID", raising=False)

    with pytest.raises(SecretReferenceInvalidError):
        await materialize_runtime_secrets(effective, organization_id=42)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        SecretNotFoundError(provider="deepgram"),
        SecretAccessDeniedError(provider="deepgram"),
        SecretBackendUnavailableError(provider="deepgram"),
    ],
)
async def test_resolution_failures_fail_closed_without_raw_fallback(error):
    persisted = OrganizationAIModelConfigurationV2.model_validate(
        _external_pipeline_dict()
    )
    effective = compile_ai_model_configuration_v2(persisted)
    with pytest.raises(type(error)) as exc_info:
        await materialize_runtime_secrets(
            effective,
            organization_id=42,
            resolver=FakeSecretResolver(error=error),
        )
    assert TEST_SECRET not in str(exc_info.value)
    with pytest.raises(SecretReferenceInvalidError):
        _ = effective.stt.api_key


@pytest.mark.asyncio
async def test_fixture_secret_is_not_logged_or_exposed_by_exceptions(caplog):
    persisted = OrganizationAIModelConfigurationV2.model_validate(
        _external_pipeline_dict()
    )
    effective = compile_ai_model_configuration_v2(persisted)
    with caplog.at_level(logging.DEBUG):
        runtime = await materialize_runtime_secrets(
            effective, organization_id=42, resolver=FakeSecretResolver()
        )
    assert TEST_SECRET not in caplog.text
    assert TEST_SECRET not in str(runtime)
    assert TEST_SECRET not in repr(runtime)


def test_api_masking_exposes_reference_state_but_no_secret():
    config = OrganizationAIModelConfigurationV2.model_validate(
        _external_pipeline_dict()
    )
    masked = mask_ai_model_configuration_v2(config)
    stt = masked["byok"]["pipeline"]["stt"]
    assert "api_key" not in stt
    assert stt["credential_source"] == "secret_ref"
    assert stt["credential_configured"] is True
    assert TEST_SECRET not in json.dumps(masked)


def test_legacy_configuration_response_schema_accepts_external_reference_state():
    from api.routes.user import UserConfigurationRequestResponseSchema

    persisted = OrganizationAIModelConfigurationV2.model_validate(
        _external_pipeline_dict()
    )
    effective = compile_ai_model_configuration_v2(persisted)
    payload = mask_user_config(effective)
    response = UserConfigurationRequestResponseSchema.model_validate(payload)
    dumped = response.model_dump(mode="json", exclude_none=True)

    assert dumped["stt"]["credential_source"] == "secret_ref"
    assert dumped["stt"]["secret_ref"]["logical_id"] == LOGICAL_REF
    assert "api_key" not in dumped["stt"]
    assert TEST_SECRET not in json.dumps(dumped)


def test_inline_to_external_merge_does_not_restore_legacy_raw_key():
    existing_payload = _external_pipeline_dict()
    for section in ("stt", "tts"):
        existing_payload["byok"]["pipeline"][section].pop("credential_source")
        existing_payload["byok"]["pipeline"][section].pop("secret_ref")
        existing_payload["byok"]["pipeline"][section]["api_key"] = "TEST_ONLY_STALE_RAW"
    existing = OrganizationAIModelConfigurationV2.model_validate(existing_payload)
    incoming = OrganizationAIModelConfigurationV2.model_validate(
        _external_pipeline_dict()
    )

    merged = merge_ai_model_configuration_v2_secrets(incoming, existing)
    dumped = merged.model_dump(mode="json", exclude_none=True)

    assert "api_key" not in dumped["byok"]["pipeline"]["stt"]
    assert "api_key" not in dumped["byok"]["pipeline"]["tts"]
    assert "TEST_ONLY_STALE_RAW" not in json.dumps(dumped)


def test_legacy_inline_to_external_merge_does_not_restore_raw_key():
    existing_payload = _external_pipeline_dict()
    stt = existing_payload["byok"]["pipeline"]["stt"]
    stt.pop("credential_source")
    stt.pop("secret_ref")
    stt["api_key"] = "TEST_ONLY_STALE_RAW"
    existing = compile_ai_model_configuration_v2(
        OrganizationAIModelConfigurationV2.model_validate(existing_payload)
    )
    incoming = {"stt": _external_pipeline_dict()["byok"]["pipeline"]["stt"]}

    merged = merge_user_configurations(existing, incoming)
    dumped = merged.model_dump(mode="json", exclude_none=True)

    assert "api_key" not in dumped["stt"]
    assert dumped["stt"]["credential_source"] == "secret_ref"
    assert "TEST_ONLY_STALE_RAW" not in json.dumps(dumped)


def test_unrelated_masked_update_preserves_external_reference():
    existing = OrganizationAIModelConfigurationV2.model_validate(
        _external_pipeline_dict()
    )
    incoming_payload = _external_pipeline_dict()
    incoming_payload["byok"]["pipeline"]["stt"]["language"] = "en"
    incoming = OrganizationAIModelConfigurationV2.model_validate(incoming_payload)

    merged = merge_ai_model_configuration_v2_secrets(incoming, existing)

    assert merged.byok.pipeline.stt.secret_ref.logical_id == LOGICAL_REF
    assert merged.byok.pipeline.stt.credential_source == "secret_ref"
    assert merged.byok.pipeline.stt.language == "en"


@pytest.mark.asyncio
async def test_provider_factory_receives_fixture_secret_only_after_resolution(caplog):
    persisted = OrganizationAIModelConfigurationV2.model_validate(
        _external_pipeline_dict()
    )
    effective = compile_ai_model_configuration_v2(persisted)
    runtime = await materialize_runtime_secrets(
        effective, organization_id=42, resolver=FakeSecretResolver()
    )
    audio_config = SimpleNamespace(transport_in_sample_rate=16000)

    with patch(
        "api.services.pipecat.service_factory.DeepgramSTTService"
    ) as constructor:
        constructor.return_value = MagicMock()
        create_stt_service(runtime, audio_config)

    assert constructor.call_args.kwargs["api_key"] == TEST_SECRET
    assert TEST_SECRET not in json.dumps(runtime.model_dump(mode="json"))
    assert TEST_SECRET not in caplog.text


@pytest.mark.parametrize(
    "url",
    [
        "http://voice.vault.azure.net",
        "https://attacker.example",
        "https://.vault.azure.net",
        "https://nested.voice.vault.azure.net",
        "https://voice.vault.azure.net/secrets/foo",
        "https://user@voice.vault.azure.net",
        "https://voice.vault.azure.net?x=1",
    ],
)
def test_vault_url_rejects_noncanonical_values(url):
    with pytest.raises(ValueError):
        validate_vault_url(url)


def test_azure_backend_uses_only_explicit_user_assigned_identity():
    policy = _policy()
    with patch("azure.identity.aio.ManagedIdentityCredential") as credential_cls, patch(
        "azure.keyvault.secrets.aio.SecretClient"
    ) as client_cls:
        resolver = AzureKeyVaultSecretResolver(
            vault_url="https://voice.vault.azure.net",
            managed_identity_client_id="00000000-0000-0000-0000-000000000001",
            policy=policy,
        )

    credential_cls.assert_called_once_with(
        client_id="00000000-0000-0000-0000-000000000001"
    )
    client_cls.assert_called_once_with(
        vault_url="https://voice.vault.azure.net",
        credential=credential_cls.return_value,
    )
    assert resolver._credential is credential_cls.return_value


@pytest.mark.asyncio
async def test_azure_backend_rejects_other_tenant_before_vault_call():
    client = SimpleNamespace(get_secret=MagicMock(), close=MagicMock())
    resolver = AzureKeyVaultSecretResolver(
        vault_url="https://voice.vault.azure.net",
        managed_identity_client_id="00000000-0000-0000-0000-000000000001",
        policy=_policy(),
        client=client,
    )

    with pytest.raises(SecretReferenceInvalidError):
        await resolver.resolve(
            SecretReference(logical_id=LOGICAL_REF),
            expected_provider="deepgram",
            organization_id=99,
        )
    client.get_secret.assert_not_called()


@pytest.mark.asyncio
async def test_azure_backend_returns_only_policy_selected_secret_and_closes():
    class Client:
        def __init__(self):
            self.requested = []
            self.closed = False

        async def get_secret(self, name):
            self.requested.append(name)
            return SimpleNamespace(value=TEST_SECRET)

        async def close(self):
            self.closed = True

    client = Client()
    resolver = AzureKeyVaultSecretResolver(
        vault_url="https://voice.vault.azure.net",
        managed_identity_client_id="00000000-0000-0000-0000-000000000001",
        policy=_policy(),
        client=client,
    )
    result = await resolver.resolve(
        SecretReference(logical_id=LOGICAL_REF),
        expected_provider="deepgram",
        organization_id=42,
    )
    await resolver.close()

    assert result == TEST_SECRET
    assert client.requested == ["agentflow-voice-deepgram-production"]
    assert client.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (type("ResourceNotFoundError", (Exception,), {})(), SecretNotFoundError),
        (
            type("ClientAuthenticationError", (Exception,), {})(),
            SecretAccessDeniedError,
        ),
        (RuntimeError(TEST_SECRET), SecretBackendUnavailableError),
    ],
)
async def test_azure_backend_maps_errors_without_sdk_detail(exc, expected):
    client = SimpleNamespace(get_secret=MagicMock(side_effect=exc), close=MagicMock())

    async def get_secret(_name):
        raise exc

    client.get_secret = get_secret
    resolver = AzureKeyVaultSecretResolver(
        vault_url="https://voice.vault.azure.net",
        managed_identity_client_id="00000000-0000-0000-0000-000000000001",
        policy=_policy(),
        client=client,
    )
    with pytest.raises(expected) as exc_info:
        await resolver.resolve(
            SecretReference(logical_id=LOGICAL_REF),
            expected_provider="deepgram",
            organization_id=42,
        )
    assert TEST_SECRET not in str(exc_info.value)
