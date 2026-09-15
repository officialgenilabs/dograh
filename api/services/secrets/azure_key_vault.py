"""Azure Key Vault resolver using one deterministic user-assigned identity."""

from __future__ import annotations

import re
from inspect import isawaitable
from urllib.parse import urlparse

from api.services.secrets.errors import (
    SecretAccessDeniedError,
    SecretBackendUnavailableError,
    SecretNotFoundError,
)
from api.services.secrets.reference import SecretReference
from api.services.secrets.resolver import SecretReferencePolicy

_AZURE_VAULT_HOST_RE = re.compile(r"^[a-z][a-z0-9-]{1,22}[a-z0-9]\.vault\.azure\.net$")


def validate_vault_url(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not _AZURE_VAULT_HOST_RE.fullmatch(parsed.hostname)
        or parsed.username
        or parsed.password
        or parsed.port is not None
        or parsed.path not in ("", "/")
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("VOICE_KEY_VAULT_URL must be a canonical Azure Key Vault URL")
    return f"https://{parsed.hostname}"


class AzureKeyVaultSecretResolver:
    """Resolve exact policy-approved refs through a dedicated Voice UAMI."""

    def __init__(
        self,
        *,
        vault_url: str,
        managed_identity_client_id: str,
        policy: SecretReferencePolicy,
        client=None,
        credential=None,
    ):
        if not managed_identity_client_id:
            raise ValueError("VOICE_UAMI_CLIENT_ID is required")
        self._vault_url = validate_vault_url(vault_url)
        self._policy = policy
        self._credential = credential
        self._client = client
        if self._client is None:
            self._credential, self._client = self._create_clients(
                managed_identity_client_id
            )

    def _create_clients(self, managed_identity_client_id: str):
        # Imports remain local so unit tests can inject a fake client without Azure.
        from azure.identity.aio import ManagedIdentityCredential
        from azure.keyvault.secrets.aio import SecretClient

        credential = ManagedIdentityCredential(client_id=managed_identity_client_id)
        client = SecretClient(vault_url=self._vault_url, credential=credential)
        return credential, client

    async def resolve(
        self,
        reference: SecretReference,
        *,
        expected_provider: str,
        organization_id: int,
    ) -> str:
        secret_name = self._policy.authorize(
            reference,
            expected_provider=expected_provider,
            organization_id=organization_id,
        )
        try:
            secret = await self._client.get_secret(secret_name)
        except Exception as exc:
            # Inspect type/status only. Never include the SDK exception string/body.
            class_name = type(exc).__name__
            status_code = getattr(exc, "status_code", None)
            if class_name == "ResourceNotFoundError" or status_code == 404:
                raise SecretNotFoundError(provider=expected_provider) from None
            if class_name == "ClientAuthenticationError" or status_code in (401, 403):
                raise SecretAccessDeniedError(provider=expected_provider) from None
            raise SecretBackendUnavailableError(provider=expected_provider) from None

        value = getattr(secret, "value", None)
        if not isinstance(value, str) or not value:
            raise SecretNotFoundError(provider=expected_provider)
        return value

    async def close(self) -> None:
        for resource in (self._client, self._credential):
            close = getattr(resource, "close", None)
            if close is None:
                continue
            result = close()
            if isawaitable(result):
                await result
