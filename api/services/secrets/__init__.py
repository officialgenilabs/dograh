"""External provider-secret custody primitives."""

from api.services.secrets.errors import (
    SecretAccessDeniedError,
    SecretBackendUnavailableError,
    SecretNotFoundError,
    SecretReferenceInvalidError,
    SecretResolutionError,
)
from api.services.secrets.reference import SecretReference
from api.services.secrets.resolver import SecretResolver

__all__ = [
    "SecretAccessDeniedError",
    "SecretBackendUnavailableError",
    "SecretNotFoundError",
    "SecretReference",
    "SecretReferenceInvalidError",
    "SecretResolutionError",
    "SecretResolver",
]
