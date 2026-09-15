"""Safe, non-secret-bearing error contract for external secret resolution."""

from __future__ import annotations


class SecretResolutionError(RuntimeError):
    """Base error whose public string contains only safe categorical metadata."""

    code = "SECRET_RESOLUTION_FAILED"

    def __init__(self, *, provider: str | None = None):
        self.provider = provider
        detail = f" provider={provider}" if provider else ""
        super().__init__(f"{self.code}{detail}")


class SecretReferenceInvalidError(SecretResolutionError):
    code = "SECRET_REFERENCE_INVALID"


class SecretBackendUnavailableError(SecretResolutionError):
    code = "SECRET_BACKEND_UNAVAILABLE"


class SecretAccessDeniedError(SecretResolutionError):
    code = "SECRET_ACCESS_DENIED"


class SecretNotFoundError(SecretResolutionError):
    code = "SECRET_NOT_FOUND"
