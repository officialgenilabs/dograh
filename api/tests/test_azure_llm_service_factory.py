"""Tests for Azure OpenAI LLM service factory settings."""

from unittest.mock import patch

import pytest

from api.services.configuration.registry import ServiceProviders
from api.services.pipecat.service_factory import create_llm_service_from_provider


@pytest.mark.parametrize("model", ["gpt-5.3-chat", "GPT-5.3-CHAT"])
def test_azure_gpt5_omits_explicit_temperature(model):
    with patch("api.services.pipecat.service_factory.AzureLLMService") as service:
        create_llm_service_from_provider(
            provider=ServiceProviders.AZURE.value,
            model=model,
            api_key="test-azure-key",
            endpoint="https://example.openai.azure.com/",
        )

    settings = service.call_args.kwargs["settings"]
    assert settings.model == model
    assert settings.given_fields() == {"model": model}
    assert "temperature" not in settings.given_fields()


def test_azure_non_gpt5_preserves_temperature_and_connection_parameters():
    with patch("api.services.pipecat.service_factory.AzureLLMService") as service:
        create_llm_service_from_provider(
            provider=ServiceProviders.AZURE.value,
            model="gpt-4.1",
            api_key="test-azure-key",
            endpoint="https://example.openai.azure.com/",
        )

    kwargs = service.call_args.kwargs
    assert kwargs["api_key"] == "test-azure-key"
    assert kwargs["endpoint"] == "https://example.openai.azure.com/"
    assert kwargs["settings"].model == "gpt-4.1"
    assert kwargs["settings"].temperature == 0.1
    assert kwargs["settings"].given_fields() == {
        "model": "gpt-4.1",
        "temperature": 0.1,
    }
