"""The small explicit production composition root."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

from ..application.agent_execution import PersistedAgentExecutorFactory, ProviderId
from ..application.ports.handoff import FinalPatchCapture
from ..config import (
    AgentSettings,
    AppConfig,
    ConfigError,
)
from ..infrastructure.final_patch import FileSystemFinalPatchCapture
from ..providers.codex_cli.composition import CodexCliProviderRegistration
from ..providers.codex_cli.identity import PROVIDER_ID
from ..providers.codex_cli.settings import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING_EFFORT,
    CodexExecutionOverrides,
    CodexExecutionSettings,
    validate_codex_execution_settings,
)
from .providers import (
    PreparedAgentProviders,
    ProviderRegistration,
    RegisteredProviderExecutorFactory,
    prepare_agent_providers,
)


def production_provider_registry() -> MappingProxyType[
    ProviderId, ProviderRegistration
]:
    codex = CodexCliProviderRegistration()
    return MappingProxyType({codex.provider_id: codex})


def prepare_production_agents(config: AppConfig) -> PreparedAgentProviders:
    return prepare_agent_providers(
        config.agents,
        registry=production_provider_registry(),
        configuration_directory=config.configuration_directory,
    )


def production_agent_executor_factory() -> PersistedAgentExecutorFactory:
    return RegisteredProviderExecutorFactory(production_provider_registry())


def production_final_patch_capture() -> FinalPatchCapture:
    return FileSystemFinalPatchCapture()


def apply_codex_execution_overrides(
    config: AppConfig,
    overrides: CodexExecutionOverrides,
) -> AppConfig:
    """Apply CLI-only Codex choices at the production composition boundary."""

    current = config.agents.providers.get(PROVIDER_ID)
    if current is None:
        raise ConfigError("Codex CLI overrides require [agents.providers.codex-cli].")
    execution = validate_codex_execution_settings(
        CodexExecutionSettings(
            model=(
                overrides.model
                if overrides.model is not None
                else current.get("model", DEFAULT_CODEX_MODEL)  # type: ignore[arg-type]
            ),
            reasoning_effort=(
                overrides.reasoning_effort
                if overrides.reasoning_effort is not None
                else current.get("reasoning_effort", DEFAULT_CODEX_REASONING_EFFORT)  # type: ignore[arg-type]
            ),
        ),
        model_name=(
            "--model"
            if overrides.model is not None
            else "agents.providers.codex-cli.model"
        ),
        reasoning_name=(
            "--reasoning-effort"
            if overrides.reasoning_effort is not None
            else "agents.providers.codex-cli.reasoning_effort"
        ),
    )
    providers = {
        provider_id: dict(settings)
        for provider_id, settings in config.agents.providers.items()
    }
    providers[PROVIDER_ID]["model"] = execution.model
    providers[PROVIDER_ID]["reasoning_effort"] = execution.reasoning_effort
    return replace(
        config,
        agents=AgentSettings(
            assignments=config.agents.assignments,
            providers=providers,
        ),
    )


__all__ = [
    "apply_codex_execution_overrides",
    "prepare_production_agents",
    "production_agent_executor_factory",
    "production_final_patch_capture",
]
