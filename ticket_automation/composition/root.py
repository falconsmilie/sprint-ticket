"""The small explicit production composition root."""

from __future__ import annotations

from types import MappingProxyType

from ..application.agent_execution import PersistedAgentExecutorFactory, ProviderId
from ..application.ports.handoff import FinalPatchCapture
from ..config import AppConfig
from ..infrastructure.final_patch import FileSystemFinalPatchCapture
from ..providers.codex_cli.composition import CodexCliProviderRegistration
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


__all__ = [
    "prepare_production_agents",
    "production_agent_executor_factory",
    "production_final_patch_capture",
]
