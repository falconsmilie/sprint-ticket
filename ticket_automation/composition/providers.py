"""Provider-neutral composition contracts and assignment validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, cast

from ..application.agent_execution import (
    AgentCapability,
    AgentExecutor,
    AgentExecutorAssignments,
    AgentTaskKind,
    ProviderId,
    RepositoryAccess,
    required_execution_capabilities,
)
from ..application.ports.preflight import (
    PreflightCheck,
    PreflightResult,
    PreflightStatus,
)
from ..config import AgentSettings, AppConfig, ConfigError
from ..resolved_config import (
    ResolvedRunPolicy,
    ResolvedRunPolicyError,
    resolve_run_policy,
)


class ProviderRegistration(Protocol):
    """Composition operations every explicitly registered provider supplies."""

    provider_id: ProviderId
    policy_version: str
    capabilities: frozenset[AgentCapability]

    def resolve_settings(
        self,
        raw_settings: Mapping[str, object],
        *,
        configuration_directory: Path,
    ) -> object: ...

    def run_preflight(
        self,
        settings: object,
        *,
        repository_path: Path,
    ) -> tuple[PreflightCheck, ...]: ...

    def resolve_run_policy(self, settings: object) -> object: ...

    def encode_run_policy(self, policy: object) -> object: ...

    def decode_run_policy(self, payload: object) -> object: ...

    def runtime_compatibility_problem(self, policy: object) -> str | None: ...

    def create_executor(self, policy: object) -> AgentExecutor: ...


@dataclass(frozen=True)
class ConfiguredProvider:
    registration: ProviderRegistration
    settings: object
    task_kinds: tuple[AgentTaskKind, ...]


@dataclass(frozen=True)
class PreparedAgentProviders:
    assignments: Mapping[AgentTaskKind, ProviderId]
    providers: Mapping[ProviderId, ConfiguredProvider]

    def __post_init__(self) -> None:
        if set(self.assignments) != set(AgentTaskKind):
            raise ConfigError(
                "Prepared provider assignments must contain implementation, review, "
                "and correction."
            )
        assigned_provider_ids = set(self.assignments.values())
        if set(self.providers) != assigned_provider_ids:
            raise ConfigError(
                "Prepared providers must exactly match the assigned provider IDs."
            )
        for provider_id, configured in self.providers.items():
            if configured.registration.provider_id != provider_id:
                raise ConfigError(
                    f"Provider registry key {provider_id} does not match registration "
                    f"identity {configured.registration.provider_id}."
                )
            expected_tasks = tuple(
                task_kind
                for task_kind in AgentTaskKind
                if self.assignments[task_kind] == provider_id
            )
            if configured.task_kinds != expected_tasks:
                raise ConfigError(
                    f"Configured task assignments for provider {provider_id} do not "
                    "match the prepared assignment map."
                )
        object.__setattr__(
            self, "assignments", MappingProxyType(dict(self.assignments))
        )
        object.__setattr__(self, "providers", MappingProxyType(dict(self.providers)))

    def run_preflight(self, *, repository_path: Path) -> PreflightResult:
        checks: list[PreflightCheck] = []
        for provider_id, configured in self.providers.items():
            assignments = ", ".join(kind.value for kind in configured.task_kinds)
            prefix = f"Provider {provider_id} ({assignments})"
            try:
                provider_checks = configured.registration.run_preflight(
                    configured.settings,
                    repository_path=repository_path,
                )
            except (OSError, RuntimeError, ValueError) as error:
                provider_checks = (
                    PreflightCheck(
                        name="Provider preflight",
                        status=PreflightStatus.FAIL,
                        message=str(error),
                    ),
                )
            for check in provider_checks:
                checks.append(
                    PreflightCheck(
                        name=f"{prefix}: {check.name}",
                        status=check.status,
                        message=check.message,
                    )
                )
        return PreflightResult(tuple(checks))

    def resolve_run_policy(
        self,
        config: AppConfig,
        *,
        target_repository_path: Path,
    ) -> ResolvedRunPolicy:
        """Resolve the complete immutable policy before a run is created."""

        if not isinstance(config, AppConfig):
            raise ConfigError("Run policy configuration has the wrong type.")

        policies: dict[ProviderId, object] = {}
        for provider_id, configured in self.providers.items():
            try:
                policy = configured.registration.resolve_run_policy(configured.settings)
                _validate_immutable_policy(policy, provider_id=provider_id)
            except ConfigError:
                raise
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                raise ConfigError(
                    f"Could not resolve run policy for provider {provider_id}: {error}"
                ) from error
            policies[provider_id] = policy
        registrations = {
            provider_id: configured.registration
            for provider_id, configured in self.providers.items()
        }
        return resolve_run_policy(
            config,
            target_repository_path=target_repository_path,
            assignments=self.assignments,
            provider_policies=policies,
            provider_registrations=registrations,
        )

    def compatibility_problem(self, resolved_policy: object) -> str | None:
        if not isinstance(resolved_policy, ResolvedRunPolicy):
            return "Persisted run policy has the wrong type."
        return resolved_policy.runtime_compatibility_problem(
            {
                provider_id: configured.registration
                for provider_id, configured in self.providers.items()
            }
        )

    def create_executors(self, resolved_policy: object) -> AgentExecutorAssignments:
        return _compose_persisted_agents(
            resolved_policy,
            registry={
                provider_id: configured.registration
                for provider_id, configured in self.providers.items()
            },
        )


@dataclass(frozen=True)
class RegisteredProviderExecutorFactory:
    """Executor factory backed by the statically composed provider registry."""

    registry: Mapping[ProviderId, ProviderRegistration]

    def __post_init__(self) -> None:
        object.__setattr__(self, "registry", MappingProxyType(dict(self.registry)))

    def compatibility_problem(self, resolved_policy: object) -> str | None:
        if not isinstance(resolved_policy, ResolvedRunPolicy):
            return "Persisted run policy has the wrong type."
        return resolved_policy.runtime_compatibility_problem(self.registry)

    def create_executors(self, resolved_policy: object) -> AgentExecutorAssignments:
        return _compose_persisted_agents(resolved_policy, registry=self.registry)


def _compose_persisted_agents(
    resolved_policy: object,
    *,
    registry: Mapping[ProviderId, ProviderRegistration],
) -> AgentExecutorAssignments:
    """Recreate executors exclusively from a persisted resolved-run policy."""

    if not isinstance(resolved_policy, ResolvedRunPolicy):
        raise ConfigError("Persisted run policy has the wrong type.")
    try:
        policies = resolved_policy.restore_provider_policies(registry)
    except ResolvedRunPolicyError as error:
        raise ConfigError(str(error)) from error
    executors: dict[ProviderId, AgentExecutor] = {}
    for persisted in resolved_policy.provider_policies:
        registration = registry[persisted.provider_id]
        try:
            policy = policies[persisted.provider_id]
            _validate_immutable_policy(policy, provider_id=persisted.provider_id)
            executors[persisted.provider_id] = registration.create_executor(policy)
        except ConfigError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ConfigError(
                f"Could not restore provider {persisted.provider_id}: {error}"
            ) from error
    assignments = resolved_policy.assignments
    return AgentExecutorAssignments(
        implementation=executors[assignments[AgentTaskKind.IMPLEMENTATION]],
        review=executors[assignments[AgentTaskKind.REVIEW]],
        correction=executors[assignments[AgentTaskKind.CORRECTION]],
    )


def prepare_agent_providers(
    settings: AgentSettings,
    *,
    registry: Mapping[ProviderId, ProviderRegistration],
    configuration_directory: Path,
) -> PreparedAgentProviders:
    """Resolve assigned provider settings and validate task requirements."""

    assigned_by_provider: dict[ProviderId, list[AgentTaskKind]] = {}
    for task_kind, provider_id in settings.assignments.items():
        assigned_by_provider.setdefault(provider_id, []).append(task_kind)

    configured: dict[ProviderId, ConfiguredProvider] = {}
    for provider_id, task_kinds in assigned_by_provider.items():
        registration = registry.get(provider_id)
        if registration is None:
            raise ConfigError(
                f"Assigned provider {provider_id} is not registered "
                f"(assignments: {_task_names(task_kinds)})."
            )
        if registration.provider_id != provider_id:
            raise ConfigError(
                f"Provider registry key {provider_id} does not match registration "
                f"identity {registration.provider_id}."
            )
        capabilities = _validated_capabilities(
            registration.capabilities,
            provider_id=provider_id,
        )
        raw_settings = settings.providers.get(provider_id)
        if raw_settings is None:
            raise ConfigError(
                f"Assigned provider {provider_id} has no configured settings "
                f"(assignments: {_task_names(task_kinds)})."
            )
        try:
            resolved_settings = registration.resolve_settings(
                raw_settings,
                configuration_directory=configuration_directory,
            )
        except (TypeError, ValueError) as error:
            raise ConfigError(
                f"Invalid settings for provider {provider_id}: {error}"
            ) from error

        for task_kind in task_kinds:
            missing = _requirements_for(task_kind) - capabilities
            if missing:
                names = ", ".join(sorted(item.value for item in missing))
                raise ConfigError(
                    f"Provider {provider_id} cannot serve {task_kind.value}; "
                    f"missing capabilities: {names}."
                )
        configured[provider_id] = ConfiguredProvider(
            registration=registration,
            settings=resolved_settings,
            task_kinds=tuple(task_kinds),
        )

    return PreparedAgentProviders(
        assignments=settings.assignments,
        providers=configured,
    )


def _requirements_for(task_kind: AgentTaskKind) -> frozenset[AgentCapability]:
    access = (
        RepositoryAccess.READ_ONLY
        if task_kind is AgentTaskKind.REVIEW
        else RepositoryAccess.WORKSPACE_WRITE
    )
    return required_execution_capabilities(access)


def _validated_capabilities(
    capabilities: object,
    *,
    provider_id: ProviderId,
) -> frozenset[AgentCapability]:
    if not isinstance(capabilities, frozenset) or not all(
        isinstance(capability, AgentCapability) for capability in capabilities
    ):
        raise ConfigError(
            f"Provider {provider_id} capabilities must be a frozenset of "
            "AgentCapability values."
        )
    return capabilities


def _task_names(task_kinds: list[AgentTaskKind]) -> str:
    return ", ".join(kind.value for kind in task_kinds)


def _validate_immutable_policy(policy: object, *, provider_id: ProviderId) -> None:
    if _is_deeply_immutable(policy, seen=set()):
        return
    raise ConfigError(
        f"Provider {provider_id} returned a mutable run policy; provider policies "
        "must be deeply immutable."
    )


def _is_deeply_immutable(value: object, *, seen: set[int]) -> bool:
    if value is None or isinstance(value, (str, bytes, bool, int, float, Enum, Path)):
        return True
    if isinstance(value, tuple):
        return all(_is_deeply_immutable(item, seen=seen) for item in value)
    if isinstance(value, frozenset):
        return all(_is_deeply_immutable(item, seen=seen) for item in value)
    if not is_dataclass(value) or isinstance(value, type):
        return False
    identity = id(value)
    if identity in seen:
        return True
    parameters = getattr(type(value), "__dataclass_params__", None)
    if not bool(getattr(parameters, "frozen", False)):
        return False
    seen.add(identity)
    try:
        return all(
            _is_deeply_immutable(getattr(value, field.name), seen=seen)
            for field in fields(cast(Any, value))
        )
    finally:
        seen.remove(identity)


__all__ = [
    "ProviderRegistration",
    "prepare_agent_providers",
]
