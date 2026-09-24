from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ticket_automation.application.agent_execution import (
    AgentCapability,
    AgentTaskKind,
    ProviderId,
)
from ticket_automation.application.ports.preflight import (
    PreflightCheck,
    PreflightStatus,
)
from ticket_automation.composition import prepare_agent_providers
from ticket_automation.config import (
    AgentSettings,
    AppConfig,
    ConfigError,
    ProjectSettings,
    RunnerSettings,
    VerificationCommand,
    VerificationSettings,
)

ALL_CAPABILITIES = frozenset(AgentCapability)
READ_ONLY_CAPABILITIES = ALL_CAPABILITIES - {AgentCapability.WORKSPACE_WRITE_EXECUTION}


@dataclass
class StubRegistration:
    provider_id: ProviderId
    policy_version: str = "stub-v1"
    capabilities: frozenset[AgentCapability] = ALL_CAPABILITIES
    preflight_status: PreflightStatus = PreflightStatus.PASS
    compatibility_problem: str | None = None
    resolve_calls: int = 0
    preflight_calls: int = 0
    policy_calls: int = 0
    executor_calls: int = 0
    executor: StubExecutor = field(default_factory=lambda: StubExecutor())

    def resolve_settings(self, raw_settings, *, configuration_directory):
        self.resolve_calls += 1
        return StubSettings(
            values=tuple(sorted(raw_settings.items())),
            configuration_directory=configuration_directory,
        )

    def run_preflight(self, settings, *, repository_path):
        del settings, repository_path
        self.preflight_calls += 1
        return (
            PreflightCheck(
                name="connection",
                status=self.preflight_status,
                message="unavailable"
                if self.preflight_status is PreflightStatus.FAIL
                else "",
            ),
        )

    def resolve_run_policy(self, settings):
        del settings
        self.policy_calls += 1
        return StubPolicy(provider_id=self.provider_id)

    def runtime_compatibility_problem(self, policy):
        del policy
        return self.compatibility_problem

    def encode_run_policy(self, policy):
        if not isinstance(policy, StubPolicy):
            raise TypeError("wrong stub policy type")
        return {"provider_id": str(policy.provider_id)}

    def decode_run_policy(self, payload):
        if not isinstance(payload, dict) or set(payload) != {"provider_id"}:
            raise ValueError("invalid stub policy payload")
        return StubPolicy(provider_id=ProviderId(payload["provider_id"]))

    def create_executor(self, policy):
        del policy
        self.executor_calls += 1
        return self.executor


@dataclass(frozen=True)
class StubSettings:
    values: tuple[tuple[str, object], ...]
    configuration_directory: Path


@dataclass(frozen=True)
class StubPolicy:
    provider_id: ProviderId


class StubExecutor:
    def execute(self, request, *, on_invocation_start=None):
        del request, on_invocation_start
        raise AssertionError("Composition tests do not execute provider work.")


class MutablePolicyRegistration(StubRegistration):
    def resolve_run_policy(self, settings):
        del settings
        self.policy_calls += 1
        return {"provider_id": str(self.provider_id)}


def test_distinct_providers_are_selected_for_each_task_kind(tmp_path):
    registrations = {
        ProviderId(name): StubRegistration(ProviderId(name))
        for name in (
            "implementation-provider",
            "review-provider",
            "correction-provider",
        )
    }
    settings = _settings(
        implementation="implementation-provider",
        review="review-provider",
        correction="correction-provider",
    )
    prepared = prepare_agent_providers(
        settings,
        registry=registrations,
        configuration_directory=tmp_path,
    )
    policy = prepared.resolve_run_policy(
        _config(tmp_path, settings), target_repository_path=tmp_path
    )
    executors = prepared.create_executors(policy)

    assert (
        executors.implementation
        is registrations[ProviderId("implementation-provider")].executor
    )
    assert executors.review is registrations[ProviderId("review-provider")].executor
    assert (
        executors.correction
        is registrations[ProviderId("correction-provider")].executor
    )
    assert all(
        registration.policy_calls == 1 for registration in registrations.values()
    )
    assert all(
        registration.executor_calls == 1 for registration in registrations.values()
    )


def test_shared_provider_reuses_preflight_and_executor(tmp_path):
    provider_id = ProviderId("shared")
    registration = StubRegistration(provider_id)
    settings = _settings(implementation="shared", review="shared", correction="shared")
    prepared = prepare_agent_providers(
        settings,
        registry={provider_id: registration},
        configuration_directory=tmp_path,
    )
    preflight = prepared.run_preflight(repository_path=Path("repo"))
    policy = prepared.resolve_run_policy(
        _config(tmp_path, settings), target_repository_path=tmp_path
    )
    executors = prepared.create_executors(policy)

    assert preflight.passed
    assert registration.preflight_calls == 1
    assert registration.policy_calls == 1
    assert registration.executor_calls == 1
    assert executors.implementation is executors.review
    assert executors.review is executors.correction


def test_unassigned_provider_is_not_resolved_or_preflighted(tmp_path):
    assigned_id = ProviderId("assigned")
    unused_id = ProviderId("unused")
    assigned = StubRegistration(assigned_id)
    unused = StubRegistration(unused_id, preflight_status=PreflightStatus.FAIL)
    settings = AgentSettings(
        assignments={task_kind: assigned_id for task_kind in AgentTaskKind},
        providers={assigned_id: {}, unused_id: {}},
    )

    prepared = prepare_agent_providers(
        settings,
        registry={assigned_id: assigned, unused_id: unused},
        configuration_directory=tmp_path,
    )
    result = prepared.run_preflight(repository_path=Path("repo"))

    assert result.passed
    assert assigned.resolve_calls == 1
    assert assigned.preflight_calls == 1
    assert unused.resolve_calls == 0
    assert unused.preflight_calls == 0


def test_read_only_provider_is_accepted_for_review(tmp_path):
    full = StubRegistration(ProviderId("full"))
    read_only = StubRegistration(
        ProviderId("read-only"), capabilities=READ_ONLY_CAPABILITIES
    )
    prepared = prepare_agent_providers(
        _settings(implementation="full", review="read-only", correction="full"),
        registry={full.provider_id: full, read_only.provider_id: read_only},
        configuration_directory=tmp_path,
    )
    assert prepared.assignments[AgentTaskKind.REVIEW] == read_only.provider_id


@pytest.mark.parametrize(
    "task_kind", [AgentTaskKind.IMPLEMENTATION, AgentTaskKind.CORRECTION]
)
def test_read_only_provider_is_rejected_for_writable_tasks(tmp_path, task_kind):
    full = StubRegistration(ProviderId("full"))
    read_only = StubRegistration(
        ProviderId("read-only"), capabilities=READ_ONLY_CAPABILITIES
    )
    names = {kind: "full" for kind in AgentTaskKind}
    names[task_kind] = "read-only"
    with pytest.raises(
        ConfigError,
        match=rf"read-only cannot serve {task_kind.value}.*workspace-write-execution",
    ):
        prepare_agent_providers(
            _settings(
                implementation=names[AgentTaskKind.IMPLEMENTATION],
                review=names[AgentTaskKind.REVIEW],
                correction=names[AgentTaskKind.CORRECTION],
            ),
            registry={full.provider_id: full, read_only.provider_id: read_only},
            configuration_directory=tmp_path,
        )


def test_provider_preflight_failure_names_provider_and_assignments(tmp_path):
    provider_id = ProviderId("broken")
    registration = StubRegistration(provider_id, preflight_status=PreflightStatus.FAIL)
    prepared = prepare_agent_providers(
        _settings(implementation="broken", review="broken", correction="broken"),
        registry={provider_id: registration},
        configuration_directory=tmp_path,
    )
    result = prepared.run_preflight(repository_path=Path("repo"))

    assert not result.passed
    assert registration.preflight_calls == 1
    assert result.failed_checks[0].name.startswith(
        "Provider broken (implementation, review, correction)"
    )


def test_trusted_agent_settings_require_every_assignment():
    provider_id = ProviderId("shared")
    with pytest.raises(
        ConfigError, match="must contain implementation, review, and correction"
    ):
        AgentSettings(
            assignments={AgentTaskKind.REVIEW: provider_id},
            providers={provider_id: {}},
        )


def test_trusted_agent_settings_reject_unconfigured_assignment():
    provider_id = ProviderId("missing")
    with pytest.raises(ConfigError, match="assigned to unconfigured provider missing"):
        AgentSettings(
            assignments={task_kind: provider_id for task_kind in AgentTaskKind},
            providers={},
        )


def test_registry_key_must_match_registration_identity(tmp_path):
    configured_id = ProviderId("configured")
    registration = StubRegistration(ProviderId("different"))
    with pytest.raises(
        ConfigError, match="does not match registration identity different"
    ):
        prepare_agent_providers(
            _settings(
                implementation="configured",
                review="configured",
                correction="configured",
            ),
            registry={configured_id: registration},
            configuration_directory=tmp_path,
        )


@pytest.mark.parametrize(
    "capabilities",
    [
        set(ALL_CAPABILITIES),
        frozenset(capability.value for capability in ALL_CAPABILITIES),
    ],
)
def test_provider_capabilities_must_be_typed_and_immutable(tmp_path, capabilities):
    provider_id = ProviderId("invalid-capabilities")
    registration = StubRegistration(provider_id, capabilities=capabilities)

    with pytest.raises(ConfigError, match="frozenset of AgentCapability values"):
        prepare_agent_providers(
            _settings(
                implementation=str(provider_id),
                review=str(provider_id),
                correction=str(provider_id),
            ),
            registry={provider_id: registration},
            configuration_directory=tmp_path,
        )


def test_mutable_run_policy_is_rejected(tmp_path):
    provider_id = ProviderId("mutable")
    registration = MutablePolicyRegistration(provider_id)
    settings = _settings(
        implementation="mutable", review="mutable", correction="mutable"
    )
    prepared = prepare_agent_providers(
        settings,
        registry={provider_id: registration},
        configuration_directory=tmp_path,
    )

    with pytest.raises(ConfigError, match="mutable run policy"):
        prepared.resolve_run_policy(
            _config(tmp_path, settings), target_repository_path=tmp_path
        )


def test_runtime_compatibility_problem_prevents_executor_construction(tmp_path):
    provider_id = ProviderId("incompatible")
    registration = StubRegistration(
        provider_id,
        compatibility_problem="adapter version changed",
    )
    settings = _settings(
        implementation=str(provider_id),
        review=str(provider_id),
        correction=str(provider_id),
    )
    prepared = prepare_agent_providers(
        settings,
        registry={provider_id: registration},
        configuration_directory=tmp_path,
    )
    policy = prepared.resolve_run_policy(
        _config(tmp_path, settings), target_repository_path=tmp_path
    )

    with pytest.raises(ConfigError, match="adapter version changed"):
        prepared.create_executors(policy)

    assert registration.policy_calls == 1
    assert registration.executor_calls == 0


def test_trusted_agent_settings_own_nested_provider_values():
    provider_id = ProviderId("shared")
    nested: list[str] = []
    settings = AgentSettings(
        assignments={task_kind: provider_id for task_kind in AgentTaskKind},
        providers={provider_id: {"nested": (nested,)}},
    )

    nested.append("changed after construction")

    assert settings.providers[provider_id]["nested"] == ((),)


@pytest.mark.parametrize(
    "provider_settings",
    [
        {1: "non-string key"},
        {"mutable": {"set"}},
    ],
)
def test_trusted_agent_settings_reject_non_toml_provider_values(provider_settings):
    provider_id = ProviderId("shared")

    with pytest.raises(ConfigError, match="Provider settings"):
        AgentSettings(
            assignments={task_kind: provider_id for task_kind in AgentTaskKind},
            providers={provider_id: provider_settings},
        )


def _settings(*, implementation: str, review: str, correction: str) -> AgentSettings:
    names = {implementation, review, correction}
    return AgentSettings(
        assignments={
            AgentTaskKind.IMPLEMENTATION: ProviderId(implementation),
            AgentTaskKind.REVIEW: ProviderId(review),
            AgentTaskKind.CORRECTION: ProviderId(correction),
        },
        providers={ProviderId(name): {} for name in names},
    )


def _config(repository: Path, agents: AgentSettings) -> AppConfig:
    return AppConfig(
        project=ProjectSettings(
            name="provider composition",
            repo=repository.resolve(),
            protected_branches=("main",),
        ),
        runner=RunnerSettings(max_correction_rounds=1),
        agents=agents,
        verification=VerificationSettings(
            commands=(
                VerificationCommand(
                    name="tests",
                    argv=("python", "-m", "pytest"),
                    timeout_seconds=30,
                ),
            )
        ),
        source_files=(),
        configuration_directory=repository,
    )
