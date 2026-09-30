from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from tests.helpers import create_git_repo
from ticket_automation.application.agent_execution import (
    AgentCapability,
    AgentExecutionPolicy,
    AgentTaskKind,
    NetworkAccess,
    ProviderId,
    RepositoryAccess,
)
from ticket_automation.application.ports.preflight import PreflightResult
from ticket_automation.composition.providers import RegisteredProviderExecutorFactory
from ticket_automation.config import (
    AgentSettings,
    AppConfig,
    ConfigError,
    ProjectSettings,
    RunnerSettings,
    VerificationCommand,
    VerificationSettings,
)
from ticket_automation.resolved_config import (
    ResolvedRunPolicy,
    ResolvedRunPolicyError,
    ResolvedTaskPolicy,
    config_from_resolved_run_policy,
    resolve_run_policy,
)
from ticket_automation.runs import RunError, create_run_snapshot, load_run_record

ALL_CAPABILITIES = frozenset(AgentCapability)


@dataclass(frozen=True)
class StubPolicy:
    setting: str


class StubExecutor:
    def __init__(self, name: str):
        self.name = name

    def execute(self, request, *, on_invocation_start=None):
        del request, on_invocation_start
        raise AssertionError("Persistence tests do not invoke providers.")


class StubRegistration:
    policy_version = "stub-v1"
    capabilities = ALL_CAPABILITIES

    def __init__(self, provider_id: ProviderId, *, policy_version: str | None = None):
        self.provider_id = provider_id
        self.executor = StubExecutor(str(provider_id))
        self.decode_calls = 0
        self.executor_calls = 0
        self.created_policies: list[StubPolicy] = []
        if policy_version is not None:
            self.policy_version = policy_version

    def encode_run_policy(self, policy: object) -> object:
        if not isinstance(policy, StubPolicy):
            raise TypeError("stub policy has the wrong type")
        return {"setting": policy.setting}

    def decode_run_policy(self, payload: object) -> object:
        self.decode_calls += 1
        if (
            not isinstance(payload, dict)
            or set(payload) != {"setting"}
            or not isinstance(payload["setting"], str)
            or not payload["setting"]
        ):
            raise ValueError("invalid stub provider payload")
        return StubPolicy(payload["setting"])

    def runtime_compatibility_problem(self, policy: object) -> str | None:
        if not isinstance(policy, StubPolicy):
            return "wrong decoded policy type"
        return None

    def create_executor(self, policy: object):
        if not isinstance(policy, StubPolicy):
            raise TypeError("wrong decoded policy type")
        self.executor_calls += 1
        self.created_policies.append(policy)
        return self.executor


class MutableDecodedRegistration(StubRegistration):
    def decode_run_policy(self, payload: object) -> object:
        decoded = super().decode_run_policy(payload)
        assert isinstance(decoded, StubPolicy)
        return {"setting": decoded.setting}

    def runtime_compatibility_problem(self, policy: object) -> str | None:
        del policy
        return None


def test_round_trip_with_one_provider_assigned_to_every_task(tmp_path):
    provider_id = ProviderId("shared")
    registration = StubRegistration(provider_id)
    resolved = _resolve(
        tmp_path,
        assignments={kind: provider_id for kind in AgentTaskKind},
        registrations={provider_id: registration},
    )
    loaded = ResolvedRunPolicy.from_dict(resolved.to_dict())

    assert loaded == resolved
    assert loaded.assignments == {kind: provider_id for kind in AgentTaskKind}
    assert [item.provider_id for item in loaded.provider_policies] == [provider_id]
    assert loaded.task_policy(AgentTaskKind.REVIEW).execution_policy == (
        AgentExecutionPolicy(3600, NetworkAccess.DENIED)
    )
    assert loaded.task_policy(AgentTaskKind.IMPLEMENTATION).execution_policy == (
        AgentExecutionPolicy(3600, NetworkAccess.ALLOWED)
    )


def test_distinct_task_deadlines_round_trip_and_reconstruct_configuration(tmp_path):
    provider_id = ProviderId("shared")
    registration = StubRegistration(provider_id)
    config = _policy_config(
        tmp_path,
        assignments={kind: provider_id for kind in AgentTaskKind},
        registrations={provider_id: registration},
    )
    config = replace(
        config,
        agents=replace(
            config.agents,
            timeouts={
                AgentTaskKind.IMPLEMENTATION: 7200,
                AgentTaskKind.REVIEW: 5400,
                AgentTaskKind.CORRECTION: 7100,
            },
        ),
    )

    resolved = resolve_run_policy(
        config,
        target_repository_path=tmp_path,
        assignments=config.agents.assignments,
        provider_policies={provider_id: StubPolicy("value")},
        provider_registrations={provider_id: registration},
    )
    loaded = ResolvedRunPolicy.from_dict(resolved.to_dict())
    reconstructed = config_from_resolved_run_policy(loaded)

    assert {
        kind: loaded.task_policy(kind).execution_policy.timeout_seconds
        for kind in AgentTaskKind
    } == {
        AgentTaskKind.IMPLEMENTATION: 7200.0,
        AgentTaskKind.REVIEW: 5400.0,
        AgentTaskKind.CORRECTION: 7100.0,
    }
    assert reconstructed.agents.timeouts == config.agents.timeouts


def test_exact_float_boundary_deadline_serializes_and_reconstructs(tmp_path):
    provider_id = ProviderId("shared")
    registration = StubRegistration(provider_id)
    config = _policy_config(
        tmp_path,
        assignments={kind: provider_id for kind in AgentTaskKind},
        registrations={provider_id: registration},
    )
    boundary = 2**53
    config = replace(
        config,
        agents=replace(
            config.agents,
            timeouts={kind: boundary for kind in AgentTaskKind},
        ),
    )

    resolved = resolve_run_policy(
        config,
        target_repository_path=tmp_path,
        assignments=config.agents.assignments,
        provider_policies={provider_id: StubPolicy("value")},
        provider_registrations={provider_id: registration},
    )
    serialized = resolved.to_dict()
    loaded = ResolvedRunPolicy.from_dict(serialized)
    reconstructed = config_from_resolved_run_policy(loaded)

    assert (
        serialized["tasks"]["implementation"]["execution_policy"]["timeout_seconds"]
        == boundary
    )
    assert (
        loaded.task_policy(
            AgentTaskKind.IMPLEMENTATION
        ).execution_policy.timeout_seconds
        == boundary
    )
    assert reconstructed.agents.timeouts == {kind: boundary for kind in AgentTaskKind}


def test_persisted_integer_deadline_is_rejected_before_lossy_conversion(tmp_path):
    resolved, _ = _shared_policy(tmp_path)
    data = resolved.to_dict()
    data["tasks"]["implementation"]["execution_policy"]["timeout_seconds"] = 2**53 + 1

    with pytest.raises(ResolvedRunPolicyError, match="exactly representable"):
        ResolvedRunPolicy.from_dict(data)


@pytest.mark.parametrize(
    "value",
    [None, 0, -1, True, 1.5, float("inf"), 10**1000],
)
def test_persisted_task_deadline_requires_positive_whole_seconds(tmp_path, value):
    resolved, _ = _shared_policy(tmp_path)
    data = resolved.to_dict()
    data["tasks"]["implementation"]["execution_policy"]["timeout_seconds"] = value

    with pytest.raises(
        ResolvedRunPolicyError, match="timeout_seconds|execution_policy"
    ):
        ResolvedRunPolicy.from_dict(data)


def test_round_trip_and_executor_selection_with_distinct_task_providers(tmp_path):
    repository = create_git_repo(tmp_path / "repository")
    ticket = tmp_path / "TA-PERSIST-001.md"
    ticket.write_text("# Persist provider-neutral policy\n", encoding="utf-8")
    ids = {kind: ProviderId(f"{kind.value}-provider") for kind in AgentTaskKind}
    registrations = {
        provider_id: StubRegistration(provider_id) for provider_id in ids.values()
    }
    config = _policy_config(
        repository,
        assignments=ids,
        registrations=registrations,
    )
    config = replace(
        config,
        agents=replace(
            config.agents,
            timeouts={
                AgentTaskKind.IMPLEMENTATION: 7200,
                AgentTaskKind.REVIEW: 5400,
                AgentTaskKind.CORRECTION: 7100,
            },
        ),
    )
    resolved = resolve_run_policy(
        config,
        target_repository_path=repository,
        assignments=ids,
        provider_policies={
            provider_id: StubPolicy(setting=f"setting-for-{provider_id}")
            for provider_id in registrations
        },
        provider_registrations=registrations,
    )
    creation = create_run_snapshot(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        provider_preflight=lambda *, repository_path: PreflightResult(()),
        resolved_policy=resolved,
    )
    loaded = load_run_record(creation.run_dir / "run.json")

    executors = RegisteredProviderExecutorFactory(registrations).create_executors(
        loaded.resolved_policy
    )

    assert (
        executors.implementation
        is registrations[ids[AgentTaskKind.IMPLEMENTATION]].executor
    )
    assert executors.review is registrations[ids[AgentTaskKind.REVIEW]].executor
    assert executors.correction is registrations[ids[AgentTaskKind.CORRECTION]].executor
    assert len(loaded.resolved_policy.provider_policies) == 3
    assert {
        kind: loaded.resolved_policy.task_policy(kind).execution_policy.timeout_seconds
        for kind in AgentTaskKind
    } == {
        AgentTaskKind.IMPLEMENTATION: 7200.0,
        AgentTaskKind.REVIEW: 5400.0,
        AgentTaskKind.CORRECTION: 7100.0,
    }
    for provider_id, registration in registrations.items():
        assert registration.created_policies == [
            StubPolicy(setting=f"setting-for-{provider_id}")
        ]
    assert loaded == creation.run_record


def test_unsupported_resolved_policy_schema_is_rejected_directly(tmp_path):
    policy, _ = _shared_policy(tmp_path)
    data = policy.to_dict()
    data["schema_version"] = 999
    with pytest.raises(
        ResolvedRunPolicyError,
        match="Unsupported resolved policy schema version",
    ):
        ResolvedRunPolicy.from_dict(data)


def test_tampered_assignment_and_missing_provider_policy_fail_closed(tmp_path):
    resolved, _ = _shared_policy(tmp_path)
    assignment = resolved.to_dict()
    assignment["tasks"]["implementation"]["provider_id"] = "unknown"
    with pytest.raises(ResolvedRunPolicyError, match="exactly match task assignments"):
        ResolvedRunPolicy.from_dict(assignment)

    missing = resolved.to_dict()
    missing["providers"].pop("shared")
    with pytest.raises(ResolvedRunPolicyError, match="at least one provider"):
        ResolvedRunPolicy.from_dict(missing)


def test_missing_capability_fails_during_core_policy_decode(tmp_path):
    resolved, _ = _shared_policy(tmp_path)
    data = resolved.to_dict()
    capabilities = data["providers"]["shared"]["declared_capabilities"]
    capabilities.remove(AgentCapability.WORKSPACE_WRITE_EXECUTION.value)

    with pytest.raises(ResolvedRunPolicyError, match="lacks capabilities required"):
        ResolvedRunPolicy.from_dict(data)


@pytest.mark.parametrize(
    ("task_kind", "field", "value"),
    [
        ("implementation", "timeout_seconds", 0),
        ("implementation", "network_access", "denied"),
        ("review", "network_access", "allowed"),
    ],
)
def test_tampered_agent_execution_policy_fails_during_core_decode(
    tmp_path,
    task_kind,
    field,
    value,
):
    resolved, _ = _shared_policy(tmp_path)
    data = resolved.to_dict()
    data["tasks"][task_kind]["execution_policy"][field] = value

    with pytest.raises(ResolvedRunPolicyError, match="execution_policy|incompatible"):
        ResolvedRunPolicy.from_dict(data)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda execution_policy: execution_policy.pop("timeout_seconds"),
            "missing timeout_seconds",
        ),
        (
            lambda execution_policy: execution_policy.update(
                {"sandbox": "workspace-write"}
            ),
            "unexpected sandbox",
        ),
    ],
    ids=["missing-field", "unexpected-field"],
)
def test_execution_policy_codec_requires_exact_nested_fields(
    tmp_path,
    mutate,
    message,
):
    resolved, _ = _shared_policy(tmp_path)
    data = resolved.to_dict()
    execution_policy = data["tasks"]["implementation"]["execution_policy"]
    mutate(execution_policy)

    with pytest.raises(ResolvedRunPolicyError, match=message):
        ResolvedRunPolicy.from_dict(data)


@pytest.mark.parametrize(
    "location",
    ["root", "task", "provider", "verification-command"],
)
def test_core_codec_rejects_unknown_fields_outside_provider_payload(tmp_path, location):
    resolved, _ = _shared_policy(tmp_path)
    data = resolved.to_dict()
    if location == "root":
        data["codex"] = {"model": "provider-field"}
    elif location == "task":
        data["tasks"]["implementation"]["sandbox"] = "workspace-write"
    elif location == "provider":
        data["providers"]["shared"]["model"] = "provider-field"
    else:
        data["verification"]["commands"][0]["shell"] = True

    with pytest.raises(ResolvedRunPolicyError, match="unexpected"):
        ResolvedRunPolicy.from_dict(data)


def test_core_codec_requires_the_opaque_provider_payload_field(tmp_path):
    resolved, _ = _shared_policy(tmp_path)
    data = resolved.to_dict()
    del data["providers"]["shared"]["payload"]

    with pytest.raises(ResolvedRunPolicyError, match="missing payload"):
        ResolvedRunPolicy.from_dict(data)


def test_unknown_provider_and_adapter_version_mismatch_fail_before_execution(tmp_path):
    resolved, registration = _shared_policy(tmp_path)

    with pytest.raises(ConfigError, match="not registered"):
        RegisteredProviderExecutorFactory({}).create_executors(resolved)
    replacement = StubRegistration(registration.provider_id, policy_version="stub-v2")
    with pytest.raises(ConfigError, match="adapter policy version is incompatible"):
        RegisteredProviderExecutorFactory(
            {replacement.provider_id: replacement}
        ).create_executors(resolved)
    assert replacement.executor_calls == 0

    replacement = StubRegistration(registration.provider_id)
    replacement.capabilities = ALL_CAPABILITIES - {
        AgentCapability.DIAGNOSTIC_ARTIFACT_CAPTURE
    }
    with pytest.raises(ConfigError, match="declared capabilities changed"):
        RegisteredProviderExecutorFactory(
            {replacement.provider_id: replacement}
        ).create_executors(resolved)
    assert replacement.executor_calls == 0


def test_registered_provider_capabilities_must_remain_typed_and_immutable(tmp_path):
    resolved, registration = _shared_policy(tmp_path)
    registration.capabilities = set(ALL_CAPABILITIES)

    with pytest.raises(ConfigError, match="registration is invalid"):
        RegisteredProviderExecutorFactory(
            {registration.provider_id: registration}
        ).create_executors(resolved)

    assert registration.executor_calls == 0


def test_provider_adapter_owns_persisted_payload_validation(tmp_path):
    resolved, registration = _shared_policy(tmp_path)
    data = resolved.to_dict()
    data["providers"]["shared"]["payload"] = {"tampered": True}

    loaded = ResolvedRunPolicy.from_dict(data)
    with pytest.raises(
        ConfigError,
        match="persisted policy is invalid: invalid stub provider payload",
    ):
        RegisteredProviderExecutorFactory(
            {registration.provider_id: registration}
        ).create_executors(loaded)

    assert registration.executor_calls == 0


def test_mutable_policy_decoded_by_adapter_is_rejected_before_executor_creation(
    tmp_path,
):
    provider_id = ProviderId("mutable-decoder")
    registration = MutableDecodedRegistration(provider_id)
    resolved = _resolve(
        tmp_path,
        assignments={kind: provider_id for kind in AgentTaskKind},
        registrations={provider_id: registration},
    )

    with pytest.raises(ConfigError, match="mutable run policy"):
        RegisteredProviderExecutorFactory({provider_id: registration}).create_executors(
            ResolvedRunPolicy.from_dict(resolved.to_dict())
        )

    assert registration.executor_calls == 0


def test_executor_is_constructed_from_the_decoded_persisted_payload(tmp_path):
    resolved, registration = _shared_policy(tmp_path)
    persisted_setting = resolved.provider_policies[0].payload()["setting"]

    RegisteredProviderExecutorFactory(
        {registration.provider_id: registration}
    ).create_executors(ResolvedRunPolicy.from_dict(resolved.to_dict()))

    assert registration.created_policies == [StubPolicy(persisted_setting)]


def test_run_creation_rejects_policy_from_a_different_configuration(tmp_path):
    repository = create_git_repo(tmp_path / "repository")
    ticket = tmp_path / "TA-PERSIST-001.md"
    ticket.write_text("# Persist provider-neutral policy\n", encoding="utf-8")
    resolved, _ = _shared_policy(repository)
    mismatched = replace(resolved, max_correction_rounds=1)
    config = _policy_config(
        repository,
        assignments=resolved.assignments,
        registrations={
            policy.provider_id: StubRegistration(policy.provider_id)
            for policy in resolved.provider_policies
        },
    )

    with pytest.raises(RunError, match="correction limit"):
        create_run_snapshot(
            config,
            ticket,
            runs_dir=tmp_path / "runs",
            provider_preflight=lambda *, repository_path: PreflightResult(()),
            resolved_policy=mismatched,
        )

    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("task_kind", list(AgentTaskKind))
def test_run_creation_rejects_each_task_deadline_mismatch(tmp_path, task_kind):
    repository = create_git_repo(tmp_path / "repository")
    ticket = tmp_path / "TA-PERSIST-001.md"
    ticket.write_text("# Persist provider-neutral policy\n", encoding="utf-8")
    resolved, registration = _shared_policy(repository)
    configured_timeouts = {kind: 3600 for kind in AgentTaskKind}
    configured_timeouts[task_kind] = 3601
    config = _policy_config(
        repository,
        assignments=resolved.assignments,
        registrations={registration.provider_id: registration},
    )
    config = replace(
        config,
        agents=replace(config.agents, timeouts=configured_timeouts),
    )

    with pytest.raises(RunError, match="agent task deadlines"):
        create_run_snapshot(
            config,
            ticket,
            runs_dir=tmp_path / "runs",
            provider_preflight=lambda *, repository_path: PreflightResult(()),
            resolved_policy=resolved,
        )

    assert not (tmp_path / "runs").exists()


def test_trusted_task_policy_construction_rejects_incompatible_requirements():
    with pytest.raises(
        ResolvedRunPolicyError, match="task requirements are incompatible"
    ):
        ResolvedTaskPolicy(
            task_kind=AgentTaskKind.REVIEW,
            provider_id=ProviderId("shared"),
            repository_access=RepositoryAccess.WORKSPACE_WRITE,
            required_capabilities=ALL_CAPABILITIES,
            execution_policy=AgentExecutionPolicy(3600, NetworkAccess.DENIED),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target_repository_path", "relative/repository", "must be absolute"),
        ("max_correction_rounds", 0, "must be positive"),
        ("ticket_automation_package", "other-package", "package is unsupported"),
        ("policy_asset_sha256", (), "policy_assets is incomplete"),
    ],
)
def test_trusted_run_policy_construction_rejects_invalid_core_values(
    tmp_path,
    field,
    value,
    message,
):
    resolved, _ = _shared_policy(tmp_path)

    with pytest.raises(ResolvedRunPolicyError, match=message):
        replace(resolved, **{field: value})


def _shared_policy(tmp_path: Path) -> tuple[ResolvedRunPolicy, StubRegistration]:
    provider_id = ProviderId("shared")
    registration = StubRegistration(provider_id)
    return (
        _resolve(
            tmp_path,
            assignments={kind: provider_id for kind in AgentTaskKind},
            registrations={provider_id: registration},
        ),
        registration,
    )


def _resolve(
    tmp_path: Path,
    *,
    assignments: dict[AgentTaskKind, ProviderId],
    registrations: dict[ProviderId, StubRegistration],
) -> ResolvedRunPolicy:
    config = _policy_config(
        tmp_path,
        assignments=assignments,
        registrations=registrations,
    )
    return resolve_run_policy(
        config,
        target_repository_path=tmp_path,
        assignments=assignments,
        provider_policies={
            provider_id: StubPolicy(setting=f"setting-for-{provider_id}")
            for provider_id in registrations
        },
        provider_registrations=registrations,
    )


def _policy_config(
    repository: Path,
    *,
    assignments: dict[AgentTaskKind, ProviderId],
    registrations: dict[ProviderId, StubRegistration],
) -> AppConfig:
    return AppConfig(
        project=ProjectSettings(
            name="policy test",
            repo=repository.resolve(),
            protected_branches=("protected",),
        ),
        runner=RunnerSettings(max_correction_rounds=2),
        agents=AgentSettings(
            assignments=assignments,
            providers={provider_id: {} for provider_id in registrations},
        ),
        verification=VerificationSettings(
            commands=(
                VerificationCommand(
                    name="tests",
                    argv=(sys.executable, "-c", "raise SystemExit(0)"),
                    timeout_seconds=30,
                ),
            )
        ),
        source_files=(),
        configuration_directory=repository.parent,
    )
