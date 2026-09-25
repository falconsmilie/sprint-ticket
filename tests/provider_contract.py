"""Reusable conformance suite for provider-neutral agent adapters.

Provider test modules register an adapter by subclassing ``ProviderContractTests``
and supplying the ``provider_contract`` fixture.  Provider-specific transports turn
the neutral outcomes below into deterministic adapter inputs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import cast

import pytest

from ticket_automation.application.agent_execution import (
    CORRECTION_RESULT_CONTRACT,
    IMPLEMENTATION_RESULT_CONTRACT,
    REVIEW_RESULT_CONTRACT,
    AgentCapability,
    AgentExecution,
    AgentExecutionPolicy,
    AgentExecutionRequest,
    AgentExecutionStatus,
    AgentExecutor,
    AgentFailureCategory,
    AgentTaskKind,
    ArtifactRole,
    AttemptArtifactLayout,
    InvocationStart,
    NetworkAccess,
    ProviderId,
    RepositoryAccess,
    required_execution_capabilities,
)
from ticket_automation.application.ports.preflight import (
    PreflightCheck,
    PreflightStatus,
)
from ticket_automation.composition.providers import ProviderRegistration
from ticket_automation.domain.task_results import (
    ImplementationResult,
    ReviewResult,
    TaskResult,
)
from ticket_automation.execution_evidence import (
    decode_execution_evidence,
    encode_execution_evidence,
    evidence_from_execution,
    read_execution_evidence,
    write_execution_evidence,
)


class ContractOutcome(StrEnum):
    """Provider-neutral transport outcomes supplied by an adapter fixture."""

    SUCCESS = "success"
    PROVIDER_UNAVAILABLE = "provider-unavailable"
    INVOCATION_START_FAILURE = "invocation-start-failure"
    TIMEOUT = "timeout"
    PROVIDER_REJECTION = "provider-rejection"
    NON_SUCCESSFUL_EXECUTION = "non-successful-execution"
    MISSING_RESULT = "missing-result"
    INVALID_RESULT = "invalid-result"


@dataclass(frozen=True)
class FailureExpectation:
    category: AgentFailureCategory
    invocation_start: InvocationStart


FAILURE_EXPECTATIONS: Mapping[ContractOutcome, FailureExpectation] = {
    ContractOutcome.PROVIDER_UNAVAILABLE: FailureExpectation(
        AgentFailureCategory.PROVIDER_UNAVAILABLE,
        InvocationStart.NOT_STARTED,
    ),
    ContractOutcome.INVOCATION_START_FAILURE: FailureExpectation(
        AgentFailureCategory.INVOCATION_START_FAILURE,
        InvocationStart.NOT_STARTED,
    ),
    ContractOutcome.TIMEOUT: FailureExpectation(
        AgentFailureCategory.TIMEOUT,
        InvocationStart.STARTED,
    ),
    ContractOutcome.PROVIDER_REJECTION: FailureExpectation(
        AgentFailureCategory.PROVIDER_REJECTION_OR_SERVICE_FAILURE,
        InvocationStart.STARTED,
    ),
    ContractOutcome.NON_SUCCESSFUL_EXECUTION: FailureExpectation(
        AgentFailureCategory.NON_SUCCESSFUL_EXECUTION,
        InvocationStart.STARTED,
    ),
    ContractOutcome.MISSING_RESULT: FailureExpectation(
        AgentFailureCategory.MISSING_RESULT,
        InvocationStart.STARTED,
    ),
    ContractOutcome.INVALID_RESULT: FailureExpectation(
        AgentFailureCategory.INVALID_RESULT,
        InvocationStart.STARTED,
    ),
}


@dataclass
class TransportObservation:
    """Only neutral facts exposed by a provider's deterministic fake transport."""

    invocation_attempts: int = 0
    invocation_starts: int = 0
    prompts: tuple[str, ...] = ()

    def record_prompt(self, prompt: str) -> None:
        self.prompts = (*self.prompts, prompt)


@dataclass(frozen=True)
class AdapterProbe:
    executor: AgentExecutor
    observation: TransportObservation


ProbeFactory = Callable[[ContractOutcome, AgentTaskKind], AdapterProbe]
PromptExpectation = Callable[[str], str]


@dataclass(frozen=True)
class ProviderContractFixture:
    """Everything the common suite needs from one provider adapter."""

    provider_id: ProviderId
    declared_capabilities: frozenset[AgentCapability]
    registration: ProviderRegistration
    valid_settings: Mapping[str, object]
    invalid_settings: Mapping[str, object]
    configuration_directory: Path
    repository_path: Path
    artifact_root: Path
    make_probe: ProbeFactory
    make_capability_limited_probe: Callable[[AgentTaskKind], AdapterProbe]
    expected_transport_prompt: PromptExpectation


_CORE_CAPABILITIES = frozenset(
    {
        AgentCapability.STRUCTURED_RESULT,
        AgentCapability.ISOLATED_INVOCATION,
        AgentCapability.DIAGNOSTIC_ARTIFACT_CAPTURE,
        AgentCapability.NETWORK_POLICY_CONTROL,
    }
)


class ContractInterruption(RuntimeError):
    """Sentinel used to verify interruption propagation through the port."""


def supported_task_kinds(
    capabilities: frozenset[AgentCapability],
) -> frozenset[AgentTaskKind]:
    supported: set[AgentTaskKind] = set()
    if AgentCapability.READ_ONLY_EXECUTION in capabilities:
        supported.add(AgentTaskKind.REVIEW)
    if AgentCapability.WORKSPACE_WRITE_EXECUTION in capabilities:
        supported.update({AgentTaskKind.IMPLEMENTATION, AgentTaskKind.CORRECTION})
    return frozenset(supported)


def assert_provider_declaration(fixture: ProviderContractFixture) -> None:
    """Validate identity and capabilities without provider-name assumptions."""

    registration = fixture.registration
    assert isinstance(fixture.provider_id, ProviderId), "provider identity is untyped"
    assert registration.provider_id == fixture.provider_id, (
        "registration identity differs from the stable provider identity"
    )
    capabilities = registration.capabilities
    assert isinstance(capabilities, frozenset), "capabilities must be immutable"
    assert all(isinstance(item, AgentCapability) for item in capabilities), (
        "capabilities must contain AgentCapability values"
    )
    assert capabilities == fixture.declared_capabilities, (
        "fixture and registration capability declarations differ"
    )
    assert _CORE_CAPABILITIES <= capabilities, (
        "provider is missing a required core capability"
    )
    assert supported_task_kinds(capabilities), (
        "provider must declare at least one repository access capability"
    )


def assert_execution_conforms(
    *,
    provider_id: ProviderId,
    request: AgentExecutionRequest[TaskResult],
    execution: AgentExecution[TaskResult],
    failure: FailureExpectation | None = None,
) -> None:
    """Validate one returned execution and every referenced generic artifact."""

    assert isinstance(execution, AgentExecution), (
        "executor must return an AgentExecution envelope"
    )
    assert execution.provider_id == provider_id, "execution provider identity changed"
    assert execution.task_kind is request.task_kind, "execution task kind changed"
    assert execution.started_at.tzinfo is not None
    assert execution.ended_at.tzinfo is not None
    assert execution.duration_seconds >= 0

    if failure is None:
        assert execution.status is AgentExecutionStatus.SUCCESS
        assert execution.invocation_start is InvocationStart.STARTED
        assert request.result_contract.accepts(execution.result), (
            "execution result does not satisfy the requested typed result contract"
        )
        assert execution.failure_category is None
        assert execution.failure_message is None
    else:
        assert execution.status is AgentExecutionStatus.FAILED
        assert execution.result is None
        assert execution.failure_category is failure.category, (
            "provider-native failure mapped to the wrong neutral category"
        )
        assert execution.invocation_start is failure.invocation_start, (
            "failure has incorrect invocation-start certainty"
        )
        assert execution.failure_message is not None
        assert execution.failure_message.strip()

    roles = tuple(reference.role for reference in execution.artifacts)
    assert len(roles) == len(set(roles)), "artifact roles must not be duplicated"
    layout = request.artifact_layout
    assert layout is not None
    for reference in execution.artifacts:
        layout.resolve(reference, require_exists=True)
    if failure is None:
        assert roles.count(ArtifactRole.TYPED_RESULT) == 1, (
            "successful execution must reference its typed result artifact"
        )


def _access_for(task_kind: AgentTaskKind) -> RepositoryAccess:
    if task_kind is AgentTaskKind.REVIEW:
        return RepositoryAccess.READ_ONLY
    return RepositoryAccess.WORKSPACE_WRITE


def _result_type_for(task_kind: AgentTaskKind) -> type[TaskResult]:
    if task_kind is AgentTaskKind.REVIEW:
        return ReviewResult
    return ImplementationResult


def _request(
    fixture: ProviderContractFixture,
    task_kind: AgentTaskKind,
    *,
    access: RepositoryAccess | None = None,
    prompt: str = "Neutral application prompt.\nPreserve this text exactly.",
) -> AgentExecutionRequest[TaskResult]:
    requested_access = access or _access_for(task_kind)
    contract = {
        AgentTaskKind.IMPLEMENTATION: IMPLEMENTATION_RESULT_CONTRACT,
        AgentTaskKind.REVIEW: REVIEW_RESULT_CONTRACT,
        AgentTaskKind.CORRECTION: CORRECTION_RESULT_CONTRACT,
    }[task_kind]
    attempt = (
        fixture.artifact_root
        / task_kind.value
        / requested_access.value
        / "attempts"
        / f"001-{task_kind.value}"
    )
    layout = AttemptArtifactLayout(attempt.parents[1], attempt)
    return AgentExecutionRequest(
        task_kind=task_kind,
        repository_path=fixture.repository_path,
        repository_access=requested_access,
        prompt=prompt,
        result_contract=contract,
        artifact_directory=attempt,
        artifact_layout=layout,
        policy=AgentExecutionPolicy(
            timeout_seconds=2,
            network_access=(
                NetworkAccess.DENIED
                if requested_access is RepositoryAccess.READ_ONLY
                else NetworkAccess.ALLOWED
            ),
        ),
        required_capabilities=required_execution_capabilities(requested_access),
    )


def _representative_task(fixture: ProviderContractFixture) -> AgentTaskKind:
    supported = supported_task_kinds(fixture.declared_capabilities)
    if AgentTaskKind.REVIEW in supported:
        return AgentTaskKind.REVIEW
    return AgentTaskKind.IMPLEMENTATION


class ProviderContractTests:
    """Inherited pytest suite; subclasses provide one adapter fixture only."""

    def test_provider_identity_capabilities_and_registration_are_stable(
        self,
        provider_contract: ProviderContractFixture,
    ) -> None:
        assert_provider_declaration(provider_contract)
        task_kind = _representative_task(provider_contract)
        first = provider_contract.make_probe(ContractOutcome.SUCCESS, task_kind)
        second = provider_contract.make_probe(ContractOutcome.SUCCESS, task_kind)

        assert first.executor.capabilities == provider_contract.declared_capabilities
        assert second.executor.capabilities == provider_contract.declared_capabilities

    def test_settings_preflight_and_policy_round_trip_are_offline(
        self,
        provider_contract: ProviderContractFixture,
    ) -> None:
        registration = provider_contract.registration
        with pytest.raises((TypeError, ValueError)):
            registration.resolve_settings(
                provider_contract.invalid_settings,
                configuration_directory=provider_contract.configuration_directory,
            )

        settings = registration.resolve_settings(
            provider_contract.valid_settings,
            configuration_directory=provider_contract.configuration_directory,
        )
        checks = registration.run_preflight(
            settings,
            repository_path=provider_contract.repository_path,
        )
        assert checks
        assert all(isinstance(check, PreflightCheck) for check in checks)
        assert all(check.status is PreflightStatus.PASS for check in checks)

        policy = registration.resolve_run_policy(settings)
        restored = registration.decode_run_policy(
            registration.encode_run_policy(policy)
        )
        assert registration.runtime_compatibility_problem(restored) is None
        executor = registration.create_executor(restored)
        assert executor.capabilities == provider_contract.declared_capabilities

    @pytest.mark.parametrize("task_kind", tuple(AgentTaskKind))
    def test_supported_task_dispatch_preserves_prompt_and_returns_typed_result(
        self,
        provider_contract: ProviderContractFixture,
        task_kind: AgentTaskKind,
    ) -> None:
        if task_kind not in supported_task_kinds(
            provider_contract.declared_capabilities
        ):
            pytest.skip("task access is not declared by this provider")
        probe = provider_contract.make_probe(ContractOutcome.SUCCESS, task_kind)
        request = _request(provider_contract, task_kind)
        starts = 0

        def observe_start() -> None:
            nonlocal starts
            starts += 1

        execution = probe.executor.execute(
            request,
            on_invocation_start=observe_start,
        )

        assert_execution_conforms(
            provider_id=provider_contract.provider_id,
            request=request,
            execution=cast(AgentExecution[TaskResult], execution),
        )
        assert type(execution.result) is _result_type_for(task_kind)
        assert starts == 1
        assert probe.observation.invocation_attempts == 1
        assert probe.observation.invocation_starts == 1
        assert probe.observation.prompts == (
            provider_contract.expected_transport_prompt(request.prompt),
        )

    @pytest.mark.parametrize("task_kind", tuple(AgentTaskKind))
    def test_adapter_enforces_task_repository_access_before_invocation(
        self,
        provider_contract: ProviderContractFixture,
        task_kind: AgentTaskKind,
    ) -> None:
        if task_kind not in supported_task_kinds(
            provider_contract.declared_capabilities
        ):
            pytest.skip("task access is not declared by this provider")
        wrong_access = (
            RepositoryAccess.WORKSPACE_WRITE
            if _access_for(task_kind) is RepositoryAccess.READ_ONLY
            else RepositoryAccess.READ_ONLY
        )
        probe = provider_contract.make_probe(ContractOutcome.SUCCESS, task_kind)
        request = _request(provider_contract, task_kind, access=wrong_access)

        execution = probe.executor.execute(request)

        assert_execution_conforms(
            provider_id=provider_contract.provider_id,
            request=request,
            execution=cast(AgentExecution[TaskResult], execution),
            failure=FailureExpectation(
                AgentFailureCategory.CAPABILITY_OR_CONFIGURATION_FAILURE,
                InvocationStart.NOT_STARTED,
            ),
        )
        assert probe.observation.invocation_attempts == 0
        assert probe.observation.invocation_starts == 0

    def test_missing_required_capability_is_rejected_before_invocation(
        self,
        provider_contract: ProviderContractFixture,
    ) -> None:
        task_kind = (
            AgentTaskKind.IMPLEMENTATION
            if AgentCapability.WORKSPACE_WRITE_EXECUTION
            in provider_contract.declared_capabilities
            else AgentTaskKind.REVIEW
        )
        probe = provider_contract.make_capability_limited_probe(task_kind)
        request = _request(provider_contract, task_kind)
        assert request.missing_capabilities(probe.executor.capabilities)

        execution = probe.executor.execute(request)

        assert_execution_conforms(
            provider_id=provider_contract.provider_id,
            request=request,
            execution=cast(AgentExecution[TaskResult], execution),
            failure=FailureExpectation(
                AgentFailureCategory.CAPABILITY_OR_CONFIGURATION_FAILURE,
                InvocationStart.NOT_STARTED,
            ),
        )
        assert probe.observation.invocation_attempts == 0
        assert probe.observation.invocation_starts == 0

    @pytest.mark.parametrize("outcome", tuple(FAILURE_EXPECTATIONS))
    def test_failure_mapping_and_invocation_start_evidence_are_deterministic(
        self,
        provider_contract: ProviderContractFixture,
        outcome: ContractOutcome,
    ) -> None:
        task_kind = _representative_task(provider_contract)
        probe = provider_contract.make_probe(outcome, task_kind)
        request = _request(provider_contract, task_kind)
        observed_starts = 0

        def observe_start() -> None:
            nonlocal observed_starts
            observed_starts += 1

        execution = probe.executor.execute(
            request,
            on_invocation_start=observe_start,
        )
        expected = FAILURE_EXPECTATIONS[outcome]

        assert_execution_conforms(
            provider_id=provider_contract.provider_id,
            request=request,
            execution=cast(AgentExecution[TaskResult], execution),
            failure=expected,
        )
        expected_starts = int(expected.invocation_start is InvocationStart.STARTED)
        assert observed_starts == expected_starts
        assert probe.observation.invocation_starts == expected_starts
        evidence = decode_execution_evidence(
            encode_execution_evidence(evidence_from_execution(request, execution)),
            layout=request.artifact_layout,
        )
        assert evidence.status is AgentExecutionStatus.FAILED
        assert evidence.failure_category is expected.category
        assert evidence.invocation_start is expected.invocation_start

    def test_execution_round_trips_through_generic_evidence_without_native_knowledge(
        self,
        provider_contract: ProviderContractFixture,
    ) -> None:
        task_kind = _representative_task(provider_contract)
        probe = provider_contract.make_probe(ContractOutcome.SUCCESS, task_kind)
        request = _request(provider_contract, task_kind)
        execution = cast(AgentExecution[TaskResult], probe.executor.execute(request))
        assert_execution_conforms(
            provider_id=provider_contract.provider_id,
            request=request,
            execution=execution,
        )

        evidence = evidence_from_execution(request, execution)
        decoded = decode_execution_evidence(
            encode_execution_evidence(evidence),
            layout=request.artifact_layout,
        )
        assert decoded.provider_id == provider_contract.provider_id
        assert decoded.task_kind is task_kind
        assert decoded.status is AgentExecutionStatus.SUCCESS
        assert decoded.invocation_start is InvocationStart.STARTED
        assert decoded.typed_result == execution.result
        assert decoded.typed_result_artifact is not None

        assert request.artifact_layout is not None
        reference = write_execution_evidence(request.artifact_layout, evidence)
        persisted = read_execution_evidence(request.artifact_layout)
        assert reference.role is ArtifactRole.EXECUTION_EVIDENCE
        assert persisted == decoded

    def test_optional_native_artifacts_and_opaque_metadata_do_not_drive_decisions(
        self,
        provider_contract: ProviderContractFixture,
    ) -> None:
        task_kind = _representative_task(provider_contract)
        probe = provider_contract.make_probe(ContractOutcome.SUCCESS, task_kind)
        request = _request(provider_contract, task_kind)
        execution = cast(AgentExecution[TaskResult], probe.executor.execute(request))
        typed = tuple(
            reference
            for reference in execution.artifacts
            if reference.role is ArtifactRole.TYPED_RESULT
        )
        minimal = replace(
            execution,
            artifacts=typed,
            provider_metadata={
                "opaque": {"provider-private": ("ignored", 17, True)},
            },
        )

        evidence = evidence_from_execution(request, minimal)
        decoded = decode_execution_evidence(encode_execution_evidence(evidence))

        assert minimal.successful is execution.successful
        assert minimal.task_kind is execution.task_kind
        assert minimal.result == execution.result
        assert decoded.status is execution.status
        assert decoded.failure_category is execution.failure_category
        assert decoded.artifacts == ()
        assert decoded.provider_metadata["opaque"] == {
            "provider-private": ("ignored", 17, True)
        }

    def test_invocation_start_observer_interruption_propagates(
        self,
        provider_contract: ProviderContractFixture,
    ) -> None:
        task_kind = _representative_task(provider_contract)
        probe = provider_contract.make_probe(ContractOutcome.SUCCESS, task_kind)
        request = _request(provider_contract, task_kind)

        def interrupt() -> None:
            raise ContractInterruption("stop after invocation start")

        with pytest.raises(ContractInterruption, match="stop after invocation start"):
            probe.executor.execute(request, on_invocation_start=interrupt)

        assert probe.observation.invocation_attempts == 1
        assert probe.observation.invocation_starts == 1


__all__ = [
    "FAILURE_EXPECTATIONS",
    "AdapterProbe",
    "ContractOutcome",
    "FailureExpectation",
    "ProviderContractFixture",
    "ProviderContractTests",
    "TransportObservation",
    "assert_execution_conforms",
    "assert_provider_declaration",
    "supported_task_kinds",
]
