from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from tests.provider_contract import (
    AdapterProbe,
    FailureExpectation,
    ProviderContractFixture,
    assert_execution_conforms,
    assert_provider_declaration,
)
from ticket_automation.application.agent_execution import (
    IMPLEMENTATION_RESULT_CONTRACT,
    AgentCapability,
    AgentExecution,
    AgentExecutionPolicy,
    AgentExecutionRequest,
    AgentExecutionStatus,
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
from ticket_automation.domain.task_results import (
    ImplementationResult,
    ImplementationStatus,
    ReviewResult,
    ReviewVerdict,
    TaskResult,
)

_PROVIDER_ID = ProviderId("non-conforming-fake")
_CAPABILITIES = frozenset(AgentCapability)


@dataclass
class FakeRegistration:
    provider_id: ProviderId = _PROVIDER_ID
    capabilities: object = _CAPABILITIES
    policy_version: str = "fake-v1"


@dataclass
class NonConformingExecutor:
    output: AgentExecution[TaskResult]
    capabilities: object = _CAPABILITIES

    def execute(self, request, *, on_invocation_start=None):
        del request, on_invocation_start
        return self.output


def _implementation_result() -> ImplementationResult:
    return ImplementationResult(
        status=ImplementationStatus.COMPLETED,
        summary="Completed by the non-conforming fake.",
        tests_run=(),
        assumptions=(),
        known_issues=(),
    )


def _request(tmp_path: Path) -> AgentExecutionRequest[TaskResult]:
    run_root = tmp_path / "run"
    attempt = run_root / "attempts" / "001-implementation"
    return AgentExecutionRequest(
        task_kind=AgentTaskKind.IMPLEMENTATION,
        repository_path=tmp_path / "repository",
        repository_access=RepositoryAccess.WORKSPACE_WRITE,
        prompt="Provider-neutral contract prompt.",
        result_contract=IMPLEMENTATION_RESULT_CONTRACT,
        artifact_directory=attempt,
        artifact_layout=AttemptArtifactLayout(run_root, attempt),
        policy=AgentExecutionPolicy(2, NetworkAccess.DENIED),
        required_capabilities=required_execution_capabilities(
            RepositoryAccess.WORKSPACE_WRITE
        ),
    )


def _successful_execution(
    request: AgentExecutionRequest[TaskResult],
    *,
    provider_id: ProviderId = _PROVIDER_ID,
    write_artifact: bool = True,
) -> AgentExecution[TaskResult]:
    assert request.artifact_layout is not None
    result_path = request.artifact_layout.path("typed-output.json")
    if write_artifact:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("{}\n", encoding="utf-8")
    reference = request.artifact_layout.reference(
        ArtifactRole.TYPED_RESULT,
        result_path,
        "application/json",
    )
    now = datetime(2026, 9, 24, 12, tzinfo=UTC)
    return AgentExecution(
        provider_id=provider_id,
        task_kind=request.task_kind,
        status=AgentExecutionStatus.SUCCESS,
        invocation_start=InvocationStart.STARTED,
        started_at=now,
        ended_at=now,
        duration_seconds=0,
        result=_implementation_result(),
        artifacts=(reference,),
    )


def _contract_fixture(tmp_path: Path, registration: Any) -> ProviderContractFixture:
    def unused_probe(outcome, task_kind) -> AdapterProbe:
        del outcome, task_kind
        raise AssertionError("declaration proof must not create an executor")

    return ProviderContractFixture(
        provider_id=_PROVIDER_ID,
        declared_capabilities=_CAPABILITIES,
        registration=registration,
        valid_settings={},
        invalid_settings={},
        configuration_directory=tmp_path,
        repository_path=tmp_path,
        artifact_root=tmp_path,
        make_probe=unused_probe,
        make_capability_limited_probe=lambda task_kind: unused_probe(None, task_kind),
        expected_transport_prompt=lambda prompt: prompt,
    )


def test_harness_rejects_non_conforming_identity(tmp_path: Path) -> None:
    request = _request(tmp_path)
    fake = NonConformingExecutor(
        _successful_execution(request, provider_id=ProviderId("unstable-identity"))
    )

    with pytest.raises(AssertionError, match="provider identity changed"):
        assert_execution_conforms(
            provider_id=_PROVIDER_ID,
            request=request,
            execution=fake.execute(request),
        )


def test_harness_rejects_non_conforming_capability_declaration(
    tmp_path: Path,
) -> None:
    registration = FakeRegistration(capabilities=set(_CAPABILITIES))

    with pytest.raises(AssertionError, match="capabilities must be immutable"):
        assert_provider_declaration(_contract_fixture(tmp_path, registration))


def test_harness_rejects_non_conforming_result_type(tmp_path: Path) -> None:
    request = _request(tmp_path)
    execution = _successful_execution(request)
    object.__setattr__(
        execution,
        "result",
        ReviewResult(
            verdict=ReviewVerdict.PASS,
            summary="Wrong result type.",
            findings=(),
        ),
    )
    fake = NonConformingExecutor(execution)

    with pytest.raises(AssertionError, match="typed result contract"):
        assert_execution_conforms(
            provider_id=_PROVIDER_ID,
            request=request,
            execution=fake.execute(request),
        )


def test_harness_rejects_non_conforming_failure_category(tmp_path: Path) -> None:
    request = _request(tmp_path)
    now = datetime(2026, 9, 24, 12, tzinfo=UTC)
    execution: AgentExecution[TaskResult] = AgentExecution(
        provider_id=_PROVIDER_ID,
        task_kind=request.task_kind,
        status=AgentExecutionStatus.FAILED,
        invocation_start=InvocationStart.STARTED,
        started_at=now,
        ended_at=now,
        duration_seconds=0,
        failure_category=AgentFailureCategory.NON_SUCCESSFUL_EXECUTION,
        failure_message="Wrong neutral mapping.",
    )
    fake = NonConformingExecutor(execution)

    with pytest.raises(AssertionError, match="wrong neutral category"):
        assert_execution_conforms(
            provider_id=_PROVIDER_ID,
            request=request,
            execution=fake.execute(request),
            failure=FailureExpectation(
                AgentFailureCategory.TIMEOUT,
                InvocationStart.STARTED,
            ),
        )


def test_harness_rejects_non_conforming_artifact_reference(tmp_path: Path) -> None:
    request = _request(tmp_path)
    fake = NonConformingExecutor(_successful_execution(request, write_artifact=False))

    with pytest.raises(ValueError, match="does not exist"):
        assert_execution_conforms(
            provider_id=_PROVIDER_ID,
            request=request,
            execution=cast(AgentExecution[TaskResult], fake.execute(request)),
        )
