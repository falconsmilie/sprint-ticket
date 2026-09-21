from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.fake_agent_executor import InMemoryAgentExecutor
from ticket_automation.application.agent_execution import (
    CORRECTION_RESULT_CONTRACT,
    IMPLEMENTATION_RESULT_CONTRACT,
    REVIEW_RESULT_CONTRACT,
    AgentCapability,
    AgentContractError,
    AgentExecution,
    AgentExecutionPolicy,
    AgentExecutionRequest,
    AgentExecutionStatus,
    AgentFailureCategory,
    AgentResultContract,
    AgentTaskKind,
    ArtifactReference,
    InvocationStart,
    NetworkAccess,
    ProviderId,
    RepositoryAccess,
)
from ticket_automation.domain.task_results import (
    ImplementationResult,
    ImplementationStatus,
    ReviewResult,
    ReviewVerdict,
    TaskResult,
)

_READ_CAPABILITIES = frozenset(
    {
        AgentCapability.DIAGNOSTIC_ARTIFACT_CAPTURE,
        AgentCapability.ISOLATED_INVOCATION,
        AgentCapability.READ_ONLY_EXECUTION,
        AgentCapability.NETWORK_POLICY_CONTROL,
        AgentCapability.STRUCTURED_RESULT,
    }
)
_WRITE_CAPABILITIES = frozenset(
    {
        AgentCapability.DIAGNOSTIC_ARTIFACT_CAPTURE,
        AgentCapability.ISOLATED_INVOCATION,
        AgentCapability.WORKSPACE_WRITE_EXECUTION,
        AgentCapability.NETWORK_POLICY_CONTROL,
        AgentCapability.STRUCTURED_RESULT,
    }
)


def _implementation_result(summary: str = "Implemented") -> ImplementationResult:
    return ImplementationResult(
        status=ImplementationStatus.COMPLETED,
        summary=summary,
        tests_run=(),
        assumptions=(),
        known_issues=(),
    )


def _review_result() -> ReviewResult:
    return ReviewResult(verdict=ReviewVerdict.PASS, summary="Approved", findings=())


def _request(
    task_kind: AgentTaskKind,
    *,
    artifact_directory: Path | None = None,
) -> AgentExecutionRequest[TaskResult]:
    if task_kind is AgentTaskKind.REVIEW:
        access = RepositoryAccess.READ_ONLY
        contract = REVIEW_RESULT_CONTRACT
        capabilities = _READ_CAPABILITIES
    elif task_kind is AgentTaskKind.IMPLEMENTATION:
        access = RepositoryAccess.WORKSPACE_WRITE
        contract = IMPLEMENTATION_RESULT_CONTRACT
        capabilities = _WRITE_CAPABILITIES
    else:
        access = RepositoryAccess.WORKSPACE_WRITE
        contract = CORRECTION_RESULT_CONTRACT
        capabilities = _WRITE_CAPABILITIES
    return AgentExecutionRequest(
        task_kind=task_kind,
        repository_path=Path("repository"),
        repository_access=access,
        prompt="Perform the requested work.",
        result_contract=contract,
        artifact_directory=(artifact_directory or Path("artifacts") / task_kind.value),
        policy=AgentExecutionPolicy(
            timeout_seconds=60,
            network_access=NetworkAccess.DENIED,
        ),
        required_capabilities=capabilities,
    )


def test_request_is_immutable_and_normalizes_capabilities():
    request = _request(AgentTaskKind.IMPLEMENTATION)

    assert request.required_capabilities == _WRITE_CAPABILITIES
    with pytest.raises(FrozenInstanceError):
        request.prompt = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "timeout_seconds", [0, -1, float("nan"), float("inf"), True, "60"]
)
def test_execution_policy_rejects_invalid_timeout(timeout_seconds: object):
    with pytest.raises(AgentContractError):
        AgentExecutionPolicy(  # type: ignore[arg-type]
            timeout_seconds, NetworkAccess.DENIED
        )


@pytest.mark.parametrize("timeout_seconds", [None, 0.5, 60])
def test_execution_policy_accepts_trusted_timeout(timeout_seconds: float | None):
    policy = AgentExecutionPolicy(timeout_seconds, NetworkAccess.ALLOWED)

    assert policy.timeout_seconds == timeout_seconds
    if timeout_seconds is not None:
        assert type(policy.timeout_seconds) is float


@pytest.mark.parametrize("value", ["", " ", " provider", "provider "])
def test_provider_id_rejects_ambiguous_values(value: str):
    with pytest.raises(AgentContractError):
        ProviderId(value)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"prompt": " "}, "prompt must be a non-empty string"),
        ({"task_kind": "implementation"}, "task_kind must be"),
        ({"repository_path": "repository"}, "repository_path must be"),
        ({"repository_access": "workspace-write"}, "repository_access must be"),
        ({"artifact_directory": "artifacts"}, "artifact_directory must be"),
        ({"policy": "default"}, "policy must be"),
        (
            {"required_capabilities": frozenset({"structured-result"})},
            "only AgentCapability values",
        ),
        (
            {
                "required_capabilities": frozenset(
                    {
                        AgentCapability.NETWORK_POLICY_CONTROL,
                        AgentCapability.STRUCTURED_RESULT,
                    }
                )
            },
            "workspace-write-execution is required",
        ),
        (
            {
                "required_capabilities": frozenset(
                    {
                        AgentCapability.NETWORK_POLICY_CONTROL,
                        AgentCapability.WORKSPACE_WRITE_EXECUTION,
                    }
                )
            },
            "structured-result is required",
        ),
    ],
)
def test_request_invariants(changes: dict[str, object], message: str):
    values: dict[str, object] = {
        "task_kind": AgentTaskKind.IMPLEMENTATION,
        "repository_path": Path("repository"),
        "repository_access": RepositoryAccess.WORKSPACE_WRITE,
        "prompt": "Implement",
        "result_contract": IMPLEMENTATION_RESULT_CONTRACT,
        "artifact_directory": Path("artifacts"),
        "policy": AgentExecutionPolicy(60, NetworkAccess.DENIED),
        "required_capabilities": _WRITE_CAPABILITIES,
    }
    values.update(changes)

    with pytest.raises(AgentContractError, match=message):
        AgentExecutionRequest(**values)  # type: ignore[arg-type]


def test_request_rejects_result_contract_for_a_different_task():
    with pytest.raises(AgentContractError, match="task kind must match"):
        AgentExecutionRequest(
            task_kind=AgentTaskKind.IMPLEMENTATION,
            repository_path=Path("repository"),
            repository_access=RepositoryAccess.WORKSPACE_WRITE,
            prompt="Implement",
            result_contract=REVIEW_RESULT_CONTRACT,
            artifact_directory=Path("artifacts"),
            policy=AgentExecutionPolicy(60, NetworkAccess.DENIED),
            required_capabilities=_WRITE_CAPABILITIES,
        )


def test_result_contract_rejects_wrong_domain_type_for_task():
    with pytest.raises(AgentContractError, match="review tasks require ReviewResult"):
        AgentResultContract(AgentTaskKind.REVIEW, ImplementationResult)


@pytest.mark.parametrize(
    "changes",
    [
        {"name": ""},
        {"path": "events.log"},
        {"media_type": ""},
    ],
)
def test_artifact_reference_rejects_invalid_values(changes: dict[str, object]):
    values: dict[str, object] = {
        "name": "diagnostics",
        "path": Path("events.log"),
        "media_type": "text/plain",
    }
    values.update(changes)

    with pytest.raises(AgentContractError):
        ArtifactReference(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("missing", sorted(_WRITE_CAPABILITIES, key=str))
def test_capability_mismatch_is_explicit_and_precedes_fake_invocation(
    missing: AgentCapability,
    tmp_path: Path,
):
    request = _request(
        AgentTaskKind.IMPLEMENTATION,
        artifact_directory=tmp_path / "artifacts",
    )
    available = _WRITE_CAPABILITIES - {missing}
    executor = InMemoryAgentExecutor(
        {AgentTaskKind.IMPLEMENTATION: _implementation_result()},
        capabilities=available,
    )

    assert request.missing_capabilities(available) == frozenset({missing})
    execution = executor.execute(request)
    assert execution.status is AgentExecutionStatus.FAILED
    assert execution.failure_category is (
        AgentFailureCategory.CAPABILITY_OR_CONFIGURATION_FAILURE
    )
    assert execution.invocation_start is InvocationStart.NOT_STARTED
    assert execution.invocation_started is False
    assert executor.requests == []


@pytest.mark.parametrize(
    ("task_kind", "expected_type"),
    [
        (AgentTaskKind.IMPLEMENTATION, ImplementationResult),
        (AgentTaskKind.REVIEW, ReviewResult),
        (AgentTaskKind.CORRECTION, ImplementationResult),
    ],
)
def test_in_memory_executor_demonstrates_each_task_kind(
    task_kind: AgentTaskKind, expected_type: type[TaskResult], tmp_path: Path
):
    results: dict[AgentTaskKind, TaskResult] = {
        AgentTaskKind.IMPLEMENTATION: _implementation_result(),
        AgentTaskKind.REVIEW: _review_result(),
        AgentTaskKind.CORRECTION: _implementation_result("Corrected"),
    }
    executor = InMemoryAgentExecutor(
        results,
        capabilities=_READ_CAPABILITIES | _WRITE_CAPABILITIES,
    )

    execution = executor.execute(
        _request(task_kind, artifact_directory=tmp_path / task_kind.value)
    )

    assert execution.successful
    assert type(execution.result) is expected_type
    assert execution.task_kind is task_kind
    assert execution.invocation_started is True


def test_in_memory_executor_writes_every_referenced_artifact(tmp_path: Path):
    request = _request(
        AgentTaskKind.IMPLEMENTATION,
        artifact_directory=tmp_path / "artifacts",
    )
    executor = InMemoryAgentExecutor(
        {AgentTaskKind.IMPLEMENTATION: _implementation_result()},
        capabilities=_WRITE_CAPABILITIES,
    )

    execution = executor.execute(request)

    assert execution.artifacts
    assert all(artifact.path.is_file() for artifact in execution.artifacts)
    assert {artifact.name for artifact in execution.artifacts} == {
        "request-copy",
        "in-memory-log",
        "in-memory-details",
        "typed-output",
    }


def test_success_rejects_result_task_mismatch():
    now = datetime.now(UTC)

    with pytest.raises(AgentContractError, match="does not match its task kind"):
        AgentExecution(
            provider_id=ProviderId("test"),
            task_kind=AgentTaskKind.REVIEW,
            status=AgentExecutionStatus.SUCCESS,
            invocation_start=InvocationStart.STARTED,
            started_at=now,
            ended_at=now,
            duration_seconds=0,
            result=_implementation_result(),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"result": _implementation_result()},
        {"failure_category": None},
        {"failure_message": None},
    ],
)
def test_failed_execution_requires_consistent_failure_state(changes: dict[str, object]):
    now = datetime.now(UTC)
    values: dict[str, object] = {
        "provider_id": ProviderId("test"),
        "task_kind": AgentTaskKind.IMPLEMENTATION,
        "status": AgentExecutionStatus.FAILED,
        "invocation_start": InvocationStart.STARTED,
        "started_at": now,
        "ended_at": now + timedelta(seconds=1),
        "duration_seconds": 1,
        "failure_category": AgentFailureCategory.NON_SUCCESSFUL_EXECUTION,
        "failure_message": "Execution failed.",
    }
    values.update(changes)

    with pytest.raises(AgentContractError):
        AgentExecution(**values)  # type: ignore[arg-type]


_PRE_INVOCATION_FAILURES = (
    AgentFailureCategory.PROVIDER_UNAVAILABLE,
    AgentFailureCategory.INVOCATION_START_FAILURE,
    AgentFailureCategory.CAPABILITY_OR_CONFIGURATION_FAILURE,
)
_POST_INVOCATION_FAILURES = tuple(
    category
    for category in AgentFailureCategory
    if category not in _PRE_INVOCATION_FAILURES
)


@pytest.mark.parametrize("category", _PRE_INVOCATION_FAILURES)
def test_pre_invocation_failure_requires_not_started_state(
    category: AgentFailureCategory,
):
    now = datetime.now(UTC)

    with pytest.raises(AgentContractError, match="before invocation start"):
        AgentExecution(
            provider_id=ProviderId("test"),
            task_kind=AgentTaskKind.REVIEW,
            status=AgentExecutionStatus.FAILED,
            invocation_start=InvocationStart.STARTED,
            started_at=now,
            ended_at=now,
            duration_seconds=0,
            failure_category=category,
            failure_message="Unsupported policy.",
        )


@pytest.mark.parametrize("category", _POST_INVOCATION_FAILURES)
def test_post_invocation_failure_rejects_not_started_state(
    category: AgentFailureCategory,
):
    now = datetime.now(UTC)

    with pytest.raises(AgentContractError, match="started or uncertain invocation"):
        AgentExecution(
            provider_id=ProviderId("test"),
            task_kind=AgentTaskKind.REVIEW,
            status=AgentExecutionStatus.FAILED,
            invocation_start=InvocationStart.NOT_STARTED,
            started_at=now,
            ended_at=now,
            duration_seconds=0,
            failure_category=category,
            failure_message="Provider did not complete the request.",
        )


@pytest.mark.parametrize(
    ("category", "invocation_start"),
    [
        *(
            (category, InvocationStart.NOT_STARTED)
            for category in _PRE_INVOCATION_FAILURES
        ),
        *(
            (category, InvocationStart.STARTED)
            for category in _POST_INVOCATION_FAILURES
        ),
        (AgentFailureCategory.TIMEOUT, InvocationStart.UNKNOWN),
    ],
)
def test_failure_categories_accept_trusted_start_states(
    category: AgentFailureCategory, invocation_start: InvocationStart
):
    now = datetime.now(UTC)

    execution = AgentExecution(
        provider_id=ProviderId("test"),
        task_kind=AgentTaskKind.REVIEW,
        status=AgentExecutionStatus.FAILED,
        invocation_start=invocation_start,
        started_at=now,
        ended_at=now,
        duration_seconds=0,
        failure_category=category,
        failure_message="Provider did not complete the request.",
    )

    assert not execution.successful
    assert execution.failure_category is category


@pytest.mark.parametrize(
    "changes",
    [
        {"started_at": datetime(2026, 1, 1)},  # noqa: DTZ001 - intentionally naive
        {"duration_seconds": float("nan")},
        {"duration_seconds": float("inf")},
        {"duration_seconds": 1},
    ],
)
def test_execution_rejects_invalid_timing(changes: dict[str, object]):
    now = datetime.now(UTC)
    values: dict[str, object] = {
        "provider_id": ProviderId("test"),
        "task_kind": AgentTaskKind.REVIEW,
        "status": AgentExecutionStatus.SUCCESS,
        "invocation_start": InvocationStart.STARTED,
        "started_at": now,
        "ended_at": now,
        "duration_seconds": 0,
        "result": _review_result(),
    }
    values.update(changes)

    with pytest.raises(AgentContractError):
        AgentExecution(**values)  # type: ignore[arg-type]


def test_artifacts_and_provider_metadata_are_generic_and_immutable():
    now = datetime.now(UTC)
    native_metadata = {"exit_code": 0}
    execution = AgentExecution(
        provider_id=ProviderId("test"),
        task_kind=AgentTaskKind.REVIEW,
        status=AgentExecutionStatus.SUCCESS,
        invocation_start=InvocationStart.STARTED,
        started_at=now,
        ended_at=now,
        duration_seconds=0,
        result=_review_result(),
        artifacts=(ArtifactReference("diagnostics", Path("events.log"), "text/plain"),),
        provider_metadata={"native": native_metadata, "arguments": ("run",)},
    )

    native_metadata["exit_code"] = 1
    assert execution.artifacts[0].name == "diagnostics"
    assert execution.provider_metadata["arguments"] == ("run",)
    assert execution.provider_metadata["native"] == {"exit_code": 0}
    with pytest.raises(TypeError):
        execution.provider_metadata["new"] = "value"  # type: ignore[index]


@pytest.mark.parametrize("mutable_value", [["run"], {"run"}, bytearray(b"run")])
def test_provider_metadata_rejects_mutable_values(mutable_value: object):
    now = datetime.now(UTC)

    with pytest.raises(AgentContractError, match="immutable JSON-compatible"):
        AgentExecution(
            provider_id=ProviderId("test"),
            task_kind=AgentTaskKind.REVIEW,
            status=AgentExecutionStatus.SUCCESS,
            invocation_start=InvocationStart.STARTED,
            started_at=now,
            ended_at=now,
            duration_seconds=0,
            result=_review_result(),
            provider_metadata={"mutable": mutable_value},  # type: ignore[dict-item]
        )
