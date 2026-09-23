from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

import ticket_automation.application.guarded_writable_operation as guarded_module
from tests.fake_agent_executor import InMemoryAgentExecutor
from tests.helpers import GIT, create_git_repo, run_git
from ticket_automation.application.agent_execution import (
    CORRECTION_RESULT_CONTRACT,
    IMPLEMENTATION_RESULT_CONTRACT,
    REVIEW_RESULT_CONTRACT,
    AgentCapability,
    AgentExecution,
    AgentExecutionPolicy,
    AgentExecutionRequest,
    AgentExecutionStatus,
    AgentFailureCategory,
    AgentTaskKind,
    InvocationStart,
    NetworkAccess,
    ProviderId,
    RepositoryAccess,
    required_execution_capabilities,
)
from ticket_automation.application.guarded_writable_operation import (
    GuardedWritableOperation,
    GuardedWritableRejectionRequest,
    GuardedWritableRequest,
    WritableAudit,
    WritableBaseline,
    WritableFailedUncertain,
    WritableFailedUnchanged,
    WritableRejectedBeforeStart,
    WritableSafetyStopped,
    WritableSafetyViolation,
    WritableSucceeded,
)
from ticket_automation.attempts import (
    finish_phase_attempt,
    load_attempt_records,
    start_attempt,
)
from ticket_automation.domain.task_results import (
    ImplementationResult,
    ImplementationStatus,
)
from ticket_automation.git import GitRepository
from ticket_automation.git_safety import WorkspaceSnapshot
from ticket_automation.models import AttemptPhase, StageOutcome

_PROVIDER_ID = ProviderId("guard-matrix")
_NOW = datetime(2026, 9, 22, 10, 0, tzinfo=UTC)


class MatrixExecutor:
    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.calls = 0

    @property
    def capabilities(self) -> frozenset[AgentCapability]:
        return required_execution_capabilities(RepositoryAccess.WORKSPACE_WRITE)

    def execute(self, request, *, on_invocation_start=None):
        self.calls += 1
        if self.scenario == "exception-before-start":
            raise RuntimeError("unexpected executor failure")
        if self.scenario == "timeout-before-start":
            return _failure(
                request.task_kind,
                InvocationStart.NOT_STARTED,
                AgentFailureCategory.TIMEOUT,
            )
        if self.scenario == "not-started":
            return _failure(
                request.task_kind,
                InvocationStart.NOT_STARTED,
                AgentFailureCategory.INVOCATION_START_FAILURE,
            )

        if on_invocation_start is not None:
            on_invocation_start()
        if self.scenario == "exception-after-start":
            raise RuntimeError("unexpected executor failure")
        if self.scenario == "interrupt-after-start":
            raise KeyboardInterrupt("interrupted after invocation start")
        _mutate(request.repository_path, self.scenario)
        if self.scenario == "timeout-after-start":
            return _failure(
                request.task_kind,
                InvocationStart.STARTED,
                AgentFailureCategory.TIMEOUT,
            )
        if self.scenario == "invalid-result":
            return _failure(
                request.task_kind,
                InvocationStart.STARTED,
                AgentFailureCategory.INVALID_RESULT,
            )
        if self.scenario in {
            "nonzero-unchanged",
            "changed-tracked-failure",
            "changed-untracked-failure",
        }:
            return _failure(
                request.task_kind,
                InvocationStart.STARTED,
                AgentFailureCategory.NON_SUCCESSFUL_EXECUTION,
            )
        return AgentExecution(
            provider_id=_PROVIDER_ID,
            task_kind=request.task_kind,
            status=AgentExecutionStatus.SUCCESS,
            invocation_start=InvocationStart.STARTED,
            started_at=_NOW,
            ended_at=_NOW,
            duration_seconds=0,
            result=ImplementationResult(
                ImplementationStatus.COMPLETED,
                "completed",
                (),
                (),
                (),
            ),
        )


@pytest.mark.skipif(GIT is None, reason="git executable is required")
@pytest.mark.parametrize(
    ("scenario", "expected_type"),
    [
        ("not-started", WritableRejectedBeforeStart),
        ("timeout-before-start", WritableRejectedBeforeStart),
        ("timeout-after-start", WritableFailedUncertain),
        ("nonzero-unchanged", WritableFailedUnchanged),
        ("invalid-result", WritableFailedUncertain),
        ("exception-before-start", WritableFailedUncertain),
        ("exception-after-start", WritableFailedUncertain),
        ("interrupt-after-start", WritableFailedUncertain),
        ("changed-tracked-failure", WritableFailedUncertain),
        ("changed-untracked-failure", WritableFailedUncertain),
        ("tracked-success", WritableSucceeded),
        ("untracked-success", WritableSucceeded),
        ("staging-success", WritableSafetyStopped),
        ("branch-success", WritableSafetyStopped),
        ("head-success", WritableSafetyStopped),
        ("environment-success", WritableSafetyStopped),
    ],
)
@pytest.mark.parametrize(
    ("task_kind", "phase"),
    [
        (AgentTaskKind.IMPLEMENTATION, AttemptPhase.IMPLEMENTING),
        (AgentTaskKind.CORRECTION, AttemptPhase.CORRECTING),
    ],
)
def test_implementation_and_correction_share_the_writable_safety_matrix(
    tmp_path: Path,
    scenario: str,
    expected_type: type,
    task_kind: AgentTaskKind,
    phase: AttemptPhase,
) -> None:
    repository_path = create_git_repo(tmp_path / "target")
    repository = GitRepository(repository_path)
    before = WorkspaceSnapshot.capture(repository)
    request = _request(tmp_path, repository, before, task_kind, phase)
    executor = MatrixExecutor(scenario)

    outcome = GuardedWritableOperation(executor).execute(request)

    assert isinstance(outcome, expected_type)
    assert executor.calls == 1
    assert outcome.audit.before_workspace is not None
    assert outcome.audit.workspace_guard.artifact_path is not None
    assert outcome.audit.workspace_guard.artifact_path.is_file()
    if scenario in {"not-started", "timeout-before-start"}:
        assert outcome.audit.invocation_start is InvocationStart.NOT_STARTED
    if scenario in {
        "timeout-after-start",
        "changed-tracked-failure",
        "interrupt-after-start",
    }:
        assert outcome.audit.invocation_start is InvocationStart.STARTED
    if scenario == "exception-before-start":
        assert outcome.audit.invocation_start is InvocationStart.UNKNOWN
    if scenario == "exception-after-start":
        assert outcome.audit.invocation_start is InvocationStart.STARTED
    if scenario in {"changed-tracked-failure", "changed-untracked-failure"}:
        assert any(
            violation.name == "worktree"
            for violation in outcome.audit.safety_violations
        )


@pytest.mark.skipif(GIT is None, reason="git executable is required")
@pytest.mark.parametrize(
    ("task_kind", "phase"),
    [
        (AgentTaskKind.IMPLEMENTATION, AttemptPhase.IMPLEMENTING),
        (AgentTaskKind.CORRECTION, AttemptPhase.CORRECTING),
    ],
)
def test_post_snapshot_failure_stops_both_writable_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_kind: AgentTaskKind,
    phase: AttemptPhase,
) -> None:
    repository_path = create_git_repo(tmp_path / "target")
    repository = GitRepository(repository_path)
    before = WorkspaceSnapshot.capture(repository)
    request = _request(tmp_path, repository, before, task_kind, phase)
    invocation_finished = False
    original_capture = WorkspaceSnapshot.capture

    class PostSnapshotFailureExecutor(MatrixExecutor):
        def execute(self, execution_request, *, on_invocation_start=None):
            nonlocal invocation_finished
            result = super().execute(
                execution_request, on_invocation_start=on_invocation_start
            )
            invocation_finished = True
            return result

    def capture(_cls, target):
        if invocation_finished:
            raise OSError("snapshot unavailable")
        return original_capture(target)

    monkeypatch.setattr(WorkspaceSnapshot, "capture", classmethod(capture))

    outcome = GuardedWritableOperation(
        PostSnapshotFailureExecutor("tracked-success")
    ).execute(request)

    assert isinstance(outcome, WritableSafetyStopped)
    assert outcome.audit.after_workspace is None
    assert any(
        violation.name == "workspace-inspection"
        for violation in outcome.audit.safety_violations
    )


@pytest.mark.skipif(GIT is None, reason="git executable is required")
@pytest.mark.parametrize(
    ("task_kind", "phase"),
    [
        (AgentTaskKind.IMPLEMENTATION, AttemptPhase.IMPLEMENTING),
        (AgentTaskKind.CORRECTION, AttemptPhase.CORRECTING),
    ],
)
def test_pre_snapshot_failure_rejects_both_tasks_before_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_kind: AgentTaskKind,
    phase: AttemptPhase,
) -> None:
    repository_path = create_git_repo(tmp_path / "target")
    repository = GitRepository(repository_path)
    before = WorkspaceSnapshot.capture(repository)
    request = _request(tmp_path, repository, before, task_kind, phase)
    executor = MatrixExecutor("tracked-success")

    def fail_capture(_cls, target):
        del target
        raise OSError("snapshot unavailable")

    monkeypatch.setattr(WorkspaceSnapshot, "capture", classmethod(fail_capture))

    outcome = GuardedWritableOperation(executor).execute(request)

    assert isinstance(outcome, WritableSafetyStopped)
    assert outcome.audit.invocation_start is InvocationStart.NOT_STARTED
    assert executor.calls == 0


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_guard_rejects_non_workspace_write_requests_before_calling_executor(
    tmp_path: Path,
) -> None:
    repository_path = create_git_repo(tmp_path / "target")
    repository = GitRepository(repository_path)
    before = WorkspaceSnapshot.capture(repository)
    record = start_attempt(
        tmp_path / "run",
        phase=AttemptPhase.IMPLEMENTING,
        before_workspace_fingerprint=None,
    )
    read_only_request = AgentExecutionRequest(
        task_kind=AgentTaskKind.IMPLEMENTATION,
        repository_path=repository.path,
        repository_access=RepositoryAccess.READ_ONLY,
        prompt="Implement the ticket.",
        result_contract=IMPLEMENTATION_RESULT_CONTRACT,
        artifact_directory=record.artifact_directory,
        policy=AgentExecutionPolicy(60, NetworkAccess.ALLOWED),
        required_capabilities=required_execution_capabilities(
            RepositoryAccess.READ_ONLY
        ),
    )
    executor = MatrixExecutor("tracked-success")

    with pytest.raises(ValueError, match="workspace-write"):
        guarded_request = GuardedWritableRequest(
            phase=AttemptPhase.IMPLEMENTING,
            execution_request=read_only_request,
            baseline=WritableBaseline(
                repository.path,
                before.branch,
                before.head_sha or "",
                before.fingerprint,
                True,
            ),
        )
        GuardedWritableOperation(executor).execute(guarded_request)

    assert executor.calls == 0


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_service_requires_workspace_write_capability_from_executor(
    tmp_path: Path,
) -> None:
    repository = GitRepository(create_git_repo(tmp_path / "target"))
    before = WorkspaceSnapshot.capture(repository)
    request = _request(
        tmp_path,
        repository,
        before,
        AgentTaskKind.IMPLEMENTATION,
        AttemptPhase.IMPLEMENTING,
    )
    executor = InMemoryAgentExecutor(
        {},
        capabilities=frozenset(
            capability
            for capability in required_execution_capabilities(
                RepositoryAccess.WORKSPACE_WRITE
            )
            if capability is not AgentCapability.WORKSPACE_WRITE_EXECUTION
        ),
    )

    outcome = GuardedWritableOperation(executor).execute(request)

    assert isinstance(outcome, WritableRejectedBeforeStart)
    assert outcome.audit.execution is None
    assert executor.requests == []
    assert outcome.audit.failure_message is not None
    assert "lacks required writable capabilities" in outcome.audit.failure_message
    assert (
        AgentCapability.WORKSPACE_WRITE_EXECUTION
        in request.execution_request.required_capabilities
    )


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_recorded_start_cannot_be_downgraded_by_executor_result(tmp_path: Path) -> None:
    repository = GitRepository(create_git_repo(tmp_path / "target"))
    before = WorkspaceSnapshot.capture(repository)
    request = _request(
        tmp_path,
        repository,
        before,
        AgentTaskKind.IMPLEMENTATION,
        AttemptPhase.IMPLEMENTING,
    )

    class ContradictoryExecutor(MatrixExecutor):
        def execute(self, execution_request, *, on_invocation_start=None):
            if on_invocation_start is not None:
                on_invocation_start()
            return _failure(
                execution_request.task_kind,
                InvocationStart.NOT_STARTED,
                AgentFailureCategory.INVOCATION_START_FAILURE,
            )

    outcome = GuardedWritableOperation(ContradictoryExecutor("unused")).execute(request)

    assert isinstance(outcome, WritableFailedUncertain)
    assert outcome.audit.invocation_start is InvocationStart.STARTED
    assert any(
        violation.name == "writable-attempt-evidence"
        for violation in outcome.audit.safety_violations
    )
    [attempt] = load_attempt_records(tmp_path / "run")
    assert attempt.process_started is True


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_executor_result_must_match_requested_task(tmp_path: Path) -> None:
    repository = GitRepository(create_git_repo(tmp_path / "target"))
    before = WorkspaceSnapshot.capture(repository)
    request = _request(
        tmp_path,
        repository,
        before,
        AgentTaskKind.IMPLEMENTATION,
        AttemptPhase.IMPLEMENTING,
    )

    class WrongTaskExecutor(MatrixExecutor):
        def execute(self, execution_request, *, on_invocation_start=None):
            if on_invocation_start is not None:
                on_invocation_start()
            return AgentExecution(
                provider_id=_PROVIDER_ID,
                task_kind=AgentTaskKind.CORRECTION,
                status=AgentExecutionStatus.SUCCESS,
                invocation_start=InvocationStart.STARTED,
                started_at=_NOW,
                ended_at=_NOW,
                duration_seconds=0,
                result=ImplementationResult(
                    ImplementationStatus.COMPLETED,
                    "wrong task",
                    (),
                    (),
                    (),
                ),
            )

    outcome = GuardedWritableOperation(WrongTaskExecutor("unused")).execute(request)

    assert isinstance(outcome, WritableFailedUncertain)
    assert outcome.audit.failure_message is not None
    assert "wrong task kind" in outcome.audit.failure_message


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_guard_artifact_persistence_failure_stops_successful_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = GitRepository(create_git_repo(tmp_path / "target"))
    before = WorkspaceSnapshot.capture(repository)
    request = _request(
        tmp_path,
        repository,
        before,
        AgentTaskKind.IMPLEMENTATION,
        AttemptPhase.IMPLEMENTING,
    )

    def fail_persistence(_inspection):
        raise OSError("audit destination unavailable")

    monkeypatch.setattr(
        guarded_module, "write_workspace_guard_inspection", fail_persistence
    )

    outcome = GuardedWritableOperation(MatrixExecutor("tracked-success")).execute(
        request
    )

    assert isinstance(outcome, WritableSafetyStopped)
    assert outcome.audit.workspace_guard.has_inspection_failure
    assert outcome.audit.workspace_guard.artifact_path is None
    assert not (
        request.execution_request.artifact_directory / "workspace-guard.json"
    ).exists()
    persisted_attempt = load_attempt_records(tmp_path / "run")[-1]
    assert persisted_attempt.execution_path is None


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_pre_execution_attempt_evidence_failure_prevents_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = GitRepository(create_git_repo(tmp_path / "target"))
    before = WorkspaceSnapshot.capture(repository)
    request = _request(
        tmp_path,
        repository,
        before,
        AgentTaskKind.IMPLEMENTATION,
        AttemptPhase.IMPLEMENTING,
    )
    executor = MatrixExecutor("tracked-success")

    def fail_update(*args, **kwargs):
        del args, kwargs
        raise OSError("attempt ledger unavailable")

    monkeypatch.setattr(guarded_module, "update_attempt", fail_update)

    outcome = GuardedWritableOperation(executor).execute(request)

    assert isinstance(outcome, WritableSafetyStopped)
    assert executor.calls == 0
    assert outcome.audit.invocation_start is InvocationStart.NOT_STARTED
    assert any(
        violation.name == "writable-attempt-evidence"
        for violation in outcome.audit.safety_violations
    )
    persisted_guard = json.loads(
        (
            request.execution_request.artifact_directory / "workspace-guard.json"
        ).read_text(encoding="utf-8")
    )
    assert tuple(persisted_guard["inspection_errors_after"]) == (
        outcome.audit.workspace_guard.after.inspection_errors
    )


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_post_evidence_comparison_failure_returns_a_typed_safety_stop(
    tmp_path: Path,
) -> None:
    repository = GitRepository(create_git_repo(tmp_path / "target"))
    before = WorkspaceSnapshot.capture(repository)
    request = _request(
        tmp_path,
        repository,
        before,
        AgentTaskKind.IMPLEMENTATION,
        AttemptPhase.IMPLEMENTING,
    )
    executor = MatrixExecutor("tracked-success")

    def fail_clock() -> datetime:
        raise RuntimeError("audit clock unavailable")

    outcome = GuardedWritableOperation(executor, clock=fail_clock).execute(request)

    assert isinstance(outcome, WritableSafetyStopped)
    assert executor.calls == 1
    assert outcome.audit.workspace_guard.has_inspection_failure
    assert outcome.audit.workspace_guard.artifact_path is not None
    assert outcome.audit.workspace_guard.artifact_path.is_file()
    assert any(
        "workspace-guard-comparison" in error
        for error in outcome.audit.workspace_guard.after.inspection_errors
    )
    comparison_errors = tuple(
        error
        for error in outcome.audit.workspace_guard.after.inspection_errors
        if "workspace-guard-comparison" in error
    )
    assert len(comparison_errors) == 1
    persisted_guard = json.loads(
        outcome.audit.workspace_guard.artifact_path.read_text(encoding="utf-8")
    )
    assert tuple(persisted_guard["inspection_errors_after"]) == (
        outcome.audit.workspace_guard.after.inspection_errors
    )


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_interrupt_while_recording_start_returns_a_typed_uncertain_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = GitRepository(create_git_repo(tmp_path / "target"))
    before = WorkspaceSnapshot.capture(repository)
    request = _request(
        tmp_path,
        repository,
        before,
        AgentTaskKind.IMPLEMENTATION,
        AttemptPhase.IMPLEMENTING,
    )
    real_update = guarded_module.update_attempt

    def interrupt_start_update(record, **changes):
        if changes.get("process_started") is True:
            raise KeyboardInterrupt("interrupted while recording process start")
        return real_update(record, **changes)

    monkeypatch.setattr(guarded_module, "update_attempt", interrupt_start_update)

    outcome = GuardedWritableOperation(MatrixExecutor("tracked-success")).execute(
        request
    )

    assert isinstance(outcome, WritableFailedUncertain)
    assert outcome.audit.invocation_start is InvocationStart.STARTED
    assert outcome.audit.before_workspace == outcome.audit.after_workspace
    assert any(
        violation.name == "writable-attempt-evidence"
        for violation in outcome.audit.safety_violations
    )
    assert outcome.audit.workspace_guard.artifact_path is not None
    persisted_guard = json.loads(
        outcome.audit.workspace_guard.artifact_path.read_text(encoding="utf-8")
    )
    assert tuple(persisted_guard["inspection_errors_after"]) == (
        outcome.audit.workspace_guard.after.inspection_errors
    )


@pytest.mark.skipif(GIT is None, reason="git executable is required")
@pytest.mark.parametrize(
    ("field_name", "expected_executor_calls", "expected_start"),
    [
        (
            "before_workspace_fingerprint",
            0,
            InvocationStart.NOT_STARTED,
        ),
        ("after_workspace_fingerprint", 1, InvocationStart.STARTED),
        ("execution_path", 1, InvocationStart.STARTED),
    ],
)
@pytest.mark.parametrize("interruption_type", [KeyboardInterrupt, SystemExit])
def test_interrupt_during_non_start_attempt_update_returns_safety_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    expected_executor_calls: int,
    expected_start: InvocationStart,
    interruption_type: type[BaseException],
) -> None:
    repository = GitRepository(create_git_repo(tmp_path / "target"))
    before = WorkspaceSnapshot.capture(repository)
    request = _request(
        tmp_path,
        repository,
        before,
        AgentTaskKind.IMPLEMENTATION,
        AttemptPhase.IMPLEMENTING,
    )
    executor = MatrixExecutor("tracked-success")
    real_update = guarded_module.update_attempt
    interrupted = False

    def interrupt_evidence_update(record, **changes):
        nonlocal interrupted
        if not interrupted and field_name in changes:
            interrupted = True
            raise interruption_type(f"interrupted while recording {field_name}")
        return real_update(record, **changes)

    monkeypatch.setattr(guarded_module, "update_attempt", interrupt_evidence_update)

    outcome = GuardedWritableOperation(executor).execute(request)

    assert isinstance(outcome, WritableSafetyStopped)
    assert executor.calls == expected_executor_calls
    assert outcome.audit.invocation_start is expected_start
    assert any(
        violation.name == "writable-attempt-evidence"
        for violation in outcome.audit.safety_violations
    )
    assert outcome.audit.workspace_guard.artifact_path is not None
    persisted_guard = json.loads(
        outcome.audit.workspace_guard.artifact_path.read_text(encoding="utf-8")
    )
    assert any(
        interruption_type.__name__ in error
        for error in persisted_guard["inspection_errors_after"]
    )
    if field_name == "execution_path":
        assert load_attempt_records(tmp_path / "run")[-1].execution_path is None


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_interrupt_during_post_snapshot_is_captured_as_safety_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = GitRepository(create_git_repo(tmp_path / "target"))
    before = WorkspaceSnapshot.capture(repository)
    request = _request(
        tmp_path,
        repository,
        before,
        AgentTaskKind.IMPLEMENTATION,
        AttemptPhase.IMPLEMENTING,
    )
    real_capture = WorkspaceSnapshot.capture
    captures = 0

    def interrupt_second_capture(target: GitRepository) -> WorkspaceSnapshot:
        nonlocal captures
        captures += 1
        if captures == 2:
            raise KeyboardInterrupt("interrupted during post snapshot")
        return real_capture(target)

    monkeypatch.setattr(
        guarded_module.WorkspaceSnapshot,
        "capture",
        staticmethod(interrupt_second_capture),
    )

    outcome = GuardedWritableOperation(MatrixExecutor("tracked-success")).execute(
        request
    )

    assert isinstance(outcome, WritableSafetyStopped)
    assert outcome.audit.after_workspace is None
    assert outcome.audit.workspace_guard.has_inspection_failure
    assert any(
        "KeyboardInterrupt" in error
        for error in outcome.audit.workspace_guard.after.inspection_errors
    )
    assert outcome.audit.workspace_guard.artifact_path is not None
    assert outcome.audit.workspace_guard.artifact_path.is_file()


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_request_rejects_mismatched_writable_task_and_phase(tmp_path: Path) -> None:
    repository = GitRepository(create_git_repo(tmp_path / "target"))
    before = WorkspaceSnapshot.capture(repository)
    request = _request(
        tmp_path,
        repository,
        before,
        AgentTaskKind.IMPLEMENTATION,
        AttemptPhase.IMPLEMENTING,
    )
    correction_execution = replace(
        request.execution_request,
        task_kind=AgentTaskKind.CORRECTION,
        result_contract=CORRECTION_RESULT_CONTRACT,
    )

    with pytest.raises(ValueError, match="requires the implementation task"):
        GuardedWritableRequest(
            phase=AttemptPhase.IMPLEMENTING,
            execution_request=correction_execution,
            baseline=request.baseline,
        )


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_request_rejects_a_review_task_with_workspace_write_access(
    tmp_path: Path,
) -> None:
    repository = GitRepository(create_git_repo(tmp_path / "target"))
    before = WorkspaceSnapshot.capture(repository)
    request = _request(
        tmp_path,
        repository,
        before,
        AgentTaskKind.IMPLEMENTATION,
        AttemptPhase.IMPLEMENTING,
    )
    review_execution = replace(
        request.execution_request,
        task_kind=AgentTaskKind.REVIEW,
        result_contract=REVIEW_RESULT_CONTRACT,
    )

    with pytest.raises(ValueError, match="requires the implementation task"):
        GuardedWritableRequest(
            phase=AttemptPhase.IMPLEMENTING,
            execution_request=review_execution,
            baseline=request.baseline,
        )


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_service_returns_typed_safety_stop_when_attempt_is_no_longer_started(
    tmp_path: Path,
) -> None:
    repository = GitRepository(create_git_repo(tmp_path / "target"))
    before = WorkspaceSnapshot.capture(repository)
    request = _request(
        tmp_path,
        repository,
        before,
        AgentTaskKind.IMPLEMENTATION,
        AttemptPhase.IMPLEMENTING,
    )
    finish_phase_attempt(
        request.execution_request.artifact_directory.parent.parent,
        phase=AttemptPhase.IMPLEMENTING,
        stage_outcome=StageOutcome.FAILED,
    )
    historical_guard = request.execution_request.artifact_directory / (
        "workspace-guard.json"
    )
    historical_guard.write_text("historical guard\n", encoding="utf-8")
    executor = MatrixExecutor("tracked-success")

    reconstructed = GuardedWritableRequest(
        phase=request.phase,
        execution_request=request.execution_request,
        baseline=request.baseline,
    )
    outcome = GuardedWritableOperation(executor).execute(reconstructed)

    assert isinstance(outcome, WritableSafetyStopped)
    assert executor.calls == 0
    assert outcome.audit.invocation_start is InvocationStart.NOT_STARTED
    assert any(
        violation.name == "writable-attempt-evidence"
        for violation in outcome.audit.safety_violations
    )
    assert outcome.audit.workspace_guard.artifact_path is None
    assert historical_guard.read_text(encoding="utf-8") == "historical guard\n"


@pytest.mark.skipif(GIT is None, reason="git executable is required")
@pytest.mark.parametrize(
    "phase",
    [AttemptPhase.IMPLEMENTING, AttemptPhase.CORRECTING],
)
def test_stage_owned_rejection_records_complete_workspace_evidence(
    tmp_path: Path,
    phase: AttemptPhase,
) -> None:
    repository = GitRepository(create_git_repo(tmp_path / "target"))
    before = WorkspaceSnapshot.capture(repository)
    record = start_attempt(
        tmp_path / "run",
        phase=phase,
        before_workspace_fingerprint=None,
    )
    executor = MatrixExecutor("tracked-success")
    request = GuardedWritableRejectionRequest(
        phase=phase,
        artifact_directory=record.artifact_directory,
        baseline=WritableBaseline(
            repository.path,
            before.branch,
            before.head_sha or "",
        ),
        failure_message="Stage authorization failed.",
        safety_violations=(
            WritableSafetyViolation(
                "stage-authorization",
                "authorized",
                "rejected",
                "The stage rejected the operation.",
            ),
        ),
    )

    outcome = GuardedWritableOperation(executor).reject_before_start(request)

    assert isinstance(outcome, WritableSafetyStopped)
    assert executor.calls == 0
    assert outcome.audit.before_workspace is not None
    assert outcome.audit.after_workspace is not None
    persisted = load_attempt_records(tmp_path / "run")[-1]
    assert persisted.before_workspace_fingerprint is not None
    assert persisted.after_workspace_fingerprint is not None
    assert persisted.process_started is False
    assert (record.artifact_directory / "workspace-guard.json").is_file()


def test_writable_baseline_rejects_untrusted_identity_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Git object ID"):
        WritableBaseline(tmp_path, "main", "not-a-sha")
    with pytest.raises(ValueError, match="SHA-256"):
        WritableBaseline(tmp_path, "main", "a" * 40, "not-a-fingerprint")


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_writable_outcome_exposes_only_immutable_audit_evidence(tmp_path: Path) -> None:
    repository = GitRepository(create_git_repo(tmp_path / "target"))
    before = WorkspaceSnapshot.capture(repository)
    request = _request(
        tmp_path,
        repository,
        before,
        AgentTaskKind.IMPLEMENTATION,
        AttemptPhase.IMPLEMENTING,
    )

    outcome = GuardedWritableOperation(MatrixExecutor("tracked-success")).execute(
        request
    )

    assert isinstance(outcome, WritableSucceeded)
    assert not hasattr(outcome.audit, "attempt")
    assert isinstance(outcome.audit.safety_violations, tuple)
    assert isinstance(outcome.audit.changed_files, tuple)
    assert outcome.audit.workspace_guard.artifact_path is not None
    assert outcome.audit.workspace_guard.artifact_path.is_file()
    with pytest.raises(FrozenInstanceError):
        outcome.audit.failure_message = "changed"  # type: ignore[misc]


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_public_outcome_variants_require_trusted_service_construction(
    tmp_path: Path,
) -> None:
    repository = GitRepository(create_git_repo(tmp_path / "target"))
    before = WorkspaceSnapshot.capture(repository)
    request = _request(
        tmp_path,
        repository,
        before,
        AgentTaskKind.IMPLEMENTATION,
        AttemptPhase.IMPLEMENTING,
    )
    outcome = GuardedWritableOperation(MatrixExecutor("tracked-success")).execute(
        request
    )
    assert isinstance(outcome, WritableSucceeded)
    audit = outcome.audit
    constructors = (
        lambda: WritableSucceeded(outcome.result, audit),
        lambda: WritableRejectedBeforeStart(audit),
        lambda: WritableFailedUnchanged(audit),
        lambda: WritableFailedUncertain(audit),
        lambda: WritableSafetyStopped(audit),
    )

    for construct in constructors:
        with pytest.raises(TypeError, match="only be constructed"):
            construct()


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_writable_audit_rejects_mutable_nested_collections(tmp_path: Path) -> None:
    repository = GitRepository(create_git_repo(tmp_path / "target"))
    before = WorkspaceSnapshot.capture(repository)
    request = _request(
        tmp_path,
        repository,
        before,
        AgentTaskKind.IMPLEMENTATION,
        AttemptPhase.IMPLEMENTING,
    )
    outcome = GuardedWritableOperation(MatrixExecutor("tracked-success")).execute(
        request
    )
    assert isinstance(outcome, WritableSucceeded)
    audit = outcome.audit

    with pytest.raises(TypeError, match="immutable tuple"):
        WritableAudit(
            execution=audit.execution,
            before_workspace=audit.before_workspace,
            after_workspace=audit.after_workspace,
            workspace_guard=audit.workspace_guard,
            safety_violations=[],  # type: ignore[arg-type]
            changed_files=audit.changed_files,
            invocation_start=audit.invocation_start,
            failure_message=audit.failure_message,
        )
    with pytest.raises(TypeError, match="immutable tuple"):
        WritableAudit(
            execution=audit.execution,
            before_workspace=audit.before_workspace,
            after_workspace=audit.after_workspace,
            workspace_guard=audit.workspace_guard,
            safety_violations=audit.safety_violations,
            changed_files=[],  # type: ignore[arg-type]
            invocation_start=audit.invocation_start,
            failure_message=audit.failure_message,
        )


def _request(
    tmp_path: Path,
    repository: GitRepository,
    before: WorkspaceSnapshot,
    task_kind: AgentTaskKind,
    phase: AttemptPhase,
) -> GuardedWritableRequest[ImplementationResult]:
    record = start_attempt(
        tmp_path / "run",
        phase=phase,
        before_workspace_fingerprint=None,
    )
    contract = (
        IMPLEMENTATION_RESULT_CONTRACT
        if task_kind is AgentTaskKind.IMPLEMENTATION
        else CORRECTION_RESULT_CONTRACT
    )
    execution_request = AgentExecutionRequest(
        task_kind=task_kind,
        repository_path=repository.path,
        repository_access=RepositoryAccess.WORKSPACE_WRITE,
        prompt="Perform the writable task.",
        result_contract=contract,
        artifact_directory=record.artifact_directory,
        policy=AgentExecutionPolicy(60, NetworkAccess.ALLOWED),
        required_capabilities=required_execution_capabilities(
            RepositoryAccess.WORKSPACE_WRITE
        ),
    )
    return GuardedWritableRequest(
        phase=phase,
        execution_request=execution_request,
        baseline=WritableBaseline(
            repository.path,
            before.branch,
            before.head_sha or "",
            before.fingerprint,
            task_kind is AgentTaskKind.IMPLEMENTATION,
        ),
    )


def _failure(
    task_kind: AgentTaskKind,
    invocation_start: InvocationStart,
    category: AgentFailureCategory,
) -> AgentExecution[ImplementationResult]:
    return AgentExecution(
        provider_id=_PROVIDER_ID,
        task_kind=task_kind,
        status=AgentExecutionStatus.FAILED,
        invocation_start=invocation_start,
        started_at=_NOW,
        ended_at=_NOW,
        duration_seconds=0,
        failure_category=category,
        failure_message=f"scripted {category.value}",
    )


def _mutate(repository: Path, scenario: str) -> None:
    if scenario in {"tracked-success", "changed-tracked-failure"}:
        repository.joinpath("file.txt").write_text("changed\n", encoding="utf-8")
    elif scenario in {"untracked-success", "changed-untracked-failure"}:
        repository.joinpath("new.txt").write_text("new\n", encoding="utf-8")
    elif scenario == "staging-success":
        repository.joinpath("file.txt").write_text("staged\n", encoding="utf-8")
        run_git(repository, "add", "file.txt")
    elif scenario == "branch-success":
        run_git(repository, "checkout", "-b", "guard-matrix-branch")
    elif scenario == "head-success":
        repository.joinpath("file.txt").write_text("committed\n", encoding="utf-8")
        run_git(repository, "add", "file.txt")
        run_git(repository, "commit", "-m", "guard matrix")
    elif scenario == "environment-success":
        environment = repository / ".guard-env"
        environment.mkdir()
        environment.joinpath("pyvenv.cfg").write_text("home = test\n", encoding="utf-8")
