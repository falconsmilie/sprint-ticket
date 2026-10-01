"""Provider-neutral application service for guarded workspace writes."""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import InitVar, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Generic, TypeAlias, TypeVar, final

from ..application.agent_execution import (
    EXECUTION_EVIDENCE_FILE,
    WORKSPACE_GUARD_FILE,
    AgentCapability,
    AgentContractError,
    AgentExecution,
    AgentExecutionRequest,
    AgentExecutor,
    AgentFailureCategory,
    AgentTaskKind,
    AttemptArtifactLayout,
    InvocationStart,
    RepositoryAccess,
)
from ..attempts import (
    AttemptMetadata,
    AttemptRecord,
    load_attempt_records,
    update_attempt,
)
from ..audit import changed_files_including_untracked
from ..domain.task_results import TaskResult
from ..execution_evidence import evidence_from_execution, write_execution_evidence
from ..git import GitRepository
from ..git_safety import WorkspaceChange, WorkspaceSnapshot, workspace_safety_changes
from ..models import AttemptPhase, AttemptStatus
from ..persistence import timestamp_now
from ..run_ownership import RunOwnership, RunOwnershipError
from ..workspace_guard import (
    WorkspaceEnvironmentSnapshot,
    WorkspaceGuardInspection,
    capture_workspace_environment_snapshot,
    format_workspace_hygiene_violation,
    new_environments,
    write_workspace_guard_inspection,
)

ResultT = TypeVar("ResultT", bound=TaskResult)
_OUTCOME_SEAL = object()
_GIT_OBJECT_RE = re.compile(r"[0-9a-fA-F]{40,64}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_WRITABLE_TASK_BY_PHASE = {
    AttemptPhase.IMPLEMENTING: AgentTaskKind.IMPLEMENTATION,
    AttemptPhase.CORRECTING: AgentTaskKind.CORRECTION,
}


@dataclass(frozen=True)
class WritableBaseline:
    """Repository identity and workspace state authorized for one write."""

    repository_path: Path
    branch: str | None
    head_sha: str
    expected_workspace_fingerprint: str | None = None
    require_clean_worktree: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.repository_path, Path):
            raise TypeError("baseline repository_path must be a Path.")
        if self.branch is not None and (
            not isinstance(self.branch, str) or not self.branch.strip()
        ):
            raise ValueError("baseline branch must be a non-empty string or None.")
        if not isinstance(self.head_sha, str) or not _GIT_OBJECT_RE.fullmatch(
            self.head_sha
        ):
            raise ValueError("baseline head_sha must be a Git object ID.")
        if self.expected_workspace_fingerprint is not None and (
            not isinstance(self.expected_workspace_fingerprint, str)
            or not _SHA256_RE.fullmatch(self.expected_workspace_fingerprint)
        ):
            raise ValueError(
                "expected_workspace_fingerprint must be a lowercase SHA-256 digest."
            )
        if not isinstance(self.require_clean_worktree, bool):
            raise TypeError("require_clean_worktree must be a boolean.")


@dataclass(frozen=True)
class GuardedWritableRequest(Generic[ResultT]):
    """Immutable, provider-neutral input for one guarded invocation."""

    phase: AttemptPhase
    execution_request: AgentExecutionRequest[ResultT]
    baseline: WritableBaseline

    def __post_init__(self) -> None:
        _validate_common_request(
            phase=self.phase,
            artifact_directory=self.execution_request.artifact_directory,
            baseline=self.baseline,
        )
        request = self.execution_request
        expected_task = _WRITABLE_TASK_BY_PHASE[self.phase]
        if request.task_kind is not expected_task:
            raise ValueError(
                f"{self.phase.value} requires the {expected_task.value} task contract."
            )
        if request.repository_access is not RepositoryAccess.WORKSPACE_WRITE:
            raise ValueError(
                "Guarded writable execution requires workspace-write repository access."
            )
        if (
            AgentCapability.WORKSPACE_WRITE_EXECUTION
            not in request.required_capabilities
        ):
            raise ValueError(
                "Guarded writable execution requires workspace-write capability."
            )
        if request.repository_path.resolve() != self.baseline.repository_path.resolve():
            raise ValueError(
                "Execution request repository does not match the guarded baseline."
            )


@dataclass(frozen=True)
class WritableSafetyViolation:
    name: str
    expected: str
    actual: str
    message: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("name", self.name),
            ("expected", self.expected),
            ("actual", self.actual),
            ("message", self.message),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")


@dataclass(frozen=True)
class GuardedWritableRejectionRequest:
    """Immutable input for a stage-owned rejection before agent invocation."""

    phase: AttemptPhase
    artifact_directory: Path
    baseline: WritableBaseline
    failure_message: str
    artifact_layout: AttemptArtifactLayout
    safety_violations: tuple[WritableSafetyViolation, ...] = ()

    def __post_init__(self) -> None:
        _validate_common_request(
            phase=self.phase,
            artifact_directory=self.artifact_directory,
            baseline=self.baseline,
        )
        if (
            not isinstance(self.failure_message, str)
            or not self.failure_message.strip()
        ):
            raise ValueError("failure_message must be a non-empty string.")
        if not isinstance(self.safety_violations, tuple) or not all(
            isinstance(item, WritableSafetyViolation) for item in self.safety_violations
        ):
            raise TypeError("safety_violations must be a tuple of safety violations.")
        if not isinstance(self.artifact_layout, AttemptArtifactLayout):
            raise TypeError("artifact_layout must be an AttemptArtifactLayout.")
        self.artifact_layout.revalidate()
        if self.artifact_layout.attempt_root != self.artifact_directory.resolve():
            raise ValueError(
                "artifact_layout attempt root must match artifact_directory."
            )


@dataclass(frozen=True)
class WritableAudit(Generic[ResultT]):
    execution: AgentExecution[ResultT] | None
    before_workspace: WorkspaceSnapshot | None
    after_workspace: WorkspaceSnapshot | None
    workspace_guard: WorkspaceGuardInspection
    safety_violations: tuple[WritableSafetyViolation, ...]
    changed_files: tuple[str, ...]
    invocation_start: InvocationStart
    failure_message: str | None = None

    def __post_init__(self) -> None:
        if self.execution is not None and not isinstance(
            self.execution, AgentExecution
        ):
            raise TypeError("execution must be an AgentExecution or None.")
        for name, snapshot in (
            ("before_workspace", self.before_workspace),
            ("after_workspace", self.after_workspace),
        ):
            if snapshot is not None and not isinstance(snapshot, WorkspaceSnapshot):
                raise TypeError(f"{name} must be a WorkspaceSnapshot or None.")
        if not isinstance(self.workspace_guard, WorkspaceGuardInspection):
            raise TypeError("workspace_guard must be a WorkspaceGuardInspection.")
        if not isinstance(self.safety_violations, tuple) or not all(
            isinstance(item, WritableSafetyViolation) for item in self.safety_violations
        ):
            raise TypeError("safety_violations must be an immutable tuple.")
        if not isinstance(self.changed_files, tuple) or not all(
            isinstance(item, str) and item for item in self.changed_files
        ):
            raise TypeError("changed_files must be an immutable tuple of paths.")
        if not isinstance(self.invocation_start, InvocationStart):
            raise TypeError("invocation_start must be an InvocationStart value.")
        if self.failure_message is not None and (
            not isinstance(self.failure_message, str)
            or not self.failure_message.strip()
        ):
            raise ValueError("failure_message must be a non-empty string or None.")

    @property
    def after_workspace_fingerprint(self) -> str | None:
        if self.after_workspace is None:
            return None
        return self.after_workspace.fingerprint


@final
@dataclass(frozen=True)
class WritableSucceeded(Generic[ResultT]):
    result: ResultT
    audit: WritableAudit[ResultT]
    _seal: InitVar[object] = field(default=None, kw_only=True)

    def __post_init__(self, _seal: object) -> None:
        _require_trusted_outcome(_seal)
        execution = self.audit.execution
        if execution is None or not execution.successful:
            raise ValueError("Writable success requires a successful execution.")
        if execution.result is not self.result:
            raise ValueError(
                "Writable success must expose the audited execution result."
            )
        if self.audit.invocation_start is not InvocationStart.STARTED:
            raise ValueError("Writable success requires a started invocation.")
        if (
            self.audit.before_workspace is None
            or self.audit.after_workspace is None
            or not self.audit.before_workspace.inspection_complete
            or not self.audit.after_workspace.inspection_complete
        ):
            raise ValueError("Writable success requires a validated post-state.")
        if self.audit.safety_violations or self.audit.workspace_guard.requires_human:
            raise ValueError("Writable success cannot contain safety violations.")
        _require_persisted_guard(self.audit)


@final
@dataclass(frozen=True)
class WritableRejectedBeforeStart(Generic[ResultT]):
    audit: WritableAudit[ResultT]
    _seal: InitVar[object] = field(default=None, kw_only=True)

    def __post_init__(self, _seal: object) -> None:
        _require_trusted_outcome(_seal)
        if self.audit.invocation_start is not InvocationStart.NOT_STARTED:
            raise ValueError("Pre-start rejection requires a not-started invocation.")
        if self.audit.safety_violations or self.audit.workspace_guard.requires_human:
            raise ValueError("Safety failures require WritableSafetyStopped.")
        if self.audit.execution is not None and self.audit.execution.successful:
            raise ValueError(
                "Pre-start rejection cannot contain a successful execution."
            )
        if (
            self.audit.before_workspace is None
            or self.audit.after_workspace is None
            or not self.audit.before_workspace.inspection_complete
            or not self.audit.after_workspace.inspection_complete
        ):
            raise ValueError(
                "Pre-start rejection requires complete workspace evidence."
            )
        _require_persisted_guard(self.audit)


@final
@dataclass(frozen=True)
class WritableFailedUnchanged(Generic[ResultT]):
    audit: WritableAudit[ResultT]
    _seal: InitVar[object] = field(default=None, kw_only=True)

    def __post_init__(self, _seal: object) -> None:
        _require_trusted_outcome(_seal)
        if self.audit.invocation_start not in {
            InvocationStart.STARTED,
            InvocationStart.UNKNOWN,
        }:
            raise ValueError(
                "Unchanged failure requires a started or uncertain invocation."
            )
        execution = self.audit.execution
        if execution is None or execution.successful:
            raise ValueError("Unchanged failure requires a failed execution.")
        before = self.audit.before_workspace
        after = self.audit.after_workspace
        if (
            before is None
            or after is None
            or not before.inspection_complete
            or not after.inspection_complete
            or not before.matches(after)
        ):
            raise ValueError(
                "Unchanged failure requires identical workspace snapshots."
            )
        if self.audit.safety_violations or self.audit.workspace_guard.requires_human:
            raise ValueError("Safety failures require WritableSafetyStopped.")
        _require_persisted_guard(self.audit)


@final
@dataclass(frozen=True)
class WritableFailedUncertain(Generic[ResultT]):
    audit: WritableAudit[ResultT]
    _seal: InitVar[object] = field(default=None, kw_only=True)

    def __post_init__(self, _seal: object) -> None:
        _require_trusted_outcome(_seal)
        if not self.audit.failure_message:
            raise ValueError("Uncertain failure requires a failure message.")


@final
@dataclass(frozen=True)
class WritableSafetyStopped(Generic[ResultT]):
    audit: WritableAudit[ResultT]
    _seal: InitVar[object] = field(default=None, kw_only=True)

    def __post_init__(self, _seal: object) -> None:
        _require_trusted_outcome(_seal)
        if (
            not self.audit.safety_violations
            and not self.audit.workspace_guard.requires_human
        ):
            raise ValueError("Safety stop requires persisted safety evidence.")


GuardedWritableOutcome: TypeAlias = (
    WritableSucceeded[ResultT]
    | WritableRejectedBeforeStart[ResultT]
    | WritableFailedUnchanged[ResultT]
    | WritableFailedUncertain[ResultT]
    | WritableSafetyStopped[ResultT]
)


def _require_trusted_outcome(seal: object) -> None:
    if seal is not _OUTCOME_SEAL:
        raise TypeError(
            "Guarded writable outcomes can only be constructed by "
            "GuardedWritableOperation."
        )


def _require_persisted_guard(audit: WritableAudit[TaskResult]) -> None:
    if audit.workspace_guard.artifact_path is None:
        raise ValueError("This writable outcome requires persisted guard evidence.")


@dataclass
class _AttemptTracker:
    artifact_directory: Path
    artifact_layout: AttemptArtifactLayout
    record: AttemptRecord | None
    before_snapshot: WorkspaceSnapshot | None
    before_error: str | None
    run_ownership: RunOwnership | None = None
    process_started: bool = False
    evidence_errors: list[str] = field(default_factory=list)
    _start_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def revalidate(self) -> None:
        self.artifact_layout.revalidate()

    @property
    def before_complete(self) -> bool:
        return bool(
            self.before_snapshot is not None
            and self.before_snapshot.inspection_complete
        )

    def mark_started(self) -> None:
        self.record_started(True)

    def record_started(self, value: bool) -> None:
        with self._start_lock:
            if not value:
                if self.process_started:
                    self.evidence_errors.append(
                        "executor returned not-started after reporting invocation start"
                    )
                return
            if self.process_started:
                return
            self.process_started = True
        # Persist from the launch observer so a controller crash after launch
        # cannot leave the invocation recorded as never started. The attempt
        # store revalidates the complete expected record in the same short
        # commit fence as replacement, which excludes a stale observer after a
        # newer controller-owned update or finalisation.
        self._update(process_started=True, propagate_interrupt=True)

    def record_after(
        self,
        snapshot: WorkspaceSnapshot | None,
        error: str | None,
    ) -> None:
        if snapshot is not None:
            self._update(after_workspace_fingerprint=snapshot.fingerprint)
            return
        if self.record is None:
            return
        detail = error or "Could not capture a post-call canonical workspace snapshot."
        self._update(
            metadata=replace(self.record.metadata, after_workspace_error=detail)
        )

    def _update(self, *, propagate_interrupt: bool = False, **changes: object) -> None:
        if self.record is None:
            return
        with self._start_lock:
            changes.setdefault("process_started", self.process_started)
        self.revalidate()
        try:
            self.record = update_attempt(
                self.record,
                run_ownership=self.run_ownership,
                **changes,
            )
        except RunOwnershipError:
            raise
        except BaseException as error:
            # If the update raced with replacement of the bound ledger, ownership
            # loss is authoritative and must not be reduced to an evidence warning.
            self.revalidate()
            self.evidence_errors.append(
                f"attempt-ledger: {type(error).__name__}: {error}"
            )
            if propagate_interrupt and not isinstance(error, Exception):
                raise


class GuardedWritableOperation:
    """Capture, execute, audit, and classify one workspace-write operation."""

    def __init__(
        self,
        executor: AgentExecutor,
        *,
        clock: Callable[[], datetime] | None = None,
        run_ownership: RunOwnership | None = None,
    ) -> None:
        self._executor = executor
        self._clock = clock
        if run_ownership is not None and not isinstance(run_ownership, RunOwnership):
            raise TypeError("run_ownership must be a RunOwnership or None.")
        self._run_ownership = run_ownership

    def execute(
        self,
        request: GuardedWritableRequest[ResultT],
    ) -> GuardedWritableOutcome[ResultT]:
        layout = request.execution_request.artifact_layout
        assert layout is not None
        self._validate_run_artifacts(
            request.execution_request.artifact_directory,
            artifact_layout=layout,
        )
        repository = GitRepository(request.baseline.repository_path)
        tracker, environment_before = self._capture_before(
            repository,
            phase=request.phase,
            artifact_directory=request.execution_request.artifact_directory,
            artifact_layout=layout,
        )
        starting_violations = _starting_violations(
            tracker.before_snapshot,
            request.baseline,
        )
        if not tracker.before_complete or not environment_before.inspection_complete:
            starting_violations = _merge_violations(
                starting_violations,
                (
                    _inspection_violation(
                        "before", tracker.before_snapshot, tracker.before_error
                    ),
                ),
            )
        starting_violations = _merge_violations(
            starting_violations,
            _attempt_evidence_violations(tracker),
        )
        if starting_violations:
            audit = self._audit_without_invocation(
                repository=repository,
                tracker=tracker,
                environment_before=environment_before,
                phase=request.phase,
                baseline=request.baseline,
                failure_message=(
                    "Writable execution was rejected by its starting safety checks."
                ),
                starting_violations=starting_violations,
            )
            return WritableSafetyStopped(audit, _seal=_OUTCOME_SEAL)

        capability_problem = self._executor_capability_problem(
            request.execution_request
        )
        if capability_problem is not None:
            audit = self._audit_without_invocation(
                repository=repository,
                tracker=tracker,
                environment_before=environment_before,
                phase=request.phase,
                baseline=request.baseline,
                failure_message=capability_problem,
                starting_violations=(),
            )
            if audit.safety_violations or audit.workspace_guard.requires_human:
                return WritableSafetyStopped(audit, _seal=_OUTCOME_SEAL)
            return WritableRejectedBeforeStart(audit, _seal=_OUTCOME_SEAL)

        execution: AgentExecution[ResultT] | None = None
        execution_error: BaseException | None = None
        after_workspace: WorkspaceSnapshot | None = None
        after_error: str | None = None
        try:
            self._validate_run_artifacts(
                request.execution_request.artifact_directory,
                artifact_layout=layout,
            )
            execution = self._executor.execute(
                request.execution_request,
                on_invocation_start=tracker.mark_started,
            )
            if execution.invocation_start is not InvocationStart.UNKNOWN:
                tracker.record_started(
                    execution.invocation_start is InvocationStart.STARTED
                )
            self._validate_run_artifacts(
                request.execution_request.artifact_directory,
                artifact_layout=layout,
            )
            contract_problem = _execution_contract_problem(
                execution, request.execution_request
            )
            if contract_problem is not None:
                raise ValueError(contract_problem)
            write_execution_evidence(
                layout,
                evidence_from_execution(request.execution_request, execution),
            )
            layout.revalidate()
            tracker._update(execution_path=EXECUTION_EVIDENCE_FILE)
        except RunOwnershipError:
            raise
        except BaseException as error:  # noqa: BLE001 - interruption must fail closed.
            execution_error = error
        finally:
            self._validate_run_artifacts(
                request.execution_request.artifact_directory,
                artifact_layout=layout,
            )
            after_workspace, after_error = self._capture_workspace(repository)
            tracker.revalidate()
            environment_after = self._capture_environment(repository)
            tracker.revalidate()
            environment_after = _with_workspace_inspection_errors(
                environment_after,
                snapshot=after_workspace,
                capture_error=after_error,
                point="after",
            )
            tracker.record_after(after_workspace, after_error)
            guard = self._build_and_persist_guard_evidence(
                tracker,
                before=environment_before,
                after=environment_after,
                phase=request.phase,
            )

        invocation_start = _invocation_start(execution, tracker, execution_error)
        post_violations = _post_violations(
            after_workspace, after_error, request.baseline
        )
        post_violations = _merge_violations(
            post_violations, _attempt_evidence_violations(tracker)
        )
        changed_files, change_error = _changed_files(
            repository, request.baseline.head_sha
        )
        tracker.revalidate()
        if change_error is not None:
            post_violations += (
                WritableSafetyViolation(
                    name="worktree-inspection",
                    expected="baseline-relative source diff inspection succeeds",
                    actual=change_error,
                    message=(
                        "Could not inspect source changes after writable agent invocation."
                    ),
                ),
            )
        unchanged = _workspace_unchanged(tracker.before_snapshot, after_workspace)
        if (
            (
                execution_error is not None
                or execution is None
                or not execution.successful
            )
            and changed_files
            and not unchanged
        ):
            post_violations += (
                WritableSafetyViolation(
                    name="worktree",
                    expected=(
                        "no baseline-relative source changes after failed writable "
                        "invocation"
                    ),
                    actual=_format_files(changed_files),
                    message=(
                        "Baseline-relative source changes exist after failed writable "
                        "agent invocation."
                    ),
                ),
            )
        failure_message = (
            f"{type(execution_error).__name__}: {execution_error}"
            if execution_error is not None
            else None
            if execution is None
            else execution.failure_message
        )
        audit = WritableAudit(
            execution=execution,
            before_workspace=tracker.before_snapshot,
            after_workspace=after_workspace,
            workspace_guard=guard,
            safety_violations=post_violations,
            changed_files=changed_files,
            invocation_start=invocation_start,
            failure_message=failure_message,
        )

        if execution_error is not None:
            return WritableFailedUncertain(audit, _seal=_OUTCOME_SEAL)
        if execution is not None and execution.successful:
            if post_violations or guard.requires_human:
                return WritableSafetyStopped(audit, _seal=_OUTCOME_SEAL)
            result = execution.result
            assert result is not None
            return WritableSucceeded(result, audit, _seal=_OUTCOME_SEAL)

        untrusted_completion = bool(
            execution is not None
            and execution.failure_category
            in {
                AgentFailureCategory.MISSING_RESULT,
                AgentFailureCategory.INVALID_RESULT,
                AgentFailureCategory.TIMEOUT,
                AgentFailureCategory.INVOCATION_CLEANUP_UNCERTAIN,
            }
        )
        if (
            invocation_start is InvocationStart.NOT_STARTED
            and unchanged
            and not post_violations
            and not guard.requires_human
        ):
            return WritableRejectedBeforeStart(audit, _seal=_OUTCOME_SEAL)
        if (
            invocation_start in {InvocationStart.STARTED, InvocationStart.UNKNOWN}
            and unchanged
            and not untrusted_completion
            and not post_violations
            and not guard.requires_human
        ):
            return WritableFailedUnchanged(audit, _seal=_OUTCOME_SEAL)
        return WritableFailedUncertain(audit, _seal=_OUTCOME_SEAL)

    def reject_before_start(
        self,
        request: GuardedWritableRejectionRequest,
    ) -> WritableRejectedBeforeStart[TaskResult] | WritableSafetyStopped[TaskResult]:
        """Persist safety evidence for a stage-owned pre-invocation rejection."""

        layout = request.artifact_layout
        self._validate_run_artifacts(
            request.artifact_directory,
            artifact_layout=layout,
        )
        repository = GitRepository(request.baseline.repository_path)
        tracker, environment_before = self._capture_before(
            repository,
            phase=request.phase,
            artifact_directory=request.artifact_directory,
            artifact_layout=layout,
        )
        violations = _merge_violations(
            request.safety_violations,
            _starting_violations(tracker.before_snapshot, request.baseline),
        )
        if not tracker.before_complete or not environment_before.inspection_complete:
            violations = _merge_violations(
                violations,
                (
                    _inspection_violation(
                        "before", tracker.before_snapshot, tracker.before_error
                    ),
                ),
            )
        audit = self._audit_without_invocation(
            repository=repository,
            tracker=tracker,
            environment_before=environment_before,
            phase=request.phase,
            baseline=request.baseline,
            failure_message=request.failure_message,
            starting_violations=violations,
        )
        if audit.safety_violations or audit.workspace_guard.requires_human:
            return WritableSafetyStopped(audit, _seal=_OUTCOME_SEAL)
        return WritableRejectedBeforeStart(audit, _seal=_OUTCOME_SEAL)

    def _capture_before(
        self,
        repository: GitRepository,
        *,
        phase: AttemptPhase,
        artifact_directory: Path,
        artifact_layout: AttemptArtifactLayout,
    ) -> tuple[_AttemptTracker, WorkspaceEnvironmentSnapshot]:
        self._validate_run_artifacts(
            artifact_directory,
            artifact_layout=artifact_layout,
        )
        record, association_error = _associated_attempt(
            artifact_directory,
            phase,
            artifact_layout=artifact_layout,
            run_ownership=self._run_ownership,
        )
        artifact_layout.revalidate()
        environment = self._capture_environment(repository)
        artifact_layout.revalidate()
        snapshot, error = self._capture_workspace(repository)
        artifact_layout.revalidate()
        environment = _with_workspace_inspection_errors(
            environment,
            snapshot=snapshot,
            capture_error=error,
            point="before",
        )
        metadata = AttemptMetadata() if record is None else record.metadata
        if error is not None and record is not None:
            metadata = replace(metadata, before_workspace_error=error)
        tracker = _AttemptTracker(
            artifact_directory,
            artifact_layout,
            record,
            snapshot,
            error,
            self._run_ownership,
        )
        if association_error is not None:
            tracker.evidence_errors.append(association_error)
        tracker._update(
            before_workspace_fingerprint=(
                None if snapshot is None else snapshot.fingerprint
            ),
            metadata=metadata,
        )
        return tracker, environment

    def _audit_without_invocation(
        self,
        *,
        repository: GitRepository,
        tracker: _AttemptTracker,
        environment_before: WorkspaceEnvironmentSnapshot,
        phase: AttemptPhase,
        baseline: WritableBaseline,
        failure_message: str,
        starting_violations: tuple[WritableSafetyViolation, ...],
    ) -> WritableAudit[TaskResult]:
        self._validate_run_artifacts(
            tracker.artifact_directory,
            artifact_layout=tracker.artifact_layout,
        )
        after_workspace, after_error = self._capture_workspace(repository)
        tracker.revalidate()
        environment_after = self._capture_environment(repository)
        tracker.revalidate()
        environment_after = _with_workspace_inspection_errors(
            environment_after,
            snapshot=after_workspace,
            capture_error=after_error,
            point="after",
        )
        tracker.record_after(after_workspace, after_error)
        guard = self._build_and_persist_guard_evidence(
            tracker,
            before=environment_before,
            after=environment_after,
            phase=phase,
        )
        violations = _merge_violations(
            starting_violations,
            _post_violations(after_workspace, after_error, baseline),
        )
        violations = _merge_violations(
            violations, _attempt_evidence_violations(tracker)
        )
        changed_files, change_error = _changed_files(repository, baseline.head_sha)
        tracker.revalidate()
        if change_error is not None:
            violations += (
                WritableSafetyViolation(
                    name="worktree-inspection",
                    expected="baseline-relative source diff inspection succeeds",
                    actual=change_error,
                    message="Could not inspect source changes before writable invocation.",
                ),
            )
        return WritableAudit(
            execution=None,
            before_workspace=tracker.before_snapshot,
            after_workspace=after_workspace,
            workspace_guard=guard,
            safety_violations=violations,
            changed_files=changed_files,
            invocation_start=InvocationStart.NOT_STARTED,
            failure_message=failure_message,
        )

    def _capture_environment(
        self,
        repository: GitRepository,
    ) -> WorkspaceEnvironmentSnapshot:
        try:
            return capture_workspace_environment_snapshot(repository.path)
        except BaseException as error:  # noqa: BLE001 - interruption evidence fails closed.
            return WorkspaceEnvironmentSnapshot(
                repository_path=repository.path.absolute(),
                environments=(),
                inspection_errors=(f"scanner: {type(error).__name__}: {error}",),
            )

    def _capture_workspace(
        self,
        repository: GitRepository,
    ) -> tuple[WorkspaceSnapshot | None, str | None]:
        try:
            return WorkspaceSnapshot.capture(repository), None
        except BaseException as error:  # noqa: BLE001 - interruption evidence fails closed.
            return None, f"{type(error).__name__}: {error}"

    def _compare_environment(
        self,
        *,
        before: WorkspaceEnvironmentSnapshot,
        after: WorkspaceEnvironmentSnapshot,
        phase: AttemptPhase,
    ) -> WorkspaceGuardInspection:
        return WorkspaceGuardInspection(
            phase=phase,
            timestamp=timestamp_now(self._clock),
            artifact_path=None,
            before=before,
            after=after,
            new_environments=new_environments(before, after),
        )

    def _build_and_persist_guard_evidence(
        self,
        tracker: _AttemptTracker,
        *,
        before: WorkspaceEnvironmentSnapshot,
        after: WorkspaceEnvironmentSnapshot,
        phase: AttemptPhase,
    ) -> WorkspaceGuardInspection:
        self._validate_run_artifacts(
            tracker.artifact_directory,
            artifact_layout=tracker.artifact_layout,
        )
        try:
            inspection = self._compare_environment(
                before=before,
                after=after,
                phase=phase,
            )
            tracker.revalidate()
        except (AgentContractError, RunOwnershipError):
            raise
        except BaseException as error:  # noqa: BLE001 - post evidence must fail closed.
            detail = f"workspace-guard-comparison: {type(error).__name__}: {error}"
            tracker.evidence_errors.append(detail)
            inspection = WorkspaceGuardInspection(
                phase=phase,
                timestamp=timestamp_now(),
                artifact_path=None,
                before=before,
                after=replace(
                    after,
                    inspection_errors=(*after.inspection_errors, detail),
                ),
                new_environments=(),
            )
        return self._persist_guard_evidence(tracker, inspection)

    def _persist_guard_evidence(
        self,
        tracker: _AttemptTracker,
        inspection: WorkspaceGuardInspection,
    ) -> WorkspaceGuardInspection:
        if tracker.record is None:
            return _with_guard_evidence_errors(inspection, tracker.evidence_errors)
        path = tracker.artifact_layout.path(WORKSPACE_GUARD_FILE)
        self._validate_run_artifacts(
            path.parent,
            artifact_layout=tracker.artifact_layout,
        )
        persisted = replace(inspection, artifact_path=path)
        persisted = _with_guard_evidence_errors(persisted, tracker.evidence_errors)
        try:
            write_workspace_guard_inspection(persisted)
            tracker.revalidate()
        except BaseException as error:  # noqa: BLE001 - audit failure must fail closed.
            tracker.revalidate()
            tracker.evidence_errors.append(
                f"workspace-guard-artifact: {type(error).__name__}: {error}"
            )
            return _with_guard_evidence_errors(inspection, tracker.evidence_errors)
        return persisted

    def _validate_run_artifacts(
        self,
        artifact_directory: Path,
        *,
        artifact_layout: AttemptArtifactLayout | None = None,
    ) -> None:
        if artifact_layout is not None:
            artifact_layout.revalidate()
        if self._run_ownership is None:
            return
        self._run_ownership.validate_descendant(artifact_directory)

    def _executor_capability_problem(
        self, request: AgentExecutionRequest[TaskResult]
    ) -> str | None:
        try:
            available = self._executor.capabilities
            missing = request.missing_capabilities(available)
        except Exception as error:  # noqa: BLE001 - capability proof must fail closed.
            return (
                "Could not establish executor capabilities before writable invocation: "
                f"{type(error).__name__}: {error}"
            )
        if not missing:
            return None
        names = ", ".join(sorted(capability.value for capability in missing))
        return f"Executor lacks required writable capabilities: {names}."


def format_writable_failure_audit(
    audit: WritableAudit[TaskResult],
    *,
    operation: str,
    run_dir: Path | str,
) -> str:
    """Format provider-neutral audit facts shared by writable stage views."""

    category = (
        "EXCEPTION"
        if audit.execution is None
        else "UNKNOWN"
        if audit.execution.failure_category is None
        else audit.execution.failure_category.value
    )
    rows = [
        f"Agent failure: {category}: {audit.failure_message or 'unknown failure'}",
        f"Last operation: {operation}",
        f"Invocation start: {audit.invocation_start.value}",
        f"Git safety: {_format_failure_safety(audit.safety_violations)}",
        f"Changed files relative to baseline: {_format_files(audit.changed_files)}",
    ]
    if audit.execution is None or not audit.execution.artifacts:
        rows.append("Agent artifacts: none")
    else:
        rows.extend(
            [
                "Agent artifacts:",
                *(
                    f"  - {artifact.name}: {artifact.run_relative_path}"
                    for artifact in audit.execution.artifacts
                ),
            ]
        )
    if audit.workspace_guard.requires_human:
        rows.append(
            format_writable_guard_stop(
                audit.workspace_guard,
                operation=operation,
                run_dir=run_dir,
                git_safety=_format_failure_safety(audit.safety_violations),
            )
        )
    return " ".join(rows)


def format_writable_guard_stop(
    inspection: WorkspaceGuardInspection,
    *,
    operation: str,
    run_dir: Path | str,
    git_safety: str,
) -> str:
    if not inspection.requires_human:
        raise ValueError("A clean writable guard has no human-required stop message.")
    if inspection.has_violation:
        return format_workspace_hygiene_violation(
            inspection,
            operation=operation,
            run_dir=run_dir,
            git_safety=git_safety,
        )
    errors = tuple(
        f"before: {error}" for error in inspection.before.inspection_errors
    ) + tuple(f"after: {error}" for error in inspection.after.inspection_errors)
    artifact = (
        "writable execution record"
        if inspection.artifact_path is None
        else _relative_path(inspection.artifact_path, Path(run_dir))
    )
    details = "; ".join(errors) or "unknown scanner failure"
    return (
        "Workspace environment inspection could not complete for the writable "
        f"agent operation {operation}. TicketAutomation has stopped for human "
        "inspection. The deterministic check covers Python and Conda "
        f"environment roots inside the target repository. Details: {details}. "
        "No files were deleted automatically. "
        f"Workspace-guard artifact: {artifact}."
    )


def _validate_common_request(
    *,
    phase: object,
    artifact_directory: object,
    baseline: object,
) -> None:
    if not isinstance(phase, AttemptPhase):
        raise TypeError("phase must be an AttemptPhase value.")
    if phase not in _WRITABLE_TASK_BY_PHASE:
        raise ValueError(f"Attempt phase {phase.value} is not writable.")
    if not isinstance(artifact_directory, Path):
        raise TypeError("artifact_directory must be a Path.")
    if not isinstance(baseline, WritableBaseline):
        raise TypeError("baseline must be a WritableBaseline.")


def _associated_attempt(
    artifact_directory: Path,
    phase: AttemptPhase,
    *,
    artifact_layout: AttemptArtifactLayout,
    run_ownership: RunOwnership | None = None,
) -> tuple[AttemptRecord | None, str | None]:
    run_dir = artifact_directory.parent.parent
    expected = artifact_directory.resolve()
    try:
        matches = tuple(
            record
            for record in load_attempt_records(
                run_dir,
                run_ownership=run_ownership,
            )
            if record.artifact_directory.resolve() == expected
        )
    except RunOwnershipError:
        raise
    except BaseException as error:  # noqa: BLE001 - association evidence fails closed.
        return None, f"attempt-association: {type(error).__name__}: {error}"
    if len(matches) != 1:
        return None, "attempt-association: artifact directory has no unique attempt"
    record = replace(matches[0], artifact_layout=artifact_layout)
    if record.phase is not phase:
        return None, "attempt-association: attempt phase does not match the request"
    if record.status is not AttemptStatus.STARTED:
        return None, "attempt-association: guarded execution requires a started attempt"
    return record, None


def _starting_violations(
    snapshot: WorkspaceSnapshot | None,
    baseline: WritableBaseline,
) -> tuple[WritableSafetyViolation, ...]:
    if snapshot is None:
        return ()
    changes = list(
        workspace_safety_changes(
            snapshot,
            expected_repository_path=baseline.repository_path,
            expected_branch=baseline.branch,
            expected_head_sha=baseline.head_sha,
            require_clean_worktree=baseline.require_clean_worktree,
        )
    )
    if (
        baseline.expected_workspace_fingerprint is not None
        and not snapshot.matches_fingerprint(baseline.expected_workspace_fingerprint)
    ):
        changes.append(
            WorkspaceChange(
                name="workspace-fingerprint",
                expected=baseline.expected_workspace_fingerprint,
                actual=snapshot.fingerprint,
                message="Workspace no longer matches the authorized starting state.",
            )
        )
    return _safety_violations(changes, point="before writable execution")


def _post_violations(
    snapshot: WorkspaceSnapshot | None,
    capture_error: str | None,
    baseline: WritableBaseline,
) -> tuple[WritableSafetyViolation, ...]:
    if snapshot is None:
        return (_inspection_violation("after", snapshot, capture_error),)
    return _safety_violations(
        workspace_safety_changes(
            snapshot,
            expected_repository_path=baseline.repository_path,
            expected_branch=baseline.branch,
            expected_head_sha=baseline.head_sha,
        ),
        point="after writable execution",
    )


def _safety_violations(
    changes: list[WorkspaceChange] | tuple[WorkspaceChange, ...],
    *,
    point: str,
) -> tuple[WritableSafetyViolation, ...]:
    return tuple(
        WritableSafetyViolation(
            name=change.name,
            expected=change.expected,
            actual=change.actual,
            message=f"{change.message.rstrip('.')} {point}.",
        )
        for change in changes
    )


def _inspection_violation(
    point: str,
    snapshot: WorkspaceSnapshot | None,
    capture_error: str | None,
) -> WritableSafetyViolation:
    actual = (
        capture_error or "unavailable"
        if snapshot is None
        else "; ".join(snapshot.inspection_errors) or "incomplete"
    )
    return WritableSafetyViolation(
        name="workspace-inspection",
        expected=f"complete {point}-call canonical workspace snapshot",
        actual=actual,
        message=f"Could not establish workspace safety {point} writable execution.",
    )


def _with_workspace_inspection_errors(
    environment: WorkspaceEnvironmentSnapshot,
    *,
    snapshot: WorkspaceSnapshot | None,
    capture_error: str | None,
    point: str,
) -> WorkspaceEnvironmentSnapshot:
    errors = list(environment.inspection_errors)
    if snapshot is None:
        errors.append(
            f"{point}-canonical-workspace: "
            f"{capture_error or 'workspace snapshot unavailable'}"
        )
    elif not snapshot.inspection_complete:
        errors.extend(
            f"{point}-canonical-workspace: {error}"
            for error in snapshot.inspection_errors
        )
    return replace(environment, inspection_errors=tuple(dict.fromkeys(errors)))


def _changed_files(
    repository: GitRepository,
    baseline_sha: str,
) -> tuple[tuple[str, ...], str | None]:
    try:
        return changed_files_including_untracked(repository, baseline_sha), None
    except (OSError, RuntimeError, ValueError) as error:
        return (), f"{type(error).__name__}: {error}"


def _workspace_unchanged(
    before: WorkspaceSnapshot | None,
    after: WorkspaceSnapshot | None,
) -> bool:
    return bool(
        before is not None
        and after is not None
        and before.inspection_complete
        and after.inspection_complete
        and before.matches(after)
    )


def _execution_contract_problem(
    execution: AgentExecution[TaskResult],
    request: AgentExecutionRequest[TaskResult],
) -> str | None:
    if execution.task_kind is not request.task_kind:
        return (
            "Executor returned a result for the wrong task kind: "
            f"expected {request.task_kind.value}, got {execution.task_kind.value}."
        )
    if execution.successful and not request.result_contract.accepts(execution.result):
        return "Executor returned a successful result outside the requested contract."
    return None


def _attempt_evidence_violations(
    tracker: _AttemptTracker,
) -> tuple[WritableSafetyViolation, ...]:
    if not tracker.evidence_errors:
        return ()
    return (
        WritableSafetyViolation(
            name="writable-attempt-evidence",
            expected="consistent, persisted writable-attempt evidence",
            actual="; ".join(tracker.evidence_errors),
            message="Writable-attempt evidence is incomplete or contradictory.",
        ),
    )


def _with_guard_evidence_errors(
    inspection: WorkspaceGuardInspection,
    errors: list[str],
) -> WorkspaceGuardInspection:
    merged = list(inspection.after.inspection_errors)
    for error in errors:
        if error not in merged:
            merged.append(error)
    after = replace(
        inspection.after,
        inspection_errors=tuple(merged),
    )
    return replace(inspection, after=after)


def _invocation_start(
    execution: AgentExecution[TaskResult] | None,
    tracker: _AttemptTracker,
    execution_error: BaseException | None,
) -> InvocationStart:
    if tracker.process_started:
        return InvocationStart.STARTED
    if execution is not None:
        return execution.invocation_start
    if execution_error is not None:
        return InvocationStart.UNKNOWN
    return InvocationStart.NOT_STARTED


def _merge_violations(
    first: tuple[WritableSafetyViolation, ...],
    second: tuple[WritableSafetyViolation, ...],
) -> tuple[WritableSafetyViolation, ...]:
    values = {violation.name: violation for violation in first}
    values.update({violation.name: violation for violation in second})
    return tuple(values.values())


def _format_failure_safety(
    violations: tuple[WritableSafetyViolation, ...],
) -> str:
    if not violations:
        return "branch unchanged; HEAD unchanged; staging empty"
    return "; ".join(
        f"{violation.name} expected {violation.expected}, got {violation.actual}"
        for violation in violations
    )


def _format_files(files: tuple[str, ...]) -> str:
    if not files:
        return "none"
    shown = ", ".join(files[:5])
    hidden_count = len(files) - 5
    return shown if hidden_count <= 0 else f"{shown}, and {hidden_count} more"


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


__all__ = [
    "GuardedWritableOperation",
    "GuardedWritableOutcome",
    "GuardedWritableRejectionRequest",
    "GuardedWritableRequest",
    "WritableAudit",
    "WritableBaseline",
    "WritableFailedUncertain",
    "WritableFailedUnchanged",
    "WritableRejectedBeforeStart",
    "WritableSafetyStopped",
    "WritableSafetyViolation",
    "WritableSucceeded",
    "format_writable_failure_audit",
    "format_writable_guard_stop",
]
