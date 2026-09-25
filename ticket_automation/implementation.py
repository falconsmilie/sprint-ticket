from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .application.agent_execution import (
    IMPLEMENTATION_RESULT_CONTRACT,
    AgentExecution,
    AgentExecutionRequest,
    AgentExecutor,
    AgentTaskKind,
    AttemptArtifactLayout,
    InvocationStart,
    RepositoryAccess,
    required_execution_capabilities,
)
from .application.guarded_writable_operation import (
    GuardedWritableOperation,
    GuardedWritableRejectionRequest,
    GuardedWritableRequest,
    WritableBaseline,
    WritableFailedUncertain,
    WritableFailedUnchanged,
    WritableRejectedBeforeStart,
    WritableSafetyStopped,
    WritableSafetyViolation,
    WritableSucceeded,
    format_writable_failure_audit,
    format_writable_guard_stop,
)
from .attempts import (
    AttemptMetadata,
    StageAttempt,
    attempt_artifact_layout,
    complete_stage_attempt,
    require_stage_attempt,
    start_attempt,
)
from .config import AppConfig
from .domain.task_results import ImplementationResult, ImplementationStatus
from .git import GitRepository
from .models import (
    ATTEMPT_RESULT_ARTIFACT_NAME,
    AttemptPhase,
    StageOutcome,
    WorkflowState,
)
from .persistence_codecs import (
    write_implementation_result,
    write_stage_message,
)
from .resolved_config import config_from_resolved_run_policy
from .run_ownership import RunOwnership
from .runs import (
    BASELINE_RECORD_FILE,
    RUN_RECORD_FILE,
    RUN_TICKET_FILE,
    RunError,
    RunRecord,
    load_baseline_record,
    load_run_record,
    load_run_record_for_owner,
)
from .verification_evidence import baseline_verification_evidence_problem
from .workspace_guard import WorkspaceGuardInspection

_TICKET_PLACEHOLDER = "{{SNAPSHOTTED_TICKET}}"
_IMPLEMENTATION_PROMPT_TEMPLATE = (
    Path(__file__).resolve().parent / "application" / "prompts" / "implement.md"
)


class ImplementationError(RunError):
    """Raised when the implementation stage cannot be prepared."""


@dataclass(frozen=True)
class _ImplementationSafetyViolation:
    name: str
    expected: str
    actual: str
    message: str


@dataclass(frozen=True)
class ImplementationStageResult:
    run_dir: Path
    run_record: RunRecord
    outcome: StageOutcome
    artifact_directory: Path
    agent_execution: AgentExecution[ImplementationResult] | None
    agent_result: ImplementationResult | None
    safety_violations: tuple[_ImplementationSafetyViolation, ...]
    changed_files: tuple[str, ...]
    workspace_guard: WorkspaceGuardInspection | None
    controller_message: str
    after_workspace_fingerprint: str | None
    process_started: bool

    @property
    def source_state(self) -> WorkflowState:
        return WorkflowState.IMPLEMENTING

    @property
    def successful(self) -> bool:
        return self.outcome == StageOutcome.COMPLETED


def run_implementation_stage(
    config: AppConfig,
    run_dir: Path | str,
    *,
    agent_executor: AgentExecutor,
    attempt_record: StageAttempt | None = None,
    clock: Callable[[], datetime] | None = None,
    run_ownership: RunOwnership | None = None,
) -> ImplementationStageResult:
    run_path = Path(run_dir)
    if run_ownership is not None:
        run_path = run_ownership.validate_run_path(run_path)
    if attempt_record is not None:
        return _run_implementation_stage(
            config,
            run_path,
            agent_executor=agent_executor,
            attempt_record=attempt_record,
            clock=clock,
            run_ownership=run_ownership,
        )
    run_record = (
        load_run_record(run_path / RUN_RECORD_FILE)
        if run_ownership is None
        else load_run_record_for_owner(run_ownership)
    )
    if run_record.state is not WorkflowState.IMPLEMENTING:
        raise ImplementationError(
            "Implementation requires run state IMPLEMENTING; "
            f"found {run_record.state.value}."
        )
    owned_attempt = start_attempt(
        run_path,
        phase=AttemptPhase.IMPLEMENTING,
        before_workspace_fingerprint=None,
        clock=clock,
        run_ownership=run_ownership,
    )
    result = _run_implementation_stage(
        config,
        run_path,
        agent_executor=agent_executor,
        attempt_record=StageAttempt.from_record(owned_attempt),
        clock=clock,
        run_ownership=run_ownership,
    )
    complete_stage_attempt(
        run_path,
        owned_attempt,
        stage_outcome=result.outcome,
        after_workspace_fingerprint=result.after_workspace_fingerprint,
        process_started=result.process_started,
        metadata=AttemptMetadata(controller_message=result.controller_message),
        clock=clock,
        run_ownership=run_ownership,
    )
    return result


def _run_implementation_stage(
    config: AppConfig,
    run_dir: Path | str,
    *,
    agent_executor: AgentExecutor,
    attempt_record: StageAttempt,
    clock: Callable[[], datetime] | None = None,
    run_ownership: RunOwnership | None = None,
) -> ImplementationStageResult:
    run_path = Path(run_dir)
    if run_ownership is not None:
        run_path = run_ownership.validate_run_path(run_path)
    run_record = (
        load_run_record(run_path / RUN_RECORD_FILE)
        if run_ownership is None
        else load_run_record_for_owner(run_ownership)
    )
    # Existing runs execute exclusively from the policy captured in run.json.
    config = config_from_resolved_run_policy(run_record.resolved_policy)
    if run_record.state != WorkflowState.IMPLEMENTING:
        raise ImplementationError(
            "Implementation requires run state IMPLEMENTING; "
            f"found {run_record.state.value}."
        )
    attempt = require_stage_attempt(
        run_path,
        attempt_record,
        phase=AttemptPhase.IMPLEMENTING,
        run_ownership=run_ownership,
    )

    baseline_record = load_baseline_record(run_path / BASELINE_RECORD_FILE)
    if baseline_record.branch != run_record.starting_branch:
        raise ImplementationError("Run record and baseline branch do not match.")
    if baseline_record.head_sha != run_record.baseline_sha:
        raise ImplementationError("Run record and baseline HEAD do not match.")

    repository = GitRepository(Path(run_record.target_repository_path))
    ticket_text = _read_snapshotted_ticket(run_path / RUN_TICKET_FILE)
    prompt = _render_implementation_prompt(ticket_text)
    implementation_dir = attempt.artifact_directory
    artifact_layout = attempt_artifact_layout(
        run_path,
        attempt,
        run_ownership=run_ownership,
    )
    baseline = WritableBaseline(
        repository_path=repository.path,
        branch=run_record.starting_branch,
        head_sha=run_record.baseline_sha,
        expected_workspace_fingerprint=baseline_record.workspace_fingerprint,
        require_clean_worktree=True,
    )
    writable_operation = GuardedWritableOperation(
        agent_executor,
        clock=clock,
        run_ownership=run_ownership,
    )
    evidence_problem = baseline_verification_evidence_problem(
        run_path,
        run_record,
        baseline_record,
        verification_commands=config.verification.commands,
    )
    if evidence_problem is not None:
        rejected = writable_operation.reject_before_start(
            GuardedWritableRejectionRequest(
                phase=AttemptPhase.IMPLEMENTING,
                artifact_directory=implementation_dir,
                artifact_layout=artifact_layout,
                baseline=baseline,
                failure_message=evidence_problem,
                safety_violations=(
                    WritableSafetyViolation(
                        name="baseline-verification",
                        expected="persisted passing clean-baseline verification",
                        actual=evidence_problem,
                        message="Writable implementation is not authorized.",
                    ),
                ),
            )
        )
        audit = rejected.audit
        return _finish(
            run_record=run_record,
            run_dir=run_path,
            artifact_directory=implementation_dir,
            artifact_layout=artifact_layout,
            execution=None,
            agent_result=None,
            safety_violations=_implementation_violations(audit.safety_violations),
            changed_files=audit.changed_files,
            workspace_guard=audit.workspace_guard,
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message=evidence_problem,
            after_workspace_fingerprint=audit.after_workspace_fingerprint,
            process_started=False,
            run_ownership=run_ownership,
        )

    active_record = run_record
    execution_request = AgentExecutionRequest(
        task_kind=AgentTaskKind.IMPLEMENTATION,
        repository_path=repository.path,
        repository_access=RepositoryAccess.WORKSPACE_WRITE,
        prompt=prompt,
        result_contract=IMPLEMENTATION_RESULT_CONTRACT,
        artifact_directory=implementation_dir,
        artifact_layout=artifact_layout,
        policy=run_record.resolved_policy.task_policy(
            AgentTaskKind.IMPLEMENTATION
        ).execution_policy,
        required_capabilities=required_execution_capabilities(
            RepositoryAccess.WORKSPACE_WRITE
        ),
    )
    guarded_request = GuardedWritableRequest(
        phase=AttemptPhase.IMPLEMENTING,
        execution_request=execution_request,
        baseline=baseline,
    )
    writable_outcome = writable_operation.execute(guarded_request)
    audit = writable_outcome.audit
    workspace_guard = audit.workspace_guard
    safety_violations = _implementation_violations(audit.safety_violations)

    if isinstance(writable_outcome, WritableSafetyStopped):
        message = (
            format_writable_guard_stop(
                workspace_guard,
                operation="implementation",
                run_dir=run_path,
                git_safety=_format_failure_safety(safety_violations),
            )
            if workspace_guard.requires_human
            else "Repository no longer matches the clean implementation baseline."
            if audit.execution is None
            else "Repository safety invariants were violated."
        )
        return _finish(
            run_record=active_record,
            run_dir=run_path,
            artifact_directory=implementation_dir,
            artifact_layout=artifact_layout,
            execution=audit.execution,
            agent_result=(
                audit.execution.result
                if audit.execution is not None and audit.execution.successful
                else None
            ),
            safety_violations=safety_violations,
            changed_files=audit.changed_files,
            workspace_guard=workspace_guard,
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message=message,
            after_workspace_fingerprint=audit.after_workspace_fingerprint,
            process_started=audit.invocation_start is InvocationStart.STARTED,
            run_ownership=run_ownership,
        )

    if isinstance(
        writable_outcome,
        (WritableRejectedBeforeStart, WritableFailedUnchanged, WritableFailedUncertain),
    ):
        outcome = (
            StageOutcome.HUMAN_REQUIRED
            if isinstance(writable_outcome, WritableFailedUncertain)
            else StageOutcome.FAILED
        )
        message = (
            (
                "The writable agent invocation did not complete successfully and may "
                "have left partial source changes. Automation has stopped for human "
                "inspection. "
            )
            + format_writable_failure_audit(
                audit,
                operation="implementation",
                run_dir=run_path,
            )
            if outcome == StageOutcome.HUMAN_REQUIRED
            else audit.failure_message or "Agent execution failed."
        )
        return _finish(
            run_record=active_record,
            run_dir=run_path,
            artifact_directory=implementation_dir,
            artifact_layout=artifact_layout,
            execution=audit.execution,
            agent_result=None,
            safety_violations=safety_violations,
            changed_files=audit.changed_files,
            workspace_guard=workspace_guard,
            outcome=outcome,
            controller_message=message,
            after_workspace_fingerprint=audit.after_workspace_fingerprint,
            process_started=audit.invocation_start is InvocationStart.STARTED,
            run_ownership=run_ownership,
        )

    if not isinstance(writable_outcome, WritableSucceeded):
        raise ImplementationError("Unknown guarded writable-operation outcome.")
    execution = audit.execution
    assert execution is not None
    agent_result = _require_agent_result(writable_outcome.result)
    changed_files = audit.changed_files
    if agent_result.status is ImplementationStatus.BLOCKED:
        return _finish(
            run_record=active_record,
            run_dir=run_path,
            artifact_directory=implementation_dir,
            artifact_layout=artifact_layout,
            execution=execution,
            agent_result=agent_result,
            safety_violations=(),
            changed_files=changed_files,
            workspace_guard=workspace_guard,
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message="Implementation agent returned BLOCKED.",
            after_workspace_fingerprint=audit.after_workspace_fingerprint,
            process_started=True,
            run_ownership=run_ownership,
        )

    if not changed_files:
        return _finish(
            run_record=active_record,
            run_dir=run_path,
            artifact_directory=implementation_dir,
            artifact_layout=artifact_layout,
            execution=execution,
            agent_result=agent_result,
            safety_violations=(),
            changed_files=(),
            workspace_guard=workspace_guard,
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message="Implementation completed without repository changes.",
            after_workspace_fingerprint=audit.after_workspace_fingerprint,
            process_started=True,
            run_ownership=run_ownership,
        )

    return _finish(
        run_record=active_record,
        run_dir=run_path,
        artifact_directory=implementation_dir,
        artifact_layout=artifact_layout,
        execution=execution,
        agent_result=agent_result,
        safety_violations=(),
        changed_files=changed_files,
        workspace_guard=workspace_guard,
        outcome=StageOutcome.COMPLETED,
        controller_message="Implementation completed and Git safety checks passed.",
        after_workspace_fingerprint=audit.after_workspace_fingerprint,
        process_started=True,
        run_ownership=run_ownership,
    )


def _render_implementation_prompt(ticket_text: str) -> str:
    template = _IMPLEMENTATION_PROMPT_TEMPLATE.read_text(encoding="utf-8")
    if _TICKET_PLACEHOLDER not in template:
        raise ImplementationError(
            f"Implementation prompt template is missing {_TICKET_PLACEHOLDER}."
        )
    return template.replace(_TICKET_PLACEHOLDER, ticket_text)


def _finish(
    *,
    run_record: RunRecord,
    run_dir: Path,
    artifact_directory: Path,
    artifact_layout: AttemptArtifactLayout,
    execution: AgentExecution[ImplementationResult] | None,
    agent_result: ImplementationResult | None,
    safety_violations: tuple[_ImplementationSafetyViolation, ...],
    changed_files: tuple[str, ...],
    workspace_guard: WorkspaceGuardInspection | None = None,
    outcome: StageOutcome,
    controller_message: str,
    after_workspace_fingerprint: str | None = None,
    process_started: bool | None = None,
    run_ownership: RunOwnership | None = None,
) -> ImplementationStageResult:
    if run_ownership is not None:
        run_ownership.validate_run_path(run_dir)
        run_ownership.validate_descendant(artifact_directory)
    artifact_layout.revalidate()
    _write_agent_result(
        artifact_layout,
        agent_result,
        outcome=outcome,
        controller_message=controller_message,
    )
    process_started = (
        process_started
        if process_started is not None
        else False
        if execution is None
        else execution.invocation_start is InvocationStart.STARTED
    )
    return ImplementationStageResult(
        run_dir=run_dir,
        run_record=run_record,
        outcome=outcome,
        artifact_directory=artifact_directory,
        agent_execution=execution,
        agent_result=agent_result,
        safety_violations=safety_violations,
        changed_files=changed_files,
        workspace_guard=workspace_guard,
        controller_message=controller_message,
        after_workspace_fingerprint=after_workspace_fingerprint,
        process_started=process_started,
    )


def _write_agent_result(
    artifact_layout: AttemptArtifactLayout,
    result: ImplementationResult | None,
    *,
    outcome: StageOutcome,
    controller_message: str,
) -> None:
    path = artifact_layout.path(ATTEMPT_RESULT_ARTIFACT_NAME)
    if result is not None:
        write_implementation_result(path, result)
    else:
        write_stage_message(path, status=outcome, message=controller_message)


def _read_snapshotted_ticket(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except OSError as error:
        raise ImplementationError(
            f"Could not read snapshotted ticket: {path}: {error}"
        ) from error
    except UnicodeDecodeError as error:
        raise ImplementationError(
            f"Snapshotted ticket must be valid UTF-8 Markdown: {path}"
        ) from error


def _require_agent_result(value: Any) -> ImplementationResult:
    if not isinstance(value, ImplementationResult):
        raise ImplementationError(
            "Implementation agent result has the wrong domain type."
        )
    return value


def _implementation_violations(
    violations: tuple[WritableSafetyViolation, ...],
) -> tuple[_ImplementationSafetyViolation, ...]:
    return tuple(
        _ImplementationSafetyViolation(
            name=(
                "baseline-workspace"
                if violation.name == "workspace-fingerprint"
                else violation.name
            ),
            expected=violation.expected,
            actual=violation.actual,
            message=violation.message,
        )
        for violation in violations
    )


def _format_failure_safety(
    violations: tuple[_ImplementationSafetyViolation, ...],
) -> str:
    if not violations:
        return "branch unchanged; HEAD unchanged; staging empty"
    return "; ".join(
        f"{violation.name} expected {violation.expected}, got {violation.actual}"
        for violation in violations
    )


__all__ = [
    "ImplementationError",
    "run_implementation_stage",
]
