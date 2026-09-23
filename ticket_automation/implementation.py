from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ._verification_artifacts import _baseline_verification_evidence_problem
from .application.agent_execution import (
    IMPLEMENTATION_RESULT_CONTRACT,
    AgentExecution,
    AgentExecutionPolicy,
    AgentExecutionRequest,
    AgentExecutor,
    AgentTaskKind,
    NetworkAccess,
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
from .attempts import finish_phase_attempt, start_attempt
from .config import AppConfig
from .domain.task_results import ImplementationResult, ImplementationStatus
from .git import GitRepository
from .models import (
    ATTEMPT_RESULT_ARTIFACT_NAME,
    AttemptPhase,
    StageOutcome,
    WorkflowState,
)
from .resolved_config import config_from_resolved_run_policy
from .runs import (
    BASELINE_RECORD_FILE,
    RUN_RECORD_FILE,
    RUN_TICKET_FILE,
    RunError,
    RunRecord,
    load_baseline_record,
    load_run_record,
)
from .task_result_codecs import encode_implementation_result
from .workspace_guard import WorkspaceGuardInspection

_TICKET_PLACEHOLDER = "{{SNAPSHOTTED_TICKET}}"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_IMPLEMENTATION_PROMPT_TEMPLATE = _PROJECT_ROOT / "prompts" / "implement.md"
_AGENT_TIMEOUT_SECONDS = 60 * 60


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

    @property
    def successful(self) -> bool:
        return self.outcome == StageOutcome.COMPLETED


def run_implementation_stage(
    config: AppConfig,
    run_dir: Path | str,
    *,
    agent_executor: AgentExecutor,
    clock: Callable[[], datetime] | None = None,
) -> ImplementationStageResult:
    run_path = Path(run_dir)
    run_record = load_run_record(run_path / RUN_RECORD_FILE)
    # Existing runs execute exclusively from the policy captured in run.json.
    config = config_from_resolved_run_policy(run_record.resolved_policy)
    if run_record.state != WorkflowState.IMPLEMENTING:
        raise ImplementationError(
            "Implementation requires run state IMPLEMENTING; "
            f"found {run_record.state.value}."
        )

    baseline_record = load_baseline_record(run_path / BASELINE_RECORD_FILE)
    if baseline_record.branch != run_record.starting_branch:
        raise ImplementationError("Run record and baseline branch do not match.")
    if baseline_record.head_sha != run_record.baseline_sha:
        raise ImplementationError("Run record and baseline HEAD do not match.")

    repository = GitRepository(Path(run_record.target_repository_path))
    ticket_text = _read_snapshotted_ticket(run_path / RUN_TICKET_FILE)
    prompt = _render_implementation_prompt(ticket_text)
    attempt_record = start_attempt(
        run_path,
        phase=AttemptPhase.IMPLEMENTING,
        before_workspace_fingerprint=None,
        clock=clock,
    )
    implementation_dir = attempt_record.artifact_directory
    baseline = WritableBaseline(
        repository_path=repository.path,
        branch=run_record.starting_branch,
        head_sha=run_record.baseline_sha,
        expected_workspace_fingerprint=baseline_record.workspace_fingerprint,
        require_clean_worktree=True,
    )
    writable_operation = GuardedWritableOperation(agent_executor, clock=clock)
    evidence_problem = _baseline_verification_evidence_problem(
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
            execution=None,
            agent_result=None,
            safety_violations=_implementation_violations(audit.safety_violations),
            changed_files=audit.changed_files,
            workspace_guard=audit.workspace_guard,
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message=evidence_problem,
            after_workspace_fingerprint=audit.after_workspace_fingerprint,
            invocation_started=False,
        )

    active_record = run_record
    execution_request = AgentExecutionRequest(
        task_kind=AgentTaskKind.IMPLEMENTATION,
        repository_path=repository.path,
        repository_access=RepositoryAccess.WORKSPACE_WRITE,
        prompt=prompt,
        result_contract=IMPLEMENTATION_RESULT_CONTRACT,
        artifact_directory=implementation_dir,
        policy=AgentExecutionPolicy(
            timeout_seconds=_AGENT_TIMEOUT_SECONDS,
            network_access=NetworkAccess.ALLOWED,
        ),
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
            invocation_started=audit.invocation_started,
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
            execution=audit.execution,
            agent_result=None,
            safety_violations=safety_violations,
            changed_files=audit.changed_files,
            workspace_guard=workspace_guard,
            outcome=outcome,
            controller_message=message,
            after_workspace_fingerprint=audit.after_workspace_fingerprint,
            invocation_started=audit.invocation_started,
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
            execution=execution,
            agent_result=agent_result,
            safety_violations=(),
            changed_files=changed_files,
            workspace_guard=workspace_guard,
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message="Implementation agent returned BLOCKED.",
            after_workspace_fingerprint=audit.after_workspace_fingerprint,
            invocation_started=True,
        )

    if not changed_files:
        return _finish(
            run_record=active_record,
            run_dir=run_path,
            artifact_directory=implementation_dir,
            execution=execution,
            agent_result=agent_result,
            safety_violations=(),
            changed_files=(),
            workspace_guard=workspace_guard,
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message="Implementation completed without repository changes.",
            after_workspace_fingerprint=audit.after_workspace_fingerprint,
            invocation_started=True,
        )

    return _finish(
        run_record=active_record,
        run_dir=run_path,
        artifact_directory=implementation_dir,
        execution=execution,
        agent_result=agent_result,
        safety_violations=(),
        changed_files=changed_files,
        workspace_guard=workspace_guard,
        outcome=StageOutcome.COMPLETED,
        controller_message="Implementation completed and Git safety checks passed.",
        after_workspace_fingerprint=audit.after_workspace_fingerprint,
        invocation_started=True,
    )


def _render_implementation_prompt(ticket_text: str) -> str:
    template = _IMPLEMENTATION_PROMPT_TEMPLATE.read_text(encoding="utf-8")
    if _TICKET_PLACEHOLDER not in template:
        raise ImplementationError(
            f"Implementation prompt template is missing {_TICKET_PLACEHOLDER}."
        )
    return template.replace(_TICKET_PLACEHOLDER, ticket_text)


def format_implementation_result(result: ImplementationStageResult) -> str:
    rows = [
        f"Implementation state: {result.run_record.state.value}",
        f"Artifacts: {result.artifact_directory}",
        result.controller_message,
    ]
    if result.workspace_guard is not None and result.workspace_guard.requires_human:
        if result.workspace_guard.artifact_path is not None:
            rows.append(f"Workspace guard: {result.workspace_guard.artifact_path}")
        if result.workspace_guard.has_violation:
            rows.append("Workspace hygiene violations:")
            rows.extend(
                "  - "
                f"{environment.root_path.relative_to(result.workspace_guard.after.repository_path)}: "
                f"marker {environment.primary_marker_path.relative_to(result.workspace_guard.after.repository_path)}"
                for environment in result.workspace_guard.new_environments
            )
        if result.workspace_guard.has_inspection_failure:
            rows.append("Workspace environment inspection was incomplete.")
    if result.agent_execution is not None and not result.agent_execution.successful:
        rows.extend(_format_agent_artifacts(result.agent_execution))
    if result.safety_violations:
        rows.append("Safety violations:")
        rows.extend(
            f"  - {violation.name}: expected {violation.expected}, got {violation.actual}"
            for violation in result.safety_violations
        )
    return "\n".join(rows)


def _finish(
    *,
    run_record: RunRecord,
    run_dir: Path,
    artifact_directory: Path,
    execution: AgentExecution[ImplementationResult] | None,
    agent_result: ImplementationResult | None,
    safety_violations: tuple[_ImplementationSafetyViolation, ...],
    changed_files: tuple[str, ...],
    workspace_guard: WorkspaceGuardInspection | None = None,
    outcome: StageOutcome,
    controller_message: str,
    after_workspace_fingerprint: str | None = None,
    invocation_started: bool | None = None,
) -> ImplementationStageResult:
    _write_agent_result(
        artifact_directory,
        agent_result,
        outcome=outcome,
        controller_message=controller_message,
    )
    finish_phase_attempt(
        run_dir,
        phase=AttemptPhase.IMPLEMENTING,
        stage_outcome=outcome,
        after_workspace_fingerprint=after_workspace_fingerprint,
        process_started=(
            invocation_started
            if invocation_started is not None
            else None
            if execution is None
            else execution.invocation_started
        ),
        execution_path=None,
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
    )


def _write_agent_result(
    artifact_directory: Path,
    result: ImplementationResult | None,
    *,
    outcome: StageOutcome,
    controller_message: str,
) -> None:
    path = artifact_directory / ATTEMPT_RESULT_ARTIFACT_NAME
    payload = (
        encode_implementation_result(result)
        if result is not None
        else {"status": outcome.value, "message": controller_message}
    )
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


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


def _format_agent_artifacts(
    execution: AgentExecution[ImplementationResult],
) -> list[str]:
    if not execution.artifacts:
        return ["Agent artifacts: none"]
    return [
        "Agent artifacts:",
        *(f"  - {artifact.name}: {artifact.path}" for artifact in execution.artifacts),
    ]


__all__ = [
    "ImplementationError",
    "format_implementation_result",
    "run_implementation_stage",
]
