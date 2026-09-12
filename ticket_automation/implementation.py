from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .audit import (
    changed_files_including_untracked as _changed_files_including_untracked,
)
from .audit import (
    diff_including_untracked as _diff_including_untracked,
)
from .audit import (
    diff_stats_including_untracked as _diff_stats_including_untracked,
)
from .codex import (
    CodexExecution,
    CodexExecutionFailure,
    CodexProcessRunner,
    Sandbox,
)
from .codex import (
    execute as execute_codex,
)
from .config import AppConfig
from .git import GitCommandError, GitRepository
from .models import WorkflowState
from .runs import (
    BASELINE_RECORD_FILE,
    RUN_RECORD_FILE,
    RUN_TICKET_FILE,
    RunError,
    RunRecord,
    load_baseline_record,
    load_run_record,
    save_run_record,
)
from .workspace_guard import (
    WorkspaceGuardInspection,
    capture_workspace_environment_snapshot,
    format_workspace_hygiene_violation,
    implementation_guard_artifact_path,
    inspect_workspace_environment_change,
)


class _SafetyInspectionPhase:
    BEFORE_IMPLEMENTATION = "before implementation"
    AFTER_IMPLEMENTATION = "after implementation"


_IMPLEMENTATION_DIR_NAME = "implementation"
_DIFFS_DIR_NAME = "diffs"
_AFTER_IMPLEMENTATION_PATCH_FILE = "after-implementation.patch"
_AFTER_IMPLEMENTATION_STATS_FILE = "after-implementation.stat"
_FAILED_IMPLEMENTATION_PATCH_FILE = "failed-implementation.patch"
_FAILED_IMPLEMENTATION_STATS_FILE = "failed-implementation.stat"
_TICKET_PLACEHOLDER = "{{SNAPSHOTTED_TICKET}}"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_IMPLEMENTATION_PROMPT_TEMPLATE = _PROJECT_ROOT / "prompts" / "implement.md"
_IMPLEMENTATION_RESULT_SCHEMA = (
    _PROJECT_ROOT / "schemas" / "implementation-result.schema.json"
)
_TERMINAL_STATES = frozenset(
    {
        WorkflowState.READY_FOR_HUMAN,
        WorkflowState.HUMAN_REQUIRED,
        WorkflowState.FAILED,
    }
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
    artifact_directory: Path
    codex_execution: CodexExecution | None
    agent_result: dict[str, Any] | None
    safety_violations: tuple[_ImplementationSafetyViolation, ...]
    changed_files: tuple[str, ...]
    patch_path: Path | None
    diff_stats_path: Path | None
    workspace_guard: WorkspaceGuardInspection | None
    controller_message: str

    @property
    def successful(self) -> bool:
        return self.run_record.state == WorkflowState.IMPLEMENT


@dataclass(frozen=True)
class _FailedWritableAudit:
    safety_violations: tuple[_ImplementationSafetyViolation, ...]
    changed_files: tuple[str, ...]
    patch_path: Path | None
    diff_stats_path: Path | None
    patch_error: str | None
    workspace_guard: WorkspaceGuardInspection | None
    human_required: bool


def run_implementation_stage(
    config: AppConfig,
    run_dir: Path | str,
    *,
    codex_runner: CodexProcessRunner | None = None,
    clock: Callable[[], datetime] | None = None,
) -> ImplementationStageResult:
    run_path = Path(run_dir)
    run_record_path = run_path / RUN_RECORD_FILE
    run_record = load_run_record(run_record_path)
    if run_record.state != WorkflowState.SNAPSHOT:
        raise ImplementationError(
            f"Implementation requires run state SNAPSHOT; found {run_record.state.value}."
        )

    baseline_record = load_baseline_record(run_path / BASELINE_RECORD_FILE)
    if baseline_record.branch != run_record.starting_branch:
        raise ImplementationError("Run record and baseline branch do not match.")
    if baseline_record.head_sha != run_record.baseline_sha:
        raise ImplementationError("Run record and baseline HEAD do not match.")

    ticket_text = _read_snapshotted_ticket(run_path / RUN_TICKET_FILE)
    repository = GitRepository(Path(run_record.target_repository_path))
    implementation_dir = run_path / _IMPLEMENTATION_DIR_NAME

    starting_violations = _inspect_starting_state(repository, run_record)
    if starting_violations:
        return _finish(
            run_record=run_record,
            run_record_path=run_record_path,
            run_dir=run_path,
            artifact_directory=implementation_dir,
            execution=None,
            agent_result=None,
            safety_violations=starting_violations,
            changed_files=(),
            patch_path=None,
            diff_stats_path=None,
            state=WorkflowState.HUMAN_REQUIRED,
            controller_message="Repository no longer matches the clean implementation baseline.",
            clock=clock,
        )

    active_record = run_record.with_state(
        WorkflowState.IMPLEMENT,
        updated_timestamp=_timestamp(clock),
        last_completed_state=WorkflowState.SNAPSHOT,
    )
    save_run_record(active_record, run_record_path)
    prompt = _render_implementation_prompt(ticket_text)
    environment_snapshot = capture_workspace_environment_snapshot(repository.path)

    try:
        execution = execute_codex(
            prompt=prompt,
            repo_path=repository.path,
            sandbox=Sandbox.WORKSPACE_WRITE,
            output_schema=_IMPLEMENTATION_RESULT_SCHEMA,
            artifact_directory=implementation_dir,
            executable=config.codex.executable,
            runner=codex_runner,
        )
    except CodexExecutionFailure as error:
        workspace_guard = inspect_workspace_environment_change(
            before=environment_snapshot,
            phase=WorkflowState.IMPLEMENT.value,
            artifact_path=implementation_guard_artifact_path(run_path),
            clock=clock,
        )
        failed_audit = _audit_failed_writable_invocation(
            repository,
            run_path,
            active_record,
            execution=error.execution,
            workspace_guard=workspace_guard,
        )
        state = (
            WorkflowState.HUMAN_REQUIRED
            if failed_audit.human_required
            else WorkflowState.FAILED
        )
        message = (
            _failed_writable_message(
                operation="implementation",
                execution=error.execution,
                audit=failed_audit,
                run_dir=run_path,
            )
            if failed_audit.human_required
            else error.execution.failure_message or str(error)
        )
        return _finish(
            run_record=active_record,
            run_record_path=run_record_path,
            run_dir=run_path,
            artifact_directory=implementation_dir,
            execution=error.execution,
            agent_result=None,
            safety_violations=failed_audit.safety_violations,
            changed_files=failed_audit.changed_files,
            patch_path=failed_audit.patch_path,
            diff_stats_path=failed_audit.diff_stats_path,
            workspace_guard=workspace_guard,
            state=state,
            controller_message=message,
            clock=clock,
            last_completed_state=active_record.last_completed_state,
        )

    agent_result = _require_agent_result(execution.structured_result)
    workspace_guard = inspect_workspace_environment_change(
        before=environment_snapshot,
        phase=WorkflowState.IMPLEMENT.value,
        artifact_path=implementation_guard_artifact_path(run_path),
        clock=clock,
    )
    safety_violations = _inspect_safety(
        repository,
        active_record,
        phase=_SafetyInspectionPhase.AFTER_IMPLEMENTATION,
    )
    if workspace_guard.has_violation:
        return _finish(
            run_record=active_record,
            run_record_path=run_record_path,
            run_dir=run_path,
            artifact_directory=implementation_dir,
            execution=execution,
            agent_result=agent_result,
            safety_violations=safety_violations,
            changed_files=(),
            patch_path=None,
            diff_stats_path=None,
            workspace_guard=workspace_guard,
            state=WorkflowState.HUMAN_REQUIRED,
            controller_message=format_workspace_hygiene_violation(
                workspace_guard,
                operation="implementation",
                run_dir=run_path,
                git_safety=_format_failure_safety(safety_violations),
            ),
            clock=clock,
        )
    if safety_violations:
        return _finish(
            run_record=active_record,
            run_record_path=run_record_path,
            run_dir=run_path,
            artifact_directory=implementation_dir,
            execution=execution,
            agent_result=agent_result,
            safety_violations=safety_violations,
            changed_files=(),
            patch_path=None,
            diff_stats_path=None,
            workspace_guard=workspace_guard,
            state=WorkflowState.HUMAN_REQUIRED,
            controller_message="Repository safety invariants were violated.",
            clock=clock,
        )

    try:
        changed_files = _changed_files(repository, active_record.baseline_sha)
    except ImplementationError as error:
        return _finish(
            run_record=active_record,
            run_record_path=run_record_path,
            run_dir=run_path,
            artifact_directory=implementation_dir,
            execution=execution,
            agent_result=agent_result,
            safety_violations=(),
            changed_files=(),
            patch_path=None,
            diff_stats_path=None,
            workspace_guard=workspace_guard,
            state=WorkflowState.HUMAN_REQUIRED,
            controller_message=str(error),
            clock=clock,
        )
    agent_status = agent_result["status"]
    if agent_status == "BLOCKED":
        return _finish(
            run_record=active_record,
            run_record_path=run_record_path,
            run_dir=run_path,
            artifact_directory=implementation_dir,
            execution=execution,
            agent_result=agent_result,
            safety_violations=(),
            changed_files=changed_files,
            patch_path=None,
            diff_stats_path=None,
            workspace_guard=workspace_guard,
            state=WorkflowState.HUMAN_REQUIRED,
            controller_message="Implementation agent returned BLOCKED.",
            clock=clock,
        )

    if not changed_files:
        return _finish(
            run_record=active_record,
            run_record_path=run_record_path,
            run_dir=run_path,
            artifact_directory=implementation_dir,
            execution=execution,
            agent_result=agent_result,
            safety_violations=(),
            changed_files=(),
            patch_path=None,
            diff_stats_path=None,
            workspace_guard=workspace_guard,
            state=WorkflowState.HUMAN_REQUIRED,
            controller_message="Implementation completed without repository changes.",
            clock=clock,
        )

    try:
        patch_path, diff_stats_path = _capture_diff(
            repository,
            run_path,
            active_record.baseline_sha,
        )
    except ImplementationError as error:
        return _finish(
            run_record=active_record,
            run_record_path=run_record_path,
            run_dir=run_path,
            artifact_directory=implementation_dir,
            execution=execution,
            agent_result=agent_result,
            safety_violations=(),
            changed_files=changed_files,
            patch_path=None,
            diff_stats_path=None,
            workspace_guard=workspace_guard,
            state=WorkflowState.HUMAN_REQUIRED,
            controller_message=str(error),
            clock=clock,
        )
    return _finish(
        run_record=active_record,
        run_record_path=run_record_path,
        run_dir=run_path,
        artifact_directory=implementation_dir,
        execution=execution,
        agent_result=agent_result,
        safety_violations=(),
        changed_files=changed_files,
        patch_path=patch_path,
        diff_stats_path=diff_stats_path,
        workspace_guard=workspace_guard,
        state=WorkflowState.IMPLEMENT,
        controller_message="Implementation completed and Git safety checks passed.",
        clock=clock,
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
    if result.patch_path is not None:
        rows.append(f"Patch: {result.patch_path}")
    if result.diff_stats_path is not None:
        rows.append(f"Diff stats: {result.diff_stats_path}")
    if result.workspace_guard is not None and result.workspace_guard.has_violation:
        rows.append(f"Workspace guard: {result.workspace_guard.artifact_path}")
        rows.append("Workspace hygiene violations:")
        rows.extend(
            "  - "
            f"{environment.root_path.relative_to(result.workspace_guard.after.repository_path)}: "
            f"marker {environment.primary_marker_path.relative_to(result.workspace_guard.after.repository_path)}"
            for environment in result.workspace_guard.new_environments
        )
    if result.codex_execution is not None and not result.codex_execution.successful:
        rows.append(f"Codex execution: {result.codex_execution.execution_json_path}")
        rows.append(f"Codex events: {result.codex_execution.events_jsonl_path}")
        rows.append(f"Codex stderr: {result.codex_execution.stderr_log_path}")
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
    run_record_path: Path,
    run_dir: Path,
    artifact_directory: Path,
    execution: CodexExecution | None,
    agent_result: dict[str, Any] | None,
    safety_violations: tuple[_ImplementationSafetyViolation, ...],
    changed_files: tuple[str, ...],
    patch_path: Path | None,
    diff_stats_path: Path | None,
    workspace_guard: WorkspaceGuardInspection | None = None,
    state: WorkflowState,
    controller_message: str,
    clock: Callable[[], datetime] | None,
    last_completed_state: WorkflowState = WorkflowState.IMPLEMENT,
) -> ImplementationStageResult:
    updated_record = run_record.with_state(
        state,
        updated_timestamp=_timestamp(clock),
        last_completed_state=last_completed_state,
        terminal_reason=controller_message if state in _TERMINAL_STATES else None,
    )
    save_run_record(updated_record, run_record_path)
    return ImplementationStageResult(
        run_dir=run_dir,
        run_record=updated_record,
        artifact_directory=artifact_directory,
        codex_execution=execution,
        agent_result=agent_result,
        safety_violations=safety_violations,
        changed_files=changed_files,
        patch_path=patch_path,
        diff_stats_path=diff_stats_path,
        workspace_guard=workspace_guard,
        controller_message=controller_message,
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


def _require_agent_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ImplementationError("Implementation agent result must be a JSON object.")
    return value


def _inspect_safety(
    repository: GitRepository,
    run_record: RunRecord,
    *,
    phase: str,
) -> tuple[_ImplementationSafetyViolation, ...]:
    violations: list[_ImplementationSafetyViolation] = []
    try:
        current_branch = repository.current_branch()
        current_head = repository.head_sha()
        staged_files = repository.staged_files()
    except GitCommandError as error:
        return (
            _ImplementationSafetyViolation(
                name="git-inspection",
                expected=f"{phase} Git inspection succeeds",
                actual=str(error),
                message=f"Could not inspect repository {phase}.",
            ),
        )

    if current_branch != run_record.starting_branch:
        violations.append(
            _ImplementationSafetyViolation(
                name="branch",
                expected=run_record.starting_branch,
                actual="<detached>" if current_branch is None else current_branch,
                message=f"Current branch changed {phase}.",
            )
        )
    if current_head != run_record.baseline_sha:
        violations.append(
            _ImplementationSafetyViolation(
                name="HEAD",
                expected=run_record.baseline_sha,
                actual=current_head,
                message=f"HEAD changed {phase}.",
            )
        )
    if staged_files:
        violations.append(
            _ImplementationSafetyViolation(
                name="staging",
                expected="empty",
                actual=", ".join(staged_files[:5]),
                message=f"Staging area is not empty {phase}.",
            )
        )
    return tuple(violations)


def _inspect_starting_state(
    repository: GitRepository,
    run_record: RunRecord,
) -> tuple[_ImplementationSafetyViolation, ...]:
    violations = list(
        _inspect_safety(
            repository,
            run_record,
            phase=_SafetyInspectionPhase.BEFORE_IMPLEMENTATION,
        )
    )
    try:
        changed_files = _worktree_changed_files(repository, run_record.baseline_sha)
    except GitCommandError as error:
        violations.append(
            _ImplementationSafetyViolation(
                name="worktree-inspection",
                expected="clean baseline worktree inspection succeeds",
                actual=str(error),
                message="Could not inspect repository worktree before implementation.",
            )
        )
        return tuple(violations)

    if changed_files:
        violations.append(
            _ImplementationSafetyViolation(
                name="worktree",
                expected="clean",
                actual=_format_files(changed_files),
                message="Worktree changed before implementation started.",
            )
        )
    return tuple(violations)


def _changed_files(repository: GitRepository, baseline_sha: str) -> tuple[str, ...]:
    try:
        return _worktree_changed_files(repository, baseline_sha)
    except GitCommandError as error:
        raise ImplementationError(
            f"Could not inspect implementation diff: {error}"
        ) from error


def _capture_diff(
    repository: GitRepository,
    run_dir: Path,
    baseline_sha: str,
) -> tuple[Path, Path]:
    diffs_dir = run_dir / _DIFFS_DIR_NAME
    diffs_dir.mkdir(parents=True, exist_ok=True)
    patch_path = diffs_dir / _AFTER_IMPLEMENTATION_PATCH_FILE
    stats_path = diffs_dir / _AFTER_IMPLEMENTATION_STATS_FILE
    try:
        patch = _diff_including_untracked(repository, baseline_sha)
        stats = _diff_stats_including_untracked(repository, baseline_sha)
    except GitCommandError as error:
        raise ImplementationError(
            f"Could not capture implementation diff: {error}"
        ) from error
    patch_path.write_text(patch, encoding="utf-8", newline="\n")
    stats_path.write_text(stats, encoding="utf-8", newline="\n")
    return patch_path, stats_path


def _audit_failed_writable_invocation(
    repository: GitRepository,
    run_path: Path,
    run_record: RunRecord,
    *,
    execution: CodexExecution,
    workspace_guard: WorkspaceGuardInspection | None,
) -> _FailedWritableAudit:
    violations = list(
        _inspect_safety(
            repository,
            run_record,
            phase=_SafetyInspectionPhase.AFTER_IMPLEMENTATION,
        )
    )
    changed_files: tuple[str, ...] = ()
    try:
        changed_files = _worktree_changed_files(repository, run_record.baseline_sha)
    except (GitCommandError, OSError, ValueError) as error:
        violations.append(
            _ImplementationSafetyViolation(
                name="worktree-inspection",
                expected="baseline-relative source diff inspection succeeds",
                actual=str(error),
                message=(
                    "Could not inspect source changes after failed writable Codex "
                    "invocation."
                ),
            )
        )

    if changed_files:
        violations.append(
            _ImplementationSafetyViolation(
                name="worktree",
                expected="no baseline-relative source changes after failed writable invocation",
                actual=_format_files(changed_files),
                message=(
                    "Baseline-relative source changes exist after failed writable "
                    "Codex invocation."
                ),
            )
        )

    human_required = (
        execution.process_started
        or bool(violations)
        or bool(workspace_guard is not None and workspace_guard.has_violation)
    )
    patch_path: Path | None = None
    diff_stats_path: Path | None = None
    patch_error: str | None = None
    if human_required:
        try:
            patch_path, diff_stats_path = _capture_failed_diff(
                repository,
                run_path,
                run_record.baseline_sha,
            )
        except ImplementationError as error:
            patch_error = str(error)

    return _FailedWritableAudit(
        safety_violations=tuple(violations),
        changed_files=changed_files,
        patch_path=patch_path,
        diff_stats_path=diff_stats_path,
        patch_error=patch_error,
        workspace_guard=workspace_guard,
        human_required=human_required,
    )


def _capture_failed_diff(
    repository: GitRepository,
    run_dir: Path,
    baseline_sha: str,
) -> tuple[Path, Path]:
    diffs_dir = run_dir / _DIFFS_DIR_NAME
    patch_path = diffs_dir / _FAILED_IMPLEMENTATION_PATCH_FILE
    stats_path = diffs_dir / _FAILED_IMPLEMENTATION_STATS_FILE
    try:
        diffs_dir.mkdir(parents=True, exist_ok=True)
        patch = _diff_including_untracked(repository, baseline_sha)
        stats = _diff_stats_including_untracked(repository, baseline_sha)
        patch_path.write_text(patch, encoding="utf-8", newline="\n")
        stats_path.write_text(stats, encoding="utf-8", newline="\n")
    except (GitCommandError, OSError, ValueError) as error:
        raise ImplementationError(
            f"Could not capture failed implementation diff: {error}"
        ) from error
    return patch_path, stats_path


def _failed_writable_message(
    *,
    operation: str,
    execution: CodexExecution,
    audit: _FailedWritableAudit,
    run_dir: Path,
) -> str:
    rows = [
        (
            "The writable Codex invocation did not complete successfully and may "
            "have left partial source changes. Automation has stopped for human "
            "inspection."
        ),
        (
            "Codex failure: "
            f"{_format_optional_failure_kind(execution)}: "
            f"{execution.failure_message or 'unknown failure'}"
        ),
        f"Last operation: {operation}",
        f"Process started: {_yes_no(execution.process_started)}",
        f"Execution metadata: {execution.execution_json_path}",
        f"Events: {execution.events_jsonl_path}",
        f"Stderr: {execution.stderr_log_path}",
        f"Git safety: {_format_failure_safety(audit.safety_violations)}",
        f"Changed files relative to baseline: {_format_files(audit.changed_files)}",
    ]
    if audit.patch_path is not None:
        rows.append(f"Baseline-relative failure patch: {audit.patch_path}")
    if audit.diff_stats_path is not None:
        rows.append(f"Failure diff stats: {audit.diff_stats_path}")
    if audit.patch_error is not None:
        rows.append(f"Failure patch capture error: {audit.patch_error}")
    if audit.workspace_guard is not None and audit.workspace_guard.has_violation:
        rows.append(
            format_workspace_hygiene_violation(
                audit.workspace_guard,
                operation=operation,
                run_dir=run_dir,
                git_safety=_format_failure_safety(audit.safety_violations),
            )
        )
    return " ".join(rows)


def _format_failure_safety(
    violations: tuple[_ImplementationSafetyViolation, ...],
) -> str:
    if not violations:
        return "branch unchanged; HEAD unchanged; staging empty"
    return "; ".join(
        f"{violation.name} expected {violation.expected}, got {violation.actual}"
        for violation in violations
    )


def _format_optional_failure_kind(execution: CodexExecution) -> str:
    return "UNKNOWN" if execution.failure_kind is None else execution.failure_kind.value


def _worktree_changed_files(
    repository: GitRepository,
    baseline_sha: str,
) -> tuple[str, ...]:
    return _changed_files_including_untracked(repository, baseline_sha)


def _format_files(files: tuple[str, ...]) -> str:
    if not files:
        return "none"
    shown = ", ".join(files[:5])
    hidden_count = len(files) - 5
    if hidden_count > 0:
        shown = f"{shown}, and {hidden_count} more"
    return shown


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _timestamp(clock: Callable[[], datetime] | None) -> str:
    now = datetime.now(UTC) if clock is None else clock()
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return now.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "ImplementationError",
    "format_implementation_result",
    "run_implementation_stage",
]
