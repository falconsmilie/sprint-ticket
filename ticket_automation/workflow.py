from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from .codex import CodexProcessRunner
from .config import AppConfig, with_codex_execution_settings
from .corrections import (
    CorrectionError,
    CorrectionStageResult,
    VerificationFailure,
    correction_reason_from_dict,
    review_findings_from_result,
    run_correction_stage,
)
from .git import GitCommandError, GitRepository
from .git_safety import (
    WorkspaceSnapshot,
    _read_workspace_fingerprint,
)
from .implementation import (
    ImplementationStageResult,
    run_implementation_stage,
)
from .locking import RepositoryRunLock, acquire_repository_run_lock
from .models import StageOutcome, WorkflowState
from .preflight import PreflightResult
from .reporting import (
    ReportStageResult,
    _latest_writable_workspace_fingerprint_path,
    collect_report_context,
    generate_terminal_report_best_effort,
    inspect_git_safety,
    latest_review_result,
    latest_verification_round,
    run_report_stage,
)
from .review import (
    ReviewStageResult,
    ReviewVerdict,
    run_review_stage,
)
from .runs import (
    BASELINE_RECORD_FILE,
    RUN_RECORD_FILE,
    RUN_TICKET_FILE,
    RunError,
    RunRecord,
    create_run_snapshot,
    load_baseline_record,
    load_run_record,
    save_run_record,
)
from .verification import (
    VerificationProcessRunner,
    VerificationStageResult,
    run_verification_stage,
)

TERMINAL_STATES = frozenset(
    {
        WorkflowState.READY_FOR_HUMAN,
        WorkflowState.HUMAN_REQUIRED,
        WorkflowState.FAILED,
    }
)


@dataclass(frozen=True)
class LifecycleSafetyViolation:
    name: str
    expected: str
    actual: str
    message: str


@dataclass(frozen=True)
class LifecycleResult:
    run_dir: Path
    run_record: RunRecord
    preflight_result: PreflightResult
    implementation_result: ImplementationStageResult | None
    verification_results: tuple[VerificationStageResult, ...]
    review_results: tuple[ReviewStageResult, ...]
    correction_results: tuple[CorrectionStageResult, ...]
    report_result: ReportStageResult | None = None
    safety_violations: tuple[LifecycleSafetyViolation, ...] = ()
    controller_error: str | None = None

    @property
    def successful(self) -> bool:
        return self.run_record.state == WorkflowState.READY_FOR_HUMAN

    @property
    def terminal_state(self) -> WorkflowState:
        return self.run_record.state


def run_ticket_lifecycle(
    config: AppConfig,
    ticket_path: Path | str,
    *,
    runs_dir: Path | str,
    codex_runner: CodexProcessRunner | None = None,
    verification_runner: VerificationProcessRunner | None = None,
    clock: Callable[[], datetime] | None = None,
) -> LifecycleResult:
    with acquire_repository_run_lock(
        config.project.repo,
        run_id=None,
        current_state=WorkflowState.PREPARING.value,
        clock=clock,
    ) as repository_lock:
        return _run_ticket_lifecycle_locked(
            config,
            ticket_path,
            runs_dir=runs_dir,
            repository_lock=repository_lock,
            codex_runner=codex_runner,
            verification_runner=verification_runner,
            clock=clock,
        )


def _run_ticket_lifecycle_locked(
    config: AppConfig,
    ticket_path: Path | str,
    *,
    runs_dir: Path | str,
    repository_lock: RepositoryRunLock,
    codex_runner: CodexProcessRunner | None,
    verification_runner: VerificationProcessRunner | None,
    clock: Callable[[], datetime] | None,
) -> LifecycleResult:
    snapshot = create_run_snapshot(
        config,
        ticket_path,
        runs_dir=runs_dir,
        clock=clock,
    )
    _update_repository_lock(repository_lock, snapshot.run_record)
    implementation_result: ImplementationStageResult | None = None
    verification_results: list[VerificationStageResult] = []
    review_results: list[ReviewStageResult] = []
    correction_results: list[CorrectionStageResult] = []

    try:
        active_record = _persist_requested_transition(
            snapshot.run_dir,
            snapshot.run_record,
            WorkflowState.IMPLEMENTING,
            clock=clock,
        )
        _update_repository_lock(repository_lock, active_record)
        implementation_result = run_implementation_stage(
            config,
            snapshot.run_dir,
            codex_runner=codex_runner,
            clock=clock,
        )
        completed_record = _persist_stage_outcome(
            snapshot.run_dir,
            implementation_result.run_record,
            implementation_result.outcome,
            terminal_reason=implementation_result.controller_message,
            clock=clock,
        )
        implementation_result = replace(
            implementation_result,
            run_record=completed_record,
        )
        _update_repository_lock(repository_lock, implementation_result.run_record)
        return _drive_lifecycle(
            config,
            snapshot.run_dir,
            snapshot.preflight_result,
            implementation_result.run_record,
            implementation_result=implementation_result,
            verification_results=verification_results,
            review_results=review_results,
            correction_results=correction_results,
            codex_runner=codex_runner,
            verification_runner=verification_runner,
            repository_lock=repository_lock,
            clock=clock,
        )
    except RunError as error:
        run_record = _mark_failed(
            snapshot.run_dir,
            terminal_reason=str(error),
            clock=clock,
        )
        _update_repository_lock(repository_lock, run_record)
        generate_terminal_report_best_effort(snapshot.run_dir)
        return LifecycleResult(
            run_dir=snapshot.run_dir,
            run_record=run_record,
            preflight_result=snapshot.preflight_result,
            implementation_result=implementation_result,
            verification_results=tuple(verification_results),
            review_results=tuple(review_results),
            correction_results=tuple(correction_results),
            controller_error=str(error),
        )
    except Exception as error:  # noqa: BLE001 - internal errors must persist FAILED.
        controller_error = (
            f"Internal TicketAutomation exception: {type(error).__name__}: {error}"
        )
        run_record = _mark_failed(
            snapshot.run_dir,
            terminal_reason=controller_error,
            clock=clock,
        )
        _update_repository_lock(repository_lock, run_record)
        generate_terminal_report_best_effort(snapshot.run_dir)
        return LifecycleResult(
            run_dir=snapshot.run_dir,
            run_record=run_record,
            preflight_result=snapshot.preflight_result,
            implementation_result=implementation_result,
            verification_results=tuple(verification_results),
            review_results=tuple(review_results),
            correction_results=tuple(correction_results),
            controller_error=controller_error,
        )


def resume_ticket_lifecycle(
    config: AppConfig,
    run_id: str,
    *,
    runs_dir: Path | str,
    codex_runner: CodexProcessRunner | None = None,
    verification_runner: VerificationProcessRunner | None = None,
    clock: Callable[[], datetime] | None = None,
) -> LifecycleResult:
    run_dir = Path(runs_dir) / run_id
    if not run_dir.is_dir():
        raise RunError(f"Run directory does not exist: {run_dir}")

    preflight_result = PreflightResult(())
    run_record = load_run_record(run_dir / RUN_RECORD_FILE)
    config = with_codex_execution_settings(config, run_record.codex)
    with acquire_repository_run_lock(
        run_record.target_repository_path,
        run_id=run_record.run_id,
        current_state=run_record.state.value,
        clock=clock,
    ) as repository_lock:
        return _resume_ticket_lifecycle_locked(
            config,
            run_dir,
            preflight_result,
            run_record,
            repository_lock=repository_lock,
            codex_runner=codex_runner,
            verification_runner=verification_runner,
            clock=clock,
        )


def _resume_ticket_lifecycle_locked(
    config: AppConfig,
    run_dir: Path,
    preflight_result: PreflightResult,
    run_record: RunRecord,
    *,
    repository_lock: RepositoryRunLock,
    codex_runner: CodexProcessRunner | None,
    verification_runner: VerificationProcessRunner | None,
    clock: Callable[[], datetime] | None,
) -> LifecycleResult:
    if run_record.state in TERMINAL_STATES:
        return LifecycleResult(
            run_dir=run_dir,
            run_record=run_record,
            preflight_result=preflight_result,
            implementation_result=None,
            verification_results=(),
            review_results=(),
            correction_results=(),
        )

    resume_problem = _resume_preflight_problem(run_dir, run_record)
    if resume_problem is not None:
        run_record = _mark_human_required(
            run_dir,
            terminal_reason=resume_problem,
            clock=clock,
        )
        _update_repository_lock(repository_lock, run_record)
        generate_terminal_report_best_effort(run_dir)
        return LifecycleResult(
            run_dir=run_dir,
            run_record=run_record,
            preflight_result=preflight_result,
            implementation_result=None,
            verification_results=(),
            review_results=(),
            correction_results=(),
        )

    checkpoint_problem = _resume_checkpoint_problem(run_dir, run_record)
    if checkpoint_problem is not None:
        run_record = _mark_human_required(
            run_dir,
            terminal_reason=checkpoint_problem,
            clock=clock,
        )
        _update_repository_lock(repository_lock, run_record)
        generate_terminal_report_best_effort(run_dir)
        return LifecycleResult(
            run_dir=run_dir,
            run_record=run_record,
            preflight_result=preflight_result,
            implementation_result=None,
            verification_results=(),
            review_results=(),
            correction_results=(),
        )

    implementation_result: ImplementationStageResult | None = None
    verification_results: list[VerificationStageResult] = []
    review_results: list[ReviewStageResult] = []
    correction_results: list[CorrectionStageResult] = []

    try:
        if run_record.state == WorkflowState.PREPARED:
            run_record = _persist_requested_transition(
                run_dir,
                run_record,
                WorkflowState.IMPLEMENTING,
                clock=clock,
            )
            _update_repository_lock(repository_lock, run_record)
            implementation_result = run_implementation_stage(
                config,
                run_dir,
                codex_runner=codex_runner,
                clock=clock,
            )
            run_record = _persist_stage_outcome(
                run_dir,
                implementation_result.run_record,
                implementation_result.outcome,
                terminal_reason=implementation_result.controller_message,
                clock=clock,
            )
            implementation_result = replace(
                implementation_result,
                run_record=run_record,
            )
            _update_repository_lock(repository_lock, implementation_result.run_record)

        return _drive_lifecycle(
            config,
            run_dir,
            preflight_result,
            run_record,
            implementation_result=implementation_result,
            verification_results=verification_results,
            review_results=review_results,
            correction_results=correction_results,
            codex_runner=codex_runner,
            verification_runner=verification_runner,
            repository_lock=repository_lock,
            clock=clock,
        )
    except RunError as error:
        run_record = _mark_failed(
            run_dir,
            terminal_reason=str(error),
            clock=clock,
        )
        _update_repository_lock(repository_lock, run_record)
        generate_terminal_report_best_effort(run_dir)
        return LifecycleResult(
            run_dir=run_dir,
            run_record=run_record,
            preflight_result=preflight_result,
            implementation_result=implementation_result,
            verification_results=tuple(verification_results),
            review_results=tuple(review_results),
            correction_results=tuple(correction_results),
            controller_error=str(error),
        )
    except Exception as error:  # noqa: BLE001 - internal errors must persist FAILED.
        controller_error = (
            "Internal TicketAutomation exception during resume: "
            f"{type(error).__name__}: {error}"
        )
        run_record = _mark_failed(
            run_dir,
            terminal_reason=controller_error,
            clock=clock,
        )
        _update_repository_lock(repository_lock, run_record)
        generate_terminal_report_best_effort(run_dir)
        return LifecycleResult(
            run_dir=run_dir,
            run_record=run_record,
            preflight_result=preflight_result,
            implementation_result=implementation_result,
            verification_results=tuple(verification_results),
            review_results=tuple(review_results),
            correction_results=tuple(correction_results),
            controller_error=controller_error,
        )


def format_lifecycle_result(result: LifecycleResult) -> str:
    record = result.run_record
    context = _safe_report_context(result.run_dir, record)
    title = f"{record.ticket_id} - {_terminal_label(record.state)}"
    lines = [
        "----------------------------------------",
        title,
        "----------------------------------------",
        "",
        "Run",
        f"  {result.run_dir}",
        "",
        "Branch",
        f"  {record.starting_branch}",
        "",
        "Baseline",
        f"  {record.baseline_sha}",
        "",
        "Files changed",
        f"  {_changed_file_count(context, result)}",
        "",
        "Diff",
        f"  {_diff_line_summary(context, result)}",
        "",
        "Verification:",
        *_verification_lines(context, result),
        "",
        "Review:",
        f"  {_review_summary(context, result)}",
        "",
        "Correction rounds",
        f"  {record.current_correction_round} / {record.max_correction_rounds}",
        "",
        "Review rounds",
        f"  {record.current_review_round}",
        "",
        "Advisory findings",
        f"  {_advisory_count(context)}",
        "",
        "Git safety",
        *_git_safety_lines(result, context),
    ]
    if record.terminal_reason:
        lines.extend(["", "Reason", f"  {record.terminal_reason}"])
    if result.controller_error:
        lines.extend(["", "Controller error", f"  {result.controller_error}"])
    if result.successful:
        lines.extend(["", "No files have been staged or committed."])
    report_path = result.run_dir / "final-report.md"
    if report_path.is_file():
        lines.extend(["", "Report", f"  {report_path}"])
    lines.append("----------------------------------------")
    return "\n".join(lines)


def _drive_lifecycle(
    config: AppConfig,
    run_dir: Path,
    preflight_result: PreflightResult,
    run_record: RunRecord,
    *,
    implementation_result: ImplementationStageResult | None,
    verification_results: list[VerificationStageResult],
    review_results: list[ReviewStageResult],
    correction_results: list[CorrectionStageResult],
    codex_runner: CodexProcessRunner | None,
    verification_runner: VerificationProcessRunner | None,
    repository_lock: RepositoryRunLock,
    clock: Callable[[], datetime] | None,
) -> LifecycleResult:
    report_result: ReportStageResult | None = None

    while run_record.state not in TERMINAL_STATES:
        _update_repository_lock(repository_lock, run_record)
        if run_record.state == WorkflowState.PREPARED:
            run_record = _persist_requested_transition(
                run_dir,
                run_record,
                WorkflowState.IMPLEMENTING,
                clock=clock,
            )
            _update_repository_lock(repository_lock, run_record)
            implementation_result = run_implementation_stage(
                config,
                run_dir,
                codex_runner=codex_runner,
                clock=clock,
            )
            run_record = _persist_stage_outcome(
                run_dir,
                implementation_result.run_record,
                implementation_result.outcome,
                terminal_reason=implementation_result.controller_message,
                clock=clock,
            )
            implementation_result = replace(
                implementation_result,
                run_record=run_record,
            )
            _update_repository_lock(repository_lock, run_record)
            continue

        if run_record.state == WorkflowState.VERIFYING:
            verification = run_verification_stage(
                config,
                run_dir,
                process_runner=verification_runner,
                clock=clock,
            )
            run_record = _persist_stage_outcome(
                run_dir,
                verification.run_record,
                verification.outcome,
                terminal_reason=verification.controller_message,
                clock=clock,
            )
            verification = replace(verification, run_record=run_record)
            verification_results.append(verification)
            _update_repository_lock(repository_lock, run_record)
            continue

        if run_record.state == WorkflowState.REVIEWING:
            review = run_review_stage(
                config,
                run_dir,
                codex_runner=codex_runner,
                clock=clock,
            )
            run_record = _persist_stage_outcome(
                run_dir,
                review.run_record,
                review.outcome,
                terminal_reason=review.controller_message,
                current_review_round=review.run_record.current_review_round + 1,
                clock=clock,
            )
            review = replace(review, run_record=run_record)
            review_results.append(review)
            _update_repository_lock(repository_lock, run_record)
            continue

        if run_record.state == WorkflowState.CORRECTION_PENDING:
            if run_record.current_correction_round >= run_record.max_correction_rounds:
                run_record = _mark_human_required(
                    run_dir,
                    terminal_reason=(
                        "Maximum corrective rounds exhausted; human intervention "
                        "is required."
                    ),
                    clock=clock,
                )
                _update_repository_lock(repository_lock, run_record)
                continue

            run_record = _persist_requested_transition(
                run_dir,
                run_record,
                WorkflowState.CORRECTING,
                clock=clock,
            )
            _update_repository_lock(repository_lock, run_record)
            correction = run_correction_stage(
                config,
                run_dir,
                codex_runner=codex_runner,
                clock=clock,
            )
            run_record = _persist_stage_outcome(
                run_dir,
                correction.run_record,
                correction.outcome,
                terminal_reason=correction.controller_message,
                current_correction_round=(
                    correction.correction_round
                    if correction.advance_correction_round
                    else correction.run_record.current_correction_round
                ),
                clock=clock,
            )
            correction = replace(correction, run_record=run_record)
            correction_results.append(correction)
            _update_repository_lock(repository_lock, run_record)
            continue

        if run_record.state == WorkflowState.REPORTING:
            report_result = run_report_stage(
                run_dir,
                transition_record=lambda record, outcome, reason: (
                    _record_after_stage_outcome(
                        record,
                        outcome,
                        terminal_reason=reason,
                        clock=clock,
                    )
                ),
            )
            run_record = report_result.run_record
            save_run_record(run_record, run_dir / RUN_RECORD_FILE)
            _update_repository_lock(repository_lock, run_record)
            continue

        raise RunError(
            "Lifecycle reached an unsupported non-terminal state: "
            f"{run_record.state.value}."
        )

    if run_record.state != WorkflowState.READY_FOR_HUMAN:
        generate_terminal_report_best_effort(run_dir)
    _update_repository_lock(repository_lock, run_record)

    return LifecycleResult(
        run_dir=run_dir,
        run_record=run_record,
        preflight_result=preflight_result,
        implementation_result=implementation_result,
        verification_results=tuple(verification_results),
        review_results=tuple(review_results),
        correction_results=tuple(correction_results),
        report_result=report_result,
    )


def _persist_requested_transition(
    run_dir: Path,
    run_record: RunRecord,
    state: WorkflowState,
    *,
    clock: Callable[[], datetime] | None,
) -> RunRecord:
    updated_record = run_record.transition_to(
        state,
        updated_timestamp=_timestamp(clock),
    )
    save_run_record(updated_record, run_dir / RUN_RECORD_FILE)
    return updated_record


def _persist_stage_outcome(
    run_dir: Path,
    run_record: RunRecord,
    outcome: StageOutcome,
    *,
    terminal_reason: str,
    clock: Callable[[], datetime] | None,
    current_correction_round: int | None = None,
    current_review_round: int | None = None,
) -> RunRecord:
    updated_record = _record_after_stage_outcome(
        run_record,
        outcome,
        terminal_reason=terminal_reason,
        clock=clock,
        current_correction_round=current_correction_round,
        current_review_round=current_review_round,
    )
    save_run_record(updated_record, run_dir / RUN_RECORD_FILE)
    return updated_record


def _record_after_stage_outcome(
    run_record: RunRecord,
    outcome: StageOutcome,
    *,
    terminal_reason: str | None,
    clock: Callable[[], datetime] | None,
    current_correction_round: int | None = None,
    current_review_round: int | None = None,
) -> RunRecord:
    if outcome == StageOutcome.HUMAN_REQUIRED:
        state = WorkflowState.HUMAN_REQUIRED
    elif outcome == StageOutcome.FAILED:
        state = WorkflowState.FAILED
    elif outcome == StageOutcome.CORRECTION_REQUIRED and run_record.state in {
        WorkflowState.VERIFYING,
        WorkflowState.REVIEWING,
    }:
        state = WorkflowState.CORRECTION_PENDING
    elif outcome == StageOutcome.COMPLETED:
        try:
            state = {
                WorkflowState.IMPLEMENTING: WorkflowState.VERIFYING,
                WorkflowState.VERIFYING: WorkflowState.REVIEWING,
                WorkflowState.REVIEWING: WorkflowState.REPORTING,
                WorkflowState.CORRECTING: WorkflowState.VERIFYING,
                WorkflowState.REPORTING: WorkflowState.READY_FOR_HUMAN,
            }[run_record.state]
        except KeyError as error:
            raise RunError(
                "Stage completion is not valid from workflow state "
                f"{run_record.state.value}."
            ) from error
    else:
        raise RunError(
            f"Stage outcome {outcome.value} is not valid from workflow state "
            f"{run_record.state.value}."
        )

    return run_record.transition_to(
        state,
        updated_timestamp=_timestamp(clock),
        current_correction_round=current_correction_round,
        current_review_round=current_review_round,
        terminal_reason=(terminal_reason if state in TERMINAL_STATES else None),
    )


def _resume_preflight_problem(run_dir: Path, run_record: RunRecord) -> str | None:
    baseline_path = run_dir / BASELINE_RECORD_FILE
    ticket_path = run_dir / RUN_TICKET_FILE
    if not baseline_path.is_file():
        return f"Required baseline artifact is missing: {baseline_path}"
    if not ticket_path.is_file():
        return f"Required snapshotted ticket artifact is missing: {ticket_path}"

    try:
        baseline = load_baseline_record(baseline_path)
    except RunError as error:
        return f"Baseline artifact is not internally consistent: {error}"
    if baseline.branch != run_record.starting_branch:
        return "Run record and baseline artifact disagree on starting branch."
    if baseline.head_sha != run_record.baseline_sha:
        return "Run record and baseline artifact disagree on baseline HEAD."

    repository = GitRepository(Path(run_record.target_repository_path))
    if not repository.path.exists():
        return f"Target repository no longer exists: {repository.path}"
    try:
        if not repository.is_repository():
            return f"Target path is no longer a Git repository: {repository.path}"
    except OSError as error:
        return f"Could not inspect target repository: {error}"

    safety = inspect_git_safety(run_record)
    if not safety.safe:
        return _resume_safety_reason(safety)
    return None


def _update_repository_lock(
    repository_lock: RepositoryRunLock,
    run_record: RunRecord,
) -> None:
    repository_lock.update(
        run_id=run_record.run_id,
        current_state=run_record.state.value,
        target_repository_path=run_record.target_repository_path,
    )


def _resume_checkpoint_problem(run_dir: Path, run_record: RunRecord) -> str | None:
    if run_record.state == WorkflowState.PREPARED:
        return None

    if run_record.state == WorkflowState.PREPARING:
        return (
            "Run preparation was interrupted before a complete snapshot was persisted."
        )

    if run_record.state == WorkflowState.IMPLEMENTING:
        return (
            "Writable implementation was interrupted before completion; the "
            "working tree may contain partial source modifications."
        )

    if run_record.state == WorkflowState.VERIFYING:
        return _require_completed_writable_checkpoint(run_dir, run_record)

    if run_record.state == WorkflowState.REVIEWING:
        verification = latest_verification_round(run_dir)
        if not isinstance(verification, dict) or verification.get("status") != "PASS":
            return "Review cannot resume because the latest verification did not pass."
        return _require_current_workspace_matches_checkpoint(run_dir, run_record)

    if run_record.state == WorkflowState.CORRECTION_PENDING:
        checkpoint_problem = _require_current_workspace_matches_checkpoint(
            run_dir,
            run_record,
        )
        if checkpoint_problem is not None:
            return checkpoint_problem
        return _require_correction_source_checkpoint(run_dir, run_record)

    if run_record.state == WorkflowState.CORRECTING:
        return (
            "Writable correction was interrupted before completion; the "
            "working tree may contain partial source modifications."
        )

    if run_record.state == WorkflowState.REPORTING:
        review = latest_review_result(run_dir)
        if not isinstance(review, dict) or review.get("verdict") != "PASS":
            return "Report cannot resume because the final review did not pass."
        verification = latest_verification_round(run_dir)
        if not isinstance(verification, dict) or verification.get("status") != "PASS":
            return (
                "Report cannot resume because deterministic verification did not pass."
            )
        return _require_current_workspace_matches_checkpoint(run_dir, run_record)

    return f"Run state is not resumable in V1: {run_record.state.value}"


def _require_completed_writable_checkpoint(
    run_dir: Path,
    run_record: RunRecord,
) -> str | None:
    if run_record.current_correction_round > 0:
        return _require_completed_correction_checkpoint(run_dir, run_record)
    return _require_completed_implementation_checkpoint(run_dir, run_record)


def _require_completed_implementation_checkpoint(
    run_dir: Path,
    run_record: RunRecord,
) -> str | None:
    result_path = run_dir / "implementation" / "result.json"
    result = _read_json_dict(result_path)
    if result is None or result.get("status") != "COMPLETED":
        return "Implementation checkpoint is missing a completed agent result."
    return _require_current_workspace_matches_checkpoint(run_dir, run_record)


def _require_completed_correction_checkpoint(
    run_dir: Path,
    run_record: RunRecord,
) -> str | None:
    result_path = (
        run_dir
        / "correction-executions"
        / f"round-{run_record.current_correction_round}"
        / "result.json"
    )
    result = _read_json_dict(result_path)
    if result is None or result.get("status") != "COMPLETED":
        return "Correction checkpoint is missing a completed agent result."
    return _require_current_workspace_matches_checkpoint(run_dir, run_record)


def _require_correction_source_checkpoint(
    run_dir: Path,
    run_record: RunRecord,
) -> str | None:
    verification = latest_verification_round(run_dir)
    if isinstance(verification, dict) and verification.get("status") == "FAIL":
        if "correction_reasons" in verification:
            reasons = verification["correction_reasons"]
            if not isinstance(reasons, list):
                return (
                    "Verification correction source is not internally consistent: "
                    "correction_reasons must be a list."
                )
            try:
                parsed_reasons = tuple(
                    correction_reason_from_dict(reason)
                    for reason in reasons
                    if isinstance(reason, dict)
                )
            except CorrectionError as error:
                return (
                    "Verification correction source is not internally consistent: "
                    f"{error}"
                )
            if len(parsed_reasons) != len(reasons):
                return (
                    "Verification correction source is not internally consistent: "
                    "correction reasons must be objects."
                )
            if not all(
                isinstance(reason, VerificationFailure) for reason in parsed_reasons
            ):
                return (
                    "Verification correction source is not internally consistent: "
                    "verification correction reasons must describe verification failures."
                )
            if parsed_reasons:
                return None
        commands = verification.get("commands")
        if isinstance(commands, list) and any(
            isinstance(command, dict) and command.get("status") == "FAIL"
            for command in commands
        ):
            return None
        return (
            "Verification correction source is not internally consistent: "
            "no failed verification command was persisted."
        )

    review = latest_review_result(run_dir)
    if (
        isinstance(review, dict)
        and review.get("verdict") == ReviewVerdict.CORRECTIONS_REQUIRED.value
    ):
        try:
            findings = review_findings_from_result(review)
        except CorrectionError as error:
            return f"Review correction source is not internally consistent: {error}"
        if not findings:
            return (
                "Review correction source is not internally consistent: "
                "no required review findings were persisted."
            )
        return None

    return (
        "Correction source is not internally consistent: the latest verification "
        "did not fail and the latest review did not require corrections."
    )


def _require_current_workspace_matches_checkpoint(
    run_dir: Path,
    run_record: RunRecord,
) -> str | None:
    fingerprint_path = _latest_writable_workspace_fingerprint_path(
        run_dir,
        run_record,
    )
    if fingerprint_path is None:
        return "Required canonical workspace fingerprint checkpoint is missing."
    repository = GitRepository(Path(run_record.target_repository_path))
    try:
        expected_fingerprint = _read_workspace_fingerprint(fingerprint_path)
        current_snapshot = WorkspaceSnapshot.capture(repository)
    except (OSError, RuntimeError, ValueError) as error:
        return f"Could not compare current workspace to checkpoint: {error}"
    if not current_snapshot.inspection_complete:
        return (
            "Could not compare current workspace to checkpoint: workspace "
            "inspection was incomplete: "
            + "; ".join(current_snapshot.inspection_errors)
        )
    if not current_snapshot.matches_fingerprint(expected_fingerprint):
        return (
            "Current workspace no longer matches the last verified writable checkpoint."
        )
    return None


def _read_json_dict(path: Path) -> dict[str, object] | None:
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _mark_failed(
    run_dir: Path,
    *,
    terminal_reason: str,
    clock: Callable[[], datetime] | None,
) -> RunRecord:
    record_path = run_dir / RUN_RECORD_FILE
    run_record = load_run_record(record_path)
    updated_record = run_record.transition_to(
        WorkflowState.FAILED,
        updated_timestamp=_timestamp(clock),
        terminal_reason=terminal_reason,
    )
    save_run_record(updated_record, record_path)
    return updated_record


def _mark_human_required(
    run_dir: Path,
    *,
    terminal_reason: str,
    clock: Callable[[], datetime] | None,
) -> RunRecord:
    record_path = run_dir / RUN_RECORD_FILE
    run_record = load_run_record(record_path)
    updated_record = run_record.transition_to(
        WorkflowState.HUMAN_REQUIRED,
        updated_timestamp=_timestamp(clock),
        terminal_reason=terminal_reason,
    )
    save_run_record(updated_record, record_path)
    return updated_record


def _terminal_label(state: WorkflowState) -> str:
    if state == WorkflowState.READY_FOR_HUMAN:
        return "READY FOR HUMAN REVIEW"
    if state == WorkflowState.HUMAN_REQUIRED:
        return "HUMAN REQUIRED"
    if state == WorkflowState.FAILED:
        return "FAILED"
    return state.value


def _safe_report_context(run_dir: Path, record: RunRecord) -> dict[str, object] | None:
    try:
        return collect_report_context(run_dir, record)
    except (GitCommandError, KeyError, OSError, RunError, TypeError, ValueError):
        return None


def _changed_file_count(
    context: dict[str, object] | None,
    result: LifecycleResult,
) -> int:
    if context is not None:
        controller = context["controller"]
        if isinstance(controller, dict):
            changed_files = controller.get("changed_files", ())
            if isinstance(changed_files, tuple | list):
                return len(changed_files)
    if result.implementation_result is not None:
        return len(result.implementation_result.changed_files)
    return 0


def _diff_line_summary(
    context: dict[str, object] | None,
    result: LifecycleResult,
) -> str:
    if result.report_result is not None:
        return f"+{result.report_result.additions} / -{result.report_result.deletions}"
    if context is not None:
        controller = context["controller"]
        if isinstance(controller, dict):
            return (
                f"+{controller.get('additions', 0)} / -{controller.get('deletions', 0)}"
            )
    return "not available"


def _verification_lines(
    context: dict[str, object] | None,
    result: LifecycleResult,
) -> list[str]:
    verification: dict[str, object] | None = None
    if context is not None:
        controller = context["controller"]
        if isinstance(controller, dict):
            rounds = controller.get("verification_rounds", ())
            if isinstance(rounds, tuple | list) and rounds:
                latest = rounds[-1]
                if isinstance(latest, dict):
                    data = latest.get("data")
                    if isinstance(data, dict):
                        verification = data
    if verification is None and result.verification_results:
        round_result = result.verification_results[-1].round_result
        return [
            f"  {round_result.status.value}",
            *(
                f"  {command.name:<10} {command.status.value}"
                for command in round_result.commands
            ),
        ]
    if verification is None:
        return ["  NOT RUN"]

    status = verification.get("status", "UNKNOWN")
    commands = verification.get("commands", [])
    lines = [f"  {status}"]
    if isinstance(commands, list) and commands:
        for command in commands:
            if isinstance(command, dict):
                lines.append(
                    f"  {command.get('name', '<unnamed>'):<10} "
                    f"{command.get('status', 'UNKNOWN')}"
                )
    return lines


def _review_summary(
    context: dict[str, object] | None,
    result: LifecycleResult,
) -> str:
    if context is not None:
        controller = context["controller"]
        if isinstance(controller, dict):
            final_review = controller.get("final_review")
            if isinstance(final_review, dict):
                verdict = final_review.get("verdict")
                if isinstance(verdict, str):
                    return verdict
    if not result.review_results:
        return "NOT RUN"
    review_result = result.review_results[-1].review_result
    if review_result is None:
        return "NOT AVAILABLE"
    verdict = ReviewVerdict(review_result["verdict"])
    return verdict.value


def _advisory_count(context: dict[str, object] | None) -> int:
    if context is None:
        return 0
    controller = context["controller"]
    if not isinstance(controller, dict):
        return 0
    findings = controller.get("advisory_findings", ())
    return len(findings) if isinstance(findings, tuple | list) else 0


def _git_safety_lines(
    result: LifecycleResult,
    context: dict[str, object] | None,
) -> list[str]:
    violations = []
    if result.implementation_result is not None:
        violations.extend(result.implementation_result.safety_violations)
    for verification in result.verification_results:
        violations.extend(verification.round_result.safety_violations)
    for review in result.review_results:
        violations.extend(review.safety_violations)
    for correction in result.correction_results:
        violations.extend(correction.safety_violations)
    violations.extend(result.safety_violations)
    if violations:
        return [
            f"  {violation.name}: expected {violation.expected}, got {violation.actual}"
            for violation in violations
        ]

    if context is not None:
        controller = context["controller"]
        if isinstance(controller, dict):
            safety = controller.get("git_safety")
            if safety is not None:
                return [
                    f"  HEAD {'unchanged' if safety.head_ok else 'changed'}",
                    f"  branch {'unchanged' if safety.branch_ok else 'changed'}",
                    f"  staging {'empty' if safety.staging_ok else 'not empty'}",
                ]
    return [
        "  HEAD unchanged",
        "  branch unchanged",
        "  staging empty",
    ]


def _resume_safety_reason(safety: object) -> str:
    problems: list[str] = []
    if getattr(safety, "inspection_error", None):
        problems.append(f"Git inspection failed: {safety.inspection_error}")
    if not getattr(safety, "branch_ok", False):
        problems.append(
            f"branch is {safety.branch_actual}, expected {safety.branch_expected}"
        )
    if not getattr(safety, "head_ok", False):
        problems.append(
            f"HEAD is {safety.head_actual}, expected {safety.head_expected}"
        )
    if not getattr(safety, "staging_ok", False):
        problems.append("staging area is not empty")
    return "Resume preflight failed: " + "; ".join(problems)


def _timestamp(clock: Callable[[], datetime] | None) -> str:
    now = datetime.now(UTC) if clock is None else clock()
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return now.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "TERMINAL_STATES",
    "LifecycleResult",
    "LifecycleSafetyViolation",
    "format_lifecycle_result",
    "resume_ticket_lifecycle",
    "run_ticket_lifecycle",
]
