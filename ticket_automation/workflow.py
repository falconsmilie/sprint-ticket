from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from ._verification_artifacts import (
    _baseline_verification_evidence_problem,
    _read_verification_source_fingerprint,
    _verification_commands_fingerprint,
    _VerificationArtifactError,
)
from .attempts import (
    AttemptError,
    AttemptRecord,
    attempt_result_path,
    finish_phase_attempt,
    latest_attempt,
    latest_writable_attempt,
    load_attempt_records,
    start_attempt,
)
from .audit import diff_including_untracked
from .codex import CodexProcessRunner
from .config import AppConfig
from .corrections import (
    CorrectionStageResult,
    run_correction_stage,
)
from .failure_classification import (
    TerminalStop,
    classify_stage_stop,
    classify_unexpected_controller_failure,
)
from .git import GitCommandError, GitRepository
from .git_safety import WorkspaceSnapshot
from .implementation import (
    ImplementationStageResult,
    run_implementation_stage,
)
from .locking import RepositoryRunLock, acquire_repository_run_lock
from .models import StageOutcome, StopCategory, StopReason, WorkflowState
from .preflight import PreflightResult
from .reporting import (
    FINAL_PATCH_FILE,
    ReportError,
    ReportStageResult,
    collect_report_context,
    generate_terminal_report_best_effort,
    run_report_stage,
)
from .resolved_config import config_from_resolved_run_config
from .review import (
    ReviewStageResult,
    ReviewVerdict,
    _validate_review_result_artifact,
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
    _run_baseline_verification_stage,
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
    config = config_from_resolved_run_config(snapshot.run_record.resolved_config)
    _update_repository_lock(repository_lock, snapshot.run_record)
    implementation_result: ImplementationStageResult | None = None
    verification_results: list[VerificationStageResult] = []
    review_results: list[ReviewStageResult] = []
    correction_results: list[CorrectionStageResult] = []

    try:
        active_record = _complete_preparation(
            config,
            snapshot.run_dir,
            process_runner=verification_runner,
            clock=clock,
        )
        _update_repository_lock(repository_lock, active_record)
        return _drive_lifecycle(
            config,
            snapshot.run_dir,
            snapshot.preflight_result,
            active_record,
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
        controller_error = str(error)
        run_record = _mark_controller_exception(
            snapshot.run_dir,
            controller_error=controller_error,
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
    except Exception as error:  # noqa: BLE001 - terminal evidence must be persisted.
        controller_error = (
            f"Internal TicketAutomation exception: {type(error).__name__}: {error}"
        )
        run_record = _mark_controller_exception(
            snapshot.run_dir,
            controller_error=controller_error,
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
    # A caller may still provide the legacy configuration argument, but an
    # existing run always executes from its persisted policy snapshot.
    del config
    config = config_from_resolved_run_config(run_record.resolved_config)
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

    compatibility_problem = run_record.resolved_config.runtime_compatibility_problem()
    if compatibility_problem is not None:
        run_record = _mark_human_required(
            run_dir,
            terminal_reason=compatibility_problem,
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

    resume_problem = _resume_preflight_problem(config, run_dir, run_record)
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

    resume_problem = _resume_problem(run_dir, run_record)
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

    implementation_result: ImplementationStageResult | None = None
    verification_results: list[VerificationStageResult] = []
    review_results: list[ReviewStageResult] = []
    correction_results: list[CorrectionStageResult] = []
    try:
        if run_record.state == WorkflowState.PREPARING:
            run_record = _complete_preparation(
                config,
                run_dir,
                process_runner=verification_runner,
                clock=clock,
            )
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
                stop_category=_implementation_stop_category(implementation_result),
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
        controller_error = str(error)
        run_record = _mark_controller_exception(
            run_dir,
            controller_error=controller_error,
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
    except Exception as error:  # noqa: BLE001 - terminal evidence must be persisted.
        controller_error = (
            "Internal TicketAutomation exception during resume: "
            f"{type(error).__name__}: {error}"
        )
        run_record = _mark_controller_exception(
            run_dir,
            controller_error=controller_error,
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
        "Baseline verification:",
        *_baseline_verification_lines(context),
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
    report_path = result.run_dir / "report.md"
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
                stop_category=_implementation_stop_category(implementation_result),
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
                stop_category=_verification_stop_category(verification),
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
                stop_category=_review_stop_category(review),
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
                stop_category=_correction_stop_category(correction),
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
            try:
                snapshot = WorkspaceSnapshot.capture(
                    GitRepository(Path(run_record.target_repository_path))
                )
                snapshot_problem: str | None = None
            except (GitCommandError, OSError, RuntimeError, ValueError) as error:
                snapshot = None
                snapshot_problem = (
                    "Could not inspect the workspace before human handoff: "
                    f"{type(error).__name__}: {error}"
                )
            report_attempt = start_attempt(
                run_dir,
                phase=WorkflowState.REPORTING.value,
                before_workspace_fingerprint=(
                    None if snapshot is None else snapshot.fingerprint
                ),
                clock=clock,
            )
            handoff_problem = snapshot_problem or _final_handoff_problem(
                run_dir,
                run_record,
                snapshot=snapshot,
            )
            if handoff_problem is not None:
                _write_attempt_result(
                    report_attempt,
                    status="HUMAN_REQUIRED",
                    message=handoff_problem,
                )
                finish_phase_attempt(
                    run_dir,
                    phase=WorkflowState.REPORTING.value,
                    stage_outcome=StageOutcome.HUMAN_REQUIRED.value,
                    after_workspace_fingerprint=(
                        None if snapshot is None else snapshot.fingerprint
                    ),
                    metadata={"controller_message": handoff_problem},
                    clock=clock,
                )
                run_record = _mark_human_required(
                    run_dir,
                    terminal_reason=handoff_problem,
                    clock=clock,
                    category=StopCategory.SAFETY_VIOLATION,
                )
                _update_repository_lock(repository_lock, run_record)
                continue
            assert snapshot is not None
            try:
                _capture_final_patch(run_dir, run_record)
                after_patch_snapshot = WorkspaceSnapshot.capture(
                    GitRepository(Path(run_record.target_repository_path))
                )
            except (GitCommandError, OSError, RuntimeError, ValueError) as error:
                handoff_problem = (
                    "Could not capture the final handoff patch safely: "
                    f"{type(error).__name__}: {error}"
                )
            else:
                if not snapshot.matches(after_patch_snapshot):
                    handoff_problem = (
                        "Workspace changed while the final handoff patch was being "
                        "captured."
                    )
                else:
                    snapshot = after_patch_snapshot
            if handoff_problem is not None:
                _write_attempt_result(
                    report_attempt,
                    status="HUMAN_REQUIRED",
                    message=handoff_problem,
                )
                finish_phase_attempt(
                    run_dir,
                    phase=WorkflowState.REPORTING.value,
                    stage_outcome=StageOutcome.HUMAN_REQUIRED.value,
                    after_workspace_fingerprint=snapshot.fingerprint,
                    metadata={"controller_message": handoff_problem},
                    clock=clock,
                )
                run_record = _mark_human_required(
                    run_dir,
                    terminal_reason=handoff_problem,
                    clock=clock,
                    category=StopCategory.SAFETY_VIOLATION,
                )
                _update_repository_lock(repository_lock, run_record)
                continue
            _write_attempt_result(
                report_attempt,
                status="PASS",
                message="Final workspace and evidence consistency checks passed.",
            )
            finish_phase_attempt(
                run_dir,
                phase=WorkflowState.REPORTING.value,
                stage_outcome=StageOutcome.COMPLETED.value,
                after_workspace_fingerprint=snapshot.fingerprint,
                metadata={
                    "controller_message": (
                        "Final workspace and evidence consistency checks passed."
                    )
                },
                clock=clock,
            )
            run_record = _persist_stage_outcome(
                run_dir,
                run_record,
                StageOutcome.COMPLETED,
                terminal_reason="Final workspace and evidence consistency checks passed.",
                clock=clock,
            )
            try:
                report_result = run_report_stage(run_dir)
            except (OSError, ReportError, ValueError):
                # The controller has already made and persisted the terminal
                # decision. Rendering is retriable presentation work and cannot
                # move a READY_FOR_HUMAN run to another terminal state.
                report_result = None
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


def _final_handoff_problem(
    run_dir: Path,
    run_record: RunRecord,
    *,
    snapshot: WorkspaceSnapshot | None,
) -> str | None:
    """Controller-owned acceptance check before moving out of REPORTING."""

    try:
        load_attempt_records(run_dir)
    except AttemptError as error:
        return f"Attempt evidence is invalid before human handoff: {error}"
    if snapshot is None or not snapshot.inspection_complete:
        return "Repository safety invariants were violated before human handoff."
    if snapshot.branch != run_record.starting_branch:
        return "Repository branch changed before human handoff."
    if snapshot.head_sha != run_record.baseline_sha:
        return "Repository HEAD changed before human handoff."
    if snapshot.staged_paths:
        return "Repository has staged changes before human handoff."
    writable = latest_writable_attempt(run_dir)
    if writable is None or writable.after_workspace_fingerprint is None:
        return "No completed writable attempt has a workspace fingerprint."
    if not snapshot.matches_fingerprint(writable.after_workspace_fingerprint):
        return "Current workspace no longer matches the completed writable attempt."
    try:
        _read_verification_source_fingerprint(
            run_dir,
            run_record,
            expected_statuses=frozenset({"PASS"}),
            verification_commands=run_record.resolved_config.verification_commands,
        )
    except _VerificationArtifactError as error:
        return f"Deterministic verification evidence is not passing: {error}"
    review = _attempt_result(run_dir, WorkflowState.REVIEWING.value)
    if review is None:
        return "Final independent review evidence is missing."
    try:
        review = _validate_review_result_artifact(review)
    except (KeyError, TypeError, ValueError) as error:
        return f"Final independent review evidence is invalid: {error}"
    if review.get("verdict") != ReviewVerdict.PASS.value:
        return "Final independent review did not pass."
    return None


def _capture_final_patch(run_dir: Path, run_record: RunRecord) -> None:
    patch = diff_including_untracked(
        GitRepository(Path(run_record.target_repository_path)),
        run_record.baseline_sha,
    )
    (run_dir / FINAL_PATCH_FILE).write_text(
        patch,
        encoding="utf-8",
        newline="\n",
    )


def _write_attempt_result(
    attempt: AttemptRecord,
    *,
    status: str,
    message: str,
) -> None:
    path = attempt.artifact_directory / "result.json"
    path.write_text(
        json.dumps({"status": status, "message": message}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _attempt_result(run_dir: Path, phase: str) -> dict[str, object] | None:
    attempt = latest_attempt(run_dir, phases=(phase,), statuses=("COMPLETED",))
    if attempt is None:
        return None
    path = attempt_result_path(run_dir, attempt)
    if path is None:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _complete_preparation(
    config: AppConfig,
    run_dir: Path,
    *,
    process_runner: VerificationProcessRunner | None,
    clock: Callable[[], datetime] | None,
) -> RunRecord:
    try:
        result = _run_baseline_verification_stage(
            config,
            run_dir,
            process_runner=process_runner,
            clock=clock,
        )
    except (OSError, RunError) as error:
        return _mark_human_required(
            run_dir,
            terminal_reason=(
                "Clean baseline verification could not be completed: "
                f"{type(error).__name__}: {error}"
            ),
            clock=clock,
            category=StopCategory.BASELINE_FAILURE,
        )
    return _persist_stage_outcome(
        run_dir,
        result.run_record,
        result.outcome,
        terminal_reason=result.controller_message,
        stop_category=(
            StopCategory.BASELINE_FAILURE
            if result.outcome == StageOutcome.HUMAN_REQUIRED
            else None
        ),
        clock=clock,
    )


def _persist_stage_outcome(
    run_dir: Path,
    run_record: RunRecord,
    outcome: StageOutcome,
    *,
    terminal_reason: str,
    stop_category: StopCategory | None = None,
    clock: Callable[[], datetime] | None,
    current_correction_round: int | None = None,
    current_review_round: int | None = None,
) -> RunRecord:
    updated_record = _record_after_stage_outcome(
        run_record,
        outcome,
        terminal_reason=terminal_reason,
        stop_category=stop_category,
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
    stop_category: StopCategory | None = None,
    clock: Callable[[], datetime] | None,
    current_correction_round: int | None = None,
    current_review_round: int | None = None,
) -> RunRecord:
    stop_reason: StopReason | None = None
    if outcome == StageOutcome.HUMAN_REQUIRED:
        stop = classify_stage_stop(
            run_record,
            outcome,
            message=terminal_reason
            or "The workflow stopped because human intervention is required.",
            category=stop_category,
        )
        state = stop.state
        stop_reason = stop.reason
    elif outcome == StageOutcome.FAILED:
        stop = classify_stage_stop(
            run_record,
            outcome,
            message=terminal_reason or "An external automation operation failed.",
        )
        state = stop.state
        stop_reason = stop.reason
    elif outcome == StageOutcome.CORRECTION_REQUIRED and run_record.state in {
        WorkflowState.VERIFYING,
        WorkflowState.REVIEWING,
    }:
        state = WorkflowState.CORRECTION_PENDING
    elif outcome == StageOutcome.COMPLETED:
        try:
            state = {
                WorkflowState.PREPARING: WorkflowState.PREPARED,
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
        terminal_reason=(stop_reason.message if stop_reason is not None else None),
        stop_reason=stop_reason,
    )


def _implementation_stop_category(
    result: ImplementationStageResult,
) -> StopCategory | None:
    if result.outcome != StageOutcome.HUMAN_REQUIRED:
        return None
    if (
        result.codex_execution is not None
        and result.codex_execution.failure_kind is not None
    ):
        return StopCategory.REPOSITORY_UNCERTAIN
    if result.safety_violations or (
        result.workspace_guard is not None and result.workspace_guard.has_violation
    ):
        return StopCategory.SAFETY_VIOLATION
    if result.agent_result is not None:
        return StopCategory.HUMAN_JUDGMENT_REQUIRED
    return StopCategory.REPOSITORY_UNCERTAIN


def _correction_stop_category(
    result: CorrectionStageResult,
) -> StopCategory | None:
    if result.outcome != StageOutcome.HUMAN_REQUIRED:
        return None
    if (
        result.codex_execution is not None
        and result.codex_execution.failure_kind is not None
    ):
        return StopCategory.REPOSITORY_UNCERTAIN
    if result.safety_violations or (
        result.workspace_guard is not None and result.workspace_guard.has_violation
    ):
        return StopCategory.SAFETY_VIOLATION
    if result.agent_result is not None:
        return StopCategory.HUMAN_JUDGMENT_REQUIRED
    return StopCategory.REPOSITORY_UNCERTAIN


def _verification_stop_category(
    result: VerificationStageResult,
) -> StopCategory | None:
    if result.outcome != StageOutcome.HUMAN_REQUIRED:
        return None
    if result.round_result.safety_violations:
        return StopCategory.SAFETY_VIOLATION
    return StopCategory.VERIFICATION_INFRASTRUCTURE


def _review_stop_category(result: ReviewStageResult) -> StopCategory | None:
    if result.outcome != StageOutcome.HUMAN_REQUIRED:
        return None
    if result.safety_violations:
        return StopCategory.SAFETY_VIOLATION
    return StopCategory.HUMAN_JUDGMENT_REQUIRED


def _resume_preflight_problem(
    config: AppConfig,
    run_dir: Path,
    run_record: RunRecord,
) -> str | None:
    try:
        load_attempt_records(run_dir)
    except AttemptError as error:
        return f"Attempt evidence is invalid: {error}"
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
    if not baseline.clean_worktree or baseline.has_staged_files:
        return "Recorded repository baseline is not clean."
    try:
        ticket_sha256 = hashlib.sha256(ticket_path.read_bytes()).hexdigest()
    except OSError as error:
        return f"Could not inspect snapshotted ticket artifact: {error}"
    if ticket_sha256 != baseline.ticket_sha256:
        return "Snapshotted ticket no longer matches the recorded ticket baseline."
    if (
        _verification_commands_fingerprint(config.verification.commands)
        != baseline.verification_commands_fingerprint
    ):
        return (
            "Configured verification_commands_fingerprint no longer matches the "
            "recorded baseline."
        )

    repository = GitRepository(Path(run_record.target_repository_path))
    if not repository.path.exists():
        return f"Target repository no longer exists: {repository.path}"
    try:
        if not repository.is_repository():
            return f"Target path is no longer a Git repository: {repository.path}"
    except OSError as error:
        return f"Could not inspect target repository: {error}"

    try:
        snapshot = WorkspaceSnapshot.capture(repository)
    except (GitCommandError, OSError, RuntimeError, ValueError) as error:
        return f"Could not inspect current workspace for safe resume: {error}"
    if not snapshot.inspection_complete:
        return "Could not inspect current workspace for safe resume completely."
    if snapshot.branch != run_record.starting_branch:
        return "Current branch no longer matches the recorded run baseline."
    if snapshot.head_sha != run_record.baseline_sha:
        return "Current HEAD no longer matches the recorded run baseline."
    if snapshot.staged_paths:
        return "Current workspace has staged files; human inspection is required."
    if run_record.state in {
        WorkflowState.PREPARING,
        WorkflowState.PREPARED,
    } and not snapshot.matches_fingerprint(baseline.workspace_fingerprint):
        return "Current workspace no longer matches the recorded clean baseline."
    if run_record.state != WorkflowState.PREPARING:
        evidence_problem = _baseline_verification_evidence_problem(
            run_dir,
            run_record,
            baseline,
            verification_commands=config.verification.commands,
        )
        if evidence_problem is not None:
            return evidence_problem
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


def _resume_problem(run_dir: Path, run_record: RunRecord) -> str | None:
    if run_record.state in {WorkflowState.IMPLEMENTING, WorkflowState.CORRECTING}:
        operation = (
            "implementation"
            if run_record.state == WorkflowState.IMPLEMENTING
            else "correction"
        )
        return (
            f"Writable {operation} was interrupted before completion; the "
            "working tree may contain partial source modifications."
        )

    if run_record.state in {WorkflowState.PREPARING, WorkflowState.PREPARED}:
        baseline = load_baseline_record(run_dir / BASELINE_RECORD_FILE)
        return _require_current_workspace_matches_fingerprint(
            run_record,
            baseline.workspace_fingerprint,
            description="the recorded clean baseline",
        )

    if run_record.state in {
        WorkflowState.VERIFYING,
        WorkflowState.REVIEWING,
        WorkflowState.REPORTING,
        WorkflowState.CORRECTION_PENDING,
    }:
        writable_attempt = latest_writable_attempt(run_dir)
        if (
            writable_attempt is None
            or writable_attempt.after_workspace_fingerprint is None
        ):
            return "No completed writable attempt has a workspace fingerprint."
        problem = _require_current_workspace_matches_fingerprint(
            run_record,
            writable_attempt.after_workspace_fingerprint,
            description="the most recent completed writable attempt",
        )
        if problem is not None:
            return problem
        return None

    return f"Run state is not resumable in V1: {run_record.state.value}"


def _require_current_workspace_matches_fingerprint(
    run_record: RunRecord,
    expected_fingerprint: str,
    *,
    description: str,
) -> str | None:
    repository = GitRepository(Path(run_record.target_repository_path))
    try:
        current = WorkspaceSnapshot.capture(repository)
    except (OSError, RuntimeError, ValueError) as error:
        return f"Could not inspect current workspace for safe resume: {error}"
    if not current.inspection_complete:
        return "Could not inspect current workspace for safe resume completely."
    if not current.matches_fingerprint(expected_fingerprint):
        return f"Current workspace no longer matches {description}."
    return None


def _mark_controller_exception(
    run_dir: Path,
    *,
    controller_error: str,
    clock: Callable[[], datetime] | None,
) -> RunRecord:
    run_record = load_run_record(run_dir / RUN_RECORD_FILE)
    if run_record.state in TERMINAL_STATES:
        return run_record
    stop = classify_unexpected_controller_failure(
        run_dir,
        run_record,
        message=controller_error,
    )
    return _mark_terminal_stop(run_dir, stop=stop, clock=clock)


def _mark_terminal_stop(
    run_dir: Path,
    *,
    stop: TerminalStop,
    clock: Callable[[], datetime] | None,
) -> RunRecord:
    record_path = run_dir / RUN_RECORD_FILE
    run_record = load_run_record(record_path)
    updated_record = run_record.transition_to(
        stop.state,
        updated_timestamp=_timestamp(clock),
        terminal_reason=stop.reason.message,
        stop_reason=stop.reason,
    )
    save_run_record(updated_record, record_path)
    return updated_record


def _mark_human_required(
    run_dir: Path,
    *,
    terminal_reason: str,
    clock: Callable[[], datetime] | None,
    category: StopCategory = StopCategory.HUMAN_JUDGMENT_REQUIRED,
) -> RunRecord:
    return _mark_terminal_stop(
        run_dir,
        stop=TerminalStop(
            state=WorkflowState.HUMAN_REQUIRED,
            reason=StopReason(
                category=category,
                message=terminal_reason,
                retryable=False,
            ),
        ),
        clock=clock,
    )


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


def _baseline_verification_lines(
    context: dict[str, object] | None,
) -> list[str]:
    if context is not None:
        controller = context.get("controller")
        if isinstance(controller, dict):
            baseline = controller.get("baseline_verification")
            if isinstance(baseline, dict):
                lines = [f"  {baseline.get('status', 'UNKNOWN')}"]
                commands = baseline.get("commands")
                if isinstance(commands, list):
                    lines.extend(
                        f"  {command.get('name', '<unnamed>'):<10} "
                        f"{command.get('status', 'UNKNOWN')}"
                        for command in commands
                        if isinstance(command, dict)
                    )
                return lines
    return ["  NOT RUN"]


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
    rendered_violations = [
        f"  {violation.name}: expected {violation.expected}, got {violation.actual}"
        for violation in violations
    ]

    if context is not None:
        controller = context.get("controller")
        if isinstance(controller, dict):
            baseline = controller.get("baseline_verification")
            if isinstance(baseline, dict):
                baseline_violations = baseline.get("safety_violations")
                if isinstance(baseline_violations, list):
                    rendered_violations.extend(
                        f"  {violation.get('name', '<unnamed>')}: expected "
                        f"{violation.get('expected', '<unknown>')}, got "
                        f"{violation.get('actual', '<unknown>')}"
                        for violation in baseline_violations
                        if isinstance(violation, dict)
                    )
    if rendered_violations:
        return rendered_violations

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
