from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .codex import CodexProcessRunner
from .config import AppConfig
from .corrections import (
    CorrectionStageResult,
    run_correction_stage,
)
from .git import GitCommandError, GitRepository
from .implementation import (
    ImplementationStageResult,
    run_implementation_stage,
)
from .models import WorkflowState
from .preflight import PreflightResult
from .review import (
    ReviewStageResult,
    ReviewVerdict,
    run_review_stage,
)
from .runs import (
    RUN_RECORD_FILE,
    RunError,
    RunRecord,
    create_run_snapshot,
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
    snapshot = create_run_snapshot(
        config,
        ticket_path,
        runs_dir=runs_dir,
        clock=clock,
    )
    implementation_result: ImplementationStageResult | None = None
    verification_results: list[VerificationStageResult] = []
    review_results: list[ReviewStageResult] = []
    correction_results: list[CorrectionStageResult] = []

    try:
        implementation_result = run_implementation_stage(
            config,
            snapshot.run_dir,
            codex_runner=codex_runner,
            clock=clock,
        )
        run_record = implementation_result.run_record

        while run_record.state not in TERMINAL_STATES:
            if run_record.state == WorkflowState.IMPLEMENT:
                verification = run_verification_stage(
                    config,
                    snapshot.run_dir,
                    process_runner=verification_runner,
                    clock=clock,
                )
                verification_results.append(verification)
                run_record = verification.run_record
                continue

            if run_record.state == WorkflowState.VERIFY:
                if run_record.last_completed_state == WorkflowState.CORRECT:
                    verification = run_verification_stage(
                        config,
                        snapshot.run_dir,
                        process_runner=verification_runner,
                        clock=clock,
                    )
                    verification_results.append(verification)
                    run_record = verification.run_record
                    continue

                if run_record.last_completed_state == WorkflowState.VERIFY:
                    review = run_review_stage(
                        config,
                        snapshot.run_dir,
                        codex_runner=codex_runner,
                        clock=clock,
                    )
                    review_results.append(review)
                    run_record = review.run_record
                    continue

            if run_record.state == WorkflowState.CORRECT:
                if (
                    run_record.current_correction_round
                    >= run_record.max_correction_rounds
                ):
                    run_record = _mark_human_required(
                        snapshot.run_dir,
                        terminal_reason=(
                            "Maximum corrective rounds exhausted; human intervention "
                            "is required."
                        ),
                        clock=clock,
                    )
                    continue
                correction = run_correction_stage(
                    config,
                    snapshot.run_dir,
                    codex_runner=codex_runner,
                    clock=clock,
                )
                correction_results.append(correction)
                run_record = correction.run_record
                continue

            raise RunError(
                "Lifecycle reached an unsupported non-terminal state: "
                f"{run_record.state.value} after "
                f"{run_record.last_completed_state.value}."
            )

        run_record, safety_violations = _finalize_ready_for_human_if_safe(
            snapshot.run_dir,
            run_record,
            clock=clock,
        )
        return LifecycleResult(
            run_dir=snapshot.run_dir,
            run_record=run_record,
            preflight_result=snapshot.preflight_result,
            implementation_result=implementation_result,
            verification_results=tuple(verification_results),
            review_results=tuple(review_results),
            correction_results=tuple(correction_results),
            safety_violations=safety_violations,
        )
    except RunError as error:
        run_record = _mark_failed(
            snapshot.run_dir,
            terminal_reason=str(error),
            clock=clock,
        )
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
            "Internal TicketAutomation exception: "
            f"{type(error).__name__}: {error}"
        )
        run_record = _mark_failed(
            snapshot.run_dir,
            terminal_reason=controller_error,
            clock=clock,
        )
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


def format_lifecycle_result(result: LifecycleResult) -> str:
    record = result.run_record
    title = f"{record.ticket_id} - {_terminal_label(record.state)}"
    lines = [
        title,
        "",
        "Run:",
        f"  {result.run_dir}",
        "",
        "Branch:",
        f"  {record.starting_branch}",
        "",
        "Baseline:",
        f"  {record.baseline_sha}",
        "",
        "Verification:",
        f"  {_verification_summary(result)}",
        "",
        "Review:",
        f"  {_review_summary(result)}",
        "",
        "Correction rounds:",
        f"  {record.current_correction_round} / {record.max_correction_rounds}",
        "",
        "Review rounds:",
        f"  {record.current_review_round}",
        "",
        "Git safety:",
        *_git_safety_lines(result),
    ]
    if record.terminal_reason:
        lines.extend(["", "Reason:", f"  {record.terminal_reason}"])
    if result.controller_error:
        lines.extend(["", "Controller error:", f"  {result.controller_error}"])
    if result.successful:
        lines.extend(["", "No files have been staged or committed."])
    return "\n".join(lines)


def _finalize_ready_for_human_if_safe(
    run_dir: Path,
    run_record: RunRecord,
    *,
    clock: Callable[[], datetime] | None,
) -> tuple[RunRecord, tuple[LifecycleSafetyViolation, ...]]:
    if run_record.state != WorkflowState.READY_FOR_HUMAN:
        return run_record, ()

    violations = _inspect_lifecycle_safety(run_record)
    if not violations:
        return run_record, ()

    updated_record = run_record.with_state(
        WorkflowState.HUMAN_REQUIRED,
        updated_timestamp=_timestamp(clock),
        last_completed_state=run_record.last_completed_state,
        terminal_reason=(
            "Repository safety invariants were violated before human handoff."
        ),
    )
    save_run_record(updated_record, run_dir / RUN_RECORD_FILE)
    return updated_record, violations


def _inspect_lifecycle_safety(
    run_record: RunRecord,
) -> tuple[LifecycleSafetyViolation, ...]:
    repository = GitRepository(Path(run_record.target_repository_path))
    try:
        current_branch = repository.current_branch()
        current_head = repository.head_sha()
        staged_files = repository.staged_files()
    except GitCommandError as error:
        return (
            LifecycleSafetyViolation(
                name="git-inspection",
                expected="Git inspection succeeds",
                actual=str(error),
                message="Could not inspect repository safety invariants.",
            ),
        )

    violations: list[LifecycleSafetyViolation] = []
    if current_branch != run_record.starting_branch:
        violations.append(
            LifecycleSafetyViolation(
                name="branch",
                expected=run_record.starting_branch,
                actual="<detached>" if current_branch is None else current_branch,
                message="Current branch no longer matches the starting branch.",
            )
        )
    if current_head != run_record.baseline_sha:
        violations.append(
            LifecycleSafetyViolation(
                name="HEAD",
                expected=run_record.baseline_sha,
                actual=current_head,
                message="HEAD no longer matches the baseline SHA.",
            )
        )
    if staged_files:
        violations.append(
            LifecycleSafetyViolation(
                name="staging",
                expected="empty",
                actual=_format_files(staged_files),
                message="Staging area is not empty.",
            )
        )
    return tuple(violations)


def _mark_failed(
    run_dir: Path,
    *,
    terminal_reason: str,
    clock: Callable[[], datetime] | None,
) -> RunRecord:
    record_path = run_dir / RUN_RECORD_FILE
    run_record = load_run_record(record_path)
    updated_record = run_record.with_state(
        WorkflowState.FAILED,
        updated_timestamp=_timestamp(clock),
        last_completed_state=run_record.last_completed_state,
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
    updated_record = run_record.with_state(
        WorkflowState.HUMAN_REQUIRED,
        updated_timestamp=_timestamp(clock),
        last_completed_state=run_record.last_completed_state,
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


def _verification_summary(result: LifecycleResult) -> str:
    if not result.verification_results:
        return "NOT RUN"
    return result.verification_results[-1].round_result.status.value


def _review_summary(result: LifecycleResult) -> str:
    if not result.review_results:
        return "NOT RUN"
    review_result = result.review_results[-1].review_result
    if review_result is None:
        return "NOT AVAILABLE"
    verdict = ReviewVerdict(review_result["verdict"])
    return verdict.value


def _git_safety_lines(result: LifecycleResult) -> list[str]:
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
    return [
        "  HEAD unchanged",
        "  branch unchanged",
        "  staging empty",
    ]


def _format_files(files: tuple[str, ...]) -> str:
    if not files:
        return "empty"
    shown = ", ".join(files[:5])
    hidden_count = len(files) - 5
    if hidden_count > 0:
        shown = f"{shown}, and {hidden_count} more"
    return shown


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
    "run_ticket_lifecycle",
]
