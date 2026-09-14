from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .git import GitRepository
from .models import StageOutcome, StopCategory, StopReason, WorkflowState
from .runs import RunRecord
from .writable_attempts import (
    WritableAttempt,
    inspect_writable_attempt_after_failure,
)


@dataclass(frozen=True)
class TerminalStop:
    state: WorkflowState
    reason: StopReason


def classify_stage_stop(
    run_record: RunRecord,
    outcome: StageOutcome,
    *,
    message: str,
    category: StopCategory | None = None,
) -> TerminalStop:
    """Classify a stage's non-successful outcome without parsing its prose."""

    if outcome == StageOutcome.FAILED:
        return TerminalStop(
            state=WorkflowState.FAILED,
            reason=StopReason(
                category=StopCategory.EXTERNAL_TOOL_FAILURE,
                message=message,
                retryable=True,
            ),
        )
    if outcome != StageOutcome.HUMAN_REQUIRED:
        raise ValueError(f"Outcome does not stop the workflow: {outcome.value}.")

    category = category or StopCategory.HUMAN_JUDGMENT_REQUIRED
    return TerminalStop(
        state=WorkflowState.HUMAN_REQUIRED,
        reason=StopReason(category=category, message=message, retryable=False),
    )


def classify_writable_failure(
    repository: GitRepository,
    *,
    attempt: WritableAttempt | None,
    message: str,
    category_if_safe: StopCategory,
    retryable_if_safe: bool,
    malformed_result: bool = False,
    untrusted_completion: bool = False,
) -> TerminalStop:
    """Decide whether a writable-call failure is safe to call FAILED.

    A complete snapshot on both sides of the invocation is the only proof that
    allows a started (or indeterminate) writable process to remain a safe
    automation failure. Structured-result corruption remains a human stop even
    when the source tree is unchanged because the controller cannot trust the
    operation's declared completion.
    """

    inspection = inspect_writable_attempt_after_failure(attempt, repository)
    process_started = None if attempt is None else attempt.process_started
    repository_identical = (
        inspection.before_complete
        and inspection.after_complete
        and inspection.workspace_identical
    )
    if malformed_result or untrusted_completion or not repository_identical:
        return _repository_uncertain_stop(
            message,
            process_started=process_started,
            inspection_error=inspection.inspection_error,
        )
    return TerminalStop(
        state=WorkflowState.FAILED,
        reason=StopReason(
            category=category_if_safe,
            message=message,
            retryable=retryable_if_safe,
        ),
    )


def classify_unexpected_controller_failure(
    run_dir: Path | str,
    run_record: RunRecord,
    *,
    message: str,
) -> TerminalStop:
    """Route outer exceptions through the same writable evidence rule."""

    if run_record.state in {
        WorkflowState.IMPLEMENTING,
        WorkflowState.CORRECTING,
    }:
        repository = GitRepository(Path(run_record.target_repository_path))
        attempt = _current_writable_attempt(Path(run_dir), run_record)
        return classify_writable_failure(
            repository,
            attempt=attempt,
            message=message,
            category_if_safe=StopCategory.CONTROLLER_FAILURE,
            retryable_if_safe=False,
        )
    return TerminalStop(
        state=WorkflowState.FAILED,
        reason=StopReason(
            category=StopCategory.CONTROLLER_FAILURE,
            message=message,
            retryable=False,
        ),
    )


def _current_writable_attempt(
    run_dir: Path,
    run_record: RunRecord,
) -> WritableAttempt | None:
    # A controller crash during a writable phase is deliberately never
    # reconstructed from disk. The caller will classify it as HUMAN_REQUIRED.
    del run_dir, run_record
    return None


def _repository_uncertain_stop(
    message: str,
    *,
    process_started: bool | None,
    inspection_error: str | None,
) -> TerminalStop:
    detail_parts = [message]
    if process_started is None:
        detail_parts.append("Writable process start could not be established.")
    elif process_started:
        detail_parts.append("Writable process may have started.")
    if inspection_error is not None:
        detail_parts.append(
            f"Post-call workspace inspection failed: {inspection_error}"
        )
    if not detail_parts[-1].endswith("."):
        detail_parts[-1] += "."
    return TerminalStop(
        state=WorkflowState.HUMAN_REQUIRED,
        reason=StopReason(
            category=StopCategory.REPOSITORY_UNCERTAIN,
            message=" ".join(detail_parts),
            retryable=False,
        ),
    )


__all__ = [
    "TerminalStop",
    "classify_stage_stop",
    "classify_unexpected_controller_failure",
    "classify_writable_failure",
]
