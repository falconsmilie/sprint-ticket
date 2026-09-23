from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import StageOutcome, StopCategory, StopReason, WorkflowState
from .runs import RunRecord


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
        del run_dir
        return _repository_uncertain_stop(
            message=message,
            process_started=None,
            inspection_error=None,
        )
    return TerminalStop(
        state=WorkflowState.FAILED,
        reason=StopReason(
            category=StopCategory.CONTROLLER_FAILURE,
            message=message,
            retryable=False,
        ),
    )


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
]
