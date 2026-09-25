"""Typed contract between lifecycle orchestration and stage handlers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, TypeVar

from ...attempts import StageAttempt
from ...failure_classification import TerminalStop, classify_stage_stop
from ...models import StageOutcome, StopCategory, WorkflowState
from ...run_ownership import RunOwnership
from ...runs import RunError, RunRecord


@dataclass(frozen=True)
class StageContext:
    run_dir: Path
    run_record: RunRecord
    attempt: StageAttempt | None
    clock: Callable[[], datetime] | None
    mark_process_started: Callable[[], None]
    run_ownership: RunOwnership

    def __post_init__(self) -> None:
        if not isinstance(self.run_ownership, RunOwnership):
            raise TypeError("StageContext requires a RunOwnership token.")
        self.run_ownership.validate_run_path(self.run_dir)

    def require_attempt(self) -> StageAttempt:
        if self.attempt is None:
            raise RunError(
                f"Lifecycle state {self.run_record.state.value} requires an attempt."
            )
        return self.attempt


@dataclass(frozen=True)
class ReportingEvidence:
    """Small result artifact emitted only by reporting/handoff coordination."""

    status: StageOutcome
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, StageOutcome):
            raise TypeError("Reporting evidence status must be a StageOutcome.")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("Reporting evidence message must be non-empty.")


class StageResult(Protocol):
    @property
    def source_state(self) -> WorkflowState: ...

    @property
    def outcome(self) -> StageOutcome: ...

    @property
    def controller_message(self) -> str: ...

    @property
    def after_workspace_fingerprint(self) -> str | None: ...

    @property
    def process_started(self) -> bool: ...


class StageCompletion(Protocol):
    @property
    def outcome(self) -> StageOutcome: ...

    @property
    def message(self) -> str: ...

    @property
    def after_workspace_fingerprint(self) -> str | None: ...

    @property
    def process_started(self) -> bool: ...

    @property
    def reporting_evidence(self) -> ReportingEvidence | None: ...


class StageDecision(Protocol):
    @property
    def source_state(self) -> WorkflowState: ...

    @property
    def requested_state(self) -> WorkflowState: ...

    @property
    def result(self) -> StageResult | None: ...

    @property
    def completion(self) -> StageCompletion | None: ...

    @property
    def terminal_stop(self) -> TerminalStop | None: ...

    @property
    def current_correction_round(self) -> int | None: ...

    @property
    def current_review_round(self) -> int | None: ...


class StageHandler(Protocol):
    def handle(self, context: StageContext) -> StageDecision: ...


@dataclass(frozen=True)
class _StageCompletion:
    outcome: StageOutcome
    message: str
    after_workspace_fingerprint: str | None
    process_started: bool
    reporting_evidence: ReportingEvidence | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, StageOutcome):
            raise TypeError("Stage completion outcome must be a StageOutcome.")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("Stage completion message must be non-empty.")
        if self.after_workspace_fingerprint is not None and not isinstance(
            self.after_workspace_fingerprint, str
        ):
            raise TypeError("Stage completion fingerprint must be a string or null.")
        if not isinstance(self.process_started, bool):
            raise TypeError("Stage completion process_started must be a boolean.")


@dataclass(frozen=True)
class _ImmediateStageDecision:
    source_state: WorkflowState
    requested_state: WorkflowState
    terminal_stop: TerminalStop | None = None
    current_correction_round: int | None = None
    current_review_round: int | None = None

    @property
    def result(self) -> None:
        return None

    @property
    def completion(self) -> None:
        return None


@dataclass(frozen=True)
class _ExecutedStageDecision:
    source_state: WorkflowState
    requested_state: WorkflowState
    result: StageResult
    completion: _StageCompletion
    terminal_stop: TerminalStop | None = None
    current_correction_round: int | None = None
    current_review_round: int | None = None


_TRUSTED_DECISION_TYPES = (_ImmediateStageDecision, _ExecutedStageDecision)

_COMPLETED_TRANSITIONS = {
    WorkflowState.PREPARING: WorkflowState.PREPARED,
    WorkflowState.IMPLEMENTING: WorkflowState.VERIFYING,
    WorkflowState.VERIFYING: WorkflowState.REVIEWING,
    WorkflowState.REVIEWING: WorkflowState.REPORTING,
    WorkflowState.CORRECTING: WorkflowState.VERIFYING,
    WorkflowState.REPORTING: WorkflowState.READY_FOR_HUMAN,
}


def require_trusted_stage_decision(value: object) -> StageDecision:
    if not isinstance(value, _TRUSTED_DECISION_TYPES):
        raise RunError(
            "Stage handler returned a decision that was not built by the lifecycle "
            "decision factories."
        )
    return value


def immediate_transition(
    source_state: WorkflowState,
    requested_state: WorkflowState,
) -> StageDecision:
    """Build a typed request; domain transition legality remains controller-owned."""

    _require_state(source_state, "source_state")
    _require_state(requested_state, "requested_state")
    return _ImmediateStageDecision(source_state, requested_state)


def immediate_terminal_stop(
    source_state: WorkflowState,
    stop: TerminalStop,
) -> StageDecision:
    if source_state is not WorkflowState.CORRECTION_PENDING:
        raise ValueError("Only correction planning may stop without an attempt.")
    if stop.state is not WorkflowState.HUMAN_REQUIRED:
        raise ValueError("Correction planning must stop in HUMAN_REQUIRED.")
    return _ImmediateStageDecision(
        source_state=source_state,
        requested_state=stop.state,
        terminal_stop=stop,
    )


ResultT = TypeVar("ResultT", bound=StageResult)


def decision_for_stage_result(
    record: RunRecord,
    result: ResultT,
    *,
    stop_category: StopCategory | None = None,
    reporting_evidence: ReportingEvidence | None = None,
    current_correction_round: int | None = None,
    current_review_round: int | None = None,
) -> StageDecision:
    """Interpret one typed result and derive all persisted completion values from it."""

    _validate_stage_result(record, result)
    if reporting_evidence is not None and record.state is not WorkflowState.REPORTING:
        raise ValueError("Only reporting decisions may carry reporting evidence.")
    if (
        reporting_evidence is not None
        and reporting_evidence.message != result.controller_message
    ):
        raise ValueError("Reporting evidence must describe the stage result message.")
    if (
        reporting_evidence is not None
        and reporting_evidence.status is not result.outcome
    ):
        raise ValueError("Reporting evidence status must match the stage outcome.")

    stop = None
    if result.outcome in {StageOutcome.HUMAN_REQUIRED, StageOutcome.FAILED}:
        stop = classify_stage_stop(
            record,
            result.outcome,
            message=result.controller_message,
            category=stop_category,
        )
    requested_state = _requested_state_for_outcome(record.state, result.outcome, stop)
    _validate_round_updates(
        record.state,
        current_correction_round=current_correction_round,
        current_review_round=current_review_round,
    )
    return _ExecutedStageDecision(
        source_state=record.state,
        requested_state=requested_state,
        result=result,
        completion=_StageCompletion(
            outcome=result.outcome,
            message=result.controller_message,
            after_workspace_fingerprint=result.after_workspace_fingerprint,
            process_started=result.process_started,
            reporting_evidence=reporting_evidence,
        ),
        terminal_stop=stop,
        current_correction_round=current_correction_round,
        current_review_round=current_review_round,
    )


def _validate_stage_result(record: RunRecord, result: StageResult) -> None:
    if result.source_state is not record.state:
        raise RunError("Stage result does not belong to the current run state.")
    if not isinstance(result.outcome, StageOutcome):
        raise TypeError("Stage result outcome must be a StageOutcome.")
    if (
        not isinstance(result.controller_message, str)
        or not result.controller_message.strip()
    ):
        raise ValueError("Stage result controller_message must be non-empty.")
    if result.after_workspace_fingerprint is not None and not isinstance(
        result.after_workspace_fingerprint, str
    ):
        raise TypeError("Stage result fingerprint must be a string or null.")
    if not isinstance(result.process_started, bool):
        raise TypeError("Stage result process_started must be a boolean.")


def _validate_round_updates(
    source_state: WorkflowState,
    *,
    current_correction_round: int | None,
    current_review_round: int | None,
) -> None:
    if (
        current_correction_round is not None
        and source_state is not WorkflowState.CORRECTING
    ):
        raise ValueError("Only correction decisions may update correction rounds.")
    if current_review_round is not None and source_state is not WorkflowState.REVIEWING:
        raise ValueError("Only review decisions may update review rounds.")


def _requested_state_for_outcome(
    source_state: WorkflowState,
    outcome: StageOutcome,
    stop: TerminalStop | None,
) -> WorkflowState:
    if outcome in {StageOutcome.HUMAN_REQUIRED, StageOutcome.FAILED}:
        if stop is None:
            raise ValueError("Terminal stage outcomes require a classified stop.")
        return stop.state
    if stop is not None:
        raise ValueError("Non-terminal stage outcomes cannot carry a terminal stop.")
    if outcome is StageOutcome.CORRECTION_REQUIRED and source_state in {
        WorkflowState.VERIFYING,
        WorkflowState.REVIEWING,
    }:
        return WorkflowState.CORRECTION_PENDING
    if outcome is StageOutcome.COMPLETED:
        try:
            return _COMPLETED_TRANSITIONS[source_state]
        except KeyError as error:
            raise RunError(
                "Stage completion is not valid from workflow state "
                f"{source_state.value}."
            ) from error
    raise RunError(
        f"Stage outcome {outcome.value} is not valid from workflow state "
        f"{source_state.value}."
    )


def _require_state(value: object, field: str) -> None:
    if not isinstance(value, WorkflowState):
        raise TypeError(f"Stage decision {field} must be a WorkflowState.")


__all__ = [
    "ReportingEvidence",
    "StageAttempt",
    "StageCompletion",
    "StageContext",
    "StageDecision",
    "StageHandler",
    "StageResult",
    "decision_for_stage_result",
    "immediate_terminal_stop",
    "immediate_transition",
    "require_trusted_stage_decision",
]
