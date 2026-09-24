"""Preparation stage handler."""

from __future__ import annotations

from dataclasses import dataclass

from ...config import AppConfig
from ...models import StageOutcome, StopCategory, WorkflowState
from ...runs import RunError
from ...verification import VerificationProcessRunner, run_baseline_verification_stage
from .contracts import (
    StageContext,
    StageDecision,
    decision_for_stage_result,
)


@dataclass(frozen=True)
class PreparationFailureResult:
    controller_message: str
    outcome: StageOutcome = StageOutcome.HUMAN_REQUIRED
    after_workspace_fingerprint: str | None = None
    process_started: bool = False

    @property
    def source_state(self) -> WorkflowState:
        return WorkflowState.PREPARING


@dataclass(frozen=True)
class PreparationStageHandler:
    config: AppConfig
    process_runner: VerificationProcessRunner | None = None

    def handle(self, context: StageContext) -> StageDecision:
        attempt = context.require_attempt()
        try:
            result = run_baseline_verification_stage(
                self.config,
                context.run_dir,
                process_runner=self.process_runner,
                attempt_record=attempt,
                clock=context.clock,
            )
        except (OSError, RunError) as error:
            message = (
                "Clean baseline verification could not be completed: "
                f"{type(error).__name__}: {error}"
            )
            failure = PreparationFailureResult(message)
            return decision_for_stage_result(
                context.run_record,
                failure,
                stop_category=StopCategory.BASELINE_FAILURE,
            )
        return decision_for_stage_result(
            context.run_record,
            result,
            stop_category=(
                StopCategory.BASELINE_FAILURE
                if result.outcome is StageOutcome.HUMAN_REQUIRED
                else None
            ),
        )


__all__ = ["PreparationFailureResult", "PreparationStageHandler"]
