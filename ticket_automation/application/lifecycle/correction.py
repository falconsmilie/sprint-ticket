"""Correction planning and execution stage handler."""

from __future__ import annotations

from dataclasses import dataclass

from ...config import AppConfig
from ...correction_planner import plan_pending_correction
from ...corrections import CorrectionStageResult, run_correction_stage
from ...failure_classification import TerminalStop
from ...models import StageOutcome, StopCategory, StopReason, WorkflowState
from ..agent_execution import AgentExecutor
from .contracts import (
    StageContext,
    StageDecision,
    decision_for_stage_result,
    immediate_terminal_stop,
    immediate_transition,
)


@dataclass(frozen=True)
class CorrectionStageHandler:
    config: AppConfig
    agent_executor: AgentExecutor

    def handle(self, context: StageContext) -> StageDecision:
        record = context.run_record
        if record.state is WorkflowState.CORRECTION_PENDING:
            if record.current_correction_round >= record.max_correction_rounds:
                reason = StopReason(
                    category=StopCategory.HUMAN_JUDGMENT_REQUIRED,
                    message=(
                        "Maximum corrective rounds exhausted; human intervention "
                        "is required."
                    ),
                    retryable=False,
                )
                return immediate_terminal_stop(
                    WorkflowState.CORRECTION_PENDING,
                    TerminalStop(
                        state=WorkflowState.HUMAN_REQUIRED,
                        reason=reason,
                    ),
                )
            return immediate_transition(
                WorkflowState.CORRECTION_PENDING, WorkflowState.CORRECTING
            )

        cause_set = plan_pending_correction(context.run_dir)
        result = run_correction_stage(
            self.config,
            context.run_dir,
            cause_set=cause_set,
            agent_executor=self.agent_executor,
            attempt_record=context.require_attempt(),
            clock=context.clock,
        )
        return decision_for_stage_result(
            record,
            result,
            stop_category=_stop_category(result),
            current_correction_round=(
                result.correction_round
                if result.advance_correction_round
                else record.current_correction_round
            ),
        )


def _stop_category(result: CorrectionStageResult) -> StopCategory | None:
    if result.outcome is not StageOutcome.HUMAN_REQUIRED:
        return None
    if result.agent_execution is not None and result.agent_execution.failure_category:
        return StopCategory.REPOSITORY_UNCERTAIN
    if result.safety_violations or (
        result.workspace_guard is not None and result.workspace_guard.has_violation
    ):
        return StopCategory.SAFETY_VIOLATION
    if result.agent_result is not None:
        return StopCategory.HUMAN_JUDGMENT_REQUIRED
    return StopCategory.REPOSITORY_UNCERTAIN


__all__ = ["CorrectionStageHandler"]
