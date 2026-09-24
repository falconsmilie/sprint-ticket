"""Implementation stage handler."""

from __future__ import annotations

from dataclasses import dataclass

from ...config import AppConfig
from ...implementation import ImplementationStageResult, run_implementation_stage
from ...models import StageOutcome, StopCategory, WorkflowState
from ..agent_execution import AgentExecutor
from .contracts import (
    StageContext,
    StageDecision,
    decision_for_stage_result,
    immediate_transition,
)


@dataclass(frozen=True)
class ImplementationStageHandler:
    config: AppConfig
    agent_executor: AgentExecutor

    def handle(self, context: StageContext) -> StageDecision:
        if context.run_record.state is WorkflowState.PREPARED:
            return immediate_transition(
                WorkflowState.PREPARED, WorkflowState.IMPLEMENTING
            )
        result = run_implementation_stage(
            self.config,
            context.run_dir,
            agent_executor=self.agent_executor,
            attempt_record=context.require_attempt(),
            clock=context.clock,
        )
        return decision_for_stage_result(
            context.run_record,
            result,
            stop_category=_stop_category(result),
        )


def _stop_category(result: ImplementationStageResult) -> StopCategory | None:
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


__all__ = ["ImplementationStageHandler"]
