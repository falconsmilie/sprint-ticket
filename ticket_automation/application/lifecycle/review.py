"""Independent review stage handler."""

from __future__ import annotations

from dataclasses import dataclass

from ...config import AppConfig
from ...models import StageOutcome, StopCategory
from ...review import ReviewStageResult, run_review_stage
from ..agent_execution import AgentExecutor
from .contracts import (
    StageContext,
    StageDecision,
    decision_for_stage_result,
)


@dataclass(frozen=True)
class ReviewStageHandler:
    config: AppConfig
    agent_executor: AgentExecutor

    def handle(self, context: StageContext) -> StageDecision:
        result = run_review_stage(
            self.config,
            context.run_dir,
            agent_executor=self.agent_executor,
            attempt_record=context.require_attempt(),
            mark_process_started=context.mark_process_started,
            clock=context.clock,
            run_ownership=context.run_ownership,
        )
        return decision_for_stage_result(
            context.run_record,
            result,
            stop_category=_stop_category(result),
            current_review_round=context.run_record.current_review_round + 1,
        )


def _stop_category(result: ReviewStageResult) -> StopCategory | None:
    if result.outcome is not StageOutcome.HUMAN_REQUIRED:
        return None
    if result.safety_violations:
        return StopCategory.SAFETY_VIOLATION
    return StopCategory.HUMAN_JUDGMENT_REQUIRED


__all__ = ["ReviewStageHandler"]
