"""Verification stage handler."""

from __future__ import annotations

from dataclasses import dataclass

from ...config import AppConfig
from ...models import StageOutcome, StopCategory
from ...verification import (
    VerificationProcessRunner,
    VerificationStageResult,
    run_verification_stage,
)
from .contracts import (
    StageContext,
    StageDecision,
    decision_for_stage_result,
)


@dataclass(frozen=True)
class VerificationStageHandler:
    config: AppConfig
    process_runner: VerificationProcessRunner | None = None

    def handle(self, context: StageContext) -> StageDecision:
        result = run_verification_stage(
            self.config,
            context.run_dir,
            process_runner=self.process_runner,
            attempt_record=context.require_attempt(),
            clock=context.clock,
        )
        return decision_for_stage_result(
            context.run_record,
            result,
            stop_category=_stop_category(result),
        )


def _stop_category(result: VerificationStageResult) -> StopCategory | None:
    if result.outcome is not StageOutcome.HUMAN_REQUIRED:
        return None
    if result.round_result.safety_violations:
        return StopCategory.SAFETY_VIOLATION
    return StopCategory.VERIFICATION_INFRASTRUCTURE


__all__ = ["VerificationStageHandler"]
