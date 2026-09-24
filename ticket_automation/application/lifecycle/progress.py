"""Lifecycle result accumulation kept outside controller dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ...corrections import CorrectionStageResult
from ...implementation import ImplementationStageResult
from ...models import WorkflowState
from ...review import ReviewStageResult
from ...verification import VerificationStageResult
from .contracts import StageResult


@dataclass
class LifecycleProgress:
    implementation_result: ImplementationStageResult | None = None
    verification_results: list[VerificationStageResult] = field(default_factory=list)
    review_results: list[ReviewStageResult] = field(default_factory=list)
    correction_results: list[CorrectionStageResult] = field(default_factory=list)

    def record(self, result: StageResult | None, *, run_record) -> None:
        if isinstance(result, ImplementationStageResult):
            self.implementation_result = replace(result, run_record=run_record)
        elif (
            isinstance(result, VerificationStageResult)
            and result.source_state is WorkflowState.VERIFYING
        ):
            self.verification_results.append(replace(result, run_record=run_record))
        elif isinstance(result, ReviewStageResult):
            self.review_results.append(replace(result, run_record=run_record))
        elif isinstance(result, CorrectionStageResult):
            self.correction_results.append(replace(result, run_record=run_record))


__all__ = ["LifecycleProgress"]
