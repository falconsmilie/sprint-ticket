"""Final handoff acceptance stage handler."""

from __future__ import annotations

from dataclasses import dataclass

from ...models import StageOutcome, WorkflowState
from ...runs import RunRecord
from ..handoff_acceptance import (
    HandoffAcceptanceRequest,
    HandoffAcceptanceService,
    HandoffAccepted,
)
from ..ports.handoff import FinalPatchCapture
from .contracts import (
    ReportingEvidence,
    StageContext,
    StageDecision,
    decision_for_stage_result,
)


@dataclass(frozen=True)
class HandoffStageResult:
    run_record: RunRecord
    outcome: StageOutcome
    controller_message: str
    after_workspace_fingerprint: str | None
    process_started: bool = False

    @property
    def source_state(self) -> WorkflowState:
        return WorkflowState.REPORTING


@dataclass(frozen=True)
class ReportingStageHandler:
    patch_capture: FinalPatchCapture

    def handle(self, context: StageContext) -> StageDecision[HandoffStageResult]:
        attempt = context.require_attempt()
        handoff = HandoffAcceptanceService(patch_capture=self.patch_capture).accept(
            HandoffAcceptanceRequest(
                run_dir=context.run_dir,
                run_record=context.run_record,
                attempt=attempt,
            )
        )
        if isinstance(handoff, HandoffAccepted):
            outcome = StageOutcome.COMPLETED
            fingerprint = handoff.final_workspace.fingerprint
            stop_category = None
        else:
            outcome = StageOutcome.HUMAN_REQUIRED
            fingerprint = (
                None
                if handoff.final_workspace is None
                else handoff.final_workspace.fingerprint
            )
            stop_category = handoff.stop_category

        result = HandoffStageResult(
            run_record=context.run_record,
            outcome=outcome,
            controller_message=handoff.reason,
            after_workspace_fingerprint=fingerprint,
        )
        return decision_for_stage_result(
            context.run_record,
            result,
            reporting_evidence=ReportingEvidence(
                status=("PASS" if outcome is StageOutcome.COMPLETED else outcome.value),
                message=handoff.reason,
            ),
            stop_category=stop_category,
        )


__all__ = ["HandoffStageResult", "ReportingStageHandler"]
