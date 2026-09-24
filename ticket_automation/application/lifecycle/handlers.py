"""Composition and explicit dispatch for active lifecycle states."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from ...config import AppConfig
from ...models import WorkflowState
from ...runs import RunError
from ...verification import VerificationProcessRunner
from ..agent_execution import AgentExecutorAssignments
from ..ports.handoff import FinalPatchCapture
from .contracts import (
    StageContext,
    StageDecision,
    StageHandler,
    require_trusted_stage_decision,
)
from .correction import CorrectionStageHandler
from .implementation import ImplementationStageHandler
from .preparation import PreparationStageHandler
from .reporting import ReportingStageHandler
from .review import ReviewStageHandler
from .verification import VerificationStageHandler


def build_active_stage_handlers(
    *,
    config: AppConfig,
    executors: AgentExecutorAssignments,
    final_patch_capture: FinalPatchCapture,
    verification_runner: VerificationProcessRunner | None,
) -> Mapping[WorkflowState, StageHandler]:
    preparation = PreparationStageHandler(config, verification_runner)
    implementation = ImplementationStageHandler(config, executors.implementation)
    verification = VerificationStageHandler(config, verification_runner)
    review = ReviewStageHandler(config, executors.review)
    correction = CorrectionStageHandler(config, executors.correction)
    reporting = ReportingStageHandler(final_patch_capture)
    return MappingProxyType(
        {
            WorkflowState.PREPARING: preparation,
            WorkflowState.PREPARED: implementation,
            WorkflowState.IMPLEMENTING: implementation,
            WorkflowState.VERIFYING: verification,
            WorkflowState.REVIEWING: review,
            WorkflowState.CORRECTION_PENDING: correction,
            WorkflowState.CORRECTING: correction,
            WorkflowState.REPORTING: reporting,
        }
    )


def dispatch_stage(
    context: StageContext,
    handlers: Mapping[WorkflowState, StageHandler],
) -> StageDecision:
    """Select exactly one handler for the current typed workflow state."""

    try:
        handler = handlers[context.run_record.state]
    except KeyError as error:
        raise RunError(
            "Lifecycle reached an unsupported non-terminal state: "
            f"{context.run_record.state.value}."
        ) from error
    return require_trusted_stage_decision(handler.handle(context))


__all__ = ["build_active_stage_handlers", "dispatch_stage"]
