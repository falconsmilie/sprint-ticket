from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class WorkflowState(StrEnum):
    PREPARING = "PREPARING"
    PREPARED = "PREPARED"
    IMPLEMENTING = "IMPLEMENTING"
    VERIFYING = "VERIFYING"
    REVIEWING = "REVIEWING"
    CORRECTION_PENDING = "CORRECTION_PENDING"
    CORRECTING = "CORRECTING"
    REPORTING = "REPORTING"
    READY_FOR_HUMAN = "READY_FOR_HUMAN"
    HUMAN_REQUIRED = "HUMAN_REQUIRED"
    FAILED = "FAILED"


class StageOutcome(StrEnum):
    COMPLETED = "COMPLETED"
    CORRECTION_REQUIRED = "CORRECTION_REQUIRED"
    HUMAN_REQUIRED = "HUMAN_REQUIRED"
    FAILED = "FAILED"


_ACTIVE_FAILURE_TRANSITIONS = frozenset(
    {WorkflowState.HUMAN_REQUIRED, WorkflowState.FAILED}
)

_LEGAL_WORKFLOW_TRANSITIONS: Final[Mapping[WorkflowState, frozenset[WorkflowState]]] = (
    MappingProxyType(
        {
            WorkflowState.PREPARING: frozenset({WorkflowState.PREPARED})
            | _ACTIVE_FAILURE_TRANSITIONS,
            WorkflowState.PREPARED: frozenset({WorkflowState.IMPLEMENTING})
            | _ACTIVE_FAILURE_TRANSITIONS,
            WorkflowState.IMPLEMENTING: frozenset({WorkflowState.VERIFYING})
            | _ACTIVE_FAILURE_TRANSITIONS,
            WorkflowState.VERIFYING: frozenset(
                {WorkflowState.REVIEWING, WorkflowState.CORRECTION_PENDING}
            )
            | _ACTIVE_FAILURE_TRANSITIONS,
            WorkflowState.REVIEWING: frozenset(
                {WorkflowState.REPORTING, WorkflowState.CORRECTION_PENDING}
            )
            | _ACTIVE_FAILURE_TRANSITIONS,
            WorkflowState.CORRECTION_PENDING: frozenset({WorkflowState.CORRECTING})
            | _ACTIVE_FAILURE_TRANSITIONS,
            WorkflowState.CORRECTING: frozenset({WorkflowState.VERIFYING})
            | _ACTIVE_FAILURE_TRANSITIONS,
            WorkflowState.REPORTING: frozenset({WorkflowState.READY_FOR_HUMAN})
            | _ACTIVE_FAILURE_TRANSITIONS,
            WorkflowState.READY_FOR_HUMAN: frozenset(),
            WorkflowState.HUMAN_REQUIRED: frozenset(),
            WorkflowState.FAILED: frozenset(),
        }
    )
)


def _validate_workflow_transition(
    current: WorkflowState,
    requested: WorkflowState,
) -> None:
    if requested not in _LEGAL_WORKFLOW_TRANSITIONS[current]:
        raise ValueError(
            f"Invalid workflow transition: {current.value} -> {requested.value}."
        )
