"""Canonical lifecycle vocabulary and phase policy.

Enum values are the exact persisted tokens. Persistence readers accept only
those values; this module intentionally defines no aliases for older schemas.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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


class AttemptPhase(StrEnum):
    PREPARING = "PREPARING"
    IMPLEMENTING = "IMPLEMENTING"
    VERIFYING = "VERIFYING"
    REVIEWING = "REVIEWING"
    CORRECTING = "CORRECTING"
    REPORTING = "REPORTING"


class AttemptStatus(StrEnum):
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    HUMAN_REQUIRED = "HUMAN_REQUIRED"
    FAILED = "FAILED"


class StageOutcome(StrEnum):
    COMPLETED = "COMPLETED"
    CORRECTION_REQUIRED = "CORRECTION_REQUIRED"
    HUMAN_REQUIRED = "HUMAN_REQUIRED"
    FAILED = "FAILED"


class StopCategory(StrEnum):
    """Stable classification for a terminal automation stop."""

    PREPARATION_REJECTED = "PREPARATION_REJECTED"
    BASELINE_FAILURE = "BASELINE_FAILURE"
    EXTERNAL_TOOL_FAILURE = "EXTERNAL_TOOL_FAILURE"
    VERIFICATION_INFRASTRUCTURE = "VERIFICATION_INFRASTRUCTURE"
    SAFETY_VIOLATION = "SAFETY_VIOLATION"
    REPOSITORY_UNCERTAIN = "REPOSITORY_UNCERTAIN"
    HUMAN_JUDGMENT_REQUIRED = "HUMAN_JUDGMENT_REQUIRED"
    CONTROLLER_FAILURE = "CONTROLLER_FAILURE"


class ResultArtifactRole(StrEnum):
    BASELINE_VERIFICATION = "baseline_verification"
    IMPLEMENTATION_RESULT = "implementation_result"
    VERIFICATION_ROUND = "verification_round"
    REVIEW_RESULT = "review_result"
    CORRECTION_RESULT = "correction_result"
    HANDOFF_RESULT = "handoff_result"


ATTEMPT_RESULT_ARTIFACT_NAME = "result.json"


@dataclass(frozen=True)
class PhaseDefinition:
    active_state: WorkflowState
    writes_target_repository: bool
    automatically_retry_interrupted: bool
    result_artifact_role: ResultArtifactRole
    result_artifact_name: str
    display_name: str
    slug: str

    def __post_init__(self) -> None:
        if not isinstance(self.active_state, WorkflowState):
            raise TypeError("Phase active_state must be a WorkflowState value.")
        if not isinstance(self.writes_target_repository, bool):
            raise TypeError("Phase writes_target_repository must be a boolean.")
        if not isinstance(self.automatically_retry_interrupted, bool):
            raise TypeError("Phase automatically_retry_interrupted must be a boolean.")
        if not isinstance(self.result_artifact_role, ResultArtifactRole):
            raise TypeError(
                "Phase result_artifact_role must be a ResultArtifactRole value."
            )
        for field, value in (
            ("result_artifact_name", self.result_artifact_name),
            ("display_name", self.display_name),
            ("slug", self.slug),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Phase {field} must be a non-empty string.")


PHASE_DEFINITIONS: Final[Mapping[AttemptPhase, PhaseDefinition]] = MappingProxyType(
    {
        AttemptPhase.PREPARING: PhaseDefinition(
            active_state=WorkflowState.PREPARING,
            writes_target_repository=False,
            automatically_retry_interrupted=True,
            result_artifact_role=ResultArtifactRole.BASELINE_VERIFICATION,
            result_artifact_name=ATTEMPT_RESULT_ARTIFACT_NAME,
            display_name="preparation",
            slug="preparation",
        ),
        AttemptPhase.IMPLEMENTING: PhaseDefinition(
            active_state=WorkflowState.IMPLEMENTING,
            writes_target_repository=True,
            automatically_retry_interrupted=False,
            result_artifact_role=ResultArtifactRole.IMPLEMENTATION_RESULT,
            result_artifact_name=ATTEMPT_RESULT_ARTIFACT_NAME,
            display_name="implementation",
            slug="implementation",
        ),
        AttemptPhase.VERIFYING: PhaseDefinition(
            active_state=WorkflowState.VERIFYING,
            writes_target_repository=False,
            automatically_retry_interrupted=True,
            result_artifact_role=ResultArtifactRole.VERIFICATION_ROUND,
            result_artifact_name=ATTEMPT_RESULT_ARTIFACT_NAME,
            display_name="verification",
            slug="verification",
        ),
        AttemptPhase.REVIEWING: PhaseDefinition(
            active_state=WorkflowState.REVIEWING,
            writes_target_repository=False,
            automatically_retry_interrupted=True,
            result_artifact_role=ResultArtifactRole.REVIEW_RESULT,
            result_artifact_name=ATTEMPT_RESULT_ARTIFACT_NAME,
            display_name="review",
            slug="review",
        ),
        AttemptPhase.CORRECTING: PhaseDefinition(
            active_state=WorkflowState.CORRECTING,
            writes_target_repository=True,
            automatically_retry_interrupted=False,
            result_artifact_role=ResultArtifactRole.CORRECTION_RESULT,
            result_artifact_name=ATTEMPT_RESULT_ARTIFACT_NAME,
            display_name="correction",
            slug="correction",
        ),
        AttemptPhase.REPORTING: PhaseDefinition(
            active_state=WorkflowState.REPORTING,
            writes_target_repository=False,
            automatically_retry_interrupted=True,
            result_artifact_role=ResultArtifactRole.HANDOFF_RESULT,
            result_artifact_name=ATTEMPT_RESULT_ARTIFACT_NAME,
            display_name="reporting",
            slug="reporting",
        ),
    }
)

TERMINAL_WORKFLOW_STATES: Final[frozenset[WorkflowState]] = frozenset(
    {
        WorkflowState.READY_FOR_HUMAN,
        WorkflowState.HUMAN_REQUIRED,
        WorkflowState.FAILED,
    }
)

ATTEMPT_STATUS_BY_STAGE_OUTCOME: Final[Mapping[StageOutcome, AttemptStatus]] = (
    MappingProxyType(
        {
            StageOutcome.COMPLETED: AttemptStatus.COMPLETED,
            StageOutcome.CORRECTION_REQUIRED: AttemptStatus.COMPLETED,
            StageOutcome.HUMAN_REQUIRED: AttemptStatus.HUMAN_REQUIRED,
            StageOutcome.FAILED: AttemptStatus.FAILED,
        }
    )
)


def phase_for_active_state(state: WorkflowState) -> AttemptPhase | None:
    for phase, definition in PHASE_DEFINITIONS.items():
        if definition.active_state is state:
            return phase
    return None


@dataclass(frozen=True)
class StopReason:
    """Persisted explanation for a non-successful terminal workflow state."""

    category: StopCategory
    message: str
    retryable: bool

    def __post_init__(self) -> None:
        if not isinstance(self.category, StopCategory):
            raise TypeError("Stop reason category must be a StopCategory value.")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("Stop reason message must be a non-empty string.")
        if not isinstance(self.retryable, bool):
            raise TypeError("Stop reason retryable must be a boolean.")


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


def validate_workflow_transition(
    current: WorkflowState,
    requested: WorkflowState,
) -> None:
    if requested not in _LEGAL_WORKFLOW_TRANSITIONS[current]:
        raise ValueError(
            f"Invalid workflow transition: {current.value} -> {requested.value}."
        )
