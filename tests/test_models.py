import pytest

from ticket_automation.models import (
    PHASE_DEFINITIONS,
    AttemptPhase,
    AttemptStatus,
    PhaseDefinition,
    ResultArtifactRole,
    WorkflowState,
    phase_for_active_state,
)


def test_workflow_state_values():
    assert [state.value for state in WorkflowState] == [
        "PREPARING",
        "PREPARED",
        "IMPLEMENTING",
        "VERIFYING",
        "REVIEWING",
        "CORRECTION_PENDING",
        "CORRECTING",
        "REPORTING",
        "READY_FOR_HUMAN",
        "HUMAN_REQUIRED",
        "FAILED",
    ]


def test_phase_catalog_is_complete_and_maps_active_states_once():
    expected = {
        AttemptPhase.PREPARING: (
            WorkflowState.PREPARING,
            False,
            True,
            ResultArtifactRole.BASELINE_VERIFICATION,
            "result.json",
            "preparation",
            "preparation",
        ),
        AttemptPhase.IMPLEMENTING: (
            WorkflowState.IMPLEMENTING,
            True,
            False,
            ResultArtifactRole.IMPLEMENTATION_RESULT,
            "result.json",
            "implementation",
            "implementation",
        ),
        AttemptPhase.VERIFYING: (
            WorkflowState.VERIFYING,
            False,
            True,
            ResultArtifactRole.VERIFICATION_ROUND,
            "result.json",
            "verification",
            "verification",
        ),
        AttemptPhase.REVIEWING: (
            WorkflowState.REVIEWING,
            False,
            True,
            ResultArtifactRole.REVIEW_RESULT,
            "result.json",
            "review",
            "review",
        ),
        AttemptPhase.CORRECTING: (
            WorkflowState.CORRECTING,
            True,
            False,
            ResultArtifactRole.CORRECTION_RESULT,
            "result.json",
            "correction",
            "correction",
        ),
        AttemptPhase.REPORTING: (
            WorkflowState.REPORTING,
            False,
            True,
            ResultArtifactRole.HANDOFF_RESULT,
            "result.json",
            "reporting",
            "reporting",
        ),
    }

    assert set(PHASE_DEFINITIONS) == set(AttemptPhase)
    assert {
        phase: (
            definition.active_state,
            definition.writes_target_repository,
            definition.automatically_retry_interrupted,
            definition.result_artifact_role,
            definition.result_artifact_name,
            definition.display_name,
            definition.slug,
        )
        for phase, definition in PHASE_DEFINITIONS.items()
    } == expected
    assert {
        phase_for_active_state(definition.active_state)
        for definition in PHASE_DEFINITIONS.values()
    } == set(AttemptPhase)
    assert len({definition.slug for definition in PHASE_DEFINITIONS.values()}) == len(
        AttemptPhase
    )


def test_phase_catalog_and_definitions_are_immutable():
    with pytest.raises(TypeError):
        PHASE_DEFINITIONS[AttemptPhase.PREPARING] = PHASE_DEFINITIONS[  # type: ignore[index]
            AttemptPhase.PREPARING
        ]
    with pytest.raises(AttributeError):
        PHASE_DEFINITIONS[AttemptPhase.PREPARING].slug = "changed"  # type: ignore[misc]


def test_attempt_status_values_are_canonical_transport_values():
    assert [status.value for status in AttemptStatus] == [
        "STARTED",
        "COMPLETED",
        "HUMAN_REQUIRED",
        "FAILED",
    ]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("active_state", "PREPARING", TypeError),
        ("writes_target_repository", 1, TypeError),
        ("automatically_retry_interrupted", 1, TypeError),
        ("result_artifact_role", "baseline_verification", TypeError),
        ("result_artifact_name", "", ValueError),
        ("display_name", "", ValueError),
        ("slug", "", ValueError),
    ],
)
def test_phase_definition_rejects_invalid_trusted_construction(
    field: str,
    value: object,
    error: type[Exception],
) -> None:
    values = {
        "active_state": WorkflowState.PREPARING,
        "writes_target_repository": False,
        "automatically_retry_interrupted": True,
        "result_artifact_role": ResultArtifactRole.BASELINE_VERIFICATION,
        "result_artifact_name": "result.json",
        "display_name": "preparation",
        "slug": "preparation",
    }
    values[field] = value

    with pytest.raises(error):
        PhaseDefinition(**values)  # type: ignore[arg-type]
