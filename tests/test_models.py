from ticket_automation.models import WorkflowState


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
