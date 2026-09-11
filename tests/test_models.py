from ticket_automation.models import WorkflowState


def test_workflow_state_values():
    assert [state.value for state in WorkflowState] == [
        "PREFLIGHT",
        "SNAPSHOT",
        "IMPLEMENT",
        "VERIFY",
        "REVIEW",
        "CORRECT",
        "REPORT",
        "READY_FOR_HUMAN",
        "HUMAN_REQUIRED",
        "FAILED",
    ]
