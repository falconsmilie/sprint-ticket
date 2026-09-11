import unittest

from ticket_automation.models import WorkflowState


class WorkflowStateTests(unittest.TestCase):
    def test_workflow_state_values(self):
        self.assertEqual(
            [state.value for state in WorkflowState],
            [
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
            ],
        )

