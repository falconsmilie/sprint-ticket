from __future__ import annotations

from enum import StrEnum


class WorkflowState(StrEnum):
    PREFLIGHT = "PREFLIGHT"
    SNAPSHOT = "SNAPSHOT"
    IMPLEMENT = "IMPLEMENT"
    VERIFY = "VERIFY"
    REVIEW = "REVIEW"
    CORRECT = "CORRECT"
    REPORT = "REPORT"
    READY_FOR_HUMAN = "READY_FOR_HUMAN"
    HUMAN_REQUIRED = "HUMAN_REQUIRED"
    FAILED = "FAILED"
