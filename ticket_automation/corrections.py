from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class CorrectionReasonKind(StrEnum):
    VERIFICATION_FAILURE = "VerificationFailure"
    REVIEW_FINDING = "ReviewFinding"


@dataclass(frozen=True)
class VerificationFailure:
    gate_name: str
    command: tuple[str, ...]
    failure_summary: str
    stdout_excerpt: str
    stderr_excerpt: str
    exit_code: int | None
    log_path: Path

    @property
    def kind(self) -> CorrectionReasonKind:
        return CorrectionReasonKind.VERIFICATION_FAILURE

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "gate_name": self.gate_name,
            "command": list(self.command),
            "failure_summary": self.failure_summary,
            "stdout_excerpt": self.stdout_excerpt,
            "stderr_excerpt": self.stderr_excerpt,
            "exit_code": self.exit_code,
            "log_path": str(self.log_path),
        }


@dataclass(frozen=True)
class ReviewFinding:
    finding_id: str
    summary: str
    details: str

    @property
    def kind(self) -> CorrectionReasonKind:
        return CorrectionReasonKind.REVIEW_FINDING

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "finding_id": self.finding_id,
            "summary": self.summary,
            "details": self.details,
        }


CorrectionReason = VerificationFailure | ReviewFinding


__all__ = [
    "CorrectionReason",
    "CorrectionReasonKind",
    "ReviewFinding",
    "VerificationFailure",
]
