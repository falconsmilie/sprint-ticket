"""Outbound port for publishing a report from persisted terminal state."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ...models import StageOutcome
from ...run_ownership import RunOwnership
from ...runs import RunRecord


class GitSafetyPublication(Protocol):
    branch_expected: str
    branch_actual: str
    head_expected: str
    head_actual: str
    staged_files: tuple[str, ...]
    inspection_error: str | None

    @property
    def safe(self) -> bool: ...


class ReportPublication(Protocol):
    run_dir: Path
    run_record: RunRecord
    outcome: StageOutcome
    final_patch_path: Path
    final_report_path: Path
    changed_files: tuple[str, ...]
    additions: int
    deletions: int
    git_safety: GitSafetyPublication
    controller_message: str

    @property
    def successful(self) -> bool: ...


class TerminalReportPublisher(Protocol):
    def publish(
        self,
        run_dir: Path,
        run_record: RunRecord,
        *,
        run_ownership: RunOwnership | None = None,
    ) -> ReportPublication | None: ...


__all__ = [
    "GitSafetyPublication",
    "ReportPublication",
    "TerminalReportPublisher",
]
