"""Compatibility facade for report presentation APIs.

Report implementation lives in :mod:`ticket_automation.presentation.reporting`.
This module preserves the established import path for programmatic callers.
"""

from .presentation.lifecycle import format_lifecycle_result
from .presentation.reporting import (
    FINAL_PATCH_FILE,
    FINAL_REPORT_FILE,
    AgentReportView,
    AttemptReportView,
    ControllerReportView,
    FilesystemTerminalReportPublisher,
    ReportError,
    ReportStageResult,
    ReportViewModel,
    changed_files_including_untracked,
    collect_report_context,
    count_patch_changes,
    diff_including_untracked,
    diff_stats_including_untracked,
    generate_terminal_report_best_effort,
    inspect_git_safety,
    latest_review_result,
    latest_verification_round,
    render_final_report,
    run_report_stage,
)

__all__ = [
    "FINAL_PATCH_FILE",
    "FINAL_REPORT_FILE",
    "AgentReportView",
    "AttemptReportView",
    "ControllerReportView",
    "FilesystemTerminalReportPublisher",
    "ReportError",
    "ReportStageResult",
    "ReportViewModel",
    "changed_files_including_untracked",
    "collect_report_context",
    "count_patch_changes",
    "diff_including_untracked",
    "diff_stats_including_untracked",
    "format_lifecycle_result",
    "generate_terminal_report_best_effort",
    "inspect_git_safety",
    "latest_review_result",
    "latest_verification_round",
    "render_final_report",
    "run_report_stage",
]
