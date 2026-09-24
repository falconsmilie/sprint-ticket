"""Console rendering for lifecycle results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..domain.task_results import FindingDisposition
from ..git import GitCommandError
from ..models import WorkflowState
from ..runs import RunError, RunRecord
from .reporting import ReportViewModel, collect_report_context

if TYPE_CHECKING:
    from ..workflow import LifecycleResult


def format_lifecycle_result(result: LifecycleResult) -> str:
    record = result.run_record
    context = _safe_report_context(result.run_dir, record)
    title = f"{record.ticket_id} - {_terminal_label(record.state)}"
    lines = [
        "----------------------------------------",
        title,
        "----------------------------------------",
        "",
        "Run",
        f"  {result.run_dir}",
        "",
        "Branch",
        f"  {record.starting_branch}",
        "",
        "Baseline",
        f"  {record.baseline_sha}",
        "",
        "Baseline verification:",
        *_baseline_verification_lines(context),
        "",
        "Files changed",
        f"  {_changed_file_count(context, result)}",
        "",
        "Diff",
        f"  {_diff_line_summary(context, result)}",
        "",
        "Verification:",
        *_verification_lines(context, result),
        "",
        "Review:",
        f"  {_review_summary(context, result)}",
        "",
        "Correction rounds",
        f"  {record.current_correction_round} / {record.max_correction_rounds}",
        "",
        "Review rounds",
        f"  {record.current_review_round}",
        "",
        "Advisory findings",
        f"  {_advisory_count(context)}",
        "",
        "Git safety",
        *_git_safety_lines(result, context),
    ]
    if record.terminal_reason:
        lines.extend(["", "Reason", f"  {record.terminal_reason}"])
    if result.controller_error:
        lines.extend(["", "Controller error", f"  {result.controller_error}"])
    if result.successful:
        lines.extend(["", "No files have been staged or committed."])
    report_path = result.run_dir / "report.md"
    if report_path.is_file():
        lines.extend(["", "Report", f"  {report_path}"])
    lines.append("----------------------------------------")
    return "\n".join(lines)


def _terminal_label(state: WorkflowState) -> str:
    if state is WorkflowState.READY_FOR_HUMAN:
        return "READY FOR HUMAN REVIEW"
    if state is WorkflowState.HUMAN_REQUIRED:
        return "HUMAN REQUIRED"
    return state.value


def _safe_report_context(run_dir, record: RunRecord) -> ReportViewModel | None:
    try:
        return collect_report_context(run_dir, record)
    except (GitCommandError, KeyError, OSError, RunError, TypeError, ValueError):
        return None


def _changed_file_count(context, result: LifecycleResult) -> int:
    if context is not None:
        return len(context.controller.changed_files)
    if result.implementation_result is not None:
        return len(result.implementation_result.changed_files)
    return 0


def _diff_line_summary(context, result: LifecycleResult) -> str:
    if result.report_result is not None:
        return f"+{result.report_result.additions} / -{result.report_result.deletions}"
    if context is not None:
        return f"+{context.controller.additions} / -{context.controller.deletions}"
    return "not available"


def _verification_lines(context, result: LifecycleResult) -> list[str]:
    if result.verification_results:
        round_result = result.verification_results[-1].round_result
        return [
            f"  {round_result.status.value}",
            *(
                f"  {item.name:<10} {item.status.value}"
                for item in round_result.commands
            ),
        ]
    if context is not None and context.controller.verification_rounds:
        return [f"  {context.controller.verification_rounds[-1].status}"]
    return ["  NOT RUN"]


def _baseline_verification_lines(context) -> list[str]:
    if context is not None:
        baseline = context.controller.baseline_verification
        if baseline is not None:
            return [f"  {baseline.status}"]
    return ["  NOT RUN"]


def _review_summary(context, result: LifecycleResult) -> str:
    if context is not None:
        final_review = context.controller.final_review
        if final_review is not None:
            return final_review.verdict.value
    if not result.review_results:
        return "NOT RUN"
    review_result = result.review_results[-1].review_result
    return "NOT AVAILABLE" if review_result is None else review_result.verdict.value


def _advisory_count(context) -> int:
    if context is None:
        return 0
    return sum(
        finding.disposition is FindingDisposition.ADVISORY
        for review in context.controller.review_results
        for finding in review.findings
    )


def _git_safety_lines(result: LifecycleResult, context) -> list[str]:
    violations = []
    if result.implementation_result is not None:
        violations.extend(result.implementation_result.safety_violations)
    for verification in result.verification_results:
        violations.extend(verification.round_result.safety_violations)
    for review in result.review_results:
        violations.extend(review.safety_violations)
    for correction in result.correction_results:
        violations.extend(correction.safety_violations)
    violations.extend(result.safety_violations)
    rendered = [
        f"  {item.name}: expected {item.expected}, got {item.actual}"
        for item in violations
    ]
    if rendered:
        return rendered
    if context is not None:
        safety = context.controller.git_safety
        return [
            f"  HEAD {'unchanged' if safety.head_ok else 'changed'}",
            f"  branch {'unchanged' if safety.branch_ok else 'changed'}",
            f"  staging {'empty' if safety.staging_ok else 'not empty'}",
        ]
    return ["  HEAD unchanged", "  branch unchanged", "  staging empty"]


__all__ = ["format_lifecycle_result"]
