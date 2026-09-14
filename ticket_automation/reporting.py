"""Render human handoff material from controller-owned evidence.

This module has no state-transition or acceptance authority.  The workflow
controller decides terminal state from the canonical workspace snapshot before
asking this renderer to describe the result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .attempts import (
    attempt_result_path,
    latest_attempt,
    latest_writable_attempt,
    load_attempt_records,
)
from .audit import (
    changed_files_including_untracked,
    diff_including_untracked,
    diff_stats_including_untracked,
)
from .git import GitCommandError, GitRepository
from .git_safety import WorkspaceSnapshot
from .models import StageOutcome, WorkflowState
from .runs import RUN_RECORD_FILE, RunError, RunRecord, load_run_record

FINAL_PATCH_FILE = "final.patch"
FINAL_REPORT_FILE = "report.md"


class ReportError(RunError):
    pass


@dataclass(frozen=True)
class GitSafetyStatus:
    branch_expected: str
    branch_actual: str
    head_expected: str
    head_actual: str
    staged_files: tuple[str, ...]
    inspection_error: str | None = None
    _workspace_snapshot: WorkspaceSnapshot | None = None

    @property
    def branch_ok(self) -> bool:
        return (
            self.inspection_error is None and self.branch_expected == self.branch_actual
        )

    @property
    def head_ok(self) -> bool:
        return self.inspection_error is None and self.head_expected == self.head_actual

    @property
    def staging_ok(self) -> bool:
        return self.inspection_error is None and not self.staged_files

    @property
    def safe(self) -> bool:
        return self.branch_ok and self.head_ok and self.staging_ok


@dataclass(frozen=True)
class ReportStageResult:
    run_dir: Path
    run_record: RunRecord
    outcome: StageOutcome
    final_patch_path: Path
    final_report_path: Path
    changed_files: tuple[str, ...]
    additions: int
    deletions: int
    git_safety: GitSafetyStatus
    controller_message: str

    @property
    def successful(self) -> bool:
        return self.outcome == StageOutcome.COMPLETED


def run_report_stage(run_dir: Path | str) -> ReportStageResult:
    run_path = Path(run_dir)
    run_record = load_run_record(run_path / RUN_RECORD_FILE)
    if run_record.state != WorkflowState.READY_FOR_HUMAN:
        raise ReportError(
            f"Report requires a READY_FOR_HUMAN run; found {run_record.state.value}."
        )
    final_patch_path = run_path / FINAL_PATCH_FILE
    try:
        patch_text = final_patch_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ReportError(
            f"The controller did not persist the final patch before reporting: {error}"
        ) from error
    context = collect_report_context(run_path, run_record, patch_text=patch_text)
    final_report_path = run_path / FINAL_REPORT_FILE
    _write_text(final_report_path, render_final_report(context))
    controller = context["controller"]
    return ReportStageResult(
        run_dir=run_path,
        run_record=run_record,
        outcome=StageOutcome.COMPLETED,
        final_patch_path=final_patch_path,
        final_report_path=final_report_path,
        changed_files=tuple(controller["changed_files"]),
        additions=int(controller["additions"]),
        deletions=int(controller["deletions"]),
        git_safety=controller["git_safety"],
        controller_message="Report rendered from persisted run evidence.",
    )


def generate_terminal_report_best_effort(run_dir: Path | str) -> Path | None:
    run_path = Path(run_dir)
    try:
        record = load_run_record(run_path / RUN_RECORD_FILE)
        try:
            patch_text = (run_path / FINAL_PATCH_FILE).read_text(encoding="utf-8")
        except OSError:
            patch_text = ""
        context = collect_report_context(run_path, record, patch_text=patch_text)
        path = run_path / FINAL_REPORT_FILE
        _write_text(path, render_final_report(context))
        return path
    except Exception:  # noqa: BLE001 - never mask the terminal state.
        return None


def collect_report_context(
    run_dir: Path | str,
    run_record: RunRecord,
    *,
    patch_text: str | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    repository = GitRepository(Path(run_record.target_repository_path))
    patch = (
        patch_text if patch_text is not None else _diff_or_empty(repository, run_record)
    )
    records = load_attempt_records(run_path)
    attempts = tuple(_attempt_view(run_path, record) for record in records)
    verification = tuple(item for item in attempts if item["phase"] == "VERIFYING")
    baseline = _latest_attempt_result(run_path, "PREPARING")
    reviews = tuple(item for item in attempts if item["phase"] == "REVIEWING")
    implementation = _latest_attempt_result(run_path, "IMPLEMENTING")
    correction_tickets = tuple(
        item["correction_ticket"]
        for item in attempts
        if isinstance(item.get("correction_ticket"), str)
    )
    changed_files = _changed_files_or_empty(repository, run_record)
    diff_stats = _diff_stats_or_empty(repository, run_record)
    safety = inspect_git_safety(run_record)
    final_review = reviews[-1]["result"] if reviews else None
    additions, deletions = count_patch_changes(patch)
    return {
        "run_dir": run_path,
        "run_record": run_record,
        "controller": {
            "attempts": attempts,
            "changed_files": changed_files,
            "diff_stats": diff_stats,
            "additions": additions,
            "deletions": deletions,
            "baseline_verification": baseline,
            "verification_rounds": verification,
            "review_results": reviews,
            "final_review": final_review,
            "correction_ticket_paths": correction_tickets,
            "git_safety": safety,
            "latest_writable_attempt": latest_writable_attempt(run_path),
            "final_patch_path": FINAL_PATCH_FILE,
            "final_report_path": FINAL_REPORT_FILE,
        },
        "agent": {"implementation": implementation},
    }


def render_final_report(context: dict[str, Any]) -> str:
    record: RunRecord = context["run_record"]
    controller: dict[str, Any] = context["controller"]
    safety: GitSafetyStatus = controller["git_safety"]
    baseline = controller["baseline_verification"]
    verification = controller["verification_rounds"]
    final_review = controller["final_review"]
    lines = [
        f"# {record.ticket_id} report",
        "",
        "## Run",
        "",
        f"- Run ID: {record.run_id}",
        f"- State: {record.state.value}",
        f"- Target repository: {record.target_repository_path}",
        f"- Starting branch: {record.starting_branch}",
        f"- Baseline SHA: {record.baseline_sha}",
        f"- Baseline verification: {_status(baseline)}",
        f"- Verification attempts: {len(verification)}",
        f"- Final review: {_review_verdict(final_review)}",
        f"- Correction rounds: {record.current_correction_round} / {record.max_correction_rounds}",
        "",
        "## Workspace",
        "",
        f"- Branch unchanged: {_yes_no(safety.branch_ok)}",
        f"- HEAD unchanged: {_yes_no(safety.head_ok)}",
        f"- Staging empty: {_yes_no(safety.staging_ok)}",
        f"- Changed files: {len(controller['changed_files'])}",
        f"- Diff line counts: +{controller['additions']} / -{controller['deletions']}",
        f"- Final patch: {controller['final_patch_path']}",
        "",
        "## Attempts",
        "",
    ]
    for attempt in controller["attempts"]:
        lines.append(
            f"- {attempt['sequence']:03d} {attempt['phase']}: {attempt['status']}"
        )
    if not controller["attempts"]:
        lines.append("- none")
    if record.terminal_reason:
        lines.extend(["", "## Human handoff", "", f"- {record.terminal_reason}"])
    lines.append("")
    return "\n".join(lines)


def inspect_git_safety(run_record: RunRecord) -> GitSafetyStatus:
    repository = GitRepository(Path(run_record.target_repository_path))
    try:
        snapshot = WorkspaceSnapshot.capture(repository)
    except (GitCommandError, OSError, RuntimeError, ValueError) as error:
        return GitSafetyStatus(
            branch_expected=run_record.starting_branch,
            branch_actual="<unavailable>",
            head_expected=run_record.baseline_sha,
            head_actual="<unavailable>",
            staged_files=(),
            inspection_error=f"{type(error).__name__}: {error}",
        )
    return GitSafetyStatus(
        branch_expected=run_record.starting_branch,
        branch_actual="<detached>" if snapshot.branch is None else snapshot.branch,
        head_expected=run_record.baseline_sha,
        head_actual=snapshot.head_sha or "<unknown>",
        staged_files=snapshot.staged_paths,
        inspection_error=(
            None
            if snapshot.inspection_complete
            else "; ".join(snapshot.inspection_errors)
        ),
        _workspace_snapshot=snapshot,
    )


def count_patch_changes(patch_text: str) -> tuple[int, int]:
    additions = sum(
        1
        for line in patch_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    deletions = sum(
        1
        for line in patch_text.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    return additions, deletions


def latest_verification_round(run_dir: Path | str) -> dict[str, Any] | None:
    return _latest_attempt_result(Path(run_dir), "VERIFYING")


def latest_review_result(run_dir: Path | str) -> dict[str, Any] | None:
    return _latest_attempt_result(Path(run_dir), "REVIEWING")


def _latest_attempt_result(run_path: Path, phase: str) -> dict[str, Any] | None:
    record = latest_attempt(run_path, phases=(phase,))
    if record is None:
        return None
    path = attempt_result_path(run_path, record)
    if path is None:
        return None
    return _read_json(path)


def _attempt_view(run_path: Path, record: Any) -> dict[str, Any]:
    result_path = attempt_result_path(run_path, record)
    correction_ticket = record.artifact_directory / "correction-ticket.md"
    return {
        "sequence": record.sequence,
        "phase": record.phase,
        "status": record.status,
        "result": None if result_path is None else _read_json(result_path),
        "correction_ticket": (
            correction_ticket.relative_to(run_path).as_posix()
            if correction_ticket.is_file()
            else None
        ),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _diff_or_empty(repository: GitRepository, record: RunRecord) -> str:
    try:
        return diff_including_untracked(repository, record.baseline_sha)
    except (GitCommandError, OSError, ValueError):
        return ""


def _changed_files_or_empty(
    repository: GitRepository, record: RunRecord
) -> tuple[str, ...]:
    try:
        return changed_files_including_untracked(repository, record.baseline_sha)
    except (GitCommandError, OSError, ValueError):
        return ()


def _diff_stats_or_empty(repository: GitRepository, record: RunRecord) -> str:
    try:
        return diff_stats_including_untracked(repository, record.baseline_sha)
    except (GitCommandError, OSError, ValueError):
        return ""


def _status(value: dict[str, Any] | None) -> str:
    return str(value.get("status", "NOT RUN")) if isinstance(value, dict) else "NOT RUN"


def _review_verdict(value: dict[str, Any] | None) -> str:
    return (
        str(value.get("verdict", "NOT RUN")) if isinstance(value, dict) else "NOT RUN"
    )


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


__all__ = [
    "FINAL_PATCH_FILE",
    "FINAL_REPORT_FILE",
    "ReportError",
    "ReportStageResult",
    "changed_files_including_untracked",
    "collect_report_context",
    "count_patch_changes",
    "diff_including_untracked",
    "diff_stats_including_untracked",
    "generate_terminal_report_best_effort",
    "inspect_git_safety",
    "latest_review_result",
    "latest_verification_round",
    "render_final_report",
    "run_report_stage",
]
