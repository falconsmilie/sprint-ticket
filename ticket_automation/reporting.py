from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audit import (
    changed_files_including_untracked,
    diff_including_untracked,
    diff_stats_including_untracked,
)
from .git import GitCommandError, GitRepository
from .git_safety import (
    WorkspaceSnapshot,
    _read_workspace_fingerprint,
    _workspace_fingerprint_path,
)
from .models import StageOutcome, WorkflowState
from .runs import RUN_RECORD_FILE, RunError, RunRecord, load_run_record
from .workspace_guard import WORKSPACE_GUARD_DIR_NAME

DIFFS_DIR_NAME = "diffs"
FINAL_PATCH_FILE = "final.patch"
FINAL_REPORT_FILE = "final-report.md"
IMPLEMENTATION_DIR_NAME = "implementation"
CORRECTIONS_DIR_NAME = "corrections"
CORRECTION_EXECUTIONS_DIR_NAME = "correction-executions"
REVIEWS_DIR_NAME = "reviews"
VERIFICATION_DIR_NAME = "verification"
RESULT_JSON = "result.json"


class ReportError(RunError):
    """Raised when the final audit report cannot be generated."""


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


def run_report_stage(
    run_dir: Path | str,
    *,
    transition_record: Callable[[RunRecord, StageOutcome, str | None], RunRecord],
) -> ReportStageResult:
    run_path = Path(run_dir)
    record_path = run_path / RUN_RECORD_FILE
    run_record = load_run_record(record_path)
    if run_record.state != WorkflowState.REPORTING:
        raise ReportError(
            f"Report requires run state REPORTING; found {run_record.state.value}."
        )

    final_patch_path, patch_text = capture_final_patch(run_path, run_record)
    context = collect_report_context(run_path, run_record, patch_text=patch_text)
    acceptance_problem = _acceptance_problem(context)
    if acceptance_problem is None:
        outcome = StageOutcome.COMPLETED
        updated_record = transition_record(
            run_record,
            outcome,
            None,
        )
        controller_message = "Final report generated; ready for human review."
    else:
        outcome = StageOutcome.HUMAN_REQUIRED
        updated_record = transition_record(
            run_record,
            outcome,
            acceptance_problem,
        )
        controller_message = acceptance_problem

    context["run_record"] = updated_record
    report_text = render_final_report(context, terminal_reason=acceptance_problem)
    final_report_path = run_path / FINAL_REPORT_FILE
    _write_text(final_report_path, report_text)

    refreshed_safety = inspect_git_safety(run_record)
    if not refreshed_safety.safe and outcome == StageOutcome.COMPLETED:
        acceptance_problem = "Repository safety invariants changed during reporting."
        outcome = StageOutcome.HUMAN_REQUIRED
        updated_record = transition_record(
            run_record,
            outcome,
            acceptance_problem,
        )
        context = collect_report_context(
            run_path, updated_record, patch_text=patch_text
        )
        context["run_record"] = updated_record
        _write_text(
            final_report_path,
            render_final_report(context, terminal_reason=acceptance_problem),
        )
        controller_message = acceptance_problem

    return ReportStageResult(
        run_dir=run_path,
        run_record=updated_record,
        outcome=outcome,
        final_patch_path=final_patch_path,
        final_report_path=final_report_path,
        changed_files=tuple(context["controller"]["changed_files"]),
        additions=int(context["controller"]["additions"]),
        deletions=int(context["controller"]["deletions"]),
        git_safety=refreshed_safety,
        controller_message=controller_message,
    )


def generate_terminal_report_best_effort(run_dir: Path | str) -> Path | None:
    run_path = Path(run_dir)
    run_record: RunRecord | None = None
    try:
        run_record = load_run_record(run_path / RUN_RECORD_FILE)
        patch_text = ""
        patch_capture_error: str | None = None
        try:
            _final_patch_path, patch_text = capture_final_patch(run_path, run_record)
        except Exception as error:  # noqa: BLE001 - best effort must not throw.
            patch_capture_error = f"Final patch unavailable: {error}"
        context = collect_report_context(
            run_path,
            run_record,
            patch_text=patch_text,
            patch_capture_error=patch_capture_error,
        )
        report_text = render_final_report(
            context,
            terminal_reason=run_record.terminal_reason,
        )
        final_report_path = run_path / FINAL_REPORT_FILE
        _write_text(final_report_path, report_text)
        return final_report_path
    except Exception as error:  # noqa: BLE001 - best effort must not throw.
        return _write_minimal_terminal_report(
            run_path,
            run_record,
            reporting_error=error,
        )


def capture_final_patch(
    run_dir: Path | str,
    run_record: RunRecord,
) -> tuple[Path, str]:
    repository = GitRepository(Path(run_record.target_repository_path))
    diffs_dir = Path(run_dir) / DIFFS_DIR_NAME
    diffs_dir.mkdir(parents=True, exist_ok=True)
    patch_path = diffs_dir / FINAL_PATCH_FILE
    try:
        patch = diff_including_untracked(repository, run_record.baseline_sha)
    except (GitCommandError, OSError, ValueError) as error:
        raise ReportError(f"Could not capture final diff: {error}") from error
    _write_text(patch_path, patch)
    return patch_path, patch


def collect_report_context(
    run_dir: Path | str,
    run_record: RunRecord,
    *,
    patch_text: str | None = None,
    patch_capture_error: str | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    repository = GitRepository(Path(run_record.target_repository_path))
    patch = patch_text
    if patch is None:
        try:
            patch = diff_including_untracked(repository, run_record.baseline_sha)
        except (GitCommandError, OSError, ValueError) as error:
            patch = ""
            patch_capture_error = f"Final patch unavailable: {error}"

    changed_files, diff_stats, git_status = _git_diff_facts(repository, run_record)
    additions, deletions = count_patch_changes(patch)
    implementation_result = _read_json_if_exists(
        run_path / IMPLEMENTATION_DIR_NAME / RESULT_JSON
    )
    verification_rounds = _read_numbered_json_files(run_path / VERIFICATION_DIR_NAME)
    review_results = _read_review_results(run_path)
    correction_ticket_paths = tuple(
        sorted(
            str(path.relative_to(run_path))
            for path in (run_path / CORRECTIONS_DIR_NAME).glob("*.md")
        )
        if (run_path / CORRECTIONS_DIR_NAME).is_dir()
        else ()
    )
    workspace_guard_inspections = _read_workspace_guard_inspections(run_path)
    final_review = review_results[-1]["data"] if review_results else None
    advisory_findings = _findings_with_disposition(final_review, "ADVISORY")
    follow_up_findings = _findings_with_disposition(final_review, "FOLLOW_UP")
    safety = inspect_git_safety(run_record)
    checkpoint_patch_path = latest_writable_checkpoint_patch(run_path, run_record)
    checkpoint_fingerprint_path = _latest_writable_workspace_fingerprint_path(
        run_path,
        run_record,
    )
    checkpoint_workspace_matches = _checkpoint_workspace_matches(
        checkpoint_fingerprint_path,
        safety._workspace_snapshot,
    )

    return {
        "run_dir": run_path,
        "run_record": run_record,
        "controller": {
            "artifact_paths": _existing_artifact_paths(run_path),
            "changed_files": changed_files,
            "diff_stats": diff_stats,
            "git_status": git_status,
            "additions": additions,
            "deletions": deletions,
            "verification_rounds": verification_rounds,
            "review_results": review_results,
            "workspace_guard_inspections": workspace_guard_inspections,
            "correction_ticket_paths": correction_ticket_paths,
            "final_review": final_review,
            "advisory_findings": advisory_findings,
            "follow_up_findings": follow_up_findings,
            "git_safety": safety,
            "latest_writable_checkpoint_patch": (
                None
                if checkpoint_patch_path is None
                else str(checkpoint_patch_path.relative_to(run_path))
            ),
            "current_workspace_matches_checkpoint": checkpoint_workspace_matches,
            "final_patch_path": str(
                (run_path / DIFFS_DIR_NAME / FINAL_PATCH_FILE).relative_to(run_path)
            ),
            "final_patch_error": patch_capture_error,
            "final_report_path": FINAL_REPORT_FILE,
        },
        "agent": {
            "implementation": implementation_result,
        },
    }


def render_final_report(
    context: dict[str, Any],
    *,
    terminal_reason: str | None,
) -> str:
    record: RunRecord = context["run_record"]
    controller: dict[str, Any] = context["controller"]
    agent: dict[str, Any] = context["agent"]
    safety: GitSafetyStatus = controller["git_safety"]
    implementation = agent["implementation"]
    final_review = controller["final_review"]

    lines = [
        f"# {record.ticket_id} final report",
        "",
        "## Run identity",
        "",
        f"- Run ID: {record.run_id}",
        f"- Ticket ID: {record.ticket_id}",
        f"- Original ticket path: {record.original_ticket_path}",
        f"- Snapshotted ticket path: {record.run_ticket_copy_path}",
        f"- Target repository path: {record.target_repository_path}",
        f"- Starting branch: {record.starting_branch}",
        f"- Baseline SHA: {record.baseline_sha}",
        f"- Workflow state: {record.state.value}",
        f"- Terminal outcome: {record.state.value}",
    ]
    if terminal_reason:
        lines.append(f"- Terminal reason: {terminal_reason}")
    elif record.terminal_reason:
        lines.append(f"- Terminal reason: {record.terminal_reason}")

    lines.extend(
        [
            "",
            "## Authoritative controller evidence",
            "",
            (
                "These facts were observed by TicketAutomation from Git, persisted "
                "runner artifacts, or structured review results."
            ),
            "",
            "### Git",
            "",
            f"- Current branch: {safety.branch_actual}",
            f"- Current HEAD: {safety.head_actual}",
            f"- Branch unchanged: {_yes_no(safety.branch_ok)}",
            f"- HEAD unchanged: {_yes_no(safety.head_ok)}",
            f"- Staging empty: {_yes_no(safety.staging_ok)}",
            (
                "- Current workspace matches last verified writable checkpoint: "
                f"{_yes_no(controller['current_workspace_matches_checkpoint'])}"
            ),
            f"- Changed files: {len(controller['changed_files'])}",
        ]
    )
    if controller["changed_files"]:
        lines.extend(f"  - {file_path}" for file_path in controller["changed_files"])
    else:
        lines.append("  - none")

    lines.extend(
        [
            f"- Diff line counts: +{controller['additions']} / -{controller['deletions']}",
            "- Diff statistics:",
            "",
            "```text",
            controller["diff_stats"] or "No diff statistics available.",
            "```",
            "- Git status:",
            "",
            "```text",
            controller["git_status"] or "No Git status available.",
            "```",
        ]
    )
    if controller["final_patch_error"]:
        lines.append(f"- Final patch: {controller['final_patch_error']}")
    else:
        lines.append(f"- Final patch: {controller['final_patch_path']}")
    if safety.inspection_error:
        lines.append(f"- Git inspection error: {safety.inspection_error}")
    checkpoint_patch = controller["latest_writable_checkpoint_patch"]
    if checkpoint_patch is not None:
        lines.append(f"- Latest writable checkpoint patch: {checkpoint_patch}")

    _append_workspace_guard(lines, controller["workspace_guard_inspections"])

    lines.extend(
        [
            "",
            "### Deterministic verification",
            "",
            f"- Verification rounds: {len(controller['verification_rounds'])}",
        ]
    )
    if controller["verification_rounds"]:
        for round_item in controller["verification_rounds"]:
            data = round_item["data"]
            lines.append(
                f"- round-{data.get('round_index', '?')}: {data.get('status', 'UNKNOWN')}"
            )
            commands = data.get("commands", [])
            if isinstance(commands, list):
                for command in commands:
                    if not isinstance(command, dict):
                        continue
                    lines.append(
                        "  - "
                        f"{command.get('name', '<unnamed>')}: "
                        f"{command.get('status', 'UNKNOWN')} "
                        f"(exit {command.get('exit_code', 'n/a')})"
                    )
    else:
        lines.append("- No deterministic verification rounds were persisted.")

    lines.extend(
        [
            "",
            "### Independent review",
            "",
            f"- Review rounds: {len(controller['review_results'])}",
            f"- Final review verdict: {_review_verdict(final_review)}",
            f"- Advisory findings: {len(controller['advisory_findings'])}",
            f"- Follow-up findings: {len(controller['follow_up_findings'])}",
        ]
    )
    _append_findings(lines, "Advisory finding detail", controller["advisory_findings"])
    _append_findings(
        lines, "Follow-up finding detail", controller["follow_up_findings"]
    )

    lines.extend(
        [
            "",
            "### Corrective work",
            "",
            f"- Corrective rounds completed: {record.current_correction_round}",
            f"- Corrective round limit: {record.max_correction_rounds}",
            f"- Generated corrective tickets: {len(controller['correction_ticket_paths'])}",
        ]
    )
    if controller["correction_ticket_paths"]:
        lines.extend(f"  - {path}" for path in controller["correction_ticket_paths"])
    else:
        lines.append("  - none")

    lines.extend(
        [
            "",
            "## Agent-reported information",
            "",
            (
                "Implementation-agent targeted validation is reported here as an "
                "agent claim. It is not treated as equivalent to deterministic "
                "runner verification."
            ),
        ]
    )
    if isinstance(implementation, dict):
        lines.extend(
            [
                "",
                f"- Implementation-agent status: {implementation.get('status', 'UNKNOWN')}",
                f"- Implementation-agent summary: {implementation.get('summary', '')}",
                "- Implementation-agent targeted validation:",
            ]
        )
        tests = implementation.get("tests_run", [])
        if isinstance(tests, list) and tests:
            for item in tests:
                lines.append(f"  - {_format_agent_test(item)}")
        else:
            lines.append("  - none reported")
    else:
        lines.extend(["", "- No implementation-agent result artifact was available."])

    lines.extend(
        [
            "",
            "## Handoff notes",
            "",
            "- TicketAutomation did not intentionally stage target-repository changes.",
            "- TicketAutomation did not intentionally commit target-repository changes.",
        ]
    )
    if record.state == WorkflowState.HUMAN_REQUIRED:
        lines.extend(
            [
                "- Automation stopped for human inspection.",
                "- Source changes may be incomplete or may require manual judgment.",
                "- Inspect the run artifacts and target repository before continuing.",
                (
                    "- What requires human inspection: "
                    f"{terminal_reason or record.terminal_reason or 'See terminal outcome.'}"
                ),
            ]
        )
    elif record.state == WorkflowState.FAILED:
        lines.extend(
            [
                "- Automation stopped because infrastructure or controller execution failed.",
                "- Existing run artifacts should be inspected before retrying.",
                (
                    "- What requires human inspection: "
                    f"{terminal_reason or record.terminal_reason or 'See terminal outcome.'}"
                ),
            ]
        )

    lines.extend(["", "### Relevant artifact/log paths", ""])
    artifact_paths = controller["artifact_paths"]
    if artifact_paths:
        lines.extend(f"- {path}" for path in artifact_paths)
    else:
        lines.append("- none available")

    lines.append("")
    return "\n".join(lines)


def inspect_git_safety(run_record: RunRecord) -> GitSafetyStatus:
    repository = GitRepository(Path(run_record.target_repository_path))
    snapshot = WorkspaceSnapshot.capture(repository)
    return GitSafetyStatus(
        branch_expected=run_record.starting_branch,
        branch_actual=("<detached>" if snapshot.branch is None else snapshot.branch),
        head_expected=run_record.baseline_sha,
        head_actual=snapshot.head_sha or "<unknown>",
        staged_files=snapshot.staged_paths,
        inspection_error=(
            None
            if snapshot.inspection_complete
            else "; ".join(snapshot.inspection_errors)
            or "workspace inspection incomplete"
        ),
        _workspace_snapshot=snapshot,
    )


def count_patch_changes(patch_text: str) -> tuple[int, int]:
    additions = 0
    deletions = 0
    for line in patch_text.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith("+"):
            additions += 1
        elif line.startswith("-"):
            deletions += 1
    return additions, deletions


def latest_verification_round(run_dir: Path | str) -> dict[str, Any] | None:
    rounds = _read_numbered_json_files(Path(run_dir) / VERIFICATION_DIR_NAME)
    return None if not rounds else rounds[-1]["data"]


def latest_review_result(run_dir: Path | str) -> dict[str, Any] | None:
    results = _read_review_results(Path(run_dir))
    return None if not results else results[-1]["data"]


def latest_writable_checkpoint_patch(
    run_dir: Path | str,
    run_record: RunRecord,
) -> Path | None:
    run_path = Path(run_dir)
    if run_record.current_correction_round > 0:
        path = (
            run_path
            / DIFFS_DIR_NAME
            / f"after-correction-{run_record.current_correction_round}.patch"
        )
        return path if path.is_file() else None
    path = run_path / DIFFS_DIR_NAME / "after-implementation.patch"
    return path if path.is_file() else None


def _latest_writable_workspace_fingerprint_path(
    run_dir: Path | str,
    run_record: RunRecord,
) -> Path | None:
    run_path = Path(run_dir)
    if run_record.current_correction_round > 0:
        patch_path = (
            run_path
            / DIFFS_DIR_NAME
            / f"after-correction-{run_record.current_correction_round}.patch"
        )
    else:
        patch_path = run_path / DIFFS_DIR_NAME / "after-implementation.patch"
    fingerprint_path = _workspace_fingerprint_path(patch_path)
    return fingerprint_path if fingerprint_path.is_file() else None


def _acceptance_problem(context: dict[str, Any]) -> str | None:
    controller = context["controller"]
    safety: GitSafetyStatus = controller["git_safety"]
    verification = (
        None
        if not controller["verification_rounds"]
        else controller["verification_rounds"][-1]["data"]
    )
    final_review = controller["final_review"]

    if not safety.safe:
        return "Repository safety invariants were violated before human handoff."
    if not controller["current_workspace_matches_checkpoint"]:
        return (
            "Current workspace no longer matches the last verified writable checkpoint."
        )
    if not isinstance(verification, dict) or verification.get("status") != "PASS":
        return "Deterministic verification is not currently passing."
    if not isinstance(final_review, dict) or final_review.get("verdict") != "PASS":
        return "Final independent review did not pass."
    return None


def _git_diff_facts(
    repository: GitRepository,
    run_record: RunRecord,
) -> tuple[tuple[str, ...], str, str]:
    changed_files: tuple[str, ...] = ()
    diff_stats = ""
    git_status = ""
    try:
        changed_files = changed_files_including_untracked(
            repository,
            run_record.baseline_sha,
        )
    except (GitCommandError, OSError, ValueError):
        pass
    try:
        diff_stats = diff_stats_including_untracked(
            repository,
            run_record.baseline_sha,
        )
    except (GitCommandError, OSError, ValueError):
        pass
    try:
        git_status = repository.status_short()
    except (GitCommandError, OSError, ValueError):
        pass
    return changed_files, diff_stats, git_status


def _checkpoint_workspace_matches(
    fingerprint_path: Path | None,
    current_snapshot: WorkspaceSnapshot | None,
) -> bool:
    if fingerprint_path is None or current_snapshot is None:
        return False
    try:
        expected = _read_workspace_fingerprint(fingerprint_path)
    except (OSError, ValueError):
        return False
    return current_snapshot.matches_fingerprint(expected)


def _read_numbered_json_files(directory: Path) -> tuple[dict[str, Any], ...]:
    if not directory.is_dir():
        return ()
    items: list[dict[str, Any]] = []
    for path in sorted(directory.glob("round-*.json"), key=_round_sort_key):
        data = _read_json_if_exists(path)
        if isinstance(data, dict):
            items.append({"path": path, "data": data})
    return tuple(items)


def _read_review_results(run_dir: Path) -> tuple[dict[str, Any], ...]:
    review_root = run_dir / REVIEWS_DIR_NAME
    if not review_root.is_dir():
        return ()
    items: list[dict[str, Any]] = []
    for path in sorted(review_root.glob("round-*"), key=_round_sort_key):
        result_path = path / RESULT_JSON
        data = _read_json_if_exists(result_path)
        if isinstance(data, dict):
            items.append({"path": result_path, "data": data})
    return tuple(items)


def _read_workspace_guard_inspections(run_dir: Path) -> tuple[dict[str, Any], ...]:
    guard_root = run_dir / WORKSPACE_GUARD_DIR_NAME
    if not guard_root.is_dir():
        return ()
    items: list[dict[str, Any]] = []
    for path in sorted(guard_root.glob("*.json")):
        data = _read_json_if_exists(path)
        if isinstance(data, dict):
            items.append({"path": path.relative_to(run_dir).as_posix(), "data": data})
    return tuple(items)


def _read_json_if_exists(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def _existing_artifact_paths(run_path: Path) -> tuple[str, ...]:
    if not run_path.is_dir():
        return ()
    paths: list[str] = []
    for path in run_path.rglob("*"):
        if path.is_file():
            paths.append(path.relative_to(run_path).as_posix())
    return tuple(sorted(paths))


def _findings_with_disposition(
    review_result: dict[str, Any] | None,
    disposition: str,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(review_result, dict):
        return ()
    findings = review_result.get("findings", [])
    if not isinstance(findings, list):
        return ()
    return tuple(
        item
        for item in findings
        if isinstance(item, dict) and item.get("disposition") == disposition
    )


def _append_findings(
    lines: list[str],
    title: str,
    findings: tuple[dict[str, Any], ...],
) -> None:
    if not findings:
        return
    lines.extend(["", f"### {title}", ""])
    for finding in findings:
        finding_id = finding.get("id", "<unknown>")
        finding_title = finding.get("title", "Untitled finding")
        lines.append(f"- {finding_id}: {finding_title}")


def _append_workspace_guard(
    lines: list[str],
    inspections: tuple[dict[str, Any], ...],
) -> None:
    lines.extend(["", "### Workspace Hygiene", ""])
    if not inspections:
        lines.append("- No workspace-environment guard artifacts were persisted.")
        return

    lines.append(f"- Workspace guard artifacts: {len(inspections)}")
    found_new_environment = False
    for item in inspections:
        path = item.get("path", "<unknown>")
        data = item.get("data", {})
        if not isinstance(data, dict):
            continue
        phase = data.get("phase", "UNKNOWN")
        new_environments = data.get("new_environments", [])
        new_count = len(new_environments) if isinstance(new_environments, list) else 0
        lines.append(
            f"- {path}: phase {phase}; newly detected environments: {new_count}"
        )
        if not isinstance(new_environments, list):
            continue
        for environment in new_environments:
            if not isinstance(environment, dict):
                continue
            found_new_environment = True
            markers = environment.get("markers", [])
            marker = (
                markers[0] if isinstance(markers, list) and markers else "<unknown>"
            )
            lines.append(
                "  - "
                f"{environment.get('root', '<unknown>')}: "
                f"{_workspace_environment_kind(environment.get('kind'))}; "
                f"marker: {marker}; "
                "did not exist before this writable operation"
            )
    if found_new_environment:
        lines.append("- TicketAutomation did not delete detected environments.")


def _workspace_environment_kind(value: object) -> str:
    if value == "conda":
        return "Conda environment"
    return "Python virtual environment"


def _format_agent_test(item: Any) -> str:
    if not isinstance(item, dict):
        return repr(item)
    command = item.get("command", "<unknown command>")
    result = item.get("result", "<unknown result>")
    return f"{command} -> {result}"


def _review_verdict(review_result: dict[str, Any] | None) -> str:
    if not isinstance(review_result, dict):
        return "NOT AVAILABLE"
    verdict = review_result.get("verdict")
    return verdict if isinstance(verdict, str) else "NOT AVAILABLE"


def _round_sort_key(path: Path) -> tuple[int, str]:
    stem = path.name
    if path.is_file():
        stem = path.stem
    marker = stem.rsplit("-", maxsplit=1)[-1]
    try:
        return int(marker), path.as_posix()
    except ValueError:
        return 1_000_000, path.as_posix()


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_minimal_terminal_report(
    run_path: Path,
    run_record: RunRecord | None,
    *,
    reporting_error: BaseException,
) -> Path | None:
    if run_record is None:
        return None
    lines = [
        f"# {run_record.ticket_id} terminal report",
        "",
        "## Run identity",
        "",
        f"- Run ID: {run_record.run_id}",
        f"- Run path: {run_path}",
        f"- Target repository path: {run_record.target_repository_path}",
        f"- State: {run_record.state.value}",
        f"- Primary failure: {run_record.terminal_reason or 'unknown'}",
        f"- Reporting fallback: {type(reporting_error).__name__}: {reporting_error}",
        "- Manual inspection required.",
        "",
    ]
    report_path = run_path / FINAL_REPORT_FILE
    try:
        _write_text(report_path, "\n".join(lines))
    except Exception:  # noqa: BLE001 - outer boundary is best effort.
        return None
    return report_path


__all__ = [
    "FINAL_PATCH_FILE",
    "FINAL_REPORT_FILE",
    "ReportError",
    "ReportStageResult",
    "capture_final_patch",
    "changed_files_including_untracked",
    "collect_report_context",
    "count_patch_changes",
    "diff_including_untracked",
    "diff_stats_including_untracked",
    "generate_terminal_report_best_effort",
    "inspect_git_safety",
    "latest_review_result",
    "latest_verification_round",
    "latest_writable_checkpoint_patch",
    "render_final_report",
    "run_report_stage",
]
