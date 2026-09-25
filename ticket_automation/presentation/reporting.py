"""Render human handoff material from controller-owned evidence.

This module has no state-transition or acceptance authority.  The workflow
controller decides terminal state from the canonical workspace snapshot before
asking this renderer to describe the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..application.agent_execution import CORRECTION_TICKET_FILE
from ..application.ports.handoff import FINAL_PATCH_FILE
from ..attempts import (
    AttemptRecord,
    attempt_artifact_layout,
    latest_attempt,
    latest_writable_attempt,
    load_attempt_records,
)
from ..audit import (
    changed_files_including_untracked,
    diff_including_untracked,
    diff_stats_including_untracked,
)
from ..domain.task_results import (
    ImplementationResult,
    ResultValidationError,
    ReviewResult,
)
from ..execution_evidence import ExecutionEvidence, read_execution_evidence
from ..git import GitCommandError, GitRepository
from ..git_safety import WorkspaceSnapshot
from ..models import (
    PHASE_DEFINITIONS,
    AttemptPhase,
    AttemptStatus,
    ResultArtifactRole,
    StageOutcome,
    WorkflowState,
)
from ..persistence import PersistenceError, atomic_write_text
from ..persistence_codecs import (
    PersistenceCodecError,
    read_implementation_result,
    read_review_result,
)
from ..run_ownership import (
    RunOwnership,
    RunOwnershipError,
    validate_unlinked_run_directory,
)
from ..runs import RUN_RECORD_FILE, RunError, RunRecord, load_run_record
from ..verification_evidence import VerificationEvidence, read_verification_evidence

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
class AttemptReportView:
    sequence: int
    phase: AttemptPhase
    status: AttemptStatus
    correction_ticket: str | None
    execution: ExecutionEvidence | None


@dataclass(frozen=True)
class ControllerReportView:
    attempts: tuple[AttemptReportView, ...]
    changed_files: tuple[str, ...]
    diff_stats: str
    additions: int
    deletions: int
    baseline_verification: VerificationEvidence | None
    verification_rounds: tuple[VerificationEvidence, ...]
    review_results: tuple[ReviewResult, ...]
    final_review: ReviewResult | None
    final_review_error: str | None
    review_result_errors: tuple[str, ...]
    correction_ticket_paths: tuple[str, ...]
    git_safety: GitSafetyStatus
    latest_writable_attempt: AttemptRecord | None
    final_patch_path: str
    final_report_path: str


@dataclass(frozen=True)
class AgentReportView:
    implementation: ImplementationResult | None
    implementation_result_error: str | None


@dataclass(frozen=True)
class ReportViewModel:
    run_dir: Path
    run_record: RunRecord
    controller: ControllerReportView
    agent: AgentReportView


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


@dataclass(frozen=True)
class FilesystemTerminalReportPublisher:
    """Publish exactly one report from the persisted terminal run record."""

    def publish(
        self,
        run_dir: Path,
        run_record: RunRecord,
        *,
        run_ownership: RunOwnership | None = None,
    ) -> ReportStageResult | None:
        controller_ownership = run_ownership
        try:
            run_ownership = (
                RunOwnership.acquire(run_dir.parent, run_record.run_id)
                if run_ownership is None
                else run_ownership
            )
            run_ownership.validate_run_path(run_dir)
        except RunOwnershipError:
            if controller_ownership is not None:
                raise
            return None
        except ValueError:
            return None
        if run_record.state is WorkflowState.READY_FOR_HUMAN:
            try:
                return run_report_stage(
                    run_dir,
                    run_ownership=run_ownership,
                )
            except (OSError, PersistenceError, ReportError, ValueError):
                return None
        generate_terminal_report_best_effort(
            run_dir,
            run_ownership=run_ownership,
        )
        return None


def run_report_stage(
    run_dir: Path | str,
    *,
    run_ownership: RunOwnership | None = None,
) -> ReportStageResult:
    try:
        run_path = (
            validate_unlinked_run_directory(run_dir)
            if run_ownership is None
            else run_ownership.validate_run_path(run_dir)
        )
    except RunOwnershipError as error:
        raise ReportError(str(error)) from error
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
    if run_ownership is not None:
        run_ownership.validate_descendant(final_report_path)
    _write_text(final_report_path, render_final_report(context))
    controller = context.controller
    return ReportStageResult(
        run_dir=run_path,
        run_record=run_record,
        outcome=StageOutcome.COMPLETED,
        final_patch_path=final_patch_path,
        final_report_path=final_report_path,
        changed_files=controller.changed_files,
        additions=controller.additions,
        deletions=controller.deletions,
        git_safety=controller.git_safety,
        controller_message="Report rendered from persisted run evidence.",
    )


def generate_terminal_report_best_effort(
    run_dir: Path | str,
    *,
    run_ownership: RunOwnership | None = None,
) -> Path | None:
    try:
        run_path = (
            validate_unlinked_run_directory(run_dir)
            if run_ownership is None
            else run_ownership.validate_run_path(run_dir)
        )
        record = load_run_record(run_path / RUN_RECORD_FILE)
        try:
            patch_text = (run_path / FINAL_PATCH_FILE).read_text(encoding="utf-8")
        except OSError:
            patch_text = ""
        context = collect_report_context(run_path, record, patch_text=patch_text)
        path = run_path / FINAL_REPORT_FILE
        if run_ownership is not None:
            run_ownership.validate_descendant(path)
        _write_text(path, render_final_report(context))
        return path
    except Exception:  # noqa: BLE001 - never mask the terminal state.
        return None


def collect_report_context(
    run_dir: Path | str,
    run_record: RunRecord,
    *,
    patch_text: str | None = None,
) -> ReportViewModel:
    run_path = Path(run_dir)
    repository = GitRepository(Path(run_record.target_repository_path))
    patch = (
        patch_text if patch_text is not None else _diff_or_empty(repository, run_record)
    )
    records = load_attempt_records(run_path)
    attempts = tuple(_attempt_view(run_path, record) for record in records)
    verification = tuple(
        evidence
        for record in records
        if _has_result_role(record, ResultArtifactRole.VERIFICATION_ROUND)
        if (evidence := _verification_view(run_path, record)) is not None
    )
    baseline = _latest_attempt_result(
        run_path, ResultArtifactRole.BASELINE_VERIFICATION
    )
    reviews, review_result_errors = _review_results(run_path, records)
    implementation, implementation_result_error = _latest_implementation_result(
        run_path
    )
    correction_tickets = tuple(
        item.correction_ticket
        for item in attempts
        if item.correction_ticket is not None
    )
    changed_files = _changed_files_or_empty(repository, run_record)
    diff_stats = _diff_stats_or_empty(repository, run_record)
    safety = inspect_git_safety(run_record)
    try:
        final_review = _latest_review_result(run_path)
        final_review_error = None
    except ResultValidationError as error:
        final_review = None
        final_review_error = str(error)
    additions, deletions = count_patch_changes(patch)
    return ReportViewModel(
        run_dir=run_path,
        run_record=run_record,
        controller=ControllerReportView(
            attempts=attempts,
            changed_files=changed_files,
            diff_stats=diff_stats,
            additions=additions,
            deletions=deletions,
            baseline_verification=baseline,
            verification_rounds=verification,
            review_results=reviews,
            final_review=final_review,
            final_review_error=final_review_error,
            review_result_errors=review_result_errors,
            correction_ticket_paths=correction_tickets,
            git_safety=safety,
            latest_writable_attempt=latest_writable_attempt(run_path),
            final_patch_path=FINAL_PATCH_FILE,
            final_report_path=FINAL_REPORT_FILE,
        ),
        agent=AgentReportView(
            implementation=implementation,
            implementation_result_error=implementation_result_error,
        ),
    )


def render_final_report(context: ReportViewModel) -> str:
    record = context.run_record
    controller = context.controller
    safety = controller.git_safety
    baseline = controller.baseline_verification
    verification = controller.verification_rounds
    final_review = controller.final_review
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
        f"- Final review: {_review_verdict(final_review, controller.final_review_error)}",
        f"- Correction rounds: {record.current_correction_round} / {record.max_correction_rounds}",
        "",
        "## Workspace",
        "",
        f"- Branch unchanged: {_yes_no(safety.branch_ok)}",
        f"- HEAD unchanged: {_yes_no(safety.head_ok)}",
        f"- Staging empty: {_yes_no(safety.staging_ok)}",
        f"- Changed files: {len(controller.changed_files)}",
        f"- Diff line counts: +{controller.additions} / -{controller.deletions}",
        f"- Final patch: {controller.final_patch_path}",
        "",
        "## Attempts",
        "",
    ]
    for attempt in controller.attempts:
        lines.append(
            f"- {attempt.sequence:03d} {attempt.phase.value}: {attempt.status.value}"
        )
        if attempt.execution is not None:
            execution = attempt.execution
            detail = (
                execution.status.value
                if execution.failure_category is None
                else f"{execution.status.value} ({execution.failure_category.value})"
            )
            lines.append(
                f"  - Provider {execution.provider_id.value}: {detail}; "
                f"invocation {execution.invocation_start.value}"
            )
    if not controller.attempts:
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


def latest_verification_round(run_dir: Path | str) -> VerificationEvidence | None:
    return _latest_attempt_result(Path(run_dir), ResultArtifactRole.VERIFICATION_ROUND)


def latest_review_result(run_dir: Path | str) -> ReviewResult | None:
    return _latest_review_result(Path(run_dir))


def _latest_review_result(run_path: Path) -> ReviewResult | None:
    record = _latest_attempt_for_role(run_path, ResultArtifactRole.REVIEW_RESULT)
    if record is None:
        return None
    try:
        return read_review_result(run_path, record)
    except PersistenceCodecError as error:
        raise ResultValidationError(str(error)) from error


def _review_results(
    run_path: Path,
    records: tuple[AttemptRecord, ...],
) -> tuple[tuple[ReviewResult, ...], tuple[str, ...]]:
    results: list[ReviewResult] = []
    errors: list[str] = []
    for record in records:
        if not _has_result_role(record, ResultArtifactRole.REVIEW_RESULT):
            continue
        try:
            result = read_review_result(run_path, record)
            if result is None:
                continue
            results.append(result)
        except PersistenceCodecError as error:
            errors.append(f"Review attempt {record.sequence}: {error}")
    return tuple(results), tuple(errors)


def _latest_implementation_result(
    run_path: Path,
) -> tuple[ImplementationResult | None, str | None]:
    record = _latest_attempt_for_role(
        run_path, ResultArtifactRole.IMPLEMENTATION_RESULT
    )
    if record is None:
        return None, None
    try:
        return read_implementation_result(run_path, record), None
    except PersistenceCodecError as error:
        return None, str(error)


def _latest_attempt_result(
    run_path: Path,
    role: ResultArtifactRole,
) -> VerificationEvidence | None:
    record = _latest_attempt_for_role(run_path, role)
    if record is None:
        return None
    return _verification_view(run_path, record)


def _verification_view(
    run_path: Path, record: AttemptRecord
) -> VerificationEvidence | None:
    try:
        return read_verification_evidence(run_path, record)
    except ValueError:
        return None


def _latest_attempt_for_role(
    run_path: Path,
    role: ResultArtifactRole,
) -> AttemptRecord | None:
    phases = tuple(
        phase
        for phase, definition in PHASE_DEFINITIONS.items()
        if definition.result_artifact_role is role
    )
    return latest_attempt(run_path, phases=phases)


def _has_result_role(record: AttemptRecord, role: ResultArtifactRole) -> bool:
    return PHASE_DEFINITIONS[record.phase].result_artifact_role is role


def _attempt_view(run_path: Path, record: AttemptRecord) -> AttemptReportView:
    layout = attempt_artifact_layout(run_path, record)
    correction_ticket = layout.path(CORRECTION_TICKET_FILE)
    execution = None
    if record.execution_path is not None:
        try:
            execution = read_execution_evidence(layout)
        except (OSError, ValueError):
            execution = None
    return AttemptReportView(
        sequence=record.sequence,
        phase=record.phase,
        status=record.status,
        correction_ticket=(
            correction_ticket.relative_to(layout.run_root).as_posix()
            if correction_ticket.is_file()
            else None
        ),
        execution=execution,
    )


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


def _status(value: VerificationEvidence | None) -> str:
    return "NOT RUN" if value is None else value.status.value


def _review_verdict(value: ReviewResult | None, error: object) -> str:
    if isinstance(error, str):
        return "INVALID"
    return "NOT RUN" if value is None else value.verdict.value


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _write_text(path: Path, text: str) -> None:
    atomic_write_text(path, text)


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
    "generate_terminal_report_best_effort",
    "inspect_git_safety",
    "latest_review_result",
    "latest_verification_round",
    "render_final_report",
    "run_report_stage",
]
