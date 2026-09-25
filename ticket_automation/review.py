from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .application.agent_execution import (
    REVIEW_RESULT_CONTRACT,
    AgentExecution,
    AgentExecutionRequest,
    AgentExecutor,
    AgentFailureCategory,
    AgentTaskKind,
    AttemptArtifactLayout,
    InvocationStart,
    RepositoryAccess,
    required_execution_capabilities,
)
from .attempts import (
    AttemptMetadata,
    StageAttempt,
    attempt_result_path,
    complete_stage_attempt,
    latest_attempt,
    require_stage_attempt,
    start_attempt,
    update_attempt,
)
from .config import AppConfig, VerificationCommand
from .domain.task_results import (
    FindingDisposition,
    ReviewFinding,
    ReviewResult,
    ReviewResultConsistencyError,
    ReviewVerdict,
)
from .execution_evidence import evidence_from_execution, write_execution_evidence
from .git import GitRepository
from .git_safety import (
    WorkspaceChange,
    WorkspaceSnapshot,
    workspace_safety_changes,
)
from .models import (
    ATTEMPT_RESULT_ARTIFACT_NAME,
    AttemptPhase,
    AttemptStatus,
    StageOutcome,
    VerificationStatus,
    WorkflowState,
)
from .persistence_codecs import (
    PersistenceCodecError,
    read_implementation_result,
    write_review_result,
    write_stage_message,
)
from .resolved_config import config_from_resolved_run_policy
from .runs import (
    BASELINE_RECORD_FILE,
    RUN_RECORD_FILE,
    RUN_TICKET_FILE,
    RunError,
    RunRecord,
    load_baseline_record,
    load_run_record,
)
from .verification_evidence import (
    read_verification_result_text,
    read_verification_source_fingerprint,
)

_REVIEW_PROMPT_TEMPLATE = (
    Path(__file__).resolve().parent / "application" / "prompts" / "review.md"
)


class ReviewError(RunError):
    """Raised when the review stage cannot be prepared."""


@dataclass(frozen=True)
class ReviewSafetyViolation:
    name: str
    expected: str
    actual: str
    message: str


@dataclass(frozen=True)
class ReviewStageResult:
    run_dir: Path
    run_record: RunRecord
    outcome: StageOutcome
    artifact_directory: Path
    agent_execution: AgentExecution[ReviewResult] | None
    review_result: ReviewResult | None
    safety_violations: tuple[ReviewSafetyViolation, ...]
    processing_error: str | None
    controller_message: str
    after_workspace_fingerprint: str | None
    process_started: bool

    @property
    def source_state(self) -> WorkflowState:
        return WorkflowState.REVIEWING

    @property
    def successful(self) -> bool:
        return self.outcome == StageOutcome.COMPLETED

    @property
    def required_findings(self) -> tuple[ReviewFinding, ...]:
        if self.review_result is None:
            return ()
        return self.review_result.required_findings


def run_review_stage(
    config: AppConfig,
    run_dir: Path | str,
    *,
    agent_executor: AgentExecutor,
    attempt_record: StageAttempt | None = None,
    mark_process_started: Callable[[], None] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> ReviewStageResult:
    run_path = Path(run_dir)
    if attempt_record is not None:
        if mark_process_started is None:
            raise ReviewError(
                "Controller-owned review attempts require a process-start callback."
            )
        return _run_review_stage(
            config,
            run_path,
            agent_executor=agent_executor,
            attempt_record=attempt_record,
            mark_process_started=mark_process_started,
            clock=clock,
        )
    run_record = load_run_record(run_path / RUN_RECORD_FILE)
    if run_record.state is not WorkflowState.REVIEWING:
        raise ReviewError(
            f"Review requires run state REVIEWING; found {run_record.state.value}."
        )
    try:
        before = WorkspaceSnapshot.capture(
            GitRepository(Path(run_record.target_repository_path))
        ).fingerprint
    except (OSError, RuntimeError, ValueError):
        before = None
    owned_attempt = start_attempt(
        run_path,
        phase=AttemptPhase.REVIEWING,
        before_workspace_fingerprint=before,
        clock=clock,
    )
    result = _run_review_stage(
        config,
        run_path,
        agent_executor=agent_executor,
        attempt_record=StageAttempt.from_record(owned_attempt),
        mark_process_started=lambda: update_attempt(
            owned_attempt, process_started=True
        ),
        clock=clock,
    )
    complete_stage_attempt(
        run_path,
        owned_attempt,
        stage_outcome=result.outcome,
        after_workspace_fingerprint=result.after_workspace_fingerprint,
        process_started=result.process_started,
        metadata=AttemptMetadata(controller_message=result.controller_message),
        clock=clock,
    )
    return result


def _run_review_stage(
    config: AppConfig,
    run_dir: Path | str,
    *,
    agent_executor: AgentExecutor,
    attempt_record: StageAttempt,
    mark_process_started: Callable[[], None],
    clock: Callable[[], datetime] | None = None,
) -> ReviewStageResult:
    run_path = Path(run_dir)
    run_record_path = run_path / RUN_RECORD_FILE
    run_record = load_run_record(run_record_path)
    # A later local configuration cannot select a different review invocation.
    config = config_from_resolved_run_policy(run_record.resolved_policy)
    if run_record.state != WorkflowState.REVIEWING:
        raise ReviewError(
            f"Review requires run state REVIEWING; found {run_record.state.value}."
        )
    attempt = require_stage_attempt(
        run_path,
        attempt_record,
        phase=AttemptPhase.REVIEWING,
    )

    baseline_record = load_baseline_record(run_path / BASELINE_RECORD_FILE)
    if baseline_record.branch != run_record.starting_branch:
        raise ReviewError("Run record and baseline branch do not match.")
    if baseline_record.head_sha != run_record.baseline_sha:
        raise ReviewError("Run record and baseline HEAD do not match.")

    repository = GitRepository(Path(run_record.target_repository_path))
    try:
        repository_snapshot = WorkspaceSnapshot.capture(repository)
        before_workspace_fingerprint = repository_snapshot.fingerprint
        snapshot_error: ReviewSafetyViolation | None = None
    except (OSError, RuntimeError, ValueError) as error:
        repository_snapshot = None
        before_workspace_fingerprint = None
        snapshot_error = ReviewSafetyViolation(
            name="repository-inspection",
            expected="complete pre-review inspection",
            actual=f"{type(error).__name__}: {error}",
            message="Repository inspection failed before independent review.",
        )
    if before_workspace_fingerprint != attempt.before_workspace_fingerprint:
        snapshot_error = ReviewSafetyViolation(
            name="attempt-workspace",
            expected=str(attempt.before_workspace_fingerprint),
            actual=str(before_workspace_fingerprint),
            message="Workspace changed after the controller started review.",
        )
    artifact_directory = attempt.artifact_directory
    if snapshot_error is not None:
        return _finish(
            run_record=run_record,
            run_dir=run_path,
            artifact_directory=artifact_directory,
            execution=None,
            review_result=None,
            safety_violations=(snapshot_error,),
            processing_error=None,
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message="Repository inspection failed before independent review.",
        )
    assert repository_snapshot is not None
    starting_violations = _inspect_review_invariants(
        repository,
        run_record,
        current_snapshot=repository_snapshot,
    )
    if starting_violations:
        return _finish(
            run_record=run_record,
            run_dir=run_path,
            artifact_directory=artifact_directory,
            execution=None,
            review_result=None,
            safety_violations=starting_violations,
            processing_error=None,
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message="Repository no longer matches the recorded review baseline.",
        )

    source_violations = _inspect_review_source_fingerprint(
        run_path,
        run_record,
        repository_snapshot,
        verification_commands=config.verification.commands,
    )
    if source_violations:
        return _finish(
            run_record=run_record,
            run_dir=run_path,
            artifact_directory=artifact_directory,
            execution=None,
            review_result=None,
            safety_violations=source_violations,
            processing_error=None,
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message=(
                "Repository no longer matches the verified review source."
            ),
        )
    try:
        prompt = _render_review_prompt(
            ticket_text=_read_snapshotted_ticket(run_path / RUN_TICKET_FILE),
            run_record=run_record,
            current_branch=(
                "<detached>"
                if repository_snapshot.branch is None
                else repository_snapshot.branch
            ),
            verification_results=_read_verification_results(run_path, run_record),
            implementation_summary=_read_implementation_summary(run_path),
        )
    except ReviewError as error:
        return _finish(
            run_record=run_record,
            run_dir=run_path,
            artifact_directory=artifact_directory,
            execution=None,
            review_result=None,
            safety_violations=(
                ReviewSafetyViolation(
                    name="review-evidence",
                    expected="readable trusted review inputs",
                    actual=str(error),
                    message="Review evidence could not be prepared safely.",
                ),
            ),
            processing_error=str(error),
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message="Review evidence could not be prepared safely.",
        )

    def mark_invocation_started() -> None:
        mark_process_started()

    request = AgentExecutionRequest(
        task_kind=AgentTaskKind.REVIEW,
        repository_path=repository.path,
        repository_access=RepositoryAccess.READ_ONLY,
        prompt=prompt,
        result_contract=REVIEW_RESULT_CONTRACT,
        artifact_directory=artifact_directory,
        artifact_layout=AttemptArtifactLayout.for_attempt(run_path, artifact_directory),
        policy=run_record.resolved_policy.task_policy(
            AgentTaskKind.REVIEW
        ).execution_policy,
        required_capabilities=required_execution_capabilities(
            RepositoryAccess.READ_ONLY
        ),
    )
    execution = agent_executor.execute(
        request,
        on_invocation_start=mark_invocation_started,
    )
    assert request.artifact_layout is not None
    write_execution_evidence(
        request.artifact_layout,
        evidence_from_execution(request, execution),
    )
    if not execution.successful:
        safety_violations = _inspect_review_invariants(
            repository,
            run_record,
            expected_snapshot=repository_snapshot,
        )
        if safety_violations:
            outcome = StageOutcome.HUMAN_REQUIRED
            processing_error = None
            message = (
                "Agent review failed and repository safety invariants were violated."
            )
        elif execution.failure_category is AgentFailureCategory.INVALID_RESULT:
            outcome = StageOutcome.HUMAN_REQUIRED
            processing_error = execution.failure_message
            message = execution.failure_message or "Agent review result was invalid."
        else:
            outcome = StageOutcome.FAILED
            processing_error = None
            message = execution.failure_message or "Agent review failed."
        return _finish(
            run_record=run_record,
            run_dir=run_path,
            artifact_directory=artifact_directory,
            execution=execution,
            review_result=None,
            safety_violations=safety_violations,
            processing_error=processing_error,
            outcome=outcome,
            controller_message=message,
        )

    review_result = _require_review_result(execution.result)
    safety_violations = _inspect_review_invariants(
        repository,
        run_record,
        expected_snapshot=repository_snapshot,
    )
    if safety_violations:
        return _finish(
            run_record=run_record,
            run_dir=run_path,
            artifact_directory=artifact_directory,
            execution=execution,
            review_result=review_result,
            safety_violations=safety_violations,
            processing_error=None,
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message="Repository safety invariants were violated during review.",
        )

    return _finish_valid_review_result(
        run_record=run_record,
        run_dir=run_path,
        artifact_directory=artifact_directory,
        execution=execution,
        review_result=review_result,
    )


def _finish_valid_review_result(
    *,
    run_record: RunRecord,
    run_dir: Path,
    artifact_directory: Path,
    execution: AgentExecution[ReviewResult] | None,
    review_result: ReviewResult,
) -> ReviewStageResult:
    if review_result.verdict == ReviewVerdict.PASS:
        outcome = StageOutcome.COMPLETED
        controller_message = "Review passed; final report can start."
    elif review_result.verdict == ReviewVerdict.CORRECTIONS_REQUIRED:
        required_findings = review_result.required_findings
        eligible_findings = tuple(
            finding for finding in required_findings if finding.correction_eligible
        )
        if len(eligible_findings) == len(required_findings):
            outcome = StageOutcome.CORRECTION_REQUIRED
            controller_message = "Review found correction-eligible required findings."
        elif not eligible_findings:
            outcome = StageOutcome.HUMAN_REQUIRED
            controller_message = (
                "Review found REQUIRED findings, but none are safely eligible for "
                "automatic correction."
            )
        else:
            outcome = StageOutcome.HUMAN_REQUIRED
            controller_message = (
                "Review contains REQUIRED findings that are not safely eligible "
                "for automatic correction."
            )
    else:
        outcome = StageOutcome.HUMAN_REQUIRED
        controller_message = "Review requires human attention."

    return _finish(
        run_record=run_record,
        run_dir=run_dir,
        artifact_directory=artifact_directory,
        execution=execution,
        review_result=review_result,
        safety_violations=(),
        processing_error=None,
        outcome=outcome,
        controller_message=controller_message,
    )


def _finish(
    *,
    run_record: RunRecord,
    run_dir: Path,
    artifact_directory: Path,
    execution: AgentExecution[ReviewResult] | None,
    review_result: ReviewResult | None,
    safety_violations: tuple[ReviewSafetyViolation, ...],
    processing_error: str | None,
    outcome: StageOutcome,
    controller_message: str,
) -> ReviewStageResult:
    _write_review_result(
        artifact_directory,
        review_result,
        outcome=outcome,
        controller_message=controller_message,
    )
    after_fingerprint: str | None = None
    try:
        after_fingerprint = WorkspaceSnapshot.capture(
            GitRepository(Path(run_record.target_repository_path))
        ).fingerprint
    except (OSError, RuntimeError, ValueError):
        pass
    process_started = (
        False
        if execution is None
        else execution.invocation_start is InvocationStart.STARTED
    )
    return ReviewStageResult(
        run_dir=run_dir,
        run_record=run_record,
        outcome=outcome,
        artifact_directory=artifact_directory,
        agent_execution=execution,
        review_result=review_result,
        safety_violations=safety_violations,
        processing_error=processing_error,
        controller_message=controller_message,
        after_workspace_fingerprint=after_fingerprint,
        process_started=process_started,
    )


def _write_review_result(
    artifact_directory: Path,
    result: ReviewResult | None,
    *,
    outcome: StageOutcome,
    controller_message: str,
) -> None:
    path = artifact_directory / ATTEMPT_RESULT_ARTIFACT_NAME
    if result is not None:
        write_review_result(path, result)
    else:
        write_stage_message(path, status=outcome, message=controller_message)


def _render_review_prompt(
    *,
    ticket_text: str,
    run_record: RunRecord,
    current_branch: str,
    verification_results: str,
    implementation_summary: str,
) -> str:
    replacements = {
        "{{ORIGINAL_TICKET}}": ticket_text,
        "{{BASELINE_SHA}}": run_record.baseline_sha,
        "{{STARTING_BRANCH}}": run_record.starting_branch,
        "{{CURRENT_BRANCH}}": current_branch,
        "{{VERIFICATION_RESULTS}}": verification_results,
        "{{IMPLEMENTATION_SUMMARY}}": implementation_summary,
    }
    template = _REVIEW_PROMPT_TEMPLATE.read_text(encoding="utf-8")
    missing = tuple(key for key in replacements if key not in template)
    if missing:
        raise ReviewError(
            "Review prompt template is missing placeholders: "
            + ", ".join(sorted(missing))
        )
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def _read_snapshotted_ticket(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except OSError as error:
        raise ReviewError(
            f"Could not read snapshotted ticket: {path}: {error}"
        ) from error
    except UnicodeDecodeError as error:
        raise ReviewError(
            f"Snapshotted ticket must be valid UTF-8 Markdown: {path}"
        ) from error


def _read_verification_results(run_path: Path, run_record: RunRecord) -> str:
    del run_record
    attempt = latest_attempt(
        run_path,
        phases=(AttemptPhase.VERIFYING,),
        statuses=(AttemptStatus.COMPLETED,),
    )
    if attempt is None:
        raise ReviewError("Missing deterministic verification results.")
    verification_path = attempt_result_path(run_path, attempt)
    try:
        return read_verification_result_text(
            run_path,
            attempt,
            expected_statuses=frozenset({VerificationStatus.PASS}),
        )
    except ValueError as error:
        raise ReviewError(
            f"Could not read deterministic verification results: {verification_path}: {error}"
        ) from error


def _read_implementation_summary(run_path: Path) -> str:
    attempt = latest_attempt(
        run_path,
        phases=(AttemptPhase.IMPLEMENTING,),
        statuses=(AttemptStatus.COMPLETED,),
    )
    if attempt is None:
        return "No implementation summary is available."
    try:
        result = read_implementation_result(run_path, attempt)
    except PersistenceCodecError:
        return (
            "Implementation summary is unavailable because the result artifact "
            "could not be read."
        )
    if result is None:
        return "No implementation summary is available."
    return result.summary


def _require_review_result(value: Any) -> ReviewResult:
    if not isinstance(value, ReviewResult):
        raise ReviewError("Review result has the wrong domain type.")
    return value


def _inspect_review_invariants(
    repository: GitRepository,
    run_record: RunRecord,
    *,
    expected_snapshot: WorkspaceSnapshot | None = None,
    current_snapshot: WorkspaceSnapshot | None = None,
) -> tuple[ReviewSafetyViolation, ...]:
    current = current_snapshot or WorkspaceSnapshot.capture(repository)
    if expected_snapshot is None:
        changes = workspace_safety_changes(
            current,
            expected_repository_path=run_record.target_repository_path,
            expected_branch=run_record.starting_branch,
            expected_head_sha=run_record.baseline_sha,
        )
    else:
        changes = expected_snapshot.compare(current)
    return _review_changes(changes)


def _review_changes(
    changes: tuple[WorkspaceChange, ...],
) -> tuple[ReviewSafetyViolation, ...]:
    return tuple(
        ReviewSafetyViolation(
            name=change.name,
            expected=change.expected,
            actual=change.actual,
            message=change.message,
        )
        for change in changes
    )


def _inspect_review_source_fingerprint(
    run_path: Path,
    run_record: RunRecord,
    current: WorkspaceSnapshot,
    *,
    verification_commands: tuple[VerificationCommand, ...],
) -> tuple[ReviewSafetyViolation, ...]:
    try:
        expected = read_verification_source_fingerprint(
            run_path,
            run_record,
            expected_statuses=frozenset({VerificationStatus.PASS}),
            verification_commands=verification_commands,
        )
    except ValueError as error:
        return (
            ReviewSafetyViolation(
                name="verification-evidence",
                expected="readable canonical workspace fingerprint",
                actual=str(error),
                message="Could not validate the verified review source.",
            ),
        )
    if current.matches_fingerprint(expected):
        return ()
    return (
        ReviewSafetyViolation(
            name="workspace-fingerprint",
            expected=expected,
            actual=(
                current.fingerprint
                if current.inspection_complete
                else "incomplete workspace inspection"
            ),
            message="Review source workspace changed after deterministic verification.",
        ),
    )


def _format_files(files: tuple[str, ...]) -> str:
    if not files:
        return "empty"
    shown = ", ".join(files[:5])
    hidden_count = len(files) - 5
    if hidden_count > 0:
        shown = f"{shown}, and {hidden_count} more"
    return shown


__all__ = [
    "FindingDisposition",
    "ReviewError",
    "ReviewResultConsistencyError",
    "ReviewSafetyViolation",
    "ReviewStageResult",
    "ReviewVerdict",
    "run_review_stage",
]
