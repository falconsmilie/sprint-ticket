from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from ._verification_artifacts import (
    _read_verification_source_fingerprint,
    _VerificationArtifactError,
)
from .attempts import (
    attempt_result_path,
    finish_phase_attempt,
    latest_attempt,
    start_attempt,
    update_attempt,
)
from .codex import (
    CodexExecution,
    CodexExecutionFailure,
    CodexProcessRunner,
    Sandbox,
    _CodexResultKind,
    _parse_codex_result,
)
from .codex import (
    _execute as _execute_codex,
)
from .config import AppConfig, VerificationCommand
from .git import GitRepository
from .git_safety import (
    WorkspaceChange,
    WorkspaceSnapshot,
    workspace_safety_changes,
)
from .models import StageOutcome, WorkflowState
from .resolved_config import config_from_resolved_run_config
from .runs import (
    BASELINE_RECORD_FILE,
    RUN_RECORD_FILE,
    RUN_TICKET_FILE,
    RunError,
    RunRecord,
    load_baseline_record,
    load_run_record,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_REVIEW_PROMPT_TEMPLATE = _PROJECT_ROOT / "prompts" / "review.md"
_REVIEW_RESULT_SCHEMA = _PROJECT_ROOT / "schemas" / "review-result.schema.json"


class ReviewError(RunError):
    """Raised when the review stage cannot be prepared."""


class ReviewResultConsistencyError(ValueError):
    """Raised when a schema-valid review result contradicts itself."""


class ReviewVerdict(StrEnum):
    PASS = "PASS"
    CORRECTIONS_REQUIRED = "CORRECTIONS_REQUIRED"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"


class FindingDisposition(StrEnum):
    REQUIRED = "REQUIRED"
    ADVISORY = "ADVISORY"
    FOLLOW_UP = "FOLLOW_UP"


class _FindingScopeRelation(StrEnum):
    TICKET = "TICKET"
    IMPLEMENTATION = "IMPLEMENTATION"
    REPOSITORY_AUTHORITY = "REPOSITORY_AUTHORITY"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    AMBIGUOUS = "AMBIGUOUS"


_REVIEW_FINDING_SCOPE_RELATIONS = frozenset(
    relation.value for relation in _FindingScopeRelation
)

_AUTOMATIC_CORRECTION_SCOPE_RELATIONS = frozenset(
    {
        _FindingScopeRelation.TICKET.value,
        _FindingScopeRelation.IMPLEMENTATION.value,
    }
)


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
    codex_execution: CodexExecution | None
    review_result: dict[str, Any] | None
    safety_violations: tuple[ReviewSafetyViolation, ...]
    processing_error: str | None
    controller_message: str

    @property
    def successful(self) -> bool:
        return self.outcome == StageOutcome.COMPLETED

    @property
    def required_findings(self) -> tuple[dict[str, Any], ...]:
        if self.review_result is None:
            return ()
        return _required_findings(self.review_result)


def run_review_stage(
    config: AppConfig,
    run_dir: Path | str,
    *,
    codex_runner: CodexProcessRunner | None = None,
    clock: Callable[[], datetime] | None = None,
) -> ReviewStageResult:
    run_path = Path(run_dir)
    run_record_path = run_path / RUN_RECORD_FILE
    run_record = load_run_record(run_record_path)
    # A later local configuration cannot select a different review invocation.
    config = config_from_resolved_run_config(run_record.resolved_config)
    if run_record.state != WorkflowState.REVIEWING:
        raise ReviewError(
            f"Review requires run state REVIEWING; found {run_record.state.value}."
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
    attempt_record = start_attempt(
        run_path,
        phase=WorkflowState.REVIEWING.value,
        before_workspace_fingerprint=before_workspace_fingerprint,
        clock=clock,
    )
    artifact_directory = attempt_record.artifact_directory
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
    try:
        update_attempt(attempt_record, process_started=True)
        execution = _execute_codex(
            prompt=prompt,
            repo_path=repository.path,
            sandbox=Sandbox.READ_ONLY,
            output_schema=_REVIEW_RESULT_SCHEMA,
            artifact_directory=artifact_directory,
            executable=config.codex.executable,
            execution_config=config.codex.execution,
            runner=codex_runner,
            _result_kind=_CodexResultKind.REVIEW,
        )
    except CodexExecutionFailure as error:
        safety_violations = _inspect_review_invariants(
            repository,
            run_record,
            expected_snapshot=repository_snapshot,
        )
        outcome = (
            StageOutcome.HUMAN_REQUIRED if safety_violations else StageOutcome.FAILED
        )
        message = (
            "Codex review failed and repository safety invariants were violated."
            if safety_violations
            else error.execution.failure_message or str(error)
        )
        return _finish(
            run_record=run_record,
            run_dir=run_path,
            artifact_directory=artifact_directory,
            execution=error.execution,
            review_result=None,
            safety_violations=safety_violations,
            processing_error=None,
            outcome=outcome,
            controller_message=message,
        )

    review_result = _require_review_result(execution.structured_result)
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


def validate_review_result_semantics(result: dict[str, Any]) -> None:
    verdict = result.get("verdict")
    required_count = len(_required_findings(result))
    if verdict == ReviewVerdict.PASS.value and required_count:
        raise ReviewResultConsistencyError(
            "PASS results must not contain REQUIRED findings."
        )
    if verdict == ReviewVerdict.CORRECTIONS_REQUIRED.value and required_count == 0:
        raise ReviewResultConsistencyError(
            "CORRECTIONS_REQUIRED results must contain at least one REQUIRED finding."
        )


def _validate_review_result_artifact(value: Any) -> dict[str, Any]:
    """Validate persisted review evidence before it influences final handoff."""

    result = _parse_codex_result(value, _result_kind=_CodexResultKind.REVIEW)
    validate_review_result_semantics(result)
    return result


def _correction_eligible_review_findings(
    result: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        finding
        for finding in _required_findings(result)
        if finding.get("scope_relation") in _AUTOMATIC_CORRECTION_SCOPE_RELATIONS
    )


def format_review_result(result: ReviewStageResult) -> str:
    rows = [
        f"Review state: {result.run_record.state.value}",
        f"Artifacts: {result.artifact_directory}",
        result.controller_message,
    ]
    if result.review_result is not None:
        rows.append(f"Verdict: {result.review_result['verdict']}")
        rows.append(f"Findings: {len(result.review_result['findings'])}")
        rows.append(f"Required findings: {len(result.required_findings)}")
    if result.processing_error:
        rows.append(f"Processing error: {result.processing_error}")
    if result.safety_violations:
        rows.append("Safety violations:")
        rows.extend(
            f"  - {violation.name}: expected {violation.expected}, got {violation.actual}"
            for violation in result.safety_violations
        )
    return "\n".join(rows)


def _finish_valid_review_result(
    *,
    run_record: RunRecord,
    run_dir: Path,
    artifact_directory: Path,
    execution: CodexExecution | None,
    review_result: dict[str, Any],
) -> ReviewStageResult:
    try:
        validate_review_result_semantics(review_result)
    except ReviewResultConsistencyError as error:
        return _finish(
            run_record=run_record,
            run_dir=run_dir,
            artifact_directory=artifact_directory,
            execution=execution,
            review_result=review_result,
            safety_violations=(),
            processing_error=str(error),
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message=f"Review result is logically contradictory: {error}",
        )

    verdict = ReviewVerdict(review_result["verdict"])
    if verdict == ReviewVerdict.PASS:
        outcome = StageOutcome.COMPLETED
        controller_message = "Review passed; final report can start."
    elif verdict == ReviewVerdict.CORRECTIONS_REQUIRED:
        required_findings = _required_findings(review_result)
        eligible_findings = _correction_eligible_review_findings(review_result)
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
    execution: CodexExecution | None,
    review_result: dict[str, Any] | None,
    safety_violations: tuple[ReviewSafetyViolation, ...],
    processing_error: str | None,
    outcome: StageOutcome,
    controller_message: str,
) -> ReviewStageResult:
    after_fingerprint: str | None = None
    try:
        after_fingerprint = WorkspaceSnapshot.capture(
            GitRepository(Path(run_record.target_repository_path))
        ).fingerprint
    except (OSError, RuntimeError, ValueError):
        pass
    finish_phase_attempt(
        run_dir,
        phase=WorkflowState.REVIEWING.value,
        stage_outcome=outcome.value,
        after_workspace_fingerprint=after_fingerprint,
        process_started=(None if execution is None else execution.process_started),
        execution_path=(None if execution is None else execution.execution_json_path),
    )
    return ReviewStageResult(
        run_dir=run_dir,
        run_record=run_record,
        outcome=outcome,
        artifact_directory=artifact_directory,
        codex_execution=execution,
        review_result=review_result,
        safety_violations=safety_violations,
        processing_error=processing_error,
        controller_message=controller_message,
    )


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
        phases=(WorkflowState.VERIFYING.value,),
        statuses=("COMPLETED",),
    )
    if attempt is None:
        raise ReviewError("Missing deterministic verification results.")
    verification_path = attempt_result_path(run_path, attempt)
    if verification_path is None:
        raise ReviewError("Verification attempt has no typed result path.")
    try:
        data = json.loads(verification_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ReviewError(
            f"Missing deterministic verification results: {verification_path}"
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewError(
            f"Could not read deterministic verification results: {verification_path}: {error}"
        ) from error
    _require_passing_verification_round(data, verification_path)
    return json.dumps(data, indent=2, sort_keys=True)


def _require_passing_verification_round(data: Any, path: Path) -> None:
    if not isinstance(data, dict):
        raise ReviewError(
            f"Deterministic verification results must be an object: {path}"
        )
    status = data.get("status")
    if status != "PASS":
        raise ReviewError(
            "Deterministic verification results must have PASS status before review: "
            f"{path} has {status!r}."
        )


def _read_implementation_summary(run_path: Path) -> str:
    attempt = latest_attempt(
        run_path,
        phases=(WorkflowState.IMPLEMENTING.value,),
        statuses=("COMPLETED",),
    )
    result_path = None if attempt is None else attempt_result_path(run_path, attempt)
    if result_path is None:
        return "No implementation summary is available."
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "No implementation summary is available."
    except (OSError, json.JSONDecodeError):
        return (
            "Implementation summary is unavailable because the result artifact "
            "could not be read."
        )

    summary = data.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary
    return "No implementation summary is available."


def _require_review_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReviewError("Review result must be a JSON object.")
    return value


def _required_findings(result: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    findings = result.get("findings", [])
    if not isinstance(findings, list):
        return ()
    return tuple(
        finding
        for finding in findings
        if isinstance(finding, dict)
        and finding.get("disposition") == FindingDisposition.REQUIRED.value
    )


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
        expected = _read_verification_source_fingerprint(
            run_path,
            run_record,
            expected_statuses=frozenset({"PASS"}),
            verification_commands=verification_commands,
        )
    except _VerificationArtifactError as error:
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
    "format_review_result",
    "run_review_stage",
    "validate_review_result_semantics",
]
