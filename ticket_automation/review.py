from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

from .codex import (
    CodexExecution,
    CodexExecutionFailure,
    CodexProcessRunner,
    Sandbox,
    execute as execute_codex,
    parse_sandbox,
)
from .config import AppConfig
from .git import GitCommandError, GitRepository
from .models import WorkflowState
from .runs import (
    BASELINE_RECORD_FILE,
    RUN_RECORD_FILE,
    RUN_TICKET_FILE,
    RunError,
    RunRecord,
    load_baseline_record,
    load_run_record,
    save_run_record,
)
from .verification import VERIFICATION_DIR_NAME


REVIEW_DIR_NAME = "reviews"
REVIEW_ROUND_OFFSET = 1
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_REVIEW_PROMPT_TEMPLATE = _PROJECT_ROOT / "prompts" / "review.md"
_REVIEW_RESULT_SCHEMA = _PROJECT_ROOT / "schemas" / "review-result.schema.json"
_IMPLEMENTATION_DIR_NAME = "implementation"
_IMPLEMENTATION_RESULT_FILE = "result.json"


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
    artifact_directory: Path
    codex_execution: CodexExecution | None
    review_result: dict[str, Any] | None
    safety_violations: tuple[ReviewSafetyViolation, ...]
    processing_error: str | None
    controller_message: str

    @property
    def successful(self) -> bool:
        return self.run_record.state == WorkflowState.REVIEW

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
    if run_record.state != WorkflowState.VERIFY:
        raise ReviewError(
            f"Review requires run state VERIFY; found {run_record.state.value}."
        )

    baseline_record = load_baseline_record(run_path / BASELINE_RECORD_FILE)
    if baseline_record.branch != run_record.starting_branch:
        raise ReviewError("Run record and baseline branch do not match.")
    if baseline_record.head_sha != run_record.baseline_sha:
        raise ReviewError("Run record and baseline HEAD do not match.")

    review_sandbox = parse_sandbox(config.codex.review_sandbox)
    if review_sandbox != Sandbox.READ_ONLY:
        raise ReviewError("Review requires codex.review_sandbox to be read-only.")

    repository = GitRepository(Path(run_record.target_repository_path))
    review_round = run_record.current_correction_round + REVIEW_ROUND_OFFSET
    artifact_directory = run_path / REVIEW_DIR_NAME / f"round-{review_round}"
    if artifact_directory.exists() and any(artifact_directory.iterdir()):
        raise ReviewError(f"Review artifacts already exist for round-{review_round}.")

    starting_violations = _inspect_review_invariants(repository, run_record)
    if starting_violations:
        return _finish(
            run_record=run_record,
            run_record_path=run_record_path,
            run_dir=run_path,
            artifact_directory=artifact_directory,
            execution=None,
            review_result=None,
            safety_violations=starting_violations,
            processing_error=None,
            state=WorkflowState.HUMAN_REQUIRED,
            controller_message="Repository no longer matches the recorded review baseline.",
            clock=clock,
        )

    prompt = _render_review_prompt(
        ticket_text=_read_snapshotted_ticket(run_path / RUN_TICKET_FILE),
        run_record=run_record,
        current_branch=_current_branch_label(repository),
        verification_results=_read_verification_results(run_path, run_record),
        implementation_summary=_read_implementation_summary(run_path),
    )

    try:
        execution = execute_codex(
            prompt=prompt,
            repo_path=repository.path,
            sandbox=Sandbox.READ_ONLY,
            output_schema=_REVIEW_RESULT_SCHEMA,
            artifact_directory=artifact_directory,
            executable=config.codex.executable,
            runner=codex_runner,
        )
    except CodexExecutionFailure as error:
        safety_violations = _inspect_review_invariants(repository, run_record)
        state = (
            WorkflowState.HUMAN_REQUIRED
            if safety_violations
            else WorkflowState.FAILED
        )
        message = (
            "Codex review failed and repository safety invariants were violated."
            if safety_violations
            else error.execution.failure_message or str(error)
        )
        return _finish(
            run_record=run_record,
            run_record_path=run_record_path,
            run_dir=run_path,
            artifact_directory=artifact_directory,
            execution=error.execution,
            review_result=None,
            safety_violations=safety_violations,
            processing_error=None,
            state=state,
            controller_message=message,
            clock=clock,
        )

    review_result = _require_review_result(execution.structured_result)
    safety_violations = _inspect_review_invariants(repository, run_record)
    if safety_violations:
        return _finish(
            run_record=run_record,
            run_record_path=run_record_path,
            run_dir=run_path,
            artifact_directory=artifact_directory,
            execution=execution,
            review_result=review_result,
            safety_violations=safety_violations,
            processing_error=None,
            state=WorkflowState.HUMAN_REQUIRED,
            controller_message="Repository safety invariants were violated during review.",
            clock=clock,
        )

    try:
        validate_review_result_semantics(review_result)
    except ReviewResultConsistencyError as error:
        return _finish(
            run_record=run_record,
            run_record_path=run_record_path,
            run_dir=run_path,
            artifact_directory=artifact_directory,
            execution=execution,
            review_result=review_result,
            safety_violations=(),
            processing_error=str(error),
            state=WorkflowState.HUMAN_REQUIRED,
            controller_message=f"Review result is logically contradictory: {error}",
            clock=clock,
        )

    verdict = ReviewVerdict(review_result["verdict"])
    if verdict == ReviewVerdict.PASS:
        state = WorkflowState.REVIEW
        controller_message = "Review passed; final reporting has not been added yet."
    elif verdict == ReviewVerdict.CORRECTIONS_REQUIRED:
        state = WorkflowState.CORRECT
        controller_message = "Review found required corrections."
    else:
        state = WorkflowState.HUMAN_REQUIRED
        controller_message = "Review requires human attention."

    return _finish(
        run_record=run_record,
        run_record_path=run_record_path,
        run_dir=run_path,
        artifact_directory=artifact_directory,
        execution=execution,
        review_result=review_result,
        safety_violations=(),
        processing_error=None,
        state=state,
        controller_message=controller_message,
        clock=clock,
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


def _finish(
    *,
    run_record: RunRecord,
    run_record_path: Path,
    run_dir: Path,
    artifact_directory: Path,
    execution: CodexExecution | None,
    review_result: dict[str, Any] | None,
    safety_violations: tuple[ReviewSafetyViolation, ...],
    processing_error: str | None,
    state: WorkflowState,
    controller_message: str,
    clock: Callable[[], datetime] | None,
) -> ReviewStageResult:
    updated_record = run_record.with_state(state, updated_timestamp=_timestamp(clock))
    save_run_record(updated_record, run_record_path)
    return ReviewStageResult(
        run_dir=run_dir,
        run_record=updated_record,
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
        raise ReviewError(f"Could not read snapshotted ticket: {path}: {error}") from error
    except UnicodeDecodeError as error:
        raise ReviewError(
            f"Snapshotted ticket must be valid UTF-8 Markdown: {path}"
        ) from error


def _read_verification_results(run_path: Path, run_record: RunRecord) -> str:
    round_index = run_record.current_correction_round
    verification_path = (
        run_path / VERIFICATION_DIR_NAME / f"round-{round_index}.json"
    )
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
        raise ReviewError(f"Deterministic verification results must be an object: {path}")
    status = data.get("status")
    if status != "PASS":
        raise ReviewError(
            "Deterministic verification results must have PASS status before review: "
            f"{path} has {status!r}."
        )


def _read_implementation_summary(run_path: Path) -> str:
    result_path = (
        run_path / _IMPLEMENTATION_DIR_NAME / _IMPLEMENTATION_RESULT_FILE
    )
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
) -> tuple[ReviewSafetyViolation, ...]:
    violations: list[ReviewSafetyViolation] = []
    try:
        current_branch = repository.current_branch()
        current_head = repository.head_sha()
        staged_files = repository.staged_files()
    except GitCommandError as error:
        return (
            ReviewSafetyViolation(
                name="git-inspection",
                expected="Git inspection succeeds",
                actual=str(error),
                message="Could not inspect repository review safety invariants.",
            ),
        )

    if current_branch != run_record.starting_branch:
        violations.append(
            ReviewSafetyViolation(
                name="branch",
                expected=run_record.starting_branch,
                actual="<detached>" if current_branch is None else current_branch,
                message="Current branch changed during review.",
            )
        )
    if current_head != run_record.baseline_sha:
        violations.append(
            ReviewSafetyViolation(
                name="HEAD",
                expected=run_record.baseline_sha,
                actual=current_head,
                message="HEAD changed during review.",
            )
        )
    if staged_files:
        violations.append(
            ReviewSafetyViolation(
                name="staging",
                expected="empty",
                actual=_format_files(staged_files),
                message="Staging area is not empty during review.",
            )
        )
    return tuple(violations)


def _current_branch_label(repository: GitRepository) -> str:
    branch = repository.current_branch()
    return "<detached>" if branch is None else branch


def _format_files(files: tuple[str, ...]) -> str:
    if not files:
        return "empty"
    shown = ", ".join(files[:5])
    hidden_count = len(files) - 5
    if hidden_count > 0:
        shown = f"{shown}, and {hidden_count} more"
    return shown


def _timestamp(clock: Callable[[], datetime] | None) -> str:
    now = datetime.now(timezone.utc) if clock is None else clock()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return (
        now.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


__all__ = [
    "FindingDisposition",
    "REVIEW_DIR_NAME",
    "REVIEW_ROUND_OFFSET",
    "ReviewError",
    "ReviewResultConsistencyError",
    "ReviewSafetyViolation",
    "ReviewStageResult",
    "ReviewVerdict",
    "format_review_result",
    "run_review_stage",
    "validate_review_result_semantics",
]
