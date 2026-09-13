from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from .codex import (
    CodexExecution,
    CodexExecutionFailure,
    CodexProcessRunner,
    CodexResultValidationError,
    Sandbox,
    parse_sandbox,
    validate_json_schema,
)
from .codex import (
    execute as execute_codex,
)
from .config import AppConfig
from .git import GitCommandError, GitRepository
from .models import StageOutcome, WorkflowState
from .runs import (
    BASELINE_RECORD_FILE,
    RUN_RECORD_FILE,
    RUN_TICKET_FILE,
    RunError,
    RunRecord,
    load_baseline_record,
    load_run_record,
)
from .verification import VERIFICATION_DIR_NAME

REVIEW_DIR_NAME = "reviews"
REVIEW_ROUND_OFFSET = 1
_REVIEW_CHECKPOINT_SCHEMA_VERSION = 1
_REVIEW_CHECKPOINT_FORMAT = "ticket_automation.review_checkpoint"
_REVIEW_CHECKPOINT_FILE = "checkpoint.json"
_INCOMPLETE_ARTIFACT_DIR_NAME = "_incomplete"
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


class _ReviewArtifactState(StrEnum):
    MISSING = "MISSING"
    PARTIAL = "PARTIAL"
    COMPLETE = "COMPLETE"


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
    del clock
    run_path = Path(run_dir)
    run_record_path = run_path / RUN_RECORD_FILE
    run_record = load_run_record(run_record_path)
    if run_record.state != WorkflowState.REVIEWING:
        raise ReviewError(
            f"Review requires run state REVIEWING; found {run_record.state.value}."
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
    review_round = run_record.current_review_round + REVIEW_ROUND_OFFSET
    artifact_directory = run_path / REVIEW_DIR_NAME / f"round-{review_round}"
    result_path = artifact_directory / "result.json"

    starting_violations = _inspect_review_invariants(repository, run_record)
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

    prompt = _render_review_prompt(
        ticket_text=_read_snapshotted_ticket(run_path / RUN_TICKET_FILE),
        run_record=run_record,
        current_branch=_current_branch_label(repository),
        verification_results=_read_verification_results(run_path, run_record),
        implementation_summary=_read_implementation_summary(run_path),
    )
    artifact_state = _review_artifact_state(artifact_directory, result_path)
    if artifact_state == _ReviewArtifactState.COMPLETE:
        review_result, adoption_problem = _load_existing_review_result(
            artifact_directory=artifact_directory,
            result_path=result_path,
            expected_prompt=prompt,
            run_record=run_record,
            review_round=review_round,
        )
        if adoption_problem is None and review_result is not None:
            safety_violations = _inspect_review_invariants(repository, run_record)
            if safety_violations:
                return _finish(
                    run_record=run_record,
                    run_dir=run_path,
                    artifact_directory=artifact_directory,
                    execution=None,
                    review_result=review_result,
                    safety_violations=safety_violations,
                    processing_error=None,
                    outcome=StageOutcome.HUMAN_REQUIRED,
                    controller_message=(
                        "Repository safety invariants were violated before "
                        "review checkpoint adoption."
                    ),
                )
            return _finish_valid_review_result(
                run_record=run_record,
                run_dir=run_path,
                artifact_directory=artifact_directory,
                execution=None,
                review_result=review_result,
            )
        _archive_review_artifacts(artifact_directory)
    elif artifact_state == _ReviewArtifactState.PARTIAL:
        _archive_review_artifacts(artifact_directory)

    _write_review_checkpoint(
        artifact_directory / _REVIEW_CHECKPOINT_FILE,
        _review_checkpoint(
            run_record,
            review_round=review_round,
            prompt=prompt,
        ),
    )

    try:
        execution = execute_codex(
            prompt=prompt,
            repo_path=repository.path,
            sandbox=Sandbox.READ_ONLY,
            output_schema=_REVIEW_RESULT_SCHEMA,
            artifact_directory=artifact_directory,
            executable=config.codex.executable,
            execution_config=config.codex.execution,
            runner=codex_runner,
        )
    except CodexExecutionFailure as error:
        safety_violations = _inspect_review_invariants(repository, run_record)
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
    safety_violations = _inspect_review_invariants(repository, run_record)
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
        outcome = StageOutcome.CORRECTION_REQUIRED
        controller_message = "Review found required corrections."
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


def _review_artifact_state(
    artifact_directory: Path,
    result_path: Path,
) -> _ReviewArtifactState:
    if result_path.is_file():
        return _ReviewArtifactState.COMPLETE
    if artifact_directory.exists() and any(artifact_directory.iterdir()):
        return _ReviewArtifactState.PARTIAL
    return _ReviewArtifactState.MISSING


def _load_existing_review_result(
    *,
    artifact_directory: Path,
    result_path: Path,
    expected_prompt: str,
    run_record: RunRecord,
    review_round: int,
) -> tuple[dict[str, Any] | None, str | None]:
    expected_checkpoint = _review_checkpoint(
        run_record,
        review_round=review_round,
        prompt=expected_prompt,
    )
    checkpoint_problem = _existing_review_checkpoint_problem(
        artifact_directory / _REVIEW_CHECKPOINT_FILE,
        expected_checkpoint=expected_checkpoint,
    )
    if checkpoint_problem is not None:
        return None, checkpoint_problem

    prompt_problem = _existing_review_prompt_problem(
        artifact_directory / "prompt.md",
        expected_prompt=expected_prompt,
    )
    if prompt_problem is not None:
        return None, prompt_problem

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, f"Could not read existing review result: {error}"
    if not isinstance(result, dict):
        return None, "Existing review result is not a JSON object."

    try:
        validate_json_schema(result, _load_review_result_schema())
    except (CodexResultValidationError, OSError, json.JSONDecodeError) as error:
        return None, f"Existing review result is not schema-valid: {error}"

    execution_problem = _existing_review_execution_problem(
        artifact_directory / "execution.json",
        result_path=result_path,
        run_record=run_record,
    )
    if execution_problem is not None:
        return None, execution_problem

    return result, None


def _existing_review_checkpoint_problem(
    checkpoint_path: Path,
    *,
    expected_checkpoint: dict[str, Any],
) -> str | None:
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "Existing review result is missing checkpoint metadata."
    except (OSError, json.JSONDecodeError) as error:
        return f"Could not read existing review checkpoint metadata: {error}"
    if not isinstance(checkpoint, dict):
        return "Existing review checkpoint metadata is not a JSON object."
    for key, expected_value in expected_checkpoint.items():
        if checkpoint.get(key) != expected_value:
            return (
                "Existing review checkpoint does not match current run metadata: "
                f"{key}."
            )
    return None


def _existing_review_prompt_problem(
    prompt_path: Path,
    *,
    expected_prompt: str,
) -> str | None:
    try:
        prompt = prompt_path.read_text(encoding="utf-8")
    except OSError as error:
        return f"Could not read existing review prompt: {error}"
    if _normalize_prompt_text(prompt) != _normalize_prompt_text(expected_prompt):
        return "Existing review prompt does not match the current run checkpoint."
    return None


def _normalize_prompt_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")


def _existing_review_execution_problem(
    execution_path: Path,
    *,
    result_path: Path,
    run_record: RunRecord,
) -> str | None:
    if not execution_path.exists():
        return None
    try:
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return f"Could not read existing review execution metadata: {error}"
    if not isinstance(execution, dict):
        return "Existing review execution metadata is not a JSON object."
    expected_repo = Path(run_record.target_repository_path).resolve()
    repo_path = execution.get("repo_path")
    if not isinstance(repo_path, str) or Path(repo_path).resolve() != expected_repo:
        return "Existing review execution metadata is for a different repository."
    if execution.get("status") != "SUCCESS":
        return "Existing review execution metadata is not successful."
    if execution.get("sandbox") != Sandbox.READ_ONLY.value:
        return "Existing review execution metadata is not read-only."
    if execution.get("result_json_present") is not True:
        return "Existing review execution metadata does not confirm result.json."
    schema_path = execution.get("output_schema_path")
    if (
        isinstance(schema_path, str)
        and Path(schema_path).resolve() != _REVIEW_RESULT_SCHEMA.resolve()
    ):
        return "Existing review execution metadata used a different result schema."
    artifact_paths = execution.get("artifact_paths")
    if isinstance(artifact_paths, dict):
        recorded_result = artifact_paths.get("result_json")
        if (
            isinstance(recorded_result, str)
            and Path(recorded_result).resolve() != result_path.resolve()
        ):
            return "Existing review execution metadata points at a different result."
    return None


def _load_review_result_schema() -> dict[str, Any]:
    schema = json.loads(_REVIEW_RESULT_SCHEMA.read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        raise ReviewError("Review result schema root must be a JSON object.")
    return schema


def _archive_review_artifacts(artifact_directory: Path) -> None:
    if not artifact_directory.exists():
        return
    archive_root = artifact_directory.parent / _INCOMPLETE_ARTIFACT_DIR_NAME
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_path = _unique_archive_path(archive_root / artifact_directory.name)
    shutil.move(str(artifact_directory), str(archive_path))


def _unique_archive_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.name}-{index}")
        if not candidate.exists():
            return candidate
    raise ReviewError(f"Could not reserve recovery artifact directory: {path}")


def _review_checkpoint(
    run_record: RunRecord,
    *,
    review_round: int,
    prompt: str,
) -> dict[str, Any]:
    return {
        "schema_version": _REVIEW_CHECKPOINT_SCHEMA_VERSION,
        "format": _REVIEW_CHECKPOINT_FORMAT,
        "stage": WorkflowState.REVIEWING.value,
        "status": "STARTED",
        "run_id": run_record.run_id,
        "review_round": review_round,
        "correction_round": run_record.current_correction_round,
        "target_repository_path": str(
            Path(run_record.target_repository_path).resolve()
        ),
        "starting_branch": run_record.starting_branch,
        "baseline_sha": run_record.baseline_sha,
        "prompt_sha256": _text_sha256(_normalize_prompt_text(prompt)),
    }


def _write_review_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(checkpoint, indent=2, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    file_descriptor = -1
    try:
        file_descriptor, temp_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            text=True,
        )
        temp_path = Path(temp_name)
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as temp_file:
            file_descriptor = -1
            temp_file.write(contents)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    except Exception:
        if file_descriptor != -1:
            os.close(file_descriptor)
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
    round_index = run_record.current_correction_round
    verification_path = run_path / VERIFICATION_DIR_NAME / f"round-{round_index}.json"
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
    result_path = run_path / _IMPLEMENTATION_DIR_NAME / _IMPLEMENTATION_RESULT_FILE
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


__all__ = [
    "REVIEW_DIR_NAME",
    "REVIEW_ROUND_OFFSET",
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
