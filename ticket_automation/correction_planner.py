"""Application policy for turning upstream outcomes into correction causes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .attempts import attempt_result_path, latest_attempt
from .corrections import (
    CorrectionCauseSet,
    CorrectionError,
    ReviewCorrectionCause,
    ReviewCorrectionScope,
    VerificationCorrectionCause,
)
from .domain.task_results import ReviewResult, ReviewVerdict
from .models import WorkflowState
from .task_result_codecs import decode_review_result
from .verification import VerificationError, VerificationFailure


def plan_verification_correction(
    failures: tuple[VerificationFailure, ...],
) -> CorrectionCauseSet:
    if not isinstance(failures, tuple):
        raise CorrectionError("Verification correction failures must be a tuple.")
    if not failures:
        raise CorrectionError("Verification correction requires at least one failure.")
    if any(not isinstance(failure, VerificationFailure) for failure in failures):
        raise CorrectionError(
            "Verification correction failures must be VerificationFailure values."
        )
    return CorrectionCauseSet(
        tuple(
            VerificationCorrectionCause(
                gate_name=failure.gate_name,
                command=failure.command,
                failure_summary=failure.failure_summary,
                stdout_excerpt=failure.stdout_excerpt,
                stderr_excerpt=failure.stderr_excerpt,
                exit_code=failure.exit_code,
                result_path=failure.result_path,
            )
            for failure in failures
        )
    )


def plan_review_correction(result: ReviewResult) -> CorrectionCauseSet:
    if not isinstance(result, ReviewResult):
        raise CorrectionError("Review correction source must be a ReviewResult.")
    if result.verdict is not ReviewVerdict.CORRECTIONS_REQUIRED:
        raise CorrectionError(
            "Review correction requires a CORRECTIONS_REQUIRED result."
        )

    required_findings = result.required_findings
    eligible_findings = tuple(
        finding
        for finding in required_findings
        if finding.correction_eligible
    )
    if len(eligible_findings) != len(required_findings):
        raise CorrectionError(
            "Review correction contains findings that are not eligible for "
            "automatic correction."
        )

    causes = tuple(
        ReviewCorrectionCause(
            finding_id=finding.id,
            summary=finding.title,
            details=finding.description,
            evidence=finding.evidence,
            required_change=finding.required_change,
            acceptance_criteria=finding.acceptance_criteria,
            scope_relation=ReviewCorrectionScope(finding.scope_relation.value),
        )
        for finding in eligible_findings
    )
    return CorrectionCauseSet(causes)


def plan_pending_correction(run_dir: Path | str) -> CorrectionCauseSet:
    run_path = Path(run_dir)
    verification_failures = _load_latest_verification_failures(run_path)
    if verification_failures:
        return plan_verification_correction(verification_failures)

    review_result = _load_latest_review_result(run_path)
    if review_result is not None:
        return plan_review_correction(review_result)

    raise CorrectionError(
        "Run is in CORRECTING state but no verification failures or REQUIRED review "
        "findings were found."
    )


def _load_latest_verification_failures(
    run_path: Path,
) -> tuple[VerificationFailure, ...]:
    attempt = latest_attempt(
        run_path,
        phases=(WorkflowState.VERIFYING.value,),
        statuses=("COMPLETED",),
    )
    if attempt is None:
        return ()
    result_path = attempt_result_path(run_path, attempt)
    if result_path is None or not result_path.is_file():
        return ()
    data = _read_json_object(result_path)
    if data.get("status") != "FAIL":
        return ()

    encoded_failures = data.get("correction_reasons")
    if not isinstance(encoded_failures, list) or not encoded_failures:
        raise CorrectionError(
            "Failed verification evidence must contain a non-empty "
            "correction_reasons list."
        )
    failures: list[VerificationFailure] = []
    for index, value in enumerate(encoded_failures, start=1):
        try:
            failures.append(VerificationFailure._from_dict(value))
        except VerificationError as error:
            raise CorrectionError(
                f"Verification correction reason {index} is invalid: {error}"
            ) from error
    return tuple(failures)


def _load_latest_review_result(run_path: Path) -> ReviewResult | None:
    attempt = latest_attempt(
        run_path,
        phases=(WorkflowState.REVIEWING.value,),
        statuses=("COMPLETED",),
    )
    if attempt is None:
        return None
    result_path = attempt_result_path(run_path, attempt)
    if result_path is None or not result_path.is_file():
        return None
    try:
        return decode_review_result(_read_json_object(result_path))
    except ValueError as error:
        raise CorrectionError(
            f"Review correction source is invalid: {error}"
        ) from error


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CorrectionError(
            f"Could not read correction source data: {path}: {error}"
        ) from error
    if not isinstance(data, dict):
        raise CorrectionError(f"Correction source data must be an object: {path}")
    return data


__all__ = [
    "plan_pending_correction",
    "plan_review_correction",
    "plan_verification_correction",
]
