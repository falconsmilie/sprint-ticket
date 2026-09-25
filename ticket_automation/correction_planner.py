"""Application policy for turning upstream outcomes into correction causes."""

from __future__ import annotations

from pathlib import Path

from .attempts import latest_attempt
from .corrections import (
    CorrectionCauseSet,
    CorrectionError,
    ReviewCorrectionCause,
    ReviewCorrectionScope,
    VerificationCorrectionCause,
)
from .domain.task_results import ReviewResult, ReviewVerdict
from .models import AttemptPhase, AttemptStatus
from .persistence_codecs import (
    PersistenceCodecError,
    read_review_result,
    read_verification_failures,
)
from .run_ownership import RunOwnership
from .verification import VerificationFailure


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
        finding for finding in required_findings if finding.correction_eligible
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


def plan_pending_correction(
    run_dir: Path | str,
    *,
    run_ownership: RunOwnership | None = None,
) -> CorrectionCauseSet:
    run_path = Path(run_dir)
    if run_ownership is not None:
        run_path = run_ownership.validate_run_path(run_path)
    verification_failures = _load_latest_verification_failures(
        run_path,
        run_ownership=run_ownership,
    )
    if verification_failures:
        plan = plan_verification_correction(verification_failures)
        if run_ownership is not None:
            run_ownership.validate_run_path(run_path)
        return plan

    review_result = _load_latest_review_result(
        run_path,
        run_ownership=run_ownership,
    )
    if review_result is not None:
        plan = plan_review_correction(review_result)
        if run_ownership is not None:
            run_ownership.validate_run_path(run_path)
        return plan

    raise CorrectionError(
        "Run is in CORRECTING state but no verification failures or REQUIRED review "
        "findings were found."
    )


def _load_latest_verification_failures(
    run_path: Path,
    *,
    run_ownership: RunOwnership | None = None,
) -> tuple[VerificationFailure, ...]:
    attempt = latest_attempt(
        run_path,
        phases=(AttemptPhase.VERIFYING,),
        statuses=(AttemptStatus.COMPLETED,),
        run_ownership=run_ownership,
    )
    if attempt is None:
        return ()
    try:
        return read_verification_failures(
            run_path,
            attempt,
            run_ownership=run_ownership,
        )
    except PersistenceCodecError as error:
        raise CorrectionError(
            f"Verification correction source is invalid: {error}"
        ) from error


def _load_latest_review_result(
    run_path: Path,
    *,
    run_ownership: RunOwnership | None = None,
) -> ReviewResult | None:
    attempt = latest_attempt(
        run_path,
        phases=(AttemptPhase.REVIEWING,),
        statuses=(AttemptStatus.COMPLETED,),
        run_ownership=run_ownership,
    )
    if attempt is None:
        return None
    try:
        return read_review_result(
            run_path,
            attempt,
            run_ownership=run_ownership,
        )
    except PersistenceCodecError as error:
        raise CorrectionError(
            f"Review correction source is invalid: {error}"
        ) from error


__all__ = [
    "plan_pending_correction",
    "plan_review_correction",
    "plan_verification_correction",
]
