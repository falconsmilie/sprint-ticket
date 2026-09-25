"""Authoritative final handoff acceptance policy and application service."""

from __future__ import annotations

import hashlib
import re
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import TypeAlias, final

from ..attempts import (
    AttemptError,
    AttemptRecord,
    StageAttempt,
    attempt_artifact_layout,
    load_attempt_records,
    require_stage_attempt,
    update_attempt,
)
from ..domain.task_results import ResultValidationError, ReviewVerdict
from ..execution_evidence import read_execution_evidence
from ..git import GitCommandError, GitRepository
from ..git_safety import WorkspaceSnapshot
from ..models import (
    PHASE_DEFINITIONS,
    AttemptPhase,
    AttemptStatus,
    StopCategory,
    VerificationStatus,
    WorkflowState,
)
from ..persistence_codecs import (
    PersistenceCodecError,
    read_implementation_result,
    read_review_result,
)
from ..runs import RunRecord
from ..verification_evidence import read_verification_source_fingerprint
from .agent_execution import (
    EXECUTION_EVIDENCE_FILE,
    AgentExecutionStatus,
    AgentTaskKind,
    InvocationStart,
)
from .ports.handoff import (
    FINAL_PATCH_FILE,
    FinalPatchCapture,
    FinalPatchCaptureRequest,
    FinalPatchReference,
)

_PASSING_HANDOFF_MESSAGE = "Final workspace and evidence consistency checks passed."
_HANDOFF_OUTCOME_SEAL = object()
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_AGENT_TASK_BY_PHASE = {
    AttemptPhase.IMPLEMENTING: AgentTaskKind.IMPLEMENTATION,
    AttemptPhase.REVIEWING: AgentTaskKind.REVIEW,
    AttemptPhase.CORRECTING: AgentTaskKind.CORRECTION,
}


@dataclass(frozen=True)
class HandoffAcceptanceRequest:
    run_dir: Path
    run_record: RunRecord
    attempt: StageAttempt | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_dir, Path):
            raise TypeError("run_dir must be a Path.")
        if not isinstance(self.run_record, RunRecord):
            raise TypeError("run_record must be a RunRecord.")
        if self.attempt is not None and not isinstance(self.attempt, StageAttempt):
            raise TypeError("attempt must be a StageAttempt or None.")


@dataclass(frozen=True)
class VerificationHandoffEvidence:
    attempt_sequence: int
    before_workspace_fingerprint: str
    after_workspace_fingerprint: str

    def __post_init__(self) -> None:
        _validate_attempt_sequence(self.attempt_sequence)
        _validate_fingerprint(
            self.before_workspace_fingerprint,
            field_name="before_workspace_fingerprint",
        )
        _validate_fingerprint(
            self.after_workspace_fingerprint,
            field_name="after_workspace_fingerprint",
        )


@dataclass(frozen=True)
class ReviewHandoffEvidence:
    attempt_sequence: int
    before_workspace_fingerprint: str | None
    after_workspace_fingerprint: str | None
    verdict: ReviewVerdict

    def __post_init__(self) -> None:
        _validate_attempt_sequence(self.attempt_sequence)
        _validate_optional_fingerprint(
            self.before_workspace_fingerprint,
            field_name="before_workspace_fingerprint",
        )
        _validate_optional_fingerprint(
            self.after_workspace_fingerprint,
            field_name="after_workspace_fingerprint",
        )
        if not isinstance(self.verdict, ReviewVerdict):
            raise TypeError("verdict must be a ReviewVerdict.")


@dataclass(frozen=True)
class HandoffAttemptEvidence:
    sequence: int
    phase: AttemptPhase
    status: AttemptStatus
    before_workspace_fingerprint: str | None
    after_workspace_fingerprint: str | None

    def __post_init__(self) -> None:
        _validate_attempt_sequence(self.sequence)
        if not isinstance(self.phase, AttemptPhase):
            raise TypeError("attempt phase must be an AttemptPhase.")
        if not isinstance(self.status, AttemptStatus):
            raise TypeError("attempt status must be an AttemptStatus.")
        _validate_optional_text(
            self.before_workspace_fingerprint,
            field_name="attempt before_workspace_fingerprint",
        )
        _validate_optional_text(
            self.after_workspace_fingerprint,
            field_name="attempt after_workspace_fingerprint",
        )


@dataclass(frozen=True)
class HandoffPolicyRequest:
    repository_path: Path
    expected_branch: str
    expected_head_sha: str
    current_correction_round: int
    max_correction_rounds: int
    current_review_round: int
    workspace: WorkspaceSnapshot | None
    attempts: tuple[HandoffAttemptEvidence, ...]
    verification: VerificationHandoffEvidence | None
    review: ReviewHandoffEvidence | None
    attempt_problem: str | None = None
    verification_problem: str | None = None
    review_problem: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.repository_path, Path):
            raise TypeError("repository_path must be a Path.")
        for value, name in (
            (self.expected_branch, "expected_branch"),
            (self.expected_head_sha, "expected_head_sha"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        if self.workspace is not None and not isinstance(
            self.workspace,
            WorkspaceSnapshot,
        ):
            raise TypeError("workspace must be a WorkspaceSnapshot or None.")
        if not isinstance(self.attempts, tuple) or any(
            not isinstance(item, HandoffAttemptEvidence) for item in self.attempts
        ):
            raise TypeError("attempts must be an immutable tuple of handoff evidence.")
        if self.verification is not None and not isinstance(
            self.verification,
            VerificationHandoffEvidence,
        ):
            raise TypeError("verification must be VerificationHandoffEvidence or None.")
        if self.review is not None and not isinstance(
            self.review,
            ReviewHandoffEvidence,
        ):
            raise TypeError("review must be ReviewHandoffEvidence or None.")
        for value, name in (
            (self.attempt_problem, "attempt_problem"),
            (self.verification_problem, "verification_problem"),
            (self.review_problem, "review_problem"),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string or None.")
        for value, name in (
            (self.current_correction_round, "current_correction_round"),
            (self.max_correction_rounds, "max_correction_rounds"),
            (self.current_review_round, "current_review_round"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")


@dataclass(frozen=True)
class HandoffPolicyProblem:
    stop_category: StopCategory
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.stop_category, StopCategory):
            raise TypeError("stop_category must be a StopCategory.")
        _validate_reason(self.reason)


@final
@dataclass(frozen=True)
class HandoffAccepted:
    initial_workspace: WorkspaceSnapshot
    final_workspace: WorkspaceSnapshot
    verification: VerificationHandoffEvidence
    review: ReviewHandoffEvidence
    patch: FinalPatchReference
    reason: str = _PASSING_HANDOFF_MESSAGE
    _seal: InitVar[object] = field(default=None, kw_only=True)

    def __post_init__(self, _seal: object) -> None:
        _require_trusted_outcome(_seal)
        if not self.initial_workspace.inspection_complete:
            raise ValueError("Accepted handoff requires a complete initial inspection.")
        if not self.initial_workspace.matches(self.final_workspace):
            raise ValueError("Accepted handoff requires an unchanged final workspace.")
        if self.review.verdict is not ReviewVerdict.PASS:
            raise ValueError("Accepted handoff requires passing review evidence.")
        _validate_reason(self.reason)


@final
@dataclass(frozen=True)
class HandoffRejected:
    stop_category: StopCategory
    reason: str
    initial_workspace: WorkspaceSnapshot | None = None
    final_workspace: WorkspaceSnapshot | None = None
    _seal: InitVar[object] = field(default=None, kw_only=True)

    def __post_init__(self, _seal: object) -> None:
        _require_trusted_outcome(_seal)
        if self.stop_category is not StopCategory.SAFETY_VIOLATION:
            raise ValueError("Handoff rejection requires the stable safety category.")
        _validate_reason(self.reason)


HandoffAcceptanceResult: TypeAlias = HandoffAccepted | HandoffRejected


class HandoffAcceptanceService:
    """Evaluate final evidence, capture the patch, and return one typed decision."""

    def __init__(
        self,
        *,
        patch_capture: FinalPatchCapture,
    ) -> None:
        self._patch_capture = patch_capture

    def accept(self, request: HandoffAcceptanceRequest) -> HandoffAcceptanceResult:
        run_record = request.run_record
        request_problem = _acceptance_request_problem(request)
        if request_problem is not None:
            return HandoffRejected(
                stop_category=StopCategory.SAFETY_VIOLATION,
                reason=request_problem,
                _seal=_HANDOFF_OUTCOME_SEAL,
            )
        repository = GitRepository(Path(run_record.target_repository_path))
        try:
            initial_workspace = WorkspaceSnapshot.capture(repository)
        except (GitCommandError, OSError, RuntimeError, TypeError, ValueError) as error:
            return HandoffRejected(
                stop_category=StopCategory.SAFETY_VIOLATION,
                reason=(
                    "Could not inspect the workspace before human handoff: "
                    f"{type(error).__name__}: {error}"
                ),
                _seal=_HANDOFF_OUTCOME_SEAL,
            )

        try:
            if request.attempt is None:
                # Preserve the established direct-service call shape. Lifecycle
                # orchestration always supplies the exact controller-owned attempt.
                reporting_attempt = _load_current_reporting_attempt(request.run_dir)
                update_attempt(
                    reporting_attempt,
                    before_workspace_fingerprint=initial_workspace.fingerprint,
                )
            else:
                reporting_attempt = require_stage_attempt(
                    request.run_dir,
                    request.attempt,
                    phase=AttemptPhase.REPORTING,
                )
                if (
                    reporting_attempt.before_workspace_fingerprint
                    != initial_workspace.fingerprint
                ):
                    raise AttemptError(
                        "Workspace changed after the controller started reporting."
                    )
        except (AttemptError, OSError, RuntimeError, TypeError, ValueError) as error:
            return HandoffRejected(
                stop_category=StopCategory.SAFETY_VIOLATION,
                reason=(
                    "Could not persist the initial handoff workspace evidence: "
                    f"{type(error).__name__}: {error}"
                ),
                initial_workspace=initial_workspace,
                final_workspace=initial_workspace,
                _seal=_HANDOFF_OUTCOME_SEAL,
            )

        policy_request = _read_policy_request(
            request,
            workspace=initial_workspace,
        )
        problem = evaluate_handoff_policy(policy_request)
        if problem is not None:
            return HandoffRejected(
                stop_category=problem.stop_category,
                reason=problem.reason,
                initial_workspace=initial_workspace,
                final_workspace=initial_workspace,
                _seal=_HANDOFF_OUTCOME_SEAL,
            )

        assert policy_request.verification is not None
        assert policy_request.review is not None
        try:
            patch = self._patch_capture.capture(
                FinalPatchCaptureRequest(
                    repository_path=repository.path,
                    baseline_sha=run_record.baseline_sha,
                    destination=request.run_dir / FINAL_PATCH_FILE,
                )
            )
            _validate_captured_patch(
                patch,
                expected_destination=request.run_dir / FINAL_PATCH_FILE,
            )
            final_workspace = WorkspaceSnapshot.capture(repository)
        except (GitCommandError, OSError, RuntimeError, TypeError, ValueError) as error:
            return HandoffRejected(
                stop_category=StopCategory.SAFETY_VIOLATION,
                reason=(
                    "Could not capture the final handoff patch safely: "
                    f"{type(error).__name__}: {error}"
                ),
                initial_workspace=initial_workspace,
                final_workspace=None,
                _seal=_HANDOFF_OUTCOME_SEAL,
            )
        if not initial_workspace.matches(final_workspace):
            return HandoffRejected(
                stop_category=StopCategory.SAFETY_VIOLATION,
                reason=(
                    "Workspace changed while the final handoff patch was being "
                    "captured."
                ),
                initial_workspace=initial_workspace,
                final_workspace=final_workspace,
                _seal=_HANDOFF_OUTCOME_SEAL,
            )
        return HandoffAccepted(
            initial_workspace=initial_workspace,
            final_workspace=final_workspace,
            verification=policy_request.verification,
            review=policy_request.review,
            patch=patch,
            _seal=_HANDOFF_OUTCOME_SEAL,
        )


def evaluate_handoff_policy(
    request: HandoffPolicyRequest,
) -> HandoffPolicyProblem | None:
    """Apply the pure, ordered final handoff policy to trusted inputs."""

    reject = _policy_rejection
    if request.attempt_problem is not None:
        return reject(request.attempt_problem)
    workspace = request.workspace
    if workspace is None or not workspace.inspection_complete:
        return reject(
            "Repository safety invariants were violated before human handoff."
        )
    if workspace.repository_path != request.repository_path.resolve(strict=False):
        return reject("Repository path changed before human handoff.")
    if workspace.branch != request.expected_branch:
        return reject("Repository branch changed before human handoff.")
    if workspace.head_sha != request.expected_head_sha:
        return reject("Repository HEAD changed before human handoff.")
    if workspace.staged_paths:
        return reject("Repository has staged changes before human handoff.")
    if request.current_correction_round > request.max_correction_rounds:
        return reject(
            "Correction-round evidence is contradictory before human handoff."
        )

    reporting = _latest_attempt(request.attempts, AttemptPhase.REPORTING)
    if reporting is None or reporting.status is not AttemptStatus.STARTED:
        return reject("Final handoff attempt is missing or incomplete.")
    if reporting.sequence != max(item.sequence for item in request.attempts):
        return reject("Final handoff attempt is not the current attempt.")

    writable_attempts = tuple(
        item
        for item in request.attempts
        if PHASE_DEFINITIONS[item.phase].writes_target_repository
    )
    if not writable_attempts:
        return reject("No completed writable attempt has a workspace fingerprint.")
    if any(item.status is not AttemptStatus.COMPLETED for item in writable_attempts):
        return reject("Writable attempt evidence is incomplete before human handoff.")
    writable = writable_attempts[-1]
    if (
        writable.status is not AttemptStatus.COMPLETED
        or writable.after_workspace_fingerprint is None
    ):
        return reject("No completed writable attempt has a workspace fingerprint.")
    if not workspace.matches_fingerprint(writable.after_workspace_fingerprint):
        return reject(
            "Current workspace no longer matches the completed writable attempt."
        )

    if any(
        item.status in {AttemptStatus.FAILED, AttemptStatus.HUMAN_REQUIRED}
        for item in request.attempts
        if not PHASE_DEFINITIONS[item.phase].writes_target_repository
    ):
        return reject(
            "Read-only attempt evidence contains a terminal outcome before human "
            "handoff."
        )

    correction_attempts = tuple(
        item for item in request.attempts if item.phase is AttemptPhase.CORRECTING
    )
    if len(correction_attempts) != request.current_correction_round:
        return reject(
            "Correction-round evidence is contradictory before human handoff."
        )
    if request.verification_problem is not None:
        return reject(request.verification_problem)
    verification = request.verification
    if verification is None:
        return reject(
            "Deterministic verification evidence is not passing: missing evidence."
        )
    verification_attempt = _latest_attempt(
        request.attempts,
        AttemptPhase.VERIFYING,
    )
    if verification_attempt is None:
        return reject(
            "Deterministic verification evidence is not passing: missing evidence."
        )
    if verification_attempt.status is not AttemptStatus.COMPLETED:
        return reject("Deterministic verification evidence is incomplete.")
    if verification_attempt.sequence != verification.attempt_sequence:
        return reject(
            "Deterministic verification evidence is stale for the final source."
        )
    if (
        verification_attempt.before_workspace_fingerprint
        != verification.before_workspace_fingerprint
        or verification_attempt.after_workspace_fingerprint
        != verification.after_workspace_fingerprint
    ):
        return reject("Deterministic verification evidence is contradictory.")
    if verification.attempt_sequence <= writable.sequence:
        return reject(
            "Deterministic verification evidence is stale for the final source."
        )
    if not workspace.matches_fingerprint(
        verification.before_workspace_fingerprint
    ) or not workspace.matches_fingerprint(verification.after_workspace_fingerprint):
        return reject(
            "Deterministic verification evidence is stale for the final source."
        )

    if request.review_problem is not None:
        return reject(request.review_problem)
    review = request.review
    if review is None:
        return reject("Final independent review evidence is missing.")
    review_attempt = _latest_attempt(request.attempts, AttemptPhase.REVIEWING)
    if review_attempt is None:
        return reject("Final independent review evidence is missing.")
    if review_attempt.status is not AttemptStatus.COMPLETED:
        return reject("Final independent review evidence is incomplete.")
    if review_attempt.sequence != review.attempt_sequence:
        return reject("Final independent review evidence is stale.")
    if (
        review_attempt.before_workspace_fingerprint
        != review.before_workspace_fingerprint
        or review_attempt.after_workspace_fingerprint
        != review.after_workspace_fingerprint
    ):
        return reject("Final independent review evidence is contradictory.")
    if review.verdict is not ReviewVerdict.PASS:
        return reject("Final independent review did not pass.")
    if review.attempt_sequence <= verification.attempt_sequence:
        return reject("Final independent review evidence is stale.")
    if review.attempt_sequence >= reporting.sequence:
        return reject("Final independent review evidence is contradictory.")
    if (
        review.before_workspace_fingerprint != verification.after_workspace_fingerprint
        or review.after_workspace_fingerprint
        != verification.after_workspace_fingerprint
    ):
        return reject(
            "Final independent review did not pass for the verified final source."
        )
    review_attempts = tuple(
        item
        for item in request.attempts
        if item.phase is AttemptPhase.REVIEWING
        and item.status is AttemptStatus.COMPLETED
    )
    if (
        request.current_review_round < 1
        or len(review_attempts) < request.current_review_round
    ):
        return reject("Review-round evidence is contradictory before human handoff.")

    latest_non_reporting = next(
        (
            item
            for item in reversed(request.attempts)
            if item.phase is not AttemptPhase.REPORTING
        ),
        None,
    )
    if (
        latest_non_reporting is None
        or latest_non_reporting.sequence != review.attempt_sequence
    ):
        return reject("Attempt completion evidence is stale or contradictory.")
    return None


def _read_policy_request(
    request: HandoffAcceptanceRequest,
    *,
    workspace: WorkspaceSnapshot,
) -> HandoffPolicyRequest:
    run_record = request.run_record
    attempt_problem: str | None = None
    verification_problem: str | None = None
    review_problem: str | None = None
    verification: VerificationHandoffEvidence | None = None
    review: ReviewHandoffEvidence | None = None
    try:
        attempt_records = load_attempt_records(request.run_dir)
        attempts = tuple(_handoff_attempt_evidence(item) for item in attempt_records)
        attempt_problem = _agent_execution_evidence_problem(
            request.run_dir,
            run_record,
            attempt_records,
        )
    except (AttemptError, OSError, TypeError, ValueError) as error:
        attempt_records = ()
        attempts = ()
        attempt_problem = f"Attempt evidence is invalid before human handoff: {error}"
    if attempt_problem is None:
        verification_attempt = _latest_attempt(attempts, AttemptPhase.VERIFYING)
        if verification_attempt is None:
            verification_problem = (
                "Deterministic verification evidence is not passing: missing evidence."
            )
        elif verification_attempt.status is not AttemptStatus.COMPLETED:
            verification_problem = "Deterministic verification evidence is incomplete."
        else:
            try:
                source_fingerprint = read_verification_source_fingerprint(
                    request.run_dir,
                    run_record,
                    expected_statuses=frozenset({VerificationStatus.PASS}),
                    verification_commands=(
                        run_record.resolved_policy.verification_commands
                    ),
                    require_all_commands_pass=True,
                    require_authoritative_pass=True,
                    expected_command_cwd=Path(
                        run_record.target_repository_path
                    ).resolve(strict=False),
                    expected_attempt_sequence=verification_attempt.sequence,
                    expected_round_index=run_record.current_correction_round,
                )
                verification = VerificationHandoffEvidence(
                    attempt_sequence=verification_attempt.sequence,
                    before_workspace_fingerprint=(
                        verification_attempt.before_workspace_fingerprint
                    ),
                    after_workspace_fingerprint=source_fingerprint,
                )
            except (TypeError, ValueError) as error:
                verification_problem = (
                    "Deterministic verification evidence is not passing: " + str(error)
                )
    if attempt_problem is None:
        review_attempt = _latest_attempt(attempts, AttemptPhase.REVIEWING)
        if review_attempt is None:
            review_problem = "Final independent review evidence is missing."
        elif review_attempt.status is not AttemptStatus.COMPLETED:
            review_problem = "Final independent review evidence is incomplete."
        else:
            try:
                result = read_review_result(
                    request.run_dir,
                    next(
                        item
                        for item in attempt_records
                        if item.sequence == review_attempt.sequence
                    ),
                )
                if result is None:
                    raise PersistenceCodecError(
                        "Final independent review result artifact is missing."
                    )
                review = ReviewHandoffEvidence(
                    attempt_sequence=review_attempt.sequence,
                    before_workspace_fingerprint=(
                        review_attempt.before_workspace_fingerprint
                    ),
                    after_workspace_fingerprint=(
                        review_attempt.after_workspace_fingerprint
                    ),
                    verdict=result.verdict,
                )
            except (
                AttemptError,
                OSError,
                PersistenceCodecError,
                ResultValidationError,
                TypeError,
                ValueError,
            ) as error:
                review_problem = "Final independent review evidence is invalid: " + str(
                    error
                )
    return HandoffPolicyRequest(
        repository_path=Path(run_record.target_repository_path).resolve(strict=False),
        expected_branch=run_record.starting_branch,
        expected_head_sha=run_record.baseline_sha,
        current_correction_round=run_record.current_correction_round,
        max_correction_rounds=run_record.max_correction_rounds,
        current_review_round=run_record.current_review_round,
        workspace=workspace,
        attempts=attempts,
        verification=verification,
        review=review,
        attempt_problem=attempt_problem,
        verification_problem=verification_problem,
        review_problem=review_problem,
    )


def _agent_execution_evidence_problem(
    run_dir: Path,
    run_record: RunRecord,
    attempts: tuple[AttemptRecord, ...],
) -> str | None:
    """Require neutral, policy-bound evidence for every completed agent task."""

    for attempt in attempts:
        task_kind = _AGENT_TASK_BY_PHASE.get(attempt.phase)
        if task_kind is None or attempt.status is not AttemptStatus.COMPLETED:
            continue
        try:
            if attempt.execution_path != EXECUTION_EVIDENCE_FILE:
                raise ValueError(
                    "attempt does not reference neutral execution evidence"
                )
            layout = attempt_artifact_layout(run_dir, attempt)
            evidence = read_execution_evidence(layout)
            expected = run_record.resolved_policy.task_policy(task_kind)
            if evidence.provider_id != expected.provider_id:
                raise ValueError("provider identity does not match resolved policy")
            if evidence.task_kind is not task_kind:
                raise ValueError("task kind does not match the attempt phase")
            if evidence.repository_access is not expected.repository_access:
                raise ValueError("repository access does not match resolved policy")
            if evidence.required_capabilities != expected.required_capabilities:
                raise ValueError("capabilities do not match resolved policy")
            if (
                evidence.status is not AgentExecutionStatus.SUCCESS
                or evidence.invocation_start is not InvocationStart.STARTED
                or not attempt.process_started
            ):
                raise ValueError("completed agent attempt is not a started success")
            if evidence.typed_result_artifact is None:
                raise ValueError("typed result artifact reference is missing")
            layout.resolve(evidence.typed_result_artifact, require_exists=True)
            if task_kind is AgentTaskKind.REVIEW:
                stage_result = read_review_result(run_dir, attempt)
            else:
                stage_result = read_implementation_result(run_dir, attempt)
            if stage_result is None or stage_result != evidence.typed_result:
                raise ValueError(
                    "typed stage result does not match neutral execution evidence"
                )
        except (AttemptError, OSError, RuntimeError, TypeError, ValueError) as error:
            return (
                f"Agent execution evidence is invalid for attempt "
                f"{attempt.sequence}: {error}"
            )
    return None


def _latest_attempt(
    attempts: tuple[HandoffAttemptEvidence, ...],
    phase: AttemptPhase,
) -> HandoffAttemptEvidence | None:
    return next(
        (item for item in reversed(attempts) if item.phase is phase),
        None,
    )


def _handoff_attempt_evidence(record: AttemptRecord) -> HandoffAttemptEvidence:
    return HandoffAttemptEvidence(
        sequence=record.sequence,
        phase=record.phase,
        status=record.status,
        before_workspace_fingerprint=record.before_workspace_fingerprint,
        after_workspace_fingerprint=record.after_workspace_fingerprint,
    )


def _acceptance_request_problem(request: HandoffAcceptanceRequest) -> str | None:
    if request.run_record.state is not WorkflowState.REPORTING:
        return "Final handoff can only be evaluated while the run is reporting."
    if request.run_dir.name != request.run_record.run_id:
        return "Final handoff run directory does not belong to the run record."
    return None


def _load_current_reporting_attempt(run_dir: Path) -> AttemptRecord:
    attempts = load_attempt_records(run_dir)
    reporting_attempt = next(
        (item for item in reversed(attempts) if item.phase is AttemptPhase.REPORTING),
        None,
    )
    if reporting_attempt is None:
        raise AttemptError("Final handoff requires a reporting attempt.")
    if reporting_attempt.status is not AttemptStatus.STARTED:
        raise AttemptError("Final handoff requires a started reporting attempt.")
    if reporting_attempt.sequence != max(item.sequence for item in attempts):
        raise AttemptError("Final handoff reporting attempt is not current.")
    return reporting_attempt


def _validate_captured_patch(
    patch: FinalPatchReference,
    *,
    expected_destination: Path,
) -> None:
    if not isinstance(patch, FinalPatchReference):
        raise TypeError("Patch capture returned an invalid reference type.")
    expected = expected_destination.resolve(strict=False)
    if patch.path.resolve(strict=False) != expected:
        raise ValueError("Patch capture returned an unexpected artifact path.")
    data = expected.read_bytes()
    if len(data) != patch.size_bytes:
        raise ValueError("Captured patch size does not match its reference.")
    if hashlib.sha256(data).hexdigest() != patch.sha256:
        raise ValueError("Captured patch digest does not match its reference.")


def _require_trusted_outcome(seal: object) -> None:
    if seal is not _HANDOFF_OUTCOME_SEAL:
        raise TypeError(
            "Handoff outcomes can only be constructed by HandoffAcceptanceService."
        )


def _validate_attempt_sequence(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("attempt_sequence must be a positive integer.")


def _validate_fingerprint(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")


def _validate_optional_fingerprint(value: str | None, *, field_name: str) -> None:
    if value is not None:
        _validate_fingerprint(value, field_name=field_name)


def _validate_optional_text(value: str | None, *, field_name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{field_name} must be a non-empty string or None.")


def _validate_reason(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("handoff reason must be a non-empty string.")


def _policy_rejection(reason: str) -> HandoffPolicyProblem:
    return HandoffPolicyProblem(
        stop_category=StopCategory.SAFETY_VIOLATION,
        reason=reason,
    )


__all__ = [
    "HandoffAcceptanceRequest",
    "HandoffAcceptanceResult",
    "HandoffAcceptanceService",
    "HandoffAccepted",
    "HandoffAttemptEvidence",
    "HandoffPolicyProblem",
    "HandoffPolicyRequest",
    "HandoffRejected",
    "ReviewHandoffEvidence",
    "VerificationHandoffEvidence",
    "evaluate_handoff_policy",
]
