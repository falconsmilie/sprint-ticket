from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path

import pytest

from tests.helpers import GIT, create_git_repo, run_git
from ticket_automation.application.handoff_acceptance import (
    HandoffAccepted,
    HandoffAttemptEvidence,
    HandoffPolicyRequest,
    HandoffRejected,
    ReviewHandoffEvidence,
    VerificationHandoffEvidence,
    evaluate_handoff_policy,
)
from ticket_automation.application.ports.handoff import (
    FinalPatchCaptureRequest,
    FinalPatchReference,
)
from ticket_automation.domain.task_results import ReviewVerdict
from ticket_automation.git_safety import WorkspaceSnapshot
from ticket_automation.infrastructure.final_patch import FileSystemFinalPatchCapture
from ticket_automation.models import (
    AttemptPhase,
    AttemptStatus,
    StopCategory,
)

_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_HEAD = "a" * 40
_OTHER_FINGERPRINT = "b" * 64


def _workspace(**changes) -> WorkspaceSnapshot:
    values = {
        "repository_path": Path("repository").resolve(),
        "branch": "main",
        "head_sha": _HEAD,
        "staged_paths": (),
        "staged_diff_sha256": _EMPTY_SHA256,
        "tracked_diff_sha256": _EMPTY_SHA256,
        "untracked_paths": (),
        "untracked_file_hashes": (),
        "environment_roots": (),
        "inspection_complete": True,
    }
    values.update(changes)
    return WorkspaceSnapshot(**values)


def _attempt(
    sequence: int,
    phase: AttemptPhase,
    status: AttemptStatus,
    *,
    before: str | None,
    after: str | None,
) -> HandoffAttemptEvidence:
    return HandoffAttemptEvidence(
        sequence=sequence,
        phase=phase,
        status=status,
        before_workspace_fingerprint=before,
        after_workspace_fingerprint=after,
    )


def _passing_policy() -> HandoffPolicyRequest:
    workspace = _workspace()
    fingerprint = workspace.fingerprint
    attempts = (
        _attempt(
            1,
            AttemptPhase.IMPLEMENTING,
            AttemptStatus.COMPLETED,
            before=None,
            after=fingerprint,
        ),
        _attempt(
            2,
            AttemptPhase.VERIFYING,
            AttemptStatus.COMPLETED,
            before=fingerprint,
            after=fingerprint,
        ),
        _attempt(
            3,
            AttemptPhase.REVIEWING,
            AttemptStatus.COMPLETED,
            before=fingerprint,
            after=fingerprint,
        ),
        _attempt(
            4,
            AttemptPhase.REPORTING,
            AttemptStatus.STARTED,
            before=None,
            after=None,
        ),
    )
    return HandoffPolicyRequest(
        repository_path=workspace.repository_path,
        expected_branch="main",
        expected_head_sha=_HEAD,
        current_correction_round=0,
        max_correction_rounds=2,
        current_review_round=1,
        workspace=workspace,
        attempts=attempts,
        verification=VerificationHandoffEvidence(2, fingerprint, fingerprint),
        review=ReviewHandoffEvidence(
            3,
            fingerprint,
            fingerprint,
            ReviewVerdict.PASS,
        ),
    )


def test_complete_current_handoff_policy_is_accepted():
    assert evaluate_handoff_policy(_passing_policy()) is None


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (
            lambda request: replace(request, attempt_problem="invalid ledger"),
            "invalid ledger",
        ),
        (
            lambda request: replace(
                request,
                workspace=replace(request.workspace, inspection_complete=False),
            ),
            "Repository safety invariants",
        ),
        (
            lambda request: replace(
                request,
                workspace=replace(request.workspace, repository_path=Path("other")),
            ),
            "Repository path changed",
        ),
        (
            lambda request: replace(
                request,
                workspace=replace(request.workspace, branch="other"),
            ),
            "Repository branch changed",
        ),
        (
            lambda request: replace(
                request,
                workspace=replace(request.workspace, head_sha="c" * 40),
            ),
            "Repository HEAD changed",
        ),
        (
            lambda request: replace(
                request,
                workspace=replace(request.workspace, staged_paths=("file.txt",)),
            ),
            "staged changes",
        ),
        (
            lambda request: replace(request, current_correction_round=3),
            "Correction-round evidence is contradictory",
        ),
        (
            lambda request: replace(request, attempts=request.attempts[:-1]),
            "Final handoff attempt is missing",
        ),
        (
            lambda request: replace(
                request,
                attempts=(
                    *request.attempts[:-1],
                    _attempt(
                        4,
                        AttemptPhase.REPORTING,
                        AttemptStatus.COMPLETED,
                        before=request.workspace.fingerprint,
                        after=request.workspace.fingerprint,
                    ),
                ),
            ),
            "Final handoff attempt is missing or incomplete",
        ),
        (
            lambda request: replace(
                request,
                attempts=request.attempts[1:],
            ),
            "No completed writable attempt",
        ),
        (
            lambda request: replace(
                request,
                attempts=(
                    *request.attempts,
                    _attempt(
                        5,
                        AttemptPhase.PREPARING,
                        AttemptStatus.STARTED,
                        before=request.workspace.fingerprint,
                        after=None,
                    ),
                ),
            ),
            "Final handoff attempt is not the current attempt",
        ),
        (
            lambda request: replace(
                request,
                attempts=(
                    replace(
                        request.attempts[0],
                        after_workspace_fingerprint=_OTHER_FINGERPRINT,
                    ),
                    *request.attempts[1:],
                ),
            ),
            "completed writable attempt",
        ),
        (
            lambda request: replace(request, verification=None),
            "verification evidence is not passing",
        ),
        (
            lambda request: _replace_verification_fingerprint(
                request,
                before=False,
            ),
            "verification evidence is stale",
        ),
        (
            lambda request: _replace_verification_fingerprint(
                request,
                before=True,
            ),
            "verification evidence is stale",
        ),
        (
            lambda request: replace(
                request,
                verification=replace(request.verification, attempt_sequence=1),
            ),
            "verification evidence is stale",
        ),
        (
            lambda request: replace(request, review=None),
            "review evidence is missing",
        ),
        (
            lambda request: replace(
                request,
                review=replace(
                    request.review,
                    verdict=ReviewVerdict.HUMAN_REVIEW_REQUIRED,
                ),
            ),
            "review did not pass",
        ),
        (
            lambda request: _replace_review_sequence(request, sequence=2),
            "review evidence is stale",
        ),
        (
            lambda request: _replace_review_sequence(request, sequence=4),
            "review evidence is contradictory",
        ),
        (
            lambda request: _replace_review_fingerprint(
                request,
                before=True,
            ),
            "review did not pass for the verified final source",
        ),
        (
            lambda request: _replace_review_fingerprint(
                request,
                before=False,
            ),
            "review did not pass for the verified final source",
        ),
    ],
    ids=[
        "invalid-attempt-evidence",
        "incomplete-inspection",
        "repository-path",
        "branch",
        "head",
        "staging",
        "correction-rounds",
        "missing-reporting-attempt",
        "completed-reporting-attempt",
        "missing-writable-attempt",
        "reporting-attempt-not-current",
        "workspace-drift",
        "missing-verification",
        "stale-verification-after-fingerprint",
        "stale-verification-before-fingerprint",
        "verification-before-final-write",
        "missing-review",
        "failed-review",
        "stale-review-order",
        "review-after-reporting",
        "stale-review-before-fingerprint",
        "stale-review-after-fingerprint",
    ],
)
def test_handoff_policy_rules_reject_independently(change, reason):
    problem = evaluate_handoff_policy(change(_passing_policy()))

    assert problem is not None
    assert problem.stop_category is StopCategory.SAFETY_VIOLATION
    assert reason in problem.reason


def _replace_verification_fingerprint(
    request: HandoffPolicyRequest,
    *,
    before: bool,
) -> HandoffPolicyRequest:
    field = "before_workspace_fingerprint" if before else "after_workspace_fingerprint"
    attempt = replace(request.attempts[1], **{field: _OTHER_FINGERPRINT})
    evidence = replace(request.verification, **{field: _OTHER_FINGERPRINT})
    return replace(
        request,
        attempts=(request.attempts[0], attempt, *request.attempts[2:]),
        verification=evidence,
    )


def _replace_review_fingerprint(
    request: HandoffPolicyRequest,
    *,
    before: bool,
) -> HandoffPolicyRequest:
    field = "before_workspace_fingerprint" if before else "after_workspace_fingerprint"
    attempt = replace(request.attempts[2], **{field: _OTHER_FINGERPRINT})
    evidence = replace(request.review, **{field: _OTHER_FINGERPRINT})
    return replace(
        request,
        attempts=(*request.attempts[:2], attempt, request.attempts[3]),
        review=evidence,
    )


def _replace_review_sequence(
    request: HandoffPolicyRequest,
    *,
    sequence: int,
) -> HandoffPolicyRequest:
    attempt = replace(request.attempts[2], sequence=sequence)
    return replace(
        request,
        attempts=(*request.attempts[:2], attempt, request.attempts[3]),
        review=replace(request.review, attempt_sequence=sequence),
    )


def test_handoff_rejects_incomplete_historical_correction_evidence():
    request = _passing_policy()
    fingerprint = request.workspace.fingerprint
    incomplete_correction = _attempt(
        2,
        AttemptPhase.CORRECTING,
        AttemptStatus.HUMAN_REQUIRED,
        before=fingerprint,
        after=fingerprint,
    )
    completed_correction = _attempt(
        3,
        AttemptPhase.CORRECTING,
        AttemptStatus.COMPLETED,
        before=fingerprint,
        after=fingerprint,
    )
    verification_attempt = replace(
        request.attempts[1],
        sequence=4,
    )
    review_attempt = replace(
        request.attempts[2],
        sequence=5,
    )
    reporting = replace(
        request.attempts[-1],
        sequence=6,
    )

    problem = evaluate_handoff_policy(
        replace(
            request,
            attempts=(
                request.attempts[0],
                incomplete_correction,
                completed_correction,
                verification_attempt,
                review_attempt,
                reporting,
            ),
            current_correction_round=2,
            verification=replace(request.verification, attempt_sequence=4),
            review=replace(request.review, attempt_sequence=5),
        )
    )

    assert problem is not None
    assert "Writable attempt evidence is incomplete" in problem.reason


def test_handoff_rejects_incomplete_historical_implementation_evidence():
    request = _passing_policy()
    fingerprint = request.workspace.fingerprint
    incomplete_implementation = replace(
        request.attempts[0],
        status=AttemptStatus.HUMAN_REQUIRED,
    )
    completed_correction = _attempt(
        2,
        AttemptPhase.CORRECTING,
        AttemptStatus.COMPLETED,
        before=fingerprint,
        after=fingerprint,
    )
    verification_attempt = replace(request.attempts[1], sequence=3)
    review_attempt = replace(request.attempts[2], sequence=4)
    reporting = replace(request.attempts[-1], sequence=5)

    problem = evaluate_handoff_policy(
        replace(
            request,
            attempts=(
                incomplete_implementation,
                completed_correction,
                verification_attempt,
                review_attempt,
                reporting,
            ),
            current_correction_round=1,
            verification=replace(request.verification, attempt_sequence=3),
            review=replace(request.review, attempt_sequence=4),
        )
    )

    assert problem is not None
    assert "Writable attempt evidence is incomplete" in problem.reason


def test_handoff_rejects_a_newer_incomplete_verification_attempt():
    request = _passing_policy()
    fingerprint = request.workspace.fingerprint
    incomplete_verification = _attempt(
        3,
        AttemptPhase.VERIFYING,
        AttemptStatus.STARTED,
        before=fingerprint,
        after=None,
    )
    review_attempt = replace(request.attempts[2], sequence=4)
    reporting = replace(request.attempts[-1], sequence=5)

    problem = evaluate_handoff_policy(
        replace(
            request,
            attempts=(
                *request.attempts[:2],
                incomplete_verification,
                review_attempt,
                reporting,
            ),
            review=replace(request.review, attempt_sequence=4),
        )
    )

    assert problem is not None
    assert "verification evidence is incomplete" in problem.reason


@pytest.mark.parametrize(
    ("phase", "status"),
    [
        (AttemptPhase.VERIFYING, AttemptStatus.FAILED),
        (AttemptPhase.REVIEWING, AttemptStatus.HUMAN_REQUIRED),
    ],
    ids=["failed-verification", "human-required-review"],
)
def test_handoff_rejects_superseded_terminal_read_only_attempt(phase, status):
    request = _passing_policy()
    fingerprint = request.workspace.fingerprint
    terminal_attempt = _attempt(
        2,
        phase,
        status,
        before=fingerprint,
        after=fingerprint,
    )
    verification_attempt = replace(request.attempts[1], sequence=3)
    review_attempt = replace(request.attempts[2], sequence=4)
    reporting = replace(request.attempts[-1], sequence=5)

    problem = evaluate_handoff_policy(
        replace(
            request,
            attempts=(
                request.attempts[0],
                terminal_attempt,
                verification_attempt,
                review_attempt,
                reporting,
            ),
            verification=replace(request.verification, attempt_sequence=3),
            review=replace(request.review, attempt_sequence=4),
        )
    )

    assert problem is not None
    assert "terminal outcome" in problem.reason


def test_handoff_accepts_completed_retry_of_the_same_review_round():
    request = _passing_policy()
    fingerprint = request.workspace.fingerprint
    retried_review = replace(request.attempts[2], sequence=4)
    reporting = replace(request.attempts[-1], sequence=5)

    problem = evaluate_handoff_policy(
        replace(
            request,
            attempts=(
                *request.attempts[:3],
                retried_review,
                reporting,
            ),
            review=ReviewHandoffEvidence(
                4,
                fingerprint,
                fingerprint,
                ReviewVerdict.PASS,
            ),
        )
    )

    assert problem is None


def test_handoff_rejects_evidence_when_review_is_not_the_final_completed_stage():
    request = _passing_policy()
    fingerprint = request.workspace.fingerprint
    later_verification = _attempt(
        4,
        AttemptPhase.VERIFYING,
        AttemptStatus.COMPLETED,
        before=fingerprint,
        after=fingerprint,
    )
    reporting = replace(
        request.attempts[-1],
        sequence=5,
    )

    problem = evaluate_handoff_policy(
        replace(
            request, attempts=(*request.attempts[:-1], later_verification, reporting)
        )
    )

    assert problem is not None
    assert "verification evidence is stale" in problem.reason


@pytest.mark.parametrize(
    "change",
    [
        lambda request: replace(request, current_correction_round=1),
        lambda request: replace(request, current_review_round=0),
    ],
    ids=["correction-counter", "review-counter"],
)
def test_handoff_rejects_attempt_counter_disagreement(change):
    problem = evaluate_handoff_policy(change(_passing_policy()))

    assert problem is not None
    assert "round evidence is contradictory" in problem.reason


def test_handoff_outcomes_require_trusted_service_construction():
    with pytest.raises(TypeError, match="only be constructed"):
        HandoffRejected(StopCategory.SAFETY_VIOLATION, "rejected")
    with pytest.raises(TypeError, match="only be constructed"):
        HandoffAccepted(None, None, None, None, None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "construct",
    [
        lambda: VerificationHandoffEvidence(1, "invalid", "0" * 64),
        lambda: HandoffAttemptEvidence(
            1,
            AttemptPhase.VERIFYING,
            AttemptStatus.COMPLETED,
            "",
            "0" * 64,
        ),
        lambda: replace(_passing_policy(), attempts=list(_passing_policy().attempts)),
        lambda: replace(_passing_policy(), verification=object()),
        lambda: replace(_passing_policy(), review=object()),
    ],
    ids=[
        "verification-fingerprint",
        "attempt-fingerprint",
        "mutable-attempts",
        "verification-type",
        "review-type",
    ],
)
def test_handoff_inputs_reject_untrusted_construction(construct):
    with pytest.raises((TypeError, ValueError)):
        construct()


@pytest.mark.parametrize(
    "reference",
    [
        lambda: FinalPatchReference(Path("final.patch"), "invalid", 0),
        lambda: FinalPatchReference(Path("final.patch"), "0" * 64, -1),
    ],
    ids=["invalid-digest", "negative-size"],
)
def test_patch_references_reject_invalid_evidence(reference):
    with pytest.raises(ValueError):
        reference()


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_final_patch_capture_replaces_a_hardlink_without_mutating_its_target(
    tmp_path: Path,
) -> None:
    repository = create_git_repo(tmp_path / "repository")
    baseline_sha = run_git(repository, "rev-parse", "HEAD")
    repository.joinpath("file.txt").write_text("changed\n", encoding="utf-8")
    external = tmp_path / "external.patch"
    external.write_text("external evidence", encoding="utf-8")
    destination = tmp_path / "run" / "final.patch"
    destination.parent.mkdir()
    os.link(external, destination)

    reference = FileSystemFinalPatchCapture().capture(
        FinalPatchCaptureRequest(
            repository_path=repository,
            baseline_sha=baseline_sha,
            destination=destination,
        )
    )

    assert external.read_text(encoding="utf-8") == "external evidence"
    assert destination.read_text(encoding="utf-8").startswith("diff --git")
    assert not os.path.samefile(external, destination)
    assert reference.path == destination
