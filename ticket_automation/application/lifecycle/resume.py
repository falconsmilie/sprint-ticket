"""Fail-closed resume policy based on typed phase and attempt evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ...attempts import (
    AttemptError,
    latest_attempt,
    latest_writable_attempt,
    load_attempt_records,
)
from ...config import AppConfig
from ...git import GitCommandError, GitRepository
from ...git_safety import WorkspaceSnapshot
from ...models import (
    PHASE_DEFINITIONS,
    AttemptStatus,
    WorkflowState,
    phase_for_active_state,
)
from ...run_ownership import RunOwnership
from ...runs import (
    BASELINE_RECORD_FILE,
    RUN_TICKET_FILE,
    RunError,
    RunRecord,
    load_baseline_record,
    load_baseline_record_for_owner,
)
from ...verification_evidence import (
    baseline_verification_evidence_problem,
    verification_commands_fingerprint,
)


def resume_preflight_problem(
    config: AppConfig,
    run_dir: Path,
    run_record: RunRecord,
    *,
    run_ownership: RunOwnership | None = None,
) -> str | None:
    if run_ownership is not None:
        run_dir = run_ownership.validate_run_path(run_dir)
    try:
        load_attempt_records(run_dir, run_ownership=run_ownership)
    except AttemptError as error:
        return f"Attempt evidence is invalid: {error}"
    baseline_path = run_dir / BASELINE_RECORD_FILE
    ticket_path = run_dir / RUN_TICKET_FILE
    if run_ownership is not None:
        run_ownership.validate_descendant(baseline_path)
        run_ownership.validate_descendant(ticket_path)
    baseline_exists = (
        baseline_path.is_file()
        if run_ownership is None
        else run_ownership.read_descendant(
            baseline_path,
            lambda source: source.is_file(),
        )
    )
    if not baseline_exists:
        return f"Required baseline artifact is missing: {baseline_path}"
    ticket_exists = (
        ticket_path.is_file()
        if run_ownership is None
        else run_ownership.read_descendant(
            ticket_path,
            lambda source: source.is_file(),
        )
    )
    if not ticket_exists:
        return f"Required snapshotted ticket artifact is missing: {ticket_path}"
    try:
        baseline = (
            load_baseline_record(baseline_path)
            if run_ownership is None
            else load_baseline_record_for_owner(run_ownership)
        )
    except RunError as error:
        return f"Baseline artifact is not internally consistent: {error}"
    if baseline.branch != run_record.starting_branch:
        return "Run record and baseline artifact disagree on starting branch."
    if baseline.head_sha != run_record.baseline_sha:
        return "Run record and baseline artifact disagree on baseline HEAD."
    if not baseline.clean_worktree or baseline.has_staged_files:
        return "Recorded repository baseline is not clean."
    try:
        ticket_contents = (
            ticket_path.read_bytes()
            if run_ownership is None
            else run_ownership.read_descendant(
                ticket_path,
                lambda source: source.read_bytes(),
            )
        )
        ticket_sha256 = hashlib.sha256(ticket_contents).hexdigest()
    except OSError as error:
        return f"Could not inspect snapshotted ticket artifact: {error}"
    if ticket_sha256 != baseline.ticket_sha256:
        return "Snapshotted ticket no longer matches the recorded ticket baseline."
    if (
        verification_commands_fingerprint(config.verification.commands)
        != baseline.verification_commands_fingerprint
    ):
        return (
            "Configured verification_commands_fingerprint no longer matches the "
            "recorded baseline."
        )
    repository = GitRepository(Path(run_record.target_repository_path))
    if not repository.path.exists():
        return f"Target repository no longer exists: {repository.path}"
    try:
        if not repository.is_repository():
            return f"Target path is no longer a Git repository: {repository.path}"
    except OSError as error:
        return f"Could not inspect target repository: {error}"
    try:
        snapshot = WorkspaceSnapshot.capture(repository)
    except (GitCommandError, OSError, RuntimeError, ValueError) as error:
        return f"Could not inspect current workspace for safe resume: {error}"
    if not snapshot.inspection_complete:
        return "Could not inspect current workspace for safe resume completely."
    if snapshot.branch != run_record.starting_branch:
        return "Current branch no longer matches the recorded run baseline."
    if snapshot.head_sha != run_record.baseline_sha:
        return "Current HEAD no longer matches the recorded run baseline."
    if snapshot.staged_paths:
        return "Current workspace has staged files; human inspection is required."
    if run_record.state in {
        WorkflowState.PREPARING,
        WorkflowState.PREPARED,
    } and not snapshot.matches_fingerprint(baseline.workspace_fingerprint):
        return "Current workspace no longer matches the recorded clean baseline."
    if run_record.state is not WorkflowState.PREPARING:
        problem = baseline_verification_evidence_problem(
            run_dir,
            run_record,
            baseline,
            verification_commands=config.verification.commands,
            run_ownership=run_ownership,
        )
        if problem is not None:
            return problem
    return None


def resume_problem(
    run_dir: Path,
    run_record: RunRecord,
    *,
    run_ownership: RunOwnership | None = None,
) -> str | None:
    if run_ownership is not None:
        run_dir = run_ownership.validate_run_path(run_dir)
    interrupted_writable = latest_attempt(
        run_dir,
        phases=(
            candidate
            for candidate, candidate_definition in PHASE_DEFINITIONS.items()
            if candidate_definition.writes_target_repository
        ),
        statuses=(AttemptStatus.STARTED,),
        run_ownership=run_ownership,
    )
    if interrupted_writable is not None:
        definition = PHASE_DEFINITIONS[interrupted_writable.phase]
        return (
            f"Writable {definition.display_name} attempt "
            f"{interrupted_writable.sequence} was interrupted before completion; "
            "the working tree may contain partial source modifications."
        )
    phase = phase_for_active_state(run_record.state)
    definition = None if phase is None else PHASE_DEFINITIONS[phase]
    if definition is not None and not definition.automatically_retry_interrupted:
        return (
            f"Writable {definition.display_name} was interrupted before completion; the "
            "working tree may contain partial source modifications."
        )
    if run_record.state in {WorkflowState.PREPARING, WorkflowState.PREPARED}:
        baseline = (
            load_baseline_record(run_dir / BASELINE_RECORD_FILE)
            if run_ownership is None
            else load_baseline_record_for_owner(run_ownership)
        )
        return _workspace_fingerprint_problem(
            run_record,
            baseline.workspace_fingerprint,
            description="the recorded clean baseline",
        )
    if (
        definition is not None and definition.automatically_retry_interrupted
    ) or run_record.state is WorkflowState.CORRECTION_PENDING:
        writable_attempt = latest_writable_attempt(
            run_dir,
            run_ownership=run_ownership,
        )
        if (
            writable_attempt is None
            or writable_attempt.after_workspace_fingerprint is None
        ):
            return "No completed writable attempt has a workspace fingerprint."
        return _workspace_fingerprint_problem(
            run_record,
            writable_attempt.after_workspace_fingerprint,
            description="the most recent completed writable attempt",
        )
    return f"Run state is not resumable in V1: {run_record.state.value}"


def _workspace_fingerprint_problem(
    run_record: RunRecord,
    expected_fingerprint: str,
    *,
    description: str,
) -> str | None:
    repository = GitRepository(Path(run_record.target_repository_path))
    try:
        current = WorkspaceSnapshot.capture(repository)
    except (OSError, RuntimeError, ValueError) as error:
        return f"Could not inspect current workspace for safe resume: {error}"
    if not current.inspection_complete:
        return "Could not inspect current workspace for safe resume completely."
    if not current.matches_fingerprint(expected_fingerprint):
        return f"Current workspace no longer matches {description}."
    return None


__all__ = ["resume_preflight_problem", "resume_problem"]
