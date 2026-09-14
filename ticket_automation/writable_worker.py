from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .attempts import AttemptRecord, update_attempt
from .codex import (
    CodexExecution,
    CodexExecutionFailure,
    CodexProcessRunner,
    Sandbox,
)
from .codex import execute as execute_codex
from .git import GitRepository
from .git_safety import WorkspaceSnapshot
from .workspace_guard import (
    WorkspaceEnvironmentSnapshot,
    WorkspaceGuardInspection,
    _compare_workspace_environment_change,
    _format_workspace_environment_inspection_failure,
    capture_workspace_environment_snapshot,
    format_workspace_hygiene_violation,
)
from .writable_attempts import (
    WritableAttempt,
    _capture_writable_attempt,
    _track_writable_process_start,
)


@dataclass(frozen=True)
class WritableCodexInvocation:
    """All safety evidence captured around exactly one writable Codex call."""

    attempt: WritableAttempt
    workspace_guard: WorkspaceGuardInspection
    before_workspace: WorkspaceSnapshot | None
    after_workspace: WorkspaceSnapshot | None
    execution: CodexExecution | None
    failure: CodexExecutionFailure | None
    invocation_permitted: bool

    @property
    def inspection_complete(self) -> bool:
        return bool(
            self.before_workspace is not None
            and self.after_workspace is not None
            and self.before_workspace.inspection_complete
            and self.after_workspace.inspection_complete
            and not self.workspace_guard.has_inspection_failure
        )


def run_writable_codex(
    *,
    repository: GitRepository,
    run_dir: Path | str,
    operation: str,
    phase: str,
    attempt_record: AttemptRecord,
    prompt: str,
    output_schema: Path | str,
    artifact_directory: Path | str,
    executable: str,
    execution_config: Any,
    runner: CodexProcessRunner | None = None,
    clock: Callable[[], datetime] | None = None,
) -> WritableCodexInvocation:
    """Run one workspace-write Codex invocation behind the common guard.

    The deterministic environment policy is deliberately narrow: it compares
    credible Python virtual-environment and Conda roots inside the target
    repository. It does not attempt to prove the absence of package caches,
    generated dependency trees, or global/external environment mutations.
    """

    environment_before = _capture_environment_snapshot(repository)
    before_workspace, before_error = _capture_workspace_snapshot(repository)
    environment_before = _with_workspace_inspection_errors(
        environment_before,
        snapshot=before_workspace,
        capture_error=before_error,
        point="before",
    )
    attempt = _capture_writable_attempt(
        operation=operation,
        attempt_record=attempt_record,
        before_snapshot=before_workspace,
        before_error=before_error,
    )
    if not attempt.before_complete or not environment_before.inspection_complete:
        guard = _compare_workspace_environment_change(
            before=environment_before,
            after=environment_before,
            phase=phase,
            clock=clock,
        )
        guard = _persist_guard_evidence_if_needed(
            attempt,
            guard,
            execution=None,
        )
        return WritableCodexInvocation(
            attempt=attempt,
            workspace_guard=guard,
            before_workspace=before_workspace,
            after_workspace=None,
            execution=None,
            failure=None,
            invocation_permitted=False,
        )

    execution: CodexExecution | None = None
    failure: CodexExecutionFailure | None = None
    after_workspace: WorkspaceSnapshot | None = None
    try:
        try:
            execution = execute_codex(
                prompt=prompt,
                repo_path=repository.path,
                sandbox=Sandbox.WORKSPACE_WRITE,
                output_schema=output_schema,
                artifact_directory=artifact_directory,
                executable=executable,
                execution_config=execution_config,
                runner=_track_writable_process_start(runner, attempt),
            )
        except CodexExecutionFailure as error:
            attempt.record_process_started(error.execution.process_started)
            failure = error
    finally:
        after_workspace, after_error = _capture_workspace_snapshot(repository)
        environment_after = _capture_environment_snapshot(repository)
        environment_after = _with_workspace_inspection_errors(
            environment_after,
            snapshot=after_workspace,
            capture_error=after_error,
            point="after",
        )
        if after_workspace is None:
            attempt.record_after_error(
                RuntimeError(
                    after_error
                    or "Could not capture a post-call canonical workspace snapshot."
                )
            )
        else:
            attempt.record_after_snapshot(after_workspace)

        guard = _compare_workspace_environment_change(
            before=environment_before,
            after=environment_after,
            phase=phase,
            clock=clock,
        )
        guard = _persist_guard_evidence_if_needed(
            attempt,
            guard,
            execution=execution,
        )
    return WritableCodexInvocation(
        attempt=attempt,
        workspace_guard=guard,
        before_workspace=before_workspace,
        after_workspace=after_workspace,
        execution=execution,
        failure=failure,
        invocation_permitted=True,
    )


def _capture_environment_snapshot(
    repository: GitRepository,
) -> WorkspaceEnvironmentSnapshot:
    # The scanner records accessible-path errors in its snapshot. A truly
    # exceptional scanner failure is represented as an incomplete baseline by
    # the caller's canonical workspace snapshot and is never allowed to start.
    try:
        return capture_workspace_environment_snapshot(repository.path)
    except Exception as error:  # noqa: BLE001 - scanner evidence must fail closed.
        return WorkspaceEnvironmentSnapshot(
            repository_path=repository.path.absolute(),
            environments=(),
            inspection_errors=(f"scanner: {type(error).__name__}: {error}",),
        )


def _capture_workspace_snapshot(
    repository: GitRepository,
) -> tuple[WorkspaceSnapshot | None, str | None]:
    try:
        return WorkspaceSnapshot.capture(repository), None
    except Exception as error:  # noqa: BLE001 - evidence collection must fail closed.
        return None, f"{type(error).__name__}: {error}"


def _with_workspace_inspection_errors(
    environment: WorkspaceEnvironmentSnapshot,
    *,
    snapshot: WorkspaceSnapshot | None,
    capture_error: str | None,
    point: str,
) -> WorkspaceEnvironmentSnapshot:
    """Make every canonical workspace inspection failure fail the guard closed."""

    errors = list(environment.inspection_errors)
    if snapshot is None:
        errors.append(
            f"{point}-canonical-workspace: "
            f"{capture_error or 'workspace snapshot unavailable'}"
        )
    elif not snapshot.inspection_complete:
        errors.extend(
            f"{point}-canonical-workspace: {error}"
            for error in snapshot.inspection_errors
        )
    return replace(environment, inspection_errors=tuple(dict.fromkeys(errors)))


def _persist_guard_evidence_if_needed(
    attempt: WritableAttempt,
    inspection: WorkspaceGuardInspection,
    *,
    execution: CodexExecution | None,
) -> WorkspaceGuardInspection:
    """Keep workspace-guard evidence with the writable execution metadata."""

    path = (
        attempt.record.artifact_directory / "execution.json"
        if execution is None
        else execution.execution_json_path
    )
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not update writable execution metadata: {error}"
        ) from error
    if not isinstance(data, dict):
        raise TypeError("Writable execution metadata must be a JSON object.")
    if execution is None:
        data = {
            "schema_version": 1,
            "format": "ticket_automation.writable_execution",
            "process_started": False,
            **data,
        }
        attempt.record = update_attempt(attempt.record, execution_path="execution.json")
    data["workspace_guard"] = inspection.to_dict()
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return inspection


def _format_writable_guard_stop(
    inspection: WorkspaceGuardInspection,
    *,
    operation: str,
    run_dir: Path | str,
    git_safety: str,
) -> str:
    """Render the shared boundary's fail-closed outcome for a stage result."""

    if not inspection.requires_human:
        raise ValueError("A clean writable guard has no human-required stop message.")
    if inspection.has_violation:
        return format_workspace_hygiene_violation(
            inspection,
            operation=operation,
            run_dir=run_dir,
            git_safety=git_safety,
        )
    return _format_workspace_environment_inspection_failure(
        inspection,
        operation=operation,
        run_dir=run_dir,
    )


__all__ = [
    "WritableCodexInvocation",
    "run_writable_codex",
]
