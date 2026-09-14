"""In-memory writable-call inspection backed by the enclosing attempt record."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .attempts import AttemptRecord, update_attempt
from .codex import (
    CodexCommand,
    CodexProcessResult,
    CodexProcessRunner,
    SubprocessCodexRunner,
)
from .git import GitRepository
from .git_safety import WorkspaceSnapshot


@dataclass
class WritableAttempt:
    """Runtime safety details for a writable call.

    The durable part is the small enclosing ``attempt.json`` record. Full
    snapshots stay in memory because an interrupted writable invocation is
    never resumed automatically.
    """

    record: AttemptRecord
    operation: str
    before_snapshot: WorkspaceSnapshot | None
    before_error: str | None
    process_started: bool = False
    after_snapshot: WorkspaceSnapshot | None = None
    after_error: str | None = None

    @property
    def artifact_path(self) -> Path:
        return self.record.path

    def mark_process_started(self) -> None:
        self.record_process_started(True)

    def record_process_started(self, value: bool) -> None:
        self.process_started = value
        self.record = update_attempt(self.record, process_started=value)

    def record_after_snapshot(self, snapshot: WorkspaceSnapshot) -> None:
        self.after_snapshot = snapshot
        self.after_error = None
        self.record = update_attempt(
            self.record,
            after_workspace_fingerprint=snapshot.fingerprint,
        )

    def record_after_error(self, error: BaseException) -> None:
        self.after_snapshot = None
        self.after_error = f"{type(error).__name__}: {error}"
        self.record = update_attempt(
            self.record,
            metadata={
                **self.record.metadata,
                "after_workspace_error": self.after_error,
            },
        )

    @property
    def before_complete(self) -> bool:
        return bool(
            self.before_snapshot is not None
            and self.before_snapshot.inspection_complete
        )


class _StartTrackingCodexRunner:
    def __init__(
        self,
        runner: CodexProcessRunner | None,
        writable_attempt: WritableAttempt,
    ):
        self._runner = runner or SubprocessCodexRunner()
        self._writable_attempt = writable_attempt

    def run(
        self,
        command: CodexCommand,
        *,
        stdin: str,
        timeout_seconds: float | None,
    ) -> CodexProcessResult:
        self._writable_attempt.mark_process_started()
        return self._runner.run(
            command,
            stdin=stdin,
            timeout_seconds=timeout_seconds,
        )


def _track_writable_process_start(
    runner: CodexProcessRunner | None,
    writable_attempt: WritableAttempt,
) -> CodexProcessRunner:
    return _StartTrackingCodexRunner(runner, writable_attempt)


@dataclass(frozen=True)
class WritableInspection:
    before_complete: bool
    after_complete: bool
    workspace_identical: bool
    inspection_error: str | None


def capture_writable_attempt(
    repository: GitRepository,
    *,
    operation: str,
    attempt_record: AttemptRecord,
) -> WritableAttempt:
    before_snapshot: WorkspaceSnapshot | None = None
    before_error: str | None = None
    try:
        before_snapshot = WorkspaceSnapshot.capture(repository)
    except Exception as error:  # noqa: BLE001 - safety evidence must fail closed.
        before_error = f"{type(error).__name__}: {error}"
    return _capture_writable_attempt(
        operation=operation,
        attempt_record=attempt_record,
        before_snapshot=before_snapshot,
        before_error=before_error,
    )


def _capture_writable_attempt(
    *,
    operation: str,
    attempt_record: AttemptRecord,
    before_snapshot: WorkspaceSnapshot | None,
    before_error: str | None,
) -> WritableAttempt:
    if before_snapshot is not None and before_error is not None:
        raise ValueError("A writable attempt cannot have both a snapshot and an error.")
    metadata = attempt_record.metadata
    if before_error is not None:
        metadata = {**metadata, "before_workspace_error": before_error}
    record = update_attempt(
        attempt_record,
        before_workspace_fingerprint=(
            None if before_snapshot is None else before_snapshot.fingerprint
        ),
        metadata=metadata,
    )
    return WritableAttempt(
        record=record,
        operation=operation,
        before_snapshot=before_snapshot,
        before_error=before_error,
    )


def inspect_writable_attempt_after_failure(
    attempt: WritableAttempt | None,
    repository: GitRepository,
) -> WritableInspection:
    if attempt is None:
        return WritableInspection(
            False, False, False, "No writable attempt is available."
        )
    if attempt.after_error is not None:
        return WritableInspection(
            attempt.before_complete,
            False,
            False,
            attempt.after_error,
        )
    if attempt.after_snapshot is None:
        try:
            attempt.record_after_snapshot(WorkspaceSnapshot.capture(repository))
        except Exception as error:  # noqa: BLE001 - safety evidence must fail closed.
            attempt.record_after_error(error)
            return WritableInspection(
                attempt.before_complete,
                False,
                False,
                attempt.after_error,
            )
    after = attempt.after_snapshot
    before = attempt.before_snapshot
    assert after is not None
    complete = bool(
        before is not None and before.inspection_complete and after.inspection_complete
    )
    return WritableInspection(
        attempt.before_complete,
        after.inspection_complete,
        bool(
            complete and before is not None and before.fingerprint == after.fingerprint
        ),
        None,
    )


__all__ = [
    "WritableAttempt",
    "WritableInspection",
    "capture_writable_attempt",
    "inspect_writable_attempt_after_failure",
]
