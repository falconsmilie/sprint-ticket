from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .codex import (
    CodexCommand,
    CodexProcessResult,
    CodexProcessRunner,
    SubprocessCodexRunner,
)
from .git import GitRepository
from .git_safety import WorkspaceSnapshot
from .workspace_guard import (
    WORKSPACE_GUARD_FORMAT,
    WORKSPACE_GUARD_SCHEMA_VERSION,
    LocalEnvironment,
    WorkspaceEnvironmentSnapshot,
    WorkspaceGuardInspection,
    new_environments,
)

WRITABLE_ATTEMPTS_DIR_NAME = "writable-attempts"
WRITABLE_ATTEMPT_FORMAT = "ticket_automation.writable_attempt"
WRITABLE_ATTEMPT_SCHEMA_VERSION = 1


@dataclass
class WritableAttempt:
    """Evidence captured immediately around one writable Codex invocation."""

    artifact_path: Path
    operation: str
    before_snapshot: WorkspaceSnapshot | None
    before_error: str | None
    process_started: bool = False
    after_snapshot: WorkspaceSnapshot | None = None
    after_error: str | None = None
    environment_guard: WorkspaceGuardInspection | None = None

    def mark_process_started(self) -> None:
        self.record_process_started(True)

    def record_process_started(self, value: bool) -> None:
        self.process_started = value
        self._persist()

    def record_after_snapshot(self, snapshot: WorkspaceSnapshot) -> None:
        self.after_snapshot = snapshot
        self.after_error = None
        self._persist()

    def record_after_error(self, error: BaseException) -> None:
        self.after_snapshot = None
        self.after_error = f"{type(error).__name__}: {error}"
        self._persist()

    def record_environment_guard(self, inspection: WorkspaceGuardInspection) -> None:
        """Persist the Python/Conda root evidence with this writable execution."""

        self.environment_guard = inspection
        self._persist()

    @property
    def before_complete(self) -> bool:
        return bool(
            self.before_snapshot is not None
            and self.before_snapshot.inspection_complete
        )

    def _persist(self) -> None:
        _atomic_write_json(self.artifact_path, self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": WRITABLE_ATTEMPT_SCHEMA_VERSION,
            "format": WRITABLE_ATTEMPT_FORMAT,
            "operation": self.operation,
            "process_started": self.process_started,
            "before_snapshot": (
                None
                if self.before_snapshot is None
                else self.before_snapshot.canonical_data()
            ),
            "before_fingerprint": (
                None
                if self.before_snapshot is None
                else self.before_snapshot.fingerprint
            ),
            "before_error": self.before_error,
            "after_snapshot": (
                None
                if self.after_snapshot is None
                else self.after_snapshot.canonical_data()
            ),
            "after_fingerprint": (
                None if self.after_snapshot is None else self.after_snapshot.fingerprint
            ),
            "after_error": self.after_error,
            "environment_guard": (
                None
                if self.environment_guard is None
                else self.environment_guard.to_dict()
            ),
        }


class _StartTrackingCodexRunner:
    """Mark durable writable-attempt evidence immediately before runner entry."""

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


def writable_attempt_path(run_dir: Path | str, *, operation: str) -> Path:
    safe_operation = operation.replace("/", "-").replace("\\", "-")
    return Path(run_dir) / WRITABLE_ATTEMPTS_DIR_NAME / f"{safe_operation}.json"


def capture_writable_attempt(
    repository: GitRepository,
    run_dir: Path | str,
    *,
    operation: str,
) -> WritableAttempt:
    before_snapshot: WorkspaceSnapshot | None = None
    before_error: str | None = None
    try:
        before_snapshot = WorkspaceSnapshot.capture(repository)
    except Exception as error:  # noqa: BLE001 - this is safety evidence collection.
        before_error = f"{type(error).__name__}: {error}"
    return _capture_writable_attempt(
        run_dir,
        operation=operation,
        before_snapshot=before_snapshot,
        before_error=before_error,
    )


def _capture_writable_attempt(
    run_dir: Path | str,
    *,
    operation: str,
    before_snapshot: WorkspaceSnapshot | None,
    before_error: str | None,
) -> WritableAttempt:
    """Persist a writable attempt from the shared boundary's canonical snapshot."""

    if before_snapshot is not None and before_error is not None:
        raise ValueError("A writable attempt cannot have both a snapshot and an error.")
    attempt = WritableAttempt(
        artifact_path=writable_attempt_path(run_dir, operation=operation),
        operation=operation,
        before_snapshot=before_snapshot,
        before_error=before_error,
    )
    attempt._persist()
    return attempt


def inspect_writable_attempt_after_failure(
    attempt: WritableAttempt | None,
    repository: GitRepository,
) -> WritableInspection:
    if attempt is None:
        return WritableInspection(
            before_complete=False,
            after_complete=False,
            workspace_identical=False,
            inspection_error="No persisted writable-attempt record is available.",
        )
    if attempt.after_error is not None:
        return WritableInspection(
            before_complete=attempt.before_complete,
            after_complete=False,
            workspace_identical=False,
            inspection_error=attempt.after_error,
        )
    if attempt.after_snapshot is not None:
        before_snapshot = attempt.before_snapshot
        after_snapshot = attempt.after_snapshot
        complete = bool(
            before_snapshot is not None
            and before_snapshot.inspection_complete
            and after_snapshot.inspection_complete
        )
        return WritableInspection(
            before_complete=attempt.before_complete,
            after_complete=after_snapshot.inspection_complete,
            workspace_identical=bool(
                complete and before_snapshot.fingerprint == after_snapshot.fingerprint
            ),
            inspection_error=None,
        )
    try:
        after_snapshot = WorkspaceSnapshot.capture(repository)
    except Exception as error:  # noqa: BLE001 - failed inspection must remain evidence.
        try:
            attempt.record_after_error(error)
        except Exception as persistence_error:  # noqa: BLE001 - preserve evidence.
            return WritableInspection(
                before_complete=attempt.before_complete,
                after_complete=False,
                workspace_identical=False,
                inspection_error=(
                    f"{type(error).__name__}: {error}; additionally could not "
                    "persist inspection evidence: "
                    f"{type(persistence_error).__name__}: {persistence_error}"
                ),
            )
        return WritableInspection(
            before_complete=attempt.before_complete,
            after_complete=False,
            workspace_identical=False,
            inspection_error=f"{type(error).__name__}: {error}",
        )

    try:
        attempt.record_after_snapshot(after_snapshot)
    except Exception as error:  # noqa: BLE001 - persistence is part of inspection.
        return WritableInspection(
            before_complete=attempt.before_complete,
            after_complete=after_snapshot.inspection_complete,
            workspace_identical=False,
            inspection_error=f"Could not persist post-call inspection: {error}",
        )

    before_snapshot = attempt.before_snapshot
    complete = bool(
        before_snapshot is not None
        and before_snapshot.inspection_complete
        and after_snapshot.inspection_complete
    )
    return WritableInspection(
        before_complete=attempt.before_complete,
        after_complete=after_snapshot.inspection_complete,
        workspace_identical=bool(
            complete and before_snapshot.fingerprint == after_snapshot.fingerprint
        ),
        inspection_error=None,
    )


def load_writable_attempt(path: Path | str) -> WritableAttempt | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if (
        data.get("schema_version") != WRITABLE_ATTEMPT_SCHEMA_VERSION
        or data.get("format") != WRITABLE_ATTEMPT_FORMAT
        or not isinstance(data.get("operation"), str)
        or not isinstance(data.get("process_started"), bool)
    ):
        return None
    before_snapshot = _snapshot_from_data(data.get("before_snapshot"))
    if before_snapshot is None:
        return None
    before_error = data.get("before_error")
    if isinstance(before_error, str):
        return None
    environment_guard_data = data.get("environment_guard")
    environment_guard = _workspace_guard_from_data(environment_guard_data)
    if environment_guard_data is not None and environment_guard is None:
        return None
    after_snapshot = _snapshot_from_data(data.get("after_snapshot"))
    after_error = data.get("after_error")
    if after_snapshot is not None and isinstance(after_error, str):
        return None
    return WritableAttempt(
        artifact_path=Path(path),
        operation=data["operation"],
        before_snapshot=before_snapshot,
        before_error=None,
        process_started=data["process_started"],
        after_snapshot=after_snapshot,
        after_error=after_error if isinstance(after_error, str) else None,
        environment_guard=environment_guard,
    )


def _snapshot_from_data(value: object) -> WorkspaceSnapshot | None:
    if not isinstance(value, dict):
        return None
    try:
        file_hashes = value["untracked_file_hashes"]
        return WorkspaceSnapshot(
            repository_path=Path(value["repository_path"]),
            branch=value["branch"],
            head_sha=value["head_sha"],
            staged_paths=tuple(value["staged_paths"]),
            staged_diff_sha256=value["staged_diff_sha256"],
            tracked_diff_sha256=value["tracked_diff_sha256"],
            untracked_paths=tuple(value["untracked_paths"]),
            untracked_file_hashes=tuple((item[0], item[1]) for item in file_hashes),
            environment_roots=tuple(value["environment_roots"]),
            inspection_complete=value["inspection_complete"],
            inspection_errors=tuple(value["inspection_errors"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _workspace_guard_from_data(value: object) -> WorkspaceGuardInspection | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        return None
    if (
        value.get("schema_version") != WORKSPACE_GUARD_SCHEMA_VERSION
        or value.get("format") != WORKSPACE_GUARD_FORMAT
        or not isinstance(value.get("phase"), str)
        or not isinstance(value.get("timestamp"), str)
        or not isinstance(value.get("target_repository_path"), str)
    ):
        return None
    repository_path = Path(value["target_repository_path"])
    before = _environment_snapshot_from_data(
        repository_path,
        value.get("environments_before"),
        value.get("inspection_errors_before"),
    )
    after = _environment_snapshot_from_data(
        repository_path,
        value.get("environments_after"),
        value.get("inspection_errors_after"),
    )
    declared_new_environments = _environments_from_data(
        repository_path,
        value.get("new_environments"),
    )
    if before is None or after is None or declared_new_environments is None:
        return None
    if declared_new_environments != new_environments(before, after):
        return None
    return WorkspaceGuardInspection(
        phase=value["phase"],
        timestamp=value["timestamp"],
        artifact_path=None,
        before=before,
        after=after,
        new_environments=declared_new_environments,
    )


def _environment_snapshot_from_data(
    repository_path: Path,
    environments: object,
    inspection_errors: object,
) -> WorkspaceEnvironmentSnapshot | None:
    parsed_environments = _environments_from_data(repository_path, environments)
    if parsed_environments is None or not isinstance(inspection_errors, (list, tuple)):
        return None
    if not all(isinstance(error, str) for error in inspection_errors):
        return None
    return WorkspaceEnvironmentSnapshot(
        repository_path=repository_path,
        environments=parsed_environments,
        inspection_errors=tuple(inspection_errors),
    )


def _environments_from_data(
    repository_path: Path,
    value: object,
) -> tuple[LocalEnvironment, ...] | None:
    if not isinstance(value, (list, tuple)):
        return None
    environments: list[LocalEnvironment] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        kind = item.get("kind")
        root_path = item.get("root_absolute_path")
        marker_paths = item.get("markers")
        if (
            kind not in {"python_venv", "conda"}
            or not isinstance(root_path, str)
            or not isinstance(marker_paths, (list, tuple))
            or not marker_paths
            or not all(isinstance(marker, str) for marker in marker_paths)
        ):
            return None
        root = Path(root_path)
        markers = tuple(repository_path / Path(marker) for marker in marker_paths)
        environments.append(
            LocalEnvironment(kind=kind, root_path=root, marker_paths=markers)
        )
    return tuple(environments)


def _atomic_write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    descriptor = -1
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            text=True,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            descriptor = -1
            json.dump(data, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if descriptor != -1:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


__all__ = [
    "WritableAttempt",
    "WritableInspection",
    "capture_writable_attempt",
    "inspect_writable_attempt_after_failure",
    "load_writable_attempt",
    "writable_attempt_path",
]
