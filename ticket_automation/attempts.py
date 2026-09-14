"""Small, append-only records for work performed during one run.

The controller owns state transitions. Attempt records only describe an
invocation that was started while the controller was in a particular state;
they are deliberately not checkpoints that can advance a run on their own.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

ATTEMPTS_DIR_NAME = "attempts"
ATTEMPT_RECORD_FILE = "attempt.json"
ATTEMPT_RECORD_FORMAT = "ticket_automation.attempt"
ATTEMPT_RECORD_SCHEMA_VERSION = 1

_ATTEMPT_DIRECTORY_PATTERN = re.compile(r"^(0*[1-9][0-9]*)-(.+)$")
_PHASES = frozenset(
    {
        "PREPARING",
        "IMPLEMENTING",
        "VERIFYING",
        "REVIEWING",
        "CORRECTING",
        "REPORTING",
    }
)
_STATUSES = frozenset({"STARTED", "COMPLETED", "HUMAN_REQUIRED", "FAILED"})

WRITABLE_PHASES = frozenset({"IMPLEMENTING", "CORRECTING"})


@dataclass(frozen=True)
class AttemptRecord:
    sequence: int
    phase: str
    status: str
    before_workspace_fingerprint: str | None
    after_workspace_fingerprint: str | None
    process_started: bool
    started_at: str
    ended_at: str | None
    result_path: str | None
    execution_path: str | None
    metadata: dict[str, Any]
    artifact_directory: Path
    schema_version: int = ATTEMPT_RECORD_SCHEMA_VERSION
    format: str = ATTEMPT_RECORD_FORMAT

    @property
    def path(self) -> Path:
        return self.artifact_directory / ATTEMPT_RECORD_FILE

    @property
    def completed(self) -> bool:
        return self.status == "COMPLETED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "format": self.format,
            "sequence": self.sequence,
            "phase": self.phase,
            "status": self.status,
            "before_workspace_fingerprint": self.before_workspace_fingerprint,
            "after_workspace_fingerprint": self.after_workspace_fingerprint,
            "process_started": self.process_started,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "result_path": self.result_path,
            "execution_path": self.execution_path,
            "metadata": self.metadata,
        }


class AttemptError(RuntimeError):
    pass


def start_attempt(
    run_dir: Path | str,
    *,
    phase: str,
    before_workspace_fingerprint: str | None,
    result_path: str | None = "result.json",
    execution_path: str | None = None,
    clock: Callable[[], datetime] | None = None,
) -> AttemptRecord:
    """Create a fully recorded attempt before exposing its final directory.

    The initial record is written in a private temporary directory and then
    renamed into ``attempts/<sequence>-<phase>``. A crash cannot leave a final
    attempt directory without its authoritative record, and any old orphaned
    directory still reserves its sequence number as diagnostic history.
    """

    run_path = Path(run_dir)
    attempts_root = run_path / ATTEMPTS_DIR_NAME
    attempts_root.mkdir(parents=True, exist_ok=True)
    while True:
        sequence = _next_sequence(attempts_root)
        directory = attempts_root / f"{sequence:03d}-{_phase_slug(phase)}"
        temporary_directory = Path(
            tempfile.mkdtemp(
                dir=attempts_root,
                prefix=f".{sequence:03d}-{_phase_slug(phase)}-",
                suffix=".tmp",
            )
        )
        record = AttemptRecord(
            sequence=sequence,
            phase=phase,
            status="STARTED",
            before_workspace_fingerprint=before_workspace_fingerprint,
            after_workspace_fingerprint=None,
            process_started=False,
            started_at=_timestamp(clock),
            ended_at=None,
            result_path=result_path,
            execution_path=execution_path,
            metadata={},
            artifact_directory=directory,
        )
        try:
            _validate_record(record)
            _atomic_write_json(
                temporary_directory / ATTEMPT_RECORD_FILE, record.to_dict()
            )
            try:
                temporary_directory.rename(directory)
            except FileExistsError:
                _remove_temporary_attempt_directory(temporary_directory)
                continue
        except Exception:
            _remove_temporary_attempt_directory(temporary_directory)
            raise
        return record


def save_attempt(record: AttemptRecord) -> None:
    _validate_record(record)
    _atomic_write_json(record.path, record.to_dict())


def update_attempt(
    record: AttemptRecord,
    *,
    status: str | None = None,
    before_workspace_fingerprint: str | None = None,
    after_workspace_fingerprint: str | None = None,
    process_started: bool | None = None,
    result_path: str | None = None,
    execution_path: str | None = None,
    metadata: dict[str, Any] | None = None,
    ended: bool = False,
    clock: Callable[[], datetime] | None = None,
) -> AttemptRecord:
    updated = replace(
        record,
        status=record.status if status is None else status,
        before_workspace_fingerprint=(
            record.before_workspace_fingerprint
            if before_workspace_fingerprint is None
            else before_workspace_fingerprint
        ),
        after_workspace_fingerprint=(
            record.after_workspace_fingerprint
            if after_workspace_fingerprint is None
            else after_workspace_fingerprint
        ),
        process_started=(
            record.process_started if process_started is None else process_started
        ),
        result_path=record.result_path if result_path is None else result_path,
        execution_path=(
            record.execution_path if execution_path is None else execution_path
        ),
        metadata=record.metadata if metadata is None else metadata,
        ended_at=_timestamp(clock) if ended else record.ended_at,
    )
    save_attempt(updated)
    return updated


def complete_attempt(
    record: AttemptRecord,
    *,
    status: str,
    after_workspace_fingerprint: str | None,
    process_started: bool | None = None,
    result_path: str | None = None,
    execution_path: str | None = None,
    metadata: dict[str, Any] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> AttemptRecord:
    return update_attempt(
        record,
        status=status,
        after_workspace_fingerprint=after_workspace_fingerprint,
        process_started=process_started,
        result_path=result_path,
        execution_path=execution_path,
        metadata=metadata,
        ended=True,
        clock=clock,
    )


def load_attempt_records(run_dir: Path | str) -> tuple[AttemptRecord, ...]:
    """Load trusted records, rejecting malformed final attempt directories.

    Directories without ``attempt.json`` are harmless crash diagnostics. They
    are never adopted and reserve their sequence number for later attempts.
    A directory that claims to contain an authoritative record must validate
    completely; otherwise callers must stop rather than infer state from it.
    """

    root = Path(run_dir) / ATTEMPTS_DIR_NAME
    if not root.is_dir():
        return ()
    records: list[AttemptRecord] = []
    for directory in sorted(root.iterdir(), key=lambda item: item.name):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        path = directory / ATTEMPT_RECORD_FILE
        if not path.exists():
            continue
        records.append(_load_attempt(path))
    ordered = tuple(sorted(records, key=lambda item: item.sequence))
    sequences = [record.sequence for record in ordered]
    if len(sequences) != len(set(sequences)):
        raise AttemptError("Attempt records contain duplicate sequence numbers.")
    return ordered


def latest_attempt(
    run_dir: Path | str,
    *,
    phases: Iterable[str] | None = None,
    statuses: Iterable[str] | None = None,
) -> AttemptRecord | None:
    phase_set = None if phases is None else frozenset(phases)
    status_set = None if statuses is None else frozenset(statuses)
    for record in reversed(load_attempt_records(run_dir)):
        if phase_set is not None and record.phase not in phase_set:
            continue
        if status_set is not None and record.status not in status_set:
            continue
        return record
    return None


def latest_writable_attempt(run_dir: Path | str) -> AttemptRecord | None:
    return latest_attempt(
        run_dir,
        phases=WRITABLE_PHASES,
        statuses=("COMPLETED",),
    )


def finish_phase_attempt(
    run_dir: Path | str,
    *,
    phase: str,
    stage_outcome: str,
    after_workspace_fingerprint: str | None = None,
    process_started: bool | None = None,
    execution_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> AttemptRecord | None:
    """Close the current phase record without giving it controller authority."""

    record = latest_attempt(run_dir, phases=(phase,), statuses=("STARTED",))
    if record is None:
        return None
    status = {
        "COMPLETED": "COMPLETED",
        "CORRECTION_REQUIRED": "COMPLETED",
        "HUMAN_REQUIRED": "HUMAN_REQUIRED",
        "FAILED": "FAILED",
    }.get(stage_outcome, "FAILED")
    relative_execution_path = None
    if execution_path is not None:
        try:
            relative_execution_path = execution_path.relative_to(
                record.artifact_directory
            ).as_posix()
        except ValueError as error:
            raise AttemptError(
                "Execution metadata must remain inside its attempt directory."
            ) from error
    return complete_attempt(
        record,
        status=status,
        after_workspace_fingerprint=after_workspace_fingerprint,
        process_started=process_started,
        execution_path=relative_execution_path,
        metadata=record.metadata if metadata is None else metadata,
        clock=clock,
    )


def attempt_result_path(run_dir: Path | str, record: AttemptRecord) -> Path | None:
    if record.result_path is None:
        return None
    return _attempt_artifact_path(run_dir, record, record.result_path)


def _attempt_artifact_path(
    run_dir: Path | str,
    record: AttemptRecord,
    relative_path: str,
) -> Path:
    expected_directory = Path(run_dir) / ATTEMPTS_DIR_NAME / _directory_name(record)
    if _path_identity(record.artifact_directory) != _path_identity(expected_directory):
        raise AttemptError(
            "Attempt record does not belong to the supplied run directory."
        )
    _validate_relative_artifact_path(relative_path, field="artifact path")
    return expected_directory / PurePosixPath(relative_path)


def _load_attempt(path: Path) -> AttemptRecord:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AttemptError(f"Could not read attempt record {path}: {error}") from error
    if not isinstance(data, dict):
        raise AttemptError(f"Attempt record {path} must be a JSON object.")
    try:
        record = AttemptRecord(
            sequence=_required_positive_int(data, "sequence"),
            phase=_required_phase(data),
            status=_required_status(data),
            before_workspace_fingerprint=_nullable_string(
                data.get("before_workspace_fingerprint")
            ),
            after_workspace_fingerprint=_nullable_string(
                data.get("after_workspace_fingerprint")
            ),
            process_started=_required_bool(data, "process_started"),
            started_at=_required_timestamp(data, "started_at"),
            ended_at=_nullable_timestamp(data.get("ended_at")),
            result_path=_nullable_relative_artifact_path(
                data.get("result_path"), "result_path"
            ),
            execution_path=_nullable_relative_artifact_path(
                data.get("execution_path"), "execution_path"
            ),
            metadata=_required_metadata(data),
            artifact_directory=path.parent,
            schema_version=data.get("schema_version"),
            format=data.get("format"),
        )
        _validate_record(record)
        return record
    except (KeyError, TypeError, ValueError, AttemptError) as error:
        raise AttemptError(f"Invalid attempt record {path}: {error}") from error


def _next_sequence(root: Path) -> int:
    sequences: list[int] = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        match = _ATTEMPT_DIRECTORY_PATTERN.fullmatch(directory.name)
        if match is not None:
            sequences.append(int(match.group(1)))
    return max(sequences, default=0) + 1


def _validate_record(record: AttemptRecord) -> None:
    if record.schema_version != ATTEMPT_RECORD_SCHEMA_VERSION:
        raise AttemptError("Attempt record has an unsupported schema version.")
    if record.format != ATTEMPT_RECORD_FORMAT:
        raise AttemptError("Attempt record has an unsupported format.")
    if (
        not isinstance(record.sequence, int)
        or isinstance(record.sequence, bool)
        or record.sequence < 1
    ):
        raise AttemptError("Attempt sequence must be a positive integer.")
    if record.phase not in _PHASES:
        raise AttemptError(f"Attempt phase is unsupported: {record.phase!r}.")
    if record.status not in _STATUSES:
        raise AttemptError(f"Attempt status is unsupported: {record.status!r}.")
    if not isinstance(record.process_started, bool):
        raise AttemptError("Attempt process_started must be a boolean.")
    _validate_timestamp(record.started_at, field="started_at")
    if record.ended_at is None:
        if record.status != "STARTED":
            raise AttemptError("Completed attempt records must have ended_at.")
    else:
        _validate_timestamp(record.ended_at, field="ended_at")
        if record.status == "STARTED":
            raise AttemptError("Started attempt records cannot have ended_at.")
    _validate_nullable_string(
        record.before_workspace_fingerprint, "before_workspace_fingerprint"
    )
    _validate_nullable_string(
        record.after_workspace_fingerprint, "after_workspace_fingerprint"
    )
    if record.result_path is not None:
        _validate_relative_artifact_path(record.result_path, field="result_path")
    if record.execution_path is not None:
        _validate_relative_artifact_path(record.execution_path, field="execution_path")
    if not isinstance(record.metadata, dict):
        raise AttemptError("Attempt metadata must be an object.")
    if record.artifact_directory.name != _directory_name(record):
        raise AttemptError("Attempt directory does not match its sequence and phase.")


def _directory_name(record: AttemptRecord) -> str:
    return f"{record.sequence:03d}-{_phase_slug(record.phase)}"


def _phase_slug(phase: str) -> str:
    names = {
        "PREPARING": "preparation",
        "IMPLEMENTING": "implementation",
        "VERIFYING": "verification",
        "REVIEWING": "review",
        "CORRECTING": "correction",
        "REPORTING": "reporting",
    }
    return names.get(phase, re.sub(r"[^a-z0-9]+", "-", phase.lower()).strip("-"))


def _timestamp(clock: Callable[[], datetime] | None) -> str:
    now = datetime.now(UTC) if clock is None else clock()
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return now.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _required_positive_int(data: dict[str, Any], key: str) -> int:
    value = data[key]
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _required_phase(data: dict[str, Any]) -> str:
    value = _required_string(data, "phase")
    if value not in _PHASES:
        raise ValueError(f"phase is unsupported: {value!r}")
    return value


def _required_status(data: dict[str, Any]) -> str:
    value = _required_string(data, "status")
    if value not in _STATUSES:
        raise ValueError(f"status is unsupported: {value!r}")
    return value


def _required_bool(data: dict[str, Any], key: str) -> bool:
    value = data[key]
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean")
    return value


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_timestamp(data: dict[str, Any], key: str) -> str:
    value = _required_string(data, key)
    _validate_timestamp(value, field=key)
    return value


def _nullable_timestamp(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("ended_at must be a string or null")
    _validate_timestamp(value, field="ended_at")
    return value


def _validate_timestamp(value: str, *, field: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise AttemptError(f"{field} must be an ISO-8601 timestamp.") from error
    if parsed.tzinfo is None:
        raise AttemptError(f"{field} must include a timezone.")


def _nullable_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("expected a string or null")
    return value


def _validate_nullable_string(value: str | None, field: str) -> None:
    if value is not None and not isinstance(value, str):
        raise AttemptError(f"{field} must be a string or null.")


def _nullable_relative_artifact_path(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string or null")
    _validate_relative_artifact_path(value, field=field)
    return value


def _validate_relative_artifact_path(value: str, *, field: str) -> None:
    if not value or "\\" in value:
        raise AttemptError(f"{field} must be a non-empty POSIX relative path.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise AttemptError(f"{field} must remain inside the attempt directory.")


def _required_metadata(data: dict[str, Any]) -> dict[str, Any]:
    value = data["metadata"]
    if not isinstance(value, dict):
        raise TypeError("metadata must be an object")
    return value


def _path_identity(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except OSError:
        return path.absolute()


def _remove_temporary_attempt_directory(path: Path) -> None:
    try:
        for item in path.iterdir():
            item.unlink(missing_ok=True)
        path.rmdir()
    except OSError:
        # A leftover hidden temporary directory is diagnostic only and does not
        # participate in sequence allocation or result loading.
        pass


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
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
    "ATTEMPTS_DIR_NAME",
    "ATTEMPT_RECORD_FILE",
    "WRITABLE_PHASES",
    "AttemptError",
    "AttemptRecord",
    "attempt_result_path",
    "complete_attempt",
    "finish_phase_attempt",
    "latest_attempt",
    "latest_writable_attempt",
    "load_attempt_records",
    "save_attempt",
    "start_attempt",
    "update_attempt",
]
