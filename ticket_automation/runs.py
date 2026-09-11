from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import AppConfig
from .git import GitCommandError, GitRepository
from .models import WorkflowState
from .preflight import PreflightResult, run_preflight


RUN_SCHEMA_VERSION = 1
BASELINE_SCHEMA_VERSION = 1
RUN_RECORD_FORMAT = "ticket_automation.run"
BASELINE_RECORD_FORMAT = "ticket_automation.baseline"
RUNS_DIR_NAME = "runs"
RUN_RECORD_FILE = "run.json"
BASELINE_RECORD_FILE = "baseline.json"
RUN_TICKET_FILE = "ticket.md"


class RunError(RuntimeError):
    """Raised when a run cannot be created or inspected."""


class TicketInputError(RunError):
    """Raised when the ticket file is missing, unreadable, or empty."""


class RunPreflightError(RunError):
    def __init__(self, result: PreflightResult):
        super().__init__("Repository preflight failed.")
        self.result = result


@dataclass(frozen=True)
class BaselineRecord:
    repository_path: str
    branch: str
    head_sha: str
    clean_worktree: bool
    has_staged_files: bool
    staging_status: str
    snapshot_timestamp: str
    schema_version: int = BASELINE_SCHEMA_VERSION
    format: str = BASELINE_RECORD_FORMAT

    @classmethod
    def capture(
        cls,
        repository: GitRepository,
        *,
        snapshot_timestamp: str,
    ) -> BaselineRecord:
        branch = repository.current_branch()
        if branch is None:
            raise RunError("Cannot snapshot a repository in detached HEAD state.")
        has_staged_files = repository.has_staged_files()
        return cls(
            repository_path=str(repository.path.resolve()),
            branch=branch,
            head_sha=repository.head_sha(),
            clean_worktree=repository.is_working_tree_clean(),
            has_staged_files=has_staged_files,
            staging_status="dirty" if has_staged_files else "clean",
            snapshot_timestamp=snapshot_timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "format": self.format,
            "repository_path": self.repository_path,
            "branch": self.branch,
            "head_sha": self.head_sha,
            "clean_worktree": self.clean_worktree,
            "has_staged_files": self.has_staged_files,
            "staging_status": self.staging_status,
            "snapshot_timestamp": self.snapshot_timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineRecord:
        schema_version = _require_exact_int(
            data,
            "schema_version",
            expected=BASELINE_SCHEMA_VERSION,
        )
        format_value = _require_exact_string(
            data,
            "format",
            expected=BASELINE_RECORD_FORMAT,
        )
        return cls(
            schema_version=schema_version,
            format=format_value,
            repository_path=_require_string(data, "repository_path"),
            branch=_require_string(data, "branch"),
            head_sha=_require_string(data, "head_sha"),
            clean_worktree=_require_bool(data, "clean_worktree"),
            has_staged_files=_require_bool(data, "has_staged_files"),
            staging_status=_require_string(data, "staging_status"),
            snapshot_timestamp=_require_string(data, "snapshot_timestamp"),
        )


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    ticket_id: str
    original_ticket_path: str
    run_ticket_copy_path: str
    target_repository_path: str
    state: WorkflowState
    starting_branch: str
    baseline_sha: str
    current_correction_round: int
    max_correction_rounds: int
    created_timestamp: str
    updated_timestamp: str
    schema_version: int = RUN_SCHEMA_VERSION
    format: str = RUN_RECORD_FORMAT

    def with_state(self, state: WorkflowState, *, updated_timestamp: str) -> RunRecord:
        return replace(self, state=state, updated_timestamp=updated_timestamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "format": self.format,
            "run_id": self.run_id,
            "ticket_id": self.ticket_id,
            "original_ticket_path": self.original_ticket_path,
            "run_ticket_copy_path": self.run_ticket_copy_path,
            "target_repository_path": self.target_repository_path,
            "state": self.state.value,
            "starting_branch": self.starting_branch,
            "baseline_sha": self.baseline_sha,
            "current_correction_round": self.current_correction_round,
            "max_correction_rounds": self.max_correction_rounds,
            "created_timestamp": self.created_timestamp,
            "updated_timestamp": self.updated_timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        schema_version = _require_exact_int(
            data,
            "schema_version",
            expected=RUN_SCHEMA_VERSION,
        )
        format_value = _require_exact_string(
            data,
            "format",
            expected=RUN_RECORD_FORMAT,
        )
        return cls(
            schema_version=schema_version,
            format=format_value,
            run_id=_require_string(data, "run_id"),
            ticket_id=_require_string(data, "ticket_id"),
            original_ticket_path=_require_string(data, "original_ticket_path"),
            run_ticket_copy_path=_require_string(data, "run_ticket_copy_path"),
            target_repository_path=_require_string(data, "target_repository_path"),
            state=WorkflowState(_require_string(data, "state")),
            starting_branch=_require_string(data, "starting_branch"),
            baseline_sha=_require_string(data, "baseline_sha"),
            current_correction_round=_require_int(data, "current_correction_round"),
            max_correction_rounds=_require_int(data, "max_correction_rounds"),
            created_timestamp=_require_string(data, "created_timestamp"),
            updated_timestamp=_require_string(data, "updated_timestamp"),
        )


@dataclass(frozen=True)
class RunCreationResult:
    run_dir: Path
    run_record: RunRecord
    baseline_record: BaselineRecord
    preflight_result: PreflightResult


def create_run_snapshot(
    config: AppConfig,
    ticket_path: Path | str,
    *,
    runs_dir: Path | str,
    clock: Callable[[], datetime] | None = None,
) -> RunCreationResult:
    source_ticket = _read_ticket(ticket_path)
    preflight_result = run_preflight(config)
    if not preflight_result.passed:
        raise RunPreflightError(preflight_result)

    repository = GitRepository(config.project.repo)
    timestamp = _timestamp(clock)
    baseline_record = _capture_baseline(repository, snapshot_timestamp=timestamp)
    ticket_id = sanitize_ticket_id(source_ticket.path.stem)
    run_id, run_dir = _reserve_run_directory(Path(runs_dir), timestamp, ticket_id)
    run_ticket_path = run_dir / RUN_TICKET_FILE
    run_record_path = run_dir / RUN_RECORD_FILE
    baseline_record_path = run_dir / BASELINE_RECORD_FILE

    try:
        run_record = RunRecord(
            run_id=run_id,
            ticket_id=ticket_id,
            original_ticket_path=str(source_ticket.path.resolve()),
            run_ticket_copy_path=str(run_ticket_path.resolve()),
            target_repository_path=baseline_record.repository_path,
            state=WorkflowState.PREFLIGHT,
            starting_branch=baseline_record.branch,
            baseline_sha=baseline_record.head_sha,
            current_correction_round=0,
            max_correction_rounds=config.runner.max_correction_rounds,
            created_timestamp=timestamp,
            updated_timestamp=timestamp,
        )
        save_run_record(run_record, run_record_path)

        _copy_ticket(source_ticket.contents, run_ticket_path)
        save_baseline_record(baseline_record, baseline_record_path)

        snapshot_timestamp = _timestamp(clock)
        run_record = run_record.with_state(
            WorkflowState.SNAPSHOT,
            updated_timestamp=snapshot_timestamp,
        )
        save_run_record(run_record, run_record_path)
    except Exception:
        _remove_incomplete_run_directory(run_dir)
        raise

    return RunCreationResult(
        run_dir=run_dir,
        run_record=run_record,
        baseline_record=baseline_record,
        preflight_result=preflight_result,
    )


def sanitize_ticket_id(value: str) -> str:
    normalized = value.strip()
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", normalized)
    normalized = normalized.strip(".-_")
    return normalized or "ticket"


def load_run_record(path: Path | str) -> RunRecord:
    data = _load_json_object(Path(path))
    return RunRecord.from_dict(data)


def save_run_record(record: RunRecord, path: Path | str) -> None:
    _atomic_write_json(Path(path), record.to_dict())


def load_baseline_record(path: Path | str) -> BaselineRecord:
    data = _load_json_object(Path(path))
    return BaselineRecord.from_dict(data)


def save_baseline_record(record: BaselineRecord, path: Path | str) -> None:
    _atomic_write_json(Path(path), record.to_dict())


def list_run_records(runs_dir: Path | str) -> tuple[RunRecord, ...]:
    root = Path(runs_dir)
    if not root.is_dir():
        return ()

    records: list[RunRecord] = []
    for run_dir in sorted(root.iterdir()):
        run_record_path = run_dir / RUN_RECORD_FILE
        if run_dir.is_dir() and run_record_path.is_file():
            records.append(load_run_record(run_record_path))
    return tuple(
        sorted(
            records,
            key=lambda record: (record.updated_timestamp, record.run_id),
            reverse=True,
        )
    )


def format_status(records: tuple[RunRecord, ...], *, runs_dir: Path | str) -> str:
    if not records:
        return f"No runs found in {Path(runs_dir)}."

    header = (
        f"{'Run ID':<36} {'Ticket ID':<16} {'State':<12} "
        f"{'Branch':<24} Updated"
    )
    rows = [header]
    for record in records:
        rows.append(
            f"{record.run_id:<36} {record.ticket_id:<16} "
            f"{record.state.value:<12} {record.starting_branch:<24} "
            f"{record.updated_timestamp}"
        )
    return "\n".join(rows)


def _timestamp(clock: Callable[[], datetime] | None) -> str:
    now = datetime.now(timezone.utc) if clock is None else clock()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_id_prefix(timestamp: str) -> str:
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return parsed.strftime("%Y%m%d-%H%M%S")


def _reserve_run_directory(runs_dir: Path, timestamp: str, ticket_id: str) -> tuple[str, Path]:
    runs_dir.mkdir(parents=True, exist_ok=True)
    base_run_id = f"{_run_id_prefix(timestamp)}_{ticket_id}"
    for index in range(1, 1000):
        run_id = base_run_id if index == 1 else f"{base_run_id}_{index}"
        run_dir = runs_dir / run_id
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_id, run_dir
    raise RunError(f"Could not reserve a unique run directory for {base_run_id}.")


def _capture_baseline(
    repository: GitRepository,
    *,
    snapshot_timestamp: str,
) -> BaselineRecord:
    try:
        return BaselineRecord.capture(
            repository,
            snapshot_timestamp=snapshot_timestamp,
        )
    except GitCommandError as error:
        raise RunError(f"Could not capture repository baseline: {error}") from error


@dataclass(frozen=True)
class _TicketSource:
    path: Path
    contents: bytes


def _read_ticket(ticket_path: Path | str) -> _TicketSource:
    path = Path(ticket_path)
    if not path.is_file():
        raise TicketInputError(f"Ticket file does not exist or is not a file: {path}")
    try:
        contents = path.read_bytes()
    except OSError as error:
        raise TicketInputError(f"Ticket file is not readable: {path}: {error}") from error
    if not contents:
        raise TicketInputError(f"Ticket file is empty: {path}")
    return _TicketSource(path=path, contents=contents)


def _copy_ticket(contents: bytes, destination: Path) -> None:
    if destination.exists():
        raise RunError(f"Run ticket copy already exists: {destination}")
    destination.write_bytes(contents)


def _remove_incomplete_run_directory(run_dir: Path) -> None:
    try:
        shutil.rmtree(run_dir)
    except FileNotFoundError:
        return


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as record_file:
        data = json.load(record_file)
    if not isinstance(data, dict):
        raise RunError(f"JSON record must be an object: {path}")
    return data


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True)
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
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as temp_file:
            file_descriptor = -1
            temp_file.write(payload)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    except Exception:
        if file_descriptor != -1:
            os.close(file_descriptor)
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise RunError(f"Run record field must be a non-empty string: {key}")
    return value


def _require_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RunError(f"Run record field must be an integer: {key}")
    return value


def _require_exact_int(data: dict[str, Any], key: str, *, expected: int) -> int:
    value = _require_int(data, key)
    if value != expected:
        raise RunError(f"Unsupported {key}: {value}; expected {expected}.")
    return value


def _require_exact_string(data: dict[str, Any], key: str, *, expected: str) -> str:
    value = _require_string(data, key)
    if value != expected:
        raise RunError(f"Unsupported {key}: {value}; expected {expected}.")
    return value


def _require_bool(data: dict[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise RunError(f"Run record field must be a boolean: {key}")
    return value


__all__ = [
    "BASELINE_RECORD_FILE",
    "BASELINE_RECORD_FORMAT",
    "BASELINE_SCHEMA_VERSION",
    "BaselineRecord",
    "RUN_RECORD_FILE",
    "RUN_RECORD_FORMAT",
    "RUN_SCHEMA_VERSION",
    "RUN_TICKET_FILE",
    "RUNS_DIR_NAME",
    "RunCreationResult",
    "RunError",
    "RunPreflightError",
    "RunRecord",
    "TicketInputError",
    "create_run_snapshot",
    "format_status",
    "list_run_records",
    "load_baseline_record",
    "load_run_record",
    "sanitize_ticket_id",
    "save_baseline_record",
    "save_run_record",
]
