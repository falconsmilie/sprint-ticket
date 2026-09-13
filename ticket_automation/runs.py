from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._verification_artifacts import _verification_commands_fingerprint
from .config import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING_EFFORT,
    SUPPORTED_CODEX_REASONING_EFFORTS,
    AppConfig,
    CodexExecutionSettings,
)
from .git import GitCommandError, GitRepository
from .git_safety import WorkspaceSnapshot, workspace_safety_changes
from .models import (
    StopCategory,
    StopReason,
    WorkflowState,
    _validate_workflow_transition,
)
from .preflight import PreflightResult, run_preflight

RUN_SCHEMA_VERSION = 3
BASELINE_SCHEMA_VERSION = 2
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
    ticket_sha256: str
    verification_commands_fingerprint: str
    workspace_fingerprint: str
    snapshot_timestamp: str
    schema_version: int = BASELINE_SCHEMA_VERSION
    format: str = BASELINE_RECORD_FORMAT

    @classmethod
    def capture(
        cls,
        repository: GitRepository,
        *,
        snapshot_timestamp: str,
        ticket_sha256: str,
        verification_commands_fingerprint: str,
    ) -> BaselineRecord:
        snapshot = WorkspaceSnapshot.capture(repository)
        violations = workspace_safety_changes(
            snapshot,
            expected_repository_path=repository.path,
            expected_branch=snapshot.branch,
            expected_head_sha=snapshot.head_sha or "<unknown>",
            require_empty_staging=True,
            require_clean_worktree=True,
        )
        if violations:
            details = "; ".join(violation.message for violation in violations)
            raise RunError(f"Cannot capture repository baseline: {details}")
        if snapshot.branch is None:
            raise RunError("Cannot snapshot a repository in detached HEAD state.")
        assert snapshot.head_sha is not None
        has_staged_files = bool(snapshot.staged_paths)
        return cls(
            repository_path=str(snapshot.repository_path),
            branch=snapshot.branch,
            head_sha=snapshot.head_sha,
            clean_worktree=snapshot.worktree_clean and not has_staged_files,
            has_staged_files=has_staged_files,
            staging_status="dirty" if has_staged_files else "clean",
            ticket_sha256=ticket_sha256,
            verification_commands_fingerprint=verification_commands_fingerprint,
            workspace_fingerprint=snapshot.fingerprint,
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
            "ticket_sha256": self.ticket_sha256,
            "verification_commands_fingerprint": (
                self.verification_commands_fingerprint
            ),
            "workspace_fingerprint": self.workspace_fingerprint,
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
            ticket_sha256=_require_sha256(data, "ticket_sha256"),
            verification_commands_fingerprint=_require_sha256(
                data,
                "verification_commands_fingerprint",
            ),
            workspace_fingerprint=_require_sha256(data, "workspace_fingerprint"),
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
    current_review_round: int
    codex: CodexExecutionSettings
    terminal_reason: str | None
    stop_reason: StopReason | None
    created_timestamp: str
    updated_timestamp: str
    schema_version: int = RUN_SCHEMA_VERSION
    format: str = RUN_RECORD_FORMAT

    def __post_init__(self) -> None:
        """Keep terminal stop evidence inseparable from terminal failure state."""

        is_failure_stop = self.state in {
            WorkflowState.HUMAN_REQUIRED,
            WorkflowState.FAILED,
        }
        if is_failure_stop:
            if self.stop_reason is None:
                raise ValueError(
                    "Terminal HUMAN_REQUIRED and FAILED records require stop_reason."
                )
            if self.terminal_reason != self.stop_reason.message:
                raise ValueError(
                    "Terminal reason must exactly match stop_reason.message."
                )
            return
        if self.stop_reason is not None or self.terminal_reason is not None:
            raise ValueError(
                "Only HUMAN_REQUIRED and FAILED records may contain a terminal stop."
            )

    def transition_to(
        self,
        state: WorkflowState,
        *,
        updated_timestamp: str,
        current_correction_round: int | None = None,
        current_review_round: int | None = None,
        terminal_reason: str | None = None,
        stop_reason: StopReason | None = None,
    ) -> RunRecord:
        _validate_workflow_transition(self.state, state)
        return replace(
            self,
            state=state,
            current_correction_round=(
                self.current_correction_round
                if current_correction_round is None
                else current_correction_round
            ),
            current_review_round=(
                self.current_review_round
                if current_review_round is None
                else current_review_round
            ),
            terminal_reason=terminal_reason,
            stop_reason=stop_reason,
            updated_timestamp=updated_timestamp,
        )

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
            "current_review_round": self.current_review_round,
            "codex": {
                "model": self.codex.model,
                "reasoning_effort": self.codex.reasoning_effort,
            },
            "terminal_reason": self.terminal_reason,
            "stop_reason": (
                None
                if self.stop_reason is None
                else {
                    "category": self.stop_reason.category.value,
                    "message": self.stop_reason.message,
                    "retryable": self.stop_reason.retryable,
                }
            ),
            "created_timestamp": self.created_timestamp,
            "updated_timestamp": self.updated_timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        schema_version = _require_run_schema_version(data)
        format_value = _require_exact_string(
            data,
            "format",
            expected=RUN_RECORD_FORMAT,
        )
        try:
            return cls(
                schema_version=schema_version,
                format=format_value,
                run_id=_require_string(data, "run_id"),
                ticket_id=_require_string(data, "ticket_id"),
                original_ticket_path=_require_string(data, "original_ticket_path"),
                run_ticket_copy_path=_require_string(data, "run_ticket_copy_path"),
                target_repository_path=_require_string(data, "target_repository_path"),
                state=_require_workflow_state(data, "state"),
                starting_branch=_require_string(data, "starting_branch"),
                baseline_sha=_require_string(data, "baseline_sha"),
                current_correction_round=_require_non_negative_int(
                    data,
                    "current_correction_round",
                ),
                max_correction_rounds=_require_int(data, "max_correction_rounds"),
                current_review_round=_optional_non_negative_int(
                    data,
                    "current_review_round",
                    default=0,
                ),
                codex=_optional_codex_execution_settings(data),
                terminal_reason=_optional_nullable_string(data, "terminal_reason"),
                stop_reason=_optional_stop_reason(data),
                created_timestamp=_require_string(data, "created_timestamp"),
                updated_timestamp=_require_string(data, "updated_timestamp"),
            )
        except (TypeError, ValueError) as error:
            raise RunError(f"Invalid terminal stop record: {error}") from error


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
    baseline_record = _capture_baseline(
        repository,
        snapshot_timestamp=timestamp,
        ticket_sha256=hashlib.sha256(source_ticket.contents).hexdigest(),
        verification_commands_fingerprint=_verification_commands_fingerprint(
            config.verification.commands
        ),
    )
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
            state=WorkflowState.PREPARING,
            starting_branch=baseline_record.branch,
            baseline_sha=baseline_record.head_sha,
            current_correction_round=0,
            max_correction_rounds=config.runner.max_correction_rounds,
            current_review_round=0,
            codex=config.codex.execution,
            terminal_reason=None,
            stop_reason=None,
            created_timestamp=timestamp,
            updated_timestamp=timestamp,
        )
        save_run_record(run_record, run_record_path)

        _copy_ticket(source_ticket.contents, run_ticket_path)
        save_baseline_record(baseline_record, baseline_record_path)

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


def format_status(
    records: tuple[RunRecord, ...],
    *,
    runs_dir: Path | str,
    active_ownerships: tuple[object, ...] = (),
) -> str:
    if not records:
        rows = [f"No runs found in {Path(runs_dir)}."]
    else:
        header = (
            f"{'Run ID':<36} {'Ticket ID':<16} {'State':<16} {'Branch':<24} "
            f"{'Corr':<9} {'Review':<8} {'Updated':<20} Reason"
        )
        rows = [header]
        for record in records:
            reason = record.terminal_reason or ""
            correction_round = (
                f"{record.current_correction_round}/{record.max_correction_rounds}"
            )
            rows.append(
                f"{record.run_id:<36} {record.ticket_id:<16} "
                f"{record.state.value:<16} {record.starting_branch:<24} "
                f"{correction_round:<9} "
                f"{record.current_review_round:<8} {record.updated_timestamp:<20} "
                f"{reason}"
            )
    if active_ownerships:
        rows.extend(["", "Active repository ownership"])
        for ownership in active_ownerships:
            run_id = _ownership_value(ownership, "run_id") or "<unknown>"
            state = _ownership_value(ownership, "current_state") or "<unknown>"
            target = (
                _ownership_value(ownership, "target_repository_path") or "<unknown>"
            )
            owner_pid = _ownership_value(ownership, "owner_pid") or "<unknown>"
            owner_host = _ownership_value(ownership, "owner_hostname") or "<unknown>"
            acquired = _ownership_value(ownership, "acquired_timestamp") or "<unknown>"
            rows.append(
                f"  {run_id}: {state} owns {target} "
                f"({owner_host} pid {owner_pid}, acquired {acquired})"
            )
    return "\n".join(rows)


def _ownership_value(ownership: object, key: str) -> object:
    if isinstance(ownership, Mapping):
        return ownership.get(key)
    return getattr(ownership, key, None)


def _timestamp(clock: Callable[[], datetime] | None) -> str:
    now = datetime.now(UTC) if clock is None else clock()
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return now.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_id_prefix(timestamp: str) -> str:
    parsed = datetime.fromisoformat(timestamp)
    return parsed.strftime("%Y%m%d-%H%M%S")


def _reserve_run_directory(
    runs_dir: Path, timestamp: str, ticket_id: str
) -> tuple[str, Path]:
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
    ticket_sha256: str,
    verification_commands_fingerprint: str,
) -> BaselineRecord:
    try:
        return BaselineRecord.capture(
            repository,
            snapshot_timestamp=snapshot_timestamp,
            ticket_sha256=ticket_sha256,
            verification_commands_fingerprint=verification_commands_fingerprint,
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
        raise TicketInputError(
            f"Ticket file is not readable: {path}: {error}"
        ) from error
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
        with os.fdopen(
            file_descriptor, "w", encoding="utf-8", newline="\n"
        ) as temp_file:
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


def _require_non_negative_int(data: dict[str, Any], key: str) -> int:
    value = _require_int(data, key)
    if value < 0:
        raise RunError(f"Run record field must be a non-negative integer: {key}")
    return value


def _optional_int(data: dict[str, Any], key: str, *, default: int) -> int:
    if key not in data:
        return default
    value = data[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise RunError(f"Run record field must be an integer: {key}")
    return value


def _optional_non_negative_int(data: dict[str, Any], key: str, *, default: int) -> int:
    value = _optional_int(data, key, default=default)
    if value < 0:
        raise RunError(f"Run record field must be a non-negative integer: {key}")
    return value


def _optional_nullable_string(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise RunError(f"Run record field must be a non-empty string or null: {key}")
    return value


def _optional_stop_reason(data: dict[str, Any]) -> StopReason | None:
    value = data.get("stop_reason")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RunError("Run record field must be an object or null: stop_reason")
    category_value = _require_nested_string(value, "stop_reason", "category")
    try:
        category = StopCategory(category_value)
    except ValueError as error:
        raise RunError(
            "Run record field has unsupported stop category: stop_reason.category"
        ) from error
    message = _require_nested_string(value, "stop_reason", "message")
    retryable = value.get("retryable")
    if not isinstance(retryable, bool):
        raise RunError("Run record field must be a boolean: stop_reason.retryable")
    return StopReason(category=category, message=message, retryable=retryable)


def _optional_codex_execution_settings(
    data: dict[str, Any],
) -> CodexExecutionSettings:
    value = data.get("codex")
    if value is None:
        return CodexExecutionSettings(
            model=DEFAULT_CODEX_MODEL,
            reasoning_effort=DEFAULT_CODEX_REASONING_EFFORT,
        )
    if not isinstance(value, dict):
        raise RunError("Run record field must be an object: codex")
    model = _require_nested_string(value, "codex", "model")
    reasoning_effort = _require_nested_string(value, "codex", "reasoning_effort")
    if reasoning_effort not in SUPPORTED_CODEX_REASONING_EFFORTS:
        supported = ", ".join(sorted(SUPPORTED_CODEX_REASONING_EFFORTS))
        raise RunError(
            f"Run record field codex.reasoning_effort must be one of: {supported}."
        )
    return CodexExecutionSettings(
        model=model,
        reasoning_effort=reasoning_effort,
    )


def _require_nested_string(data: dict[str, Any], parent: str, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RunError(f"Run record field must be a non-empty string: {parent}.{key}")
    return value.strip()


def _require_workflow_state(data: dict[str, Any], key: str) -> WorkflowState:
    return _workflow_state(_require_string(data, key), key=key)


def _workflow_state(value: str, *, key: str) -> WorkflowState:
    try:
        return WorkflowState(value)
    except ValueError as error:
        raise RunError(
            f"Run record field has unsupported workflow state: {key}"
        ) from error


def _require_run_schema_version(data: dict[str, Any]) -> int:
    value = _require_int(data, "schema_version")
    if value != RUN_SCHEMA_VERSION:
        raise RunError(
            "Unsupported run record schema version: "
            f"{value}; expected {RUN_SCHEMA_VERSION}. Start a new run."
        )
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


def _require_sha256(data: dict[str, Any], key: str) -> str:
    value = _require_string(data, key)
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RunError(f"Run record field must be a lowercase SHA-256 digest: {key}")
    return value


__all__ = [
    "BASELINE_RECORD_FILE",
    "BASELINE_RECORD_FORMAT",
    "BASELINE_SCHEMA_VERSION",
    "RUNS_DIR_NAME",
    "RUN_RECORD_FILE",
    "RUN_RECORD_FORMAT",
    "RUN_SCHEMA_VERSION",
    "RUN_TICKET_FILE",
    "BaselineRecord",
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
