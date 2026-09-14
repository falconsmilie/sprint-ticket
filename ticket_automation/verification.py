"""Typed deterministic verification with one result per attempt."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from ._verification_artifacts import (
    VERIFICATION_ROUND_FORMAT,
    VERIFICATION_SCHEMA_VERSION,
)
from .attempts import finish_phase_attempt, start_attempt
from .config import AppConfig, VerificationCommand
from .corrections import CorrectionReason, VerificationFailure
from .git import GitRepository
from .git_safety import WorkspaceChange, WorkspaceSnapshot, workspace_safety_changes
from .models import StageOutcome, WorkflowState
from .process_output import decode_human_output
from .resolved_config import config_from_resolved_run_config
from .runs import (
    BASELINE_RECORD_FILE,
    RUN_RECORD_FILE,
    RUN_TICKET_FILE,
    BaselineRecord,
    RunError,
    RunRecord,
    load_baseline_record,
    load_run_record,
)


class VerificationError(RunError):
    pass


class VerificationStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"


class VerificationErrorKind(StrEnum):
    EXECUTABLE_UNAVAILABLE = "EXECUTABLE_UNAVAILABLE"
    PROCESS_START_FAILED = "PROCESS_START_FAILED"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class VerificationProcessCommand:
    argv: tuple[str, ...]
    cwd: Path


@dataclass(frozen=True)
class VerificationProcessResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class VerificationProcessTimeout:
    stdout: str
    stderr: str
    timeout_seconds: float


class VerificationProcessTimedOut(TimeoutError):
    def __init__(self, result: VerificationProcessTimeout):
        super().__init__(
            f"Verification command timed out after {result.timeout_seconds:g} seconds."
        )
        self.result = result


class VerificationProcessRunner(Protocol):
    def run(
        self,
        command: VerificationProcessCommand,
        *,
        timeout_seconds: float | None,
    ) -> VerificationProcessResult: ...


class SubprocessVerificationRunner:
    def run(
        self,
        command: VerificationProcessCommand,
        *,
        timeout_seconds: float | None,
    ) -> VerificationProcessResult:
        try:
            process = subprocess.run(
                command.argv,
                cwd=command.cwd,
                check=False,
                capture_output=True,
                text=False,
                timeout=timeout_seconds,
                shell=False,
            )
        except subprocess.TimeoutExpired as error:
            raise VerificationProcessTimedOut(
                VerificationProcessTimeout(
                    stdout=_process_text(error.stdout),
                    stderr=_process_text(error.stderr),
                    timeout_seconds=float(timeout_seconds or 0),
                )
            ) from error
        return VerificationProcessResult(
            returncode=process.returncode,
            stdout=decode_human_output(process.stdout),
            stderr=decode_human_output(process.stderr),
        )


@dataclass(frozen=True)
class VerificationSafetyViolation:
    name: str
    expected: str
    actual: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "expected": self.expected,
            "actual": self.actual,
            "message": self.message,
        }


@dataclass(frozen=True)
class VerificationCommandResult:
    name: str
    argv: tuple[str, ...]
    cwd: Path
    started_at: str
    ended_at: str
    duration_seconds: float
    status: VerificationStatus
    exit_code: int | None
    stdout: str
    stderr: str
    error_kind: VerificationErrorKind | None = None
    error_message: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASS

    @property
    def failed(self) -> bool:
        return self.status == VerificationStatus.FAIL

    @property
    def errored(self) -> bool:
        return self.status == VerificationStatus.ERROR

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "argv": list(self.argv),
            "cwd": str(self.cwd),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "status": self.status.value,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error_kind": None if self.error_kind is None else self.error_kind.value,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class VerificationRound:
    round_index: int
    started_at: str
    ended_at: str
    duration_seconds: float
    status: VerificationStatus
    commands: tuple[VerificationCommandResult, ...]
    safety_violations: tuple[VerificationSafetyViolation, ...]
    correction_reasons: tuple[CorrectionReason, ...]
    result_path: Path
    schema_version: int = VERIFICATION_SCHEMA_VERSION
    format: str = VERIFICATION_ROUND_FORMAT

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASS

    @property
    def failed_commands(self) -> tuple[VerificationCommandResult, ...]:
        return tuple(item for item in self.commands if item.failed)

    @property
    def errored_commands(self) -> tuple[VerificationCommandResult, ...]:
        return tuple(item for item in self.commands if item.errored)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "format": self.format,
            "round_index": self.round_index,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "status": self.status.value,
            "commands": [item.to_dict() for item in self.commands],
            "safety_violations": [item.to_dict() for item in self.safety_violations],
            "correction_reasons": [item.to_dict() for item in self.correction_reasons],
        }


@dataclass(frozen=True)
class VerificationStageResult:
    run_dir: Path
    run_record: RunRecord
    outcome: StageOutcome
    artifact_directory: Path
    round_result: VerificationRound
    controller_message: str

    @property
    def successful(self) -> bool:
        return self.outcome == StageOutcome.COMPLETED


def _run_baseline_verification_stage(
    config: AppConfig,
    run_dir: Path | str,
    *,
    process_runner: VerificationProcessRunner | None = None,
    clock: Callable[[], datetime] | None = None,
) -> VerificationStageResult:
    run_path = Path(run_dir)
    run_record = load_run_record(run_path / RUN_RECORD_FILE)
    # The baseline is part of the same immutable run contract as later
    # verification. The caller's current configuration cannot alter it.
    del config
    config = config_from_resolved_run_config(run_record.resolved_config)
    if run_record.state != WorkflowState.PREPARING:
        raise VerificationError("Baseline verification requires PREPARING state.")
    baseline = load_baseline_record(run_path / BASELINE_RECORD_FILE)
    repository = GitRepository(Path(run_record.target_repository_path))
    try:
        snapshot = WorkspaceSnapshot.capture(repository)
        violations = _baseline_violations(run_path, run_record, baseline, snapshot)
        before = snapshot.fingerprint
    except (OSError, RuntimeError, ValueError) as error:
        snapshot = None
        before = None
        violations = (
            VerificationSafetyViolation(
                "repository-inspection",
                "complete clean-baseline inspection",
                f"{type(error).__name__}: {error}",
                "Clean baseline repository inspection failed.",
            ),
        )
    attempt = start_attempt(
        run_path,
        phase=WorkflowState.PREPARING.value,
        before_workspace_fingerprint=before,
        clock=clock,
    )
    result_path = attempt.artifact_directory / "result.json"
    if violations:
        round_result = _controller_error_round(result_path, 0, violations, clock=clock)
    else:
        assert snapshot is not None
        round_result = _run_round(
            config.verification.commands,
            cwd=repository.path,
            round_index=0,
            result_path=result_path,
            repository=repository,
            before_snapshot=snapshot,
            include_correction_reasons=False,
            process_runner=process_runner,
            clock=clock,
        )
    return _finish_stage(
        run_path,
        run_record,
        attempt_phase=WorkflowState.PREPARING.value,
        artifact_directory=attempt.artifact_directory,
        round_result=round_result,
        baseline=True,
    )


def run_verification_stage(
    config: AppConfig,
    run_dir: Path | str,
    *,
    process_runner: VerificationProcessRunner | None = None,
    round_index: int | None = None,
    clock: Callable[[], datetime] | None = None,
) -> VerificationStageResult:
    run_path = Path(run_dir)
    run_record = load_run_record(run_path / RUN_RECORD_FILE)
    config = config_from_resolved_run_config(run_record.resolved_config)
    if run_record.state != WorkflowState.VERIFYING:
        raise VerificationError("Verification requires VERIFYING state.")
    repository = GitRepository(Path(run_record.target_repository_path))
    selected_round = (
        run_record.current_correction_round if round_index is None else round_index
    )
    try:
        snapshot = WorkspaceSnapshot.capture(repository)
        before = snapshot.fingerprint
        violations = _verification_violations(repository, run_record, snapshot)
    except (OSError, RuntimeError, ValueError) as error:
        snapshot = None
        before = None
        violations = (
            VerificationSafetyViolation(
                "repository-inspection",
                "complete pre-verification inspection",
                f"{type(error).__name__}: {error}",
                "Repository inspection failed before deterministic verification.",
            ),
        )
    attempt = start_attempt(
        run_path,
        phase=WorkflowState.VERIFYING.value,
        before_workspace_fingerprint=before,
        clock=clock,
    )
    result_path = attempt.artifact_directory / "result.json"
    if violations:
        round_result = _controller_error_round(
            result_path, selected_round, violations, clock=clock
        )
    else:
        assert snapshot is not None
        round_result = _run_round(
            config.verification.commands,
            cwd=repository.path,
            round_index=selected_round,
            result_path=result_path,
            repository=repository,
            before_snapshot=snapshot,
            include_correction_reasons=True,
            process_runner=process_runner,
            clock=clock,
        )
    return _finish_stage(
        run_path,
        run_record,
        attempt_phase=WorkflowState.VERIFYING.value,
        artifact_directory=attempt.artifact_directory,
        round_result=round_result,
        baseline=False,
    )


def _run_round(
    commands: tuple[VerificationCommand, ...],
    *,
    cwd: Path,
    round_index: int,
    result_path: Path,
    repository: GitRepository | None,
    before_snapshot: WorkspaceSnapshot | None,
    include_correction_reasons: bool,
    process_runner: VerificationProcessRunner | None,
    clock: Callable[[], datetime] | None,
) -> VerificationRound:
    started = _utcnow(clock)
    runner = process_runner or SubprocessVerificationRunner()
    commands_result = tuple(
        _run_command(command, cwd, runner, clock) for command in commands
    )
    violations = _after_violations(repository, before_snapshot)
    ended = _utcnow(clock)
    result = VerificationRound(
        round_index=round_index,
        started_at=_timestamp(started),
        ended_at=_timestamp(ended),
        duration_seconds=(ended - started).total_seconds(),
        status=_round_status(commands_result, violations),
        commands=commands_result,
        safety_violations=violations,
        correction_reasons=(
            _correction_reasons(commands_result, result_path)
            if include_correction_reasons
            else ()
        ),
        result_path=result_path,
    )
    _write_json(result_path, result.to_dict())
    return result


def _finish_stage(
    run_path: Path,
    run_record: RunRecord,
    *,
    attempt_phase: str,
    artifact_directory: Path,
    round_result: VerificationRound,
    baseline: bool,
) -> VerificationStageResult:
    if round_result.safety_violations or round_result.errored_commands:
        outcome = StageOutcome.HUMAN_REQUIRED
        message = (
            "Verification could not complete without a safety or infrastructure error."
        )
    elif baseline and (round_result.failed_commands or not round_result.passed):
        outcome = StageOutcome.HUMAN_REQUIRED
        message = "The clean repository baseline failed deterministic verification."
    elif round_result.failed_commands:
        outcome = StageOutcome.CORRECTION_REQUIRED
        message = "Verification failed; corrective work is required."
    else:
        outcome = StageOutcome.COMPLETED
        message = (
            "Baseline verification passed."
            if baseline
            else "Verification passed; review can start."
        )
    try:
        after = WorkspaceSnapshot.capture(
            GitRepository(Path(run_record.target_repository_path))
        ).fingerprint
    except (OSError, RuntimeError, ValueError):
        after = None
    finish_phase_attempt(
        run_path,
        phase=attempt_phase,
        stage_outcome=outcome.value,
        after_workspace_fingerprint=after,
        process_started=_round_process_started(round_result.commands),
    )
    return VerificationStageResult(
        run_dir=run_path,
        run_record=run_record,
        outcome=outcome,
        artifact_directory=artifact_directory,
        round_result=round_result,
        controller_message=message,
    )


def _controller_error_round(
    result_path: Path,
    round_index: int,
    violations: tuple[VerificationSafetyViolation, ...],
    *,
    clock: Callable[[], datetime] | None,
) -> VerificationRound:
    now = _utcnow(clock)
    result = VerificationRound(
        round_index=round_index,
        started_at=_timestamp(now),
        ended_at=_timestamp(now),
        duration_seconds=0.0,
        status=VerificationStatus.ERROR,
        commands=(),
        safety_violations=violations,
        correction_reasons=(),
        result_path=result_path,
    )
    _write_json(result_path, result.to_dict())
    return result


def _round_process_started(
    commands: tuple[VerificationCommandResult, ...],
) -> bool:
    return any(
        command.status != VerificationStatus.ERROR
        or command.error_kind == VerificationErrorKind.TIMEOUT
        for command in commands
    )


def _run_command(
    command: VerificationCommand,
    cwd: Path,
    runner: VerificationProcessRunner,
    clock: Callable[[], datetime] | None,
) -> VerificationCommandResult:
    started = _utcnow(clock)
    try:
        process = runner.run(
            VerificationProcessCommand(command.argv, cwd),
            timeout_seconds=command.timeout_seconds,
        )
    except FileNotFoundError as error:
        return _command_error(
            command,
            cwd,
            started,
            VerificationErrorKind.EXECUTABLE_UNAVAILABLE,
            str(error),
            clock,
        )
    except VerificationProcessTimedOut as error:
        return _command_error(
            command,
            cwd,
            started,
            VerificationErrorKind.TIMEOUT,
            str(error),
            clock,
            stdout=error.result.stdout,
            stderr=error.result.stderr,
        )
    except OSError as error:
        return _command_error(
            command,
            cwd,
            started,
            VerificationErrorKind.PROCESS_START_FAILED,
            str(error),
            clock,
        )
    ended = _utcnow(clock)
    return VerificationCommandResult(
        command.name,
        command.argv,
        cwd,
        _timestamp(started),
        _timestamp(ended),
        (ended - started).total_seconds(),
        VerificationStatus.PASS if process.returncode == 0 else VerificationStatus.FAIL,
        process.returncode,
        process.stdout,
        process.stderr,
    )


def _command_error(
    command: VerificationCommand,
    cwd: Path,
    started: datetime,
    kind: VerificationErrorKind,
    message: str,
    clock: Callable[[], datetime] | None,
    *,
    stdout: str = "",
    stderr: str = "",
) -> VerificationCommandResult:
    ended = _utcnow(clock)
    return VerificationCommandResult(
        command.name,
        command.argv,
        cwd,
        _timestamp(started),
        _timestamp(ended),
        (ended - started).total_seconds(),
        VerificationStatus.ERROR,
        None,
        stdout,
        stderr,
        kind,
        message,
    )


def _baseline_violations(
    run_path: Path,
    record: RunRecord,
    baseline: BaselineRecord,
    snapshot: WorkspaceSnapshot,
) -> tuple[VerificationSafetyViolation, ...]:
    changes = list(
        workspace_safety_changes(
            snapshot,
            expected_repository_path=record.target_repository_path,
            expected_branch=record.starting_branch,
            expected_head_sha=record.baseline_sha,
            require_clean_worktree=True,
        )
    )
    if snapshot.fingerprint != baseline.workspace_fingerprint:
        changes.append(
            WorkspaceChange(
                "baseline-workspace",
                baseline.workspace_fingerprint,
                snapshot.fingerprint,
                "Current workspace does not match the clean baseline.",
            )
        )
    if not (run_path / RUN_TICKET_FILE).is_file():
        changes.append(
            WorkspaceChange(
                "ticket", "snapshotted ticket", "missing", "Ticket snapshot is missing."
            )
        )
    return _as_violations(tuple(changes))


def _verification_violations(
    repository: GitRepository,
    record: RunRecord,
    snapshot: WorkspaceSnapshot,
) -> tuple[VerificationSafetyViolation, ...]:
    del repository
    return _as_violations(
        workspace_safety_changes(
            snapshot,
            expected_repository_path=record.target_repository_path,
            expected_branch=record.starting_branch,
            expected_head_sha=record.baseline_sha,
        )
    )


def _after_violations(
    repository: GitRepository | None,
    before: WorkspaceSnapshot | None,
) -> tuple[VerificationSafetyViolation, ...]:
    if repository is None or before is None:
        return ()
    try:
        return _as_violations(before.compare(WorkspaceSnapshot.capture(repository)))
    except (OSError, RuntimeError, ValueError) as error:
        return (
            VerificationSafetyViolation(
                "repository-inspection",
                "complete post-verification inspection",
                f"{type(error).__name__}: {error}",
                "Repository inspection failed after verification.",
            ),
        )


def _as_violations(
    changes: tuple[WorkspaceChange, ...],
) -> tuple[VerificationSafetyViolation, ...]:
    return tuple(
        VerificationSafetyViolation(item.name, item.expected, item.actual, item.message)
        for item in changes
    )


def _round_status(
    commands: tuple[VerificationCommandResult, ...],
    violations: tuple[VerificationSafetyViolation, ...],
) -> VerificationStatus:
    if violations or any(item.errored for item in commands):
        return VerificationStatus.ERROR
    if any(item.failed for item in commands):
        return VerificationStatus.FAIL
    return VerificationStatus.PASS


def _correction_reasons(
    commands: tuple[VerificationCommandResult, ...], result_path: Path
) -> tuple[CorrectionReason, ...]:
    return tuple(
        VerificationFailure(
            gate_name=item.name,
            command=item.argv,
            failure_summary=f"Verification gate {item.name!r} exited with code {item.exit_code}.",
            stdout_excerpt=_excerpt(item.stdout),
            stderr_excerpt=_excerpt(item.stderr),
            exit_code=item.exit_code,
            result_path=result_path,
        )
        for item in commands
        if item.failed
    )


def _excerpt(value: str, limit: int = 4000) -> str:
    return value if len(value) <= limit else value[:limit] + "\n... <truncated>"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _process_text(value: bytes | str | None) -> str:
    if value is None:
        return ""
    return decode_human_output(value) if isinstance(value, bytes) else value


def _utcnow(clock: Callable[[], datetime] | None) -> datetime:
    value = datetime.now(UTC) if clock is None else clock()
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _timestamp(value: datetime) -> str:
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def format_verification_result(result: VerificationStageResult) -> str:
    return "\n".join(
        [
            f"Verification state: {result.run_record.state.value}",
            f"Artifacts: {result.artifact_directory}",
            result.controller_message,
        ]
    )


__all__ = [
    "SubprocessVerificationRunner",
    "VerificationCommandResult",
    "VerificationError",
    "VerificationErrorKind",
    "VerificationProcessCommand",
    "VerificationProcessResult",
    "VerificationProcessRunner",
    "VerificationProcessTimedOut",
    "VerificationProcessTimeout",
    "VerificationRound",
    "VerificationSafetyViolation",
    "VerificationStageResult",
    "VerificationStatus",
    "format_verification_result",
    "run_verification_stage",
]
