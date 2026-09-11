from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Protocol

from .config import AppConfig, VerificationCommand
from .corrections import CorrectionReason, VerificationFailure
from .git import GitCommandError, GitRepository
from .models import WorkflowState
from .runs import (
    RUN_RECORD_FILE,
    RunError,
    RunRecord,
    load_run_record,
    save_run_record,
)


VERIFICATION_SCHEMA_VERSION = 1
VERIFICATION_ROUND_FORMAT = "ticket_automation.verification_round"
VERIFICATION_DIR_NAME = "verification"
CORRECTION_EXCERPT_CHARS = 4000


class VerificationError(RunError):
    """Raised when verification cannot be prepared or persisted."""


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
    ) -> VerificationProcessResult:
        ...


class SubprocessVerificationRunner:
    def run(
        self,
        command: VerificationProcessCommand,
        *,
        timeout_seconds: float | None,
    ) -> VerificationProcessResult:
        try:
            completed = subprocess.run(
                command.argv,
                cwd=command.cwd,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
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
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
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
    json_path: Path
    log_path: Path
    schema_version: int = VERIFICATION_SCHEMA_VERSION
    format: str = VERIFICATION_ROUND_FORMAT

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASS

    @property
    def failed_commands(self) -> tuple[VerificationCommandResult, ...]:
        return tuple(command for command in self.commands if command.failed)

    @property
    def errored_commands(self) -> tuple[VerificationCommandResult, ...]:
        return tuple(command for command in self.commands if command.errored)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "format": self.format,
            "round_index": self.round_index,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "status": self.status.value,
            "commands": [command.to_dict() for command in self.commands],
            "safety_violations": [
                violation.to_dict() for violation in self.safety_violations
            ],
            "correction_reasons": [
                reason.to_dict() for reason in self.correction_reasons
            ],
        }


@dataclass(frozen=True)
class VerificationStageResult:
    run_dir: Path
    run_record: RunRecord
    artifact_directory: Path
    round_result: VerificationRound
    controller_message: str

    @property
    def successful(self) -> bool:
        return self.run_record.state == WorkflowState.VERIFY


def run_verification_stage(
    config: AppConfig,
    run_dir: Path | str,
    *,
    process_runner: VerificationProcessRunner | None = None,
    round_index: int | None = None,
    clock: Callable[[], datetime] | None = None,
) -> VerificationStageResult:
    run_path = Path(run_dir)
    run_record_path = run_path / RUN_RECORD_FILE
    run_record = load_run_record(run_record_path)
    if run_record.state != WorkflowState.IMPLEMENT:
        raise VerificationError(
            f"Verification requires run state IMPLEMENT; found {run_record.state.value}."
        )

    selected_round = (
        run_record.current_correction_round if round_index is None else round_index
    )
    if selected_round < 0:
        raise VerificationError("Verification round index must not be negative.")

    repository = GitRepository(Path(run_record.target_repository_path))
    artifact_directory = run_path / VERIFICATION_DIR_NAME
    round_name = f"round-{selected_round}"
    json_path = artifact_directory / f"{round_name}.json"
    log_path = artifact_directory / f"{round_name}.log"
    if json_path.exists() or log_path.exists():
        raise VerificationError(
            f"Verification artifacts already exist for {round_name}."
        )

    runner = process_runner or SubprocessVerificationRunner()
    repository_snapshot, starting_violations = _capture_starting_repository_snapshot(
        repository,
        run_record,
    )
    if starting_violations:
        round_result = _write_controller_error_round(
            round_index=selected_round,
            json_path=json_path,
            log_path=log_path,
            safety_violations=starting_violations,
            clock=clock,
        )
        updated_record = run_record.with_state(
            WorkflowState.HUMAN_REQUIRED,
            updated_timestamp=_timestamp(clock),
        )
        save_run_record(updated_record, run_record_path)
        return VerificationStageResult(
            run_dir=run_path,
            run_record=updated_record,
            artifact_directory=artifact_directory,
            round_result=round_result,
            controller_message=(
                "Verification could not start from the recorded implementation state."
            ),
        )

    round_result = run_verification_round(
        config.verification.commands,
        cwd=repository.path,
        round_index=selected_round,
        json_path=json_path,
        log_path=log_path,
        process_runner=runner,
        repository=repository,
        repository_snapshot=repository_snapshot,
        baseline_sha=run_record.baseline_sha,
        clock=clock,
    )

    if round_result.safety_violations:
        state = WorkflowState.HUMAN_REQUIRED
        controller_message = (
            "Verification changed repository state; human intervention is required."
        )
    elif round_result.errored_commands:
        state = WorkflowState.HUMAN_REQUIRED
        controller_message = (
            "Verification could not execute completely; "
            "human intervention is required."
        )
    elif round_result.failed_commands:
        state = WorkflowState.CORRECT
        controller_message = "Verification failed; corrective work is required."
    else:
        state = WorkflowState.VERIFY
        controller_message = "Verification passed; review can start."

    updated_record = run_record.with_state(state, updated_timestamp=_timestamp(clock))
    save_run_record(updated_record, run_record_path)
    return VerificationStageResult(
        run_dir=run_path,
        run_record=updated_record,
        artifact_directory=artifact_directory,
        round_result=round_result,
        controller_message=controller_message,
    )


def run_verification_round(
    commands: tuple[VerificationCommand, ...],
    *,
    cwd: Path,
    round_index: int,
    json_path: Path,
    log_path: Path,
    process_runner: VerificationProcessRunner | None = None,
    repository: GitRepository | None = None,
    repository_snapshot: "_RepositoryVerificationSnapshot | None" = None,
    baseline_sha: str | None = None,
    clock: Callable[[], datetime] | None = None,
) -> VerificationRound:
    runner = process_runner or SubprocessVerificationRunner()
    started = _utcnow(clock)
    command_results = tuple(
        _run_command(command, cwd=cwd, process_runner=runner, clock=clock)
        for command in commands
    )
    ended = _utcnow(clock)
    safety_violations = _verification_safety_violations(
        repository=repository,
        repository_snapshot=repository_snapshot,
        baseline_sha=baseline_sha,
    )
    status = _round_status(command_results, safety_violations=safety_violations)
    correction_reasons = _correction_reasons(command_results, log_path=log_path)
    round_result = VerificationRound(
        round_index=round_index,
        started_at=_format_timestamp(started),
        ended_at=_format_timestamp(ended),
        duration_seconds=_duration_seconds(started, ended),
        status=status,
        commands=command_results,
        safety_violations=safety_violations,
        correction_reasons=correction_reasons,
        json_path=json_path,
        log_path=log_path,
    )
    _write_json(json_path, round_result.to_dict())
    _write_log(log_path, round_result)
    return round_result


def format_verification_result(result: VerificationStageResult) -> str:
    rows = [
        f"Verification state: {result.run_record.state.value}",
        f"Artifacts: {result.artifact_directory}",
        result.controller_message,
        f"Round: round-{result.round_result.round_index}",
    ]
    rows.extend(
        (
            f"  - {command.name}: {command.status.value}"
            f" (exit {command.exit_code if command.exit_code is not None else 'n/a'},"
            f" {command.duration_seconds:.3f}s)"
        )
        for command in result.round_result.commands
    )
    if result.round_result.correction_reasons:
        rows.append(
            f"Correction reasons: {len(result.round_result.correction_reasons)}"
        )
    return "\n".join(rows)


def _run_command(
    command: VerificationCommand,
    *,
    cwd: Path,
    process_runner: VerificationProcessRunner,
    clock: Callable[[], datetime] | None,
) -> VerificationCommandResult:
    started = _utcnow(clock)
    process_command = VerificationProcessCommand(argv=command.argv, cwd=cwd)
    try:
        process = process_runner.run(
            process_command,
            timeout_seconds=command.timeout_seconds,
        )
    except FileNotFoundError as error:
        return _command_error(
            command,
            cwd=cwd,
            started=started,
            clock=clock,
            kind=VerificationErrorKind.EXECUTABLE_UNAVAILABLE,
            message=f"Executable unavailable: {command.argv[0]}",
            stderr=f"{error}\n",
        )
    except VerificationProcessTimedOut as error:
        return _command_error(
            command,
            cwd=cwd,
            started=started,
            clock=clock,
            kind=VerificationErrorKind.TIMEOUT,
            message=str(error),
            stdout=error.result.stdout,
            stderr=error.result.stderr,
        )
    except OSError as error:
        return _command_error(
            command,
            cwd=cwd,
            started=started,
            clock=clock,
            kind=VerificationErrorKind.PROCESS_START_FAILED,
            message=f"Could not start verification command: {error}",
            stderr=f"{error}\n",
        )

    ended = _utcnow(clock)
    status = (
        VerificationStatus.PASS
        if process.returncode == 0
        else VerificationStatus.FAIL
    )
    return VerificationCommandResult(
        name=command.name,
        argv=command.argv,
        cwd=cwd,
        started_at=_format_timestamp(started),
        ended_at=_format_timestamp(ended),
        duration_seconds=_duration_seconds(started, ended),
        status=status,
        exit_code=process.returncode,
        stdout=process.stdout,
        stderr=process.stderr,
    )


def _command_error(
    command: VerificationCommand,
    *,
    cwd: Path,
    started: datetime,
    clock: Callable[[], datetime] | None,
    kind: VerificationErrorKind,
    message: str,
    stdout: str = "",
    stderr: str = "",
) -> VerificationCommandResult:
    ended = _utcnow(clock)
    return VerificationCommandResult(
        name=command.name,
        argv=command.argv,
        cwd=cwd,
        started_at=_format_timestamp(started),
        ended_at=_format_timestamp(ended),
        duration_seconds=_duration_seconds(started, ended),
        status=VerificationStatus.ERROR,
        exit_code=None,
        stdout=stdout,
        stderr=stderr,
        error_kind=kind,
        error_message=message,
    )


def _round_status(
    command_results: tuple[VerificationCommandResult, ...],
    *,
    safety_violations: tuple[VerificationSafetyViolation, ...],
) -> VerificationStatus:
    if safety_violations:
        return VerificationStatus.ERROR
    if any(command.errored for command in command_results):
        return VerificationStatus.ERROR
    if any(command.failed for command in command_results):
        return VerificationStatus.FAIL
    return VerificationStatus.PASS


def _correction_reasons(
    command_results: tuple[VerificationCommandResult, ...],
    *,
    log_path: Path,
) -> tuple[CorrectionReason, ...]:
    return tuple(
        VerificationFailure(
            gate_name=command.name,
            command=command.argv,
            failure_summary=(
                f"Verification gate {command.name!r} exited with code "
                f"{command.exit_code}."
            ),
            stdout_excerpt=_excerpt(command.stdout),
            stderr_excerpt=_excerpt(command.stderr),
            exit_code=command.exit_code,
            log_path=log_path,
        )
        for command in command_results
        if command.failed
    )


@dataclass(frozen=True)
class _RepositoryVerificationSnapshot:
    branch: str | None
    head_sha: str
    staged_files: tuple[str, ...]
    tracked_diff: str
    untracked_files: tuple[str, ...]
    untracked_hashes: tuple[tuple[str, str], ...]

    @classmethod
    def capture(
        cls,
        repository: GitRepository,
        *,
        baseline_sha: str,
    ) -> "_RepositoryVerificationSnapshot":
        untracked_files = repository.untracked_files()
        return cls(
            branch=repository.current_branch(),
            head_sha=repository.head_sha(),
            staged_files=repository.staged_files(),
            tracked_diff=repository.diff(baseline_sha),
            untracked_files=untracked_files,
            untracked_hashes=_hash_untracked_files(repository.path, untracked_files),
        )

    def compare(
        self,
        repository: GitRepository,
        *,
        baseline_sha: str,
    ) -> tuple[VerificationSafetyViolation, ...]:
        current = _RepositoryVerificationSnapshot.capture(
            repository,
            baseline_sha=baseline_sha,
        )
        violations: list[VerificationSafetyViolation] = []

        if self.branch != current.branch:
            violations.append(
                VerificationSafetyViolation(
                    name="branch",
                    expected=_format_optional(self.branch),
                    actual=_format_optional(current.branch),
                    message="Repository branch changed during verification.",
                )
            )
        if self.head_sha != current.head_sha:
            violations.append(
                VerificationSafetyViolation(
                    name="HEAD",
                    expected=self.head_sha,
                    actual=current.head_sha,
                    message="Repository HEAD changed during verification.",
                )
            )
        if self.staged_files != current.staged_files:
            violations.append(
                VerificationSafetyViolation(
                    name="staging",
                    expected=_format_files(self.staged_files),
                    actual=_format_files(current.staged_files),
                    message="Repository staging area changed during verification.",
                )
            )
        if self.tracked_diff != current.tracked_diff:
            violations.append(
                VerificationSafetyViolation(
                    name="tracked-diff",
                    expected="unchanged",
                    actual="changed",
                    message="Tracked worktree diff changed during verification.",
                )
            )
        if self.untracked_files != current.untracked_files:
            violations.append(
                VerificationSafetyViolation(
                    name="untracked-files",
                    expected=_format_files(self.untracked_files),
                    actual=_format_files(current.untracked_files),
                    message="Untracked files changed during verification.",
                )
            )
        elif self.untracked_hashes != current.untracked_hashes:
            violations.append(
                VerificationSafetyViolation(
                    name="untracked-content",
                    expected="unchanged",
                    actual="changed",
                    message="Untracked file content changed during verification.",
                )
            )

        return tuple(violations)


def _capture_starting_repository_snapshot(
    repository: GitRepository,
    run_record: RunRecord,
) -> tuple[
    _RepositoryVerificationSnapshot | None,
    tuple[VerificationSafetyViolation, ...],
]:
    try:
        snapshot = _RepositoryVerificationSnapshot.capture(
            repository,
            baseline_sha=run_record.baseline_sha,
        )
    except (GitCommandError, OSError, ValueError) as error:
        return None, (
            VerificationSafetyViolation(
                name="git-inspection",
                expected="repository inspection succeeds",
                actual=str(error),
                message="Could not inspect repository before verification.",
            ),
        )

    violations: list[VerificationSafetyViolation] = []
    if snapshot.branch != run_record.starting_branch:
        violations.append(
            VerificationSafetyViolation(
                name="branch",
                expected=run_record.starting_branch,
                actual=_format_optional(snapshot.branch),
                message="Repository branch no longer matches the implementation run.",
            )
        )
    if snapshot.head_sha != run_record.baseline_sha:
        violations.append(
            VerificationSafetyViolation(
                name="HEAD",
                expected=run_record.baseline_sha,
                actual=snapshot.head_sha,
                message="Repository HEAD no longer matches the implementation run.",
            )
        )
    if snapshot.staged_files:
        violations.append(
            VerificationSafetyViolation(
                name="staging",
                expected="empty",
                actual=_format_files(snapshot.staged_files),
                message="Repository staging area is not empty before verification.",
            )
        )
    return snapshot, tuple(violations)


def _verification_safety_violations(
    *,
    repository: GitRepository | None,
    repository_snapshot: _RepositoryVerificationSnapshot | None,
    baseline_sha: str | None,
) -> tuple[VerificationSafetyViolation, ...]:
    if (
        repository is None
        or repository_snapshot is None
        or baseline_sha is None
    ):
        return ()
    try:
        return repository_snapshot.compare(repository, baseline_sha=baseline_sha)
    except (GitCommandError, OSError, ValueError) as error:
        return (
            VerificationSafetyViolation(
                name="git-inspection",
                expected="repository inspection succeeds",
                actual=str(error),
                message="Could not inspect repository after verification.",
            ),
        )


def _write_controller_error_round(
    *,
    round_index: int,
    json_path: Path,
    log_path: Path,
    safety_violations: tuple[VerificationSafetyViolation, ...],
    clock: Callable[[], datetime] | None,
) -> VerificationRound:
    started = _utcnow(clock)
    ended = _utcnow(clock)
    round_result = VerificationRound(
        round_index=round_index,
        started_at=_format_timestamp(started),
        ended_at=_format_timestamp(ended),
        duration_seconds=_duration_seconds(started, ended),
        status=VerificationStatus.ERROR,
        commands=(),
        safety_violations=safety_violations,
        correction_reasons=(),
        json_path=json_path,
        log_path=log_path,
    )
    _write_json(json_path, round_result.to_dict())
    _write_log(log_path, round_result)
    return round_result


def _hash_untracked_files(
    repo_path: Path,
    files: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    return tuple((file_path, _file_sha256(repo_path / file_path)) for file_path in files)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_log(path: Path, round_result: VerificationRound) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_format_round_log(round_result), encoding="utf-8", newline="\n")


def _format_round_log(round_result: VerificationRound) -> str:
    lines = [
        f"Verification round: round-{round_result.round_index}",
        f"Status: {round_result.status.value}",
        f"Started: {round_result.started_at}",
        f"Ended: {round_result.ended_at}",
        f"Duration seconds: {round_result.duration_seconds:.6f}",
    ]
    for command in round_result.commands:
        lines.extend(
            [
                "",
                f"Gate: {command.name}",
                f"Status: {command.status.value}",
                f"Argv: {json.dumps(list(command.argv))}",
                f"Cwd: {command.cwd}",
                f"Started: {command.started_at}",
                f"Ended: {command.ended_at}",
                f"Duration seconds: {command.duration_seconds:.6f}",
                (
                    "Exit code: "
                    f"{command.exit_code if command.exit_code is not None else 'n/a'}"
                ),
            ]
        )
        if command.error_kind is not None:
            lines.append(f"Error kind: {command.error_kind.value}")
        if command.error_message:
            lines.append(f"Error message: {command.error_message}")
        lines.extend(
            [
                "Stdout:",
                command.stdout.rstrip("\n"),
                "Stderr:",
                command.stderr.rstrip("\n"),
            ]
        )
    if round_result.safety_violations:
        lines.append("")
        lines.append("Safety violations:")
        for violation in round_result.safety_violations:
            lines.extend(
                [
                    f"  - {violation.name}",
                    f"    Expected: {violation.expected}",
                    f"    Actual: {violation.actual}",
                    f"    Message: {violation.message}",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _format_optional(value: str | None) -> str:
    return "<detached>" if value is None else value


def _format_files(files: tuple[str, ...]) -> str:
    if not files:
        return "empty"
    shown = ", ".join(files[:5])
    hidden_count = len(files) - 5
    if hidden_count > 0:
        shown = f"{shown}, and {hidden_count} more"
    return shown


def _excerpt(value: str) -> str:
    if len(value) <= CORRECTION_EXCERPT_CHARS:
        return value
    omission = len(value) - CORRECTION_EXCERPT_CHARS
    return f"{value[:CORRECTION_EXCERPT_CHARS]}\n... <truncated {omission} chars>"


def _utcnow(clock: Callable[[], datetime] | None) -> datetime:
    now = datetime.now(timezone.utc) if clock is None else clock()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _duration_seconds(start: datetime, end: datetime) -> float:
    return max(0.0, (end - start).total_seconds())


def _timestamp(clock: Callable[[], datetime] | None) -> str:
    return _format_timestamp(_utcnow(clock))


def _process_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


__all__ = [
    "SubprocessVerificationRunner",
    "VERIFICATION_DIR_NAME",
    "VERIFICATION_ROUND_FORMAT",
    "VERIFICATION_SCHEMA_VERSION",
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
    "run_verification_round",
    "run_verification_stage",
]
