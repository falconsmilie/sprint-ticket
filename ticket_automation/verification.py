from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from ._verification_artifacts import (
    BASELINE_VERIFICATION_DIR_NAME as _BASELINE_VERIFICATION_DIR_NAME,
)
from ._verification_artifacts import (
    BASELINE_VERIFICATION_JSON_FILE as _BASELINE_VERIFICATION_JSON_FILE,
)
from ._verification_artifacts import (
    BASELINE_VERIFICATION_LOG_FILE as _BASELINE_VERIFICATION_LOG_FILE,
)
from ._verification_artifacts import (
    VERIFICATION_CHECKPOINT_FORMAT,
    VERIFICATION_CHECKPOINT_SCHEMA_VERSION,
    VERIFICATION_DIR_NAME,
    VERIFICATION_ROUND_FORMAT,
    VERIFICATION_SCHEMA_VERSION,
    _verification_commands_fingerprint,
)
from .config import AppConfig, VerificationCommand
from .corrections import (
    CorrectionError,
    CorrectionReason,
    VerificationFailure,
    correction_reason_from_dict,
)
from .git import GitRepository
from .git_safety import (
    WorkspaceChange,
    WorkspaceSnapshot,
    workspace_safety_changes,
)
from .models import StageOutcome, WorkflowState
from .process_output import decode_human_output
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

_INCOMPLETE_ARTIFACT_DIR_NAME = "_incomplete"
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


class _VerificationArtifactState(StrEnum):
    MISSING = "MISSING"
    PARTIAL = "PARTIAL"
    COMPLETE = "COMPLETE"


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
            completed = subprocess.run(
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
            returncode=completed.returncode,
            stdout=decode_human_output(completed.stdout),
            stderr=decode_human_output(completed.stderr),
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
    checkpoint: dict[str, Any] | None = None

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
        data = {
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
        if self.checkpoint is not None:
            data["checkpoint"] = self.checkpoint
        return data


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
    if run_record.state != WorkflowState.PREPARING:
        raise VerificationError(
            "Baseline verification requires run state PREPARING; "
            f"found {run_record.state.value}."
        )

    baseline_record = load_baseline_record(run_path / BASELINE_RECORD_FILE)
    repository = GitRepository(Path(run_record.target_repository_path))
    artifact_directory = run_path / _BASELINE_VERIFICATION_DIR_NAME
    json_path = artifact_directory / _BASELINE_VERIFICATION_JSON_FILE
    log_path = artifact_directory / _BASELINE_VERIFICATION_LOG_FILE

    try:
        repository_snapshot = WorkspaceSnapshot.capture(repository)
        starting_violations = _baseline_starting_violations(
            run_path=run_path,
            run_record=run_record,
            baseline_record=baseline_record,
            repository_snapshot=repository_snapshot,
            commands=config.verification.commands,
        )
    except (OSError, RuntimeError, ValueError) as error:
        repository_snapshot = None
        starting_violations = (
            VerificationSafetyViolation(
                name="repository-inspection",
                expected="complete clean-baseline repository inspection",
                actual=f"{type(error).__name__}: {error}",
                message="Clean baseline repository inspection failed.",
            ),
        )

    artifact_state = _verification_artifact_state(json_path, log_path)
    if artifact_state == _VerificationArtifactState.COMPLETE:
        if starting_violations:
            return _finish_unadoptable_verification_checkpoint(
                run_record=run_record,
                run_dir=run_path,
                artifact_directory=artifact_directory,
                json_path=json_path,
                log_path=log_path,
                round_index=0,
                reason=(
                    "Existing baseline verification evidence cannot be adopted because "
                    "the current workspace does not match the recorded clean baseline."
                ),
                clock=clock,
            )
        assert repository_snapshot is not None
        existing_round, checkpoint_problem = _load_existing_verification_round(
            json_path=json_path,
            log_path=log_path,
            run_record=run_record,
            round_index=0,
            repository_snapshot=repository_snapshot,
            commands=config.verification.commands,
            checkpoint_stage=WorkflowState.PREPARING,
        )
        if checkpoint_problem is None and existing_round is not None:
            round_problem = _baseline_round_problem(
                existing_round,
                commands=config.verification.commands,
                cwd=repository.path,
            )
            if round_problem is not None:
                return _finish_unadoptable_verification_checkpoint(
                    run_record=run_record,
                    run_dir=run_path,
                    artifact_directory=artifact_directory,
                    json_path=json_path,
                    log_path=log_path,
                    round_index=0,
                    reason=round_problem,
                    clock=clock,
                )
            return _finish_baseline_verification_round(
                run_record=run_record,
                run_dir=run_path,
                artifact_directory=artifact_directory,
                round_result=existing_round,
            )
        return _finish_unadoptable_verification_checkpoint(
            run_record=run_record,
            run_dir=run_path,
            artifact_directory=artifact_directory,
            json_path=json_path,
            log_path=log_path,
            round_index=0,
            reason=(
                checkpoint_problem
                or "Baseline verification evidence contains correction reasons."
            ),
            clock=clock,
        )

    if artifact_state == _VerificationArtifactState.PARTIAL:
        if starting_violations:
            return _finish_unadoptable_verification_checkpoint(
                run_record=run_record,
                run_dir=run_path,
                artifact_directory=artifact_directory,
                json_path=json_path,
                log_path=log_path,
                round_index=0,
                reason=(
                    "Partial baseline verification evidence cannot be retried because "
                    "the current workspace does not match the recorded clean baseline."
                ),
                clock=clock,
            )
        _archive_incomplete_verification_artifacts(
            artifact_directory=artifact_directory,
            round_name="baseline",
            json_path=json_path,
            log_path=log_path,
        )

    if starting_violations:
        round_result = _write_controller_error_round(
            run_record=run_record,
            round_index=0,
            json_path=json_path,
            log_path=log_path,
            repository_snapshot=repository_snapshot,
            commands=config.verification.commands,
            safety_violations=starting_violations,
            checkpoint_stage=WorkflowState.PREPARING,
            clock=clock,
        )
        return _finish_baseline_verification_round(
            run_record=run_record,
            run_dir=run_path,
            artifact_directory=artifact_directory,
            round_result=round_result,
        )

    assert repository_snapshot is not None
    round_result = _run_verification_round(
        config.verification.commands,
        cwd=repository.path,
        round_index=0,
        json_path=json_path,
        log_path=log_path,
        process_runner=process_runner,
        repository=repository,
        repository_snapshot=repository_snapshot,
        baseline_sha=run_record.baseline_sha,
        run_record=run_record,
        checkpoint_stage=WorkflowState.PREPARING,
        include_correction_reasons=False,
        include_command_output_in_log=False,
        capture_inspection_errors=True,
        clock=clock,
    )
    return _finish_baseline_verification_round(
        run_record=run_record,
        run_dir=run_path,
        artifact_directory=artifact_directory,
        round_result=round_result,
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
    run_record_path = run_path / RUN_RECORD_FILE
    run_record = load_run_record(run_record_path)
    if run_record.state != WorkflowState.VERIFYING:
        raise VerificationError(
            "Verification requires run state VERIFYING; "
            f"found {run_record.state.value}."
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

    repository_snapshot, starting_violations = _capture_starting_repository_snapshot(
        repository,
        run_record,
    )
    artifact_state = _verification_artifact_state(json_path, log_path)
    if artifact_state == _VerificationArtifactState.COMPLETE:
        if starting_violations:
            return _finish_unadoptable_verification_checkpoint(
                run_record=run_record,
                run_dir=run_path,
                artifact_directory=artifact_directory,
                json_path=json_path,
                log_path=log_path,
                round_index=selected_round,
                reason=(
                    "Existing verification artifacts cannot be adopted because "
                    "repository safety invariants no longer hold."
                ),
                clock=clock,
            )
        assert repository_snapshot is not None
        existing_round, checkpoint_problem = _load_existing_verification_round(
            json_path=json_path,
            log_path=log_path,
            run_record=run_record,
            round_index=selected_round,
            repository_snapshot=repository_snapshot,
            commands=config.verification.commands,
        )
        if checkpoint_problem is None and existing_round is not None:
            return _finish_verification_round(
                run_record=run_record,
                run_dir=run_path,
                artifact_directory=artifact_directory,
                round_result=existing_round,
            )
        return _finish_unadoptable_verification_checkpoint(
            run_record=run_record,
            run_dir=run_path,
            artifact_directory=artifact_directory,
            json_path=json_path,
            log_path=log_path,
            round_index=selected_round,
            reason=checkpoint_problem or "Existing verification checkpoint is invalid.",
            clock=clock,
        )

    if artifact_state == _VerificationArtifactState.PARTIAL:
        if starting_violations:
            return _finish_unadoptable_verification_checkpoint(
                run_record=run_record,
                run_dir=run_path,
                artifact_directory=artifact_directory,
                json_path=json_path,
                log_path=log_path,
                round_index=selected_round,
                reason=(
                    "Partial verification artifacts cannot be retried because "
                    "repository safety invariants no longer hold."
                ),
                clock=clock,
            )
        _archive_incomplete_verification_artifacts(
            artifact_directory=artifact_directory,
            round_name=round_name,
            json_path=json_path,
            log_path=log_path,
        )

    if starting_violations:
        round_result = _write_controller_error_round(
            run_record=run_record,
            round_index=selected_round,
            json_path=json_path,
            log_path=log_path,
            repository_snapshot=repository_snapshot,
            commands=config.verification.commands,
            safety_violations=starting_violations,
            clock=clock,
        )
        return VerificationStageResult(
            run_dir=run_path,
            run_record=run_record,
            outcome=StageOutcome.HUMAN_REQUIRED,
            artifact_directory=artifact_directory,
            round_result=round_result,
            controller_message=(
                "Verification could not start from the recorded implementation state."
            ),
        )

    runner = process_runner or SubprocessVerificationRunner()
    assert repository_snapshot is not None
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
        run_record=run_record,
        clock=clock,
    )

    return _finish_verification_round(
        run_record=run_record,
        run_dir=run_path,
        artifact_directory=artifact_directory,
        round_result=round_result,
    )


def _finish_verification_round(
    *,
    run_record: RunRecord,
    run_dir: Path,
    artifact_directory: Path,
    round_result: VerificationRound,
) -> VerificationStageResult:
    if round_result.safety_violations:
        outcome = StageOutcome.HUMAN_REQUIRED
        controller_message = (
            "Verification changed repository state; human intervention is required."
        )
    elif round_result.errored_commands:
        outcome = StageOutcome.HUMAN_REQUIRED
        controller_message = (
            "Verification could not execute completely; human intervention is required."
        )
    elif round_result.failed_commands:
        outcome = StageOutcome.CORRECTION_REQUIRED
        controller_message = "Verification failed; corrective work is required."
    else:
        outcome = StageOutcome.COMPLETED
        controller_message = "Verification passed; review can start."

    return VerificationStageResult(
        run_dir=run_dir,
        run_record=run_record,
        outcome=outcome,
        artifact_directory=artifact_directory,
        round_result=round_result,
        controller_message=controller_message,
    )


def _finish_baseline_verification_round(
    *,
    run_record: RunRecord,
    run_dir: Path,
    artifact_directory: Path,
    round_result: VerificationRound,
) -> VerificationStageResult:
    if round_result.safety_violations:
        outcome = StageOutcome.HUMAN_REQUIRED
        controller_message = (
            "Baseline verification changed or could not inspect the clean repository; "
            "human intervention is required."
        )
    elif round_result.errored_commands:
        outcome = StageOutcome.HUMAN_REQUIRED
        controller_message = (
            "Baseline verification could not execute completely; human intervention "
            "is required."
        )
    elif round_result.failed_commands:
        outcome = StageOutcome.HUMAN_REQUIRED
        controller_message = (
            "The clean repository baseline failed deterministic verification; human "
            "intervention is required."
        )
    elif round_result.correction_reasons:
        outcome = StageOutcome.HUMAN_REQUIRED
        controller_message = (
            "Baseline verification evidence unexpectedly contains correction reasons; "
            "human intervention is required."
        )
    elif not round_result.passed:
        outcome = StageOutcome.HUMAN_REQUIRED
        controller_message = (
            "Baseline verification evidence is not a complete passing result; human "
            "intervention is required."
        )
    else:
        outcome = StageOutcome.COMPLETED
        controller_message = (
            "The clean repository baseline passed deterministic verification."
        )

    return VerificationStageResult(
        run_dir=run_dir,
        run_record=run_record,
        outcome=outcome,
        artifact_directory=artifact_directory,
        round_result=round_result,
        controller_message=controller_message,
    )


def _baseline_round_problem(
    round_result: VerificationRound,
    *,
    commands: tuple[VerificationCommand, ...],
    cwd: Path,
) -> str | None:
    if round_result.correction_reasons:
        return "Baseline verification evidence contains correction reasons."

    if not round_result.commands:
        if (
            round_result.status == VerificationStatus.ERROR
            and round_result.safety_violations
        ):
            return None
        return "Baseline verification evidence has an unexpected command set."

    if len(round_result.commands) != len(commands):
        return "Baseline verification evidence has an unexpected command set."

    expected_cwd = cwd.resolve()
    for result, command in zip(round_result.commands, commands, strict=True):
        if (
            result.name != command.name
            or result.argv != command.argv
            or result.cwd.resolve() != expected_cwd
        ):
            return (
                "Baseline verification command evidence does not match configuration."
            )
        if result.status == VerificationStatus.PASS:
            internally_consistent = (
                result.exit_code == 0
                and result.error_kind is None
                and result.error_message is None
            )
        elif result.status == VerificationStatus.FAIL:
            internally_consistent = (
                result.exit_code is not None
                and result.exit_code != 0
                and result.error_kind is None
                and result.error_message is None
            )
        else:
            internally_consistent = (
                result.exit_code is None
                and result.error_kind is not None
                and bool(result.error_message)
            )
        if not internally_consistent:
            return "Baseline verification command evidence is internally inconsistent."

    expected_status = _round_status(
        round_result.commands,
        safety_violations=round_result.safety_violations,
    )
    if round_result.status != expected_status:
        return "Baseline verification result status is internally inconsistent."
    return None


def _finish_unadoptable_verification_checkpoint(
    *,
    run_record: RunRecord,
    run_dir: Path,
    artifact_directory: Path,
    json_path: Path,
    log_path: Path,
    round_index: int,
    reason: str,
    clock: Callable[[], datetime] | None,
) -> VerificationStageResult:
    timestamp = _timestamp(clock)
    violation = VerificationSafetyViolation(
        name="verification-checkpoint",
        expected="valid verification checkpoint for this run and repository state",
        actual=reason,
        message="Existing verification artifacts require human inspection.",
    )
    round_result = VerificationRound(
        round_index=round_index,
        started_at=timestamp,
        ended_at=timestamp,
        duration_seconds=0.0,
        status=VerificationStatus.ERROR,
        commands=(),
        safety_violations=(violation,),
        correction_reasons=(),
        json_path=json_path,
        log_path=log_path,
    )
    return VerificationStageResult(
        run_dir=run_dir,
        run_record=run_record,
        outcome=StageOutcome.HUMAN_REQUIRED,
        artifact_directory=artifact_directory,
        round_result=round_result,
        controller_message=reason,
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
    repository_snapshot: WorkspaceSnapshot | None = None,
    baseline_sha: str | None = None,
    run_record: RunRecord | None = None,
    clock: Callable[[], datetime] | None = None,
) -> VerificationRound:
    return _run_verification_round(
        commands,
        cwd=cwd,
        round_index=round_index,
        json_path=json_path,
        log_path=log_path,
        process_runner=process_runner,
        repository=repository,
        repository_snapshot=repository_snapshot,
        baseline_sha=baseline_sha,
        run_record=run_record,
        checkpoint_stage=WorkflowState.VERIFYING,
        include_correction_reasons=True,
        include_command_output_in_log=True,
        capture_inspection_errors=False,
        clock=clock,
    )


def _run_verification_round(
    commands: tuple[VerificationCommand, ...],
    *,
    cwd: Path,
    round_index: int,
    json_path: Path,
    log_path: Path,
    process_runner: VerificationProcessRunner | None,
    repository: GitRepository | None,
    repository_snapshot: WorkspaceSnapshot | None,
    baseline_sha: str | None,
    run_record: RunRecord | None,
    checkpoint_stage: WorkflowState,
    include_correction_reasons: bool,
    include_command_output_in_log: bool,
    capture_inspection_errors: bool,
    clock: Callable[[], datetime] | None,
) -> VerificationRound:
    runner = process_runner or SubprocessVerificationRunner()
    started = _utcnow(clock)
    command_results = tuple(
        _run_command(command, cwd=cwd, process_runner=runner, clock=clock)
        for command in commands
    )
    ended = _utcnow(clock)
    if capture_inspection_errors:
        try:
            safety_violations = _verification_safety_violations(
                repository=repository,
                repository_snapshot=repository_snapshot,
                baseline_sha=baseline_sha,
            )
        except (OSError, RuntimeError, ValueError) as error:
            safety_violations = (
                VerificationSafetyViolation(
                    name="repository-inspection",
                    expected="complete repository inspection after verification",
                    actual=f"{type(error).__name__}: {error}",
                    message="Repository inspection failed after verification.",
                ),
            )
    else:
        safety_violations = _verification_safety_violations(
            repository=repository,
            repository_snapshot=repository_snapshot,
            baseline_sha=baseline_sha,
        )
    status = _round_status(command_results, safety_violations=safety_violations)
    correction_reasons = (
        _correction_reasons(command_results, log_path=log_path)
        if include_correction_reasons
        else ()
    )
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
        checkpoint=(
            None
            if run_record is None
            else _verification_checkpoint(
                run_record,
                round_index=round_index,
                repository_snapshot=repository_snapshot,
                commands=commands,
                stage=checkpoint_stage,
            )
        ),
    )
    _write_json(json_path, round_result.to_dict())
    _write_log(
        log_path,
        round_result,
        include_command_output=include_command_output_in_log,
    )
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
        VerificationStatus.PASS if process.returncode == 0 else VerificationStatus.FAIL
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


def _capture_starting_repository_snapshot(
    repository: GitRepository,
    run_record: RunRecord,
) -> tuple[
    WorkspaceSnapshot | None,
    tuple[VerificationSafetyViolation, ...],
]:
    snapshot = WorkspaceSnapshot.capture(repository)
    changes = workspace_safety_changes(
        snapshot,
        expected_repository_path=run_record.target_repository_path,
        expected_branch=run_record.starting_branch,
        expected_head_sha=run_record.baseline_sha,
    )
    return snapshot, _verification_changes(changes)


def _baseline_starting_violations(
    *,
    run_path: Path,
    run_record: RunRecord,
    baseline_record: BaselineRecord,
    repository_snapshot: WorkspaceSnapshot,
    commands: tuple[VerificationCommand, ...],
) -> tuple[VerificationSafetyViolation, ...]:
    changes = list(
        workspace_safety_changes(
            repository_snapshot,
            expected_repository_path=run_record.target_repository_path,
            expected_branch=run_record.starting_branch,
            expected_head_sha=run_record.baseline_sha,
            require_clean_worktree=True,
        )
    )
    if baseline_record.branch != run_record.starting_branch:
        changes.append(
            WorkspaceChange(
                name="baseline-branch",
                expected=run_record.starting_branch,
                actual=baseline_record.branch,
                message="Run and baseline records disagree on the starting branch.",
            )
        )
    if baseline_record.head_sha != run_record.baseline_sha:
        changes.append(
            WorkspaceChange(
                name="baseline-HEAD",
                expected=run_record.baseline_sha,
                actual=baseline_record.head_sha,
                message="Run and baseline records disagree on the baseline HEAD.",
            )
        )
    if not baseline_record.clean_worktree or baseline_record.has_staged_files:
        changes.append(
            WorkspaceChange(
                name="baseline-cleanliness",
                expected="clean worktree and empty staging area",
                actual=(
                    f"clean_worktree={baseline_record.clean_worktree}, "
                    f"has_staged_files={baseline_record.has_staged_files}"
                ),
                message="The recorded workspace baseline is not clean.",
            )
        )
    if repository_snapshot.fingerprint != baseline_record.workspace_fingerprint:
        changes.append(
            WorkspaceChange(
                name="baseline-workspace",
                expected=baseline_record.workspace_fingerprint,
                actual=repository_snapshot.fingerprint,
                message="Current workspace does not match the recorded clean baseline.",
            )
        )

    commands_fingerprint = _verification_commands_fingerprint(commands)
    if commands_fingerprint != baseline_record.verification_commands_fingerprint:
        changes.append(
            WorkspaceChange(
                name="baseline-verification-config",
                expected=baseline_record.verification_commands_fingerprint,
                actual=commands_fingerprint,
                message="Configured verification commands changed after run creation.",
            )
        )

    try:
        ticket_fingerprint = hashlib.sha256(
            (run_path / RUN_TICKET_FILE).read_bytes()
        ).hexdigest()
    except OSError as error:
        changes.append(
            WorkspaceChange(
                name="baseline-ticket",
                expected=baseline_record.ticket_sha256,
                actual=f"unavailable: {error}",
                message="Could not inspect the snapshotted ticket.",
            )
        )
    else:
        if ticket_fingerprint != baseline_record.ticket_sha256:
            changes.append(
                WorkspaceChange(
                    name="baseline-ticket",
                    expected=baseline_record.ticket_sha256,
                    actual=ticket_fingerprint,
                    message="Snapshotted ticket changed after run creation.",
                )
            )
    return _verification_changes(tuple(changes))


def _verification_safety_violations(
    *,
    repository: GitRepository | None,
    repository_snapshot: WorkspaceSnapshot | None,
    baseline_sha: str | None,
) -> tuple[VerificationSafetyViolation, ...]:
    if repository is None or repository_snapshot is None or baseline_sha is None:
        return ()
    current = WorkspaceSnapshot.capture(repository)
    return _verification_changes(repository_snapshot.compare(current))


def _verification_changes(
    changes: tuple[WorkspaceChange, ...],
) -> tuple[VerificationSafetyViolation, ...]:
    return tuple(
        VerificationSafetyViolation(
            name=change.name,
            expected=change.expected,
            actual=change.actual,
            message=change.message,
        )
        for change in changes
    )


def _write_controller_error_round(
    *,
    run_record: RunRecord,
    round_index: int,
    json_path: Path,
    log_path: Path,
    repository_snapshot: WorkspaceSnapshot | None,
    commands: tuple[VerificationCommand, ...],
    safety_violations: tuple[VerificationSafetyViolation, ...],
    checkpoint_stage: WorkflowState = WorkflowState.VERIFYING,
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
        checkpoint=_verification_checkpoint(
            run_record,
            round_index=round_index,
            repository_snapshot=repository_snapshot,
            commands=commands,
            stage=checkpoint_stage,
        ),
    )
    _write_json(json_path, round_result.to_dict())
    _write_log(log_path, round_result)
    return round_result


def _verification_artifact_state(
    json_path: Path,
    log_path: Path,
) -> _VerificationArtifactState:
    json_exists = json_path.is_file()
    log_exists = log_path.is_file()
    if json_exists and log_exists:
        return _VerificationArtifactState.COMPLETE
    if json_path.exists() or log_path.exists():
        return _VerificationArtifactState.PARTIAL
    return _VerificationArtifactState.MISSING


def _load_existing_verification_round(
    *,
    json_path: Path,
    log_path: Path,
    run_record: RunRecord,
    round_index: int,
    repository_snapshot: WorkspaceSnapshot,
    commands: tuple[VerificationCommand, ...],
    checkpoint_stage: WorkflowState = WorkflowState.VERIFYING,
) -> tuple[VerificationRound | None, str | None]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, f"Could not read existing verification artifact: {error}"
    if not isinstance(data, dict):
        return None, "Existing verification artifact is not a JSON object."

    metadata_problem = _verification_checkpoint_problem(
        data,
        run_record=run_record,
        round_index=round_index,
        repository_snapshot=repository_snapshot,
        commands=commands,
        checkpoint_stage=checkpoint_stage,
    )
    if metadata_problem is not None:
        return None, metadata_problem

    try:
        if log_path.stat().st_size <= 0:
            return None, "Existing verification log is empty."
    except OSError as error:
        return None, f"Could not inspect existing verification log: {error}"

    try:
        return _verification_round_from_dict(
            data, json_path=json_path, log_path=log_path
        ), None
    except (CorrectionError, TypeError, ValueError) as error:
        return None, f"Existing verification artifact is malformed: {error}"


def _verification_checkpoint_problem(
    data: dict[str, Any],
    *,
    run_record: RunRecord,
    round_index: int,
    repository_snapshot: WorkspaceSnapshot,
    commands: tuple[VerificationCommand, ...],
    checkpoint_stage: WorkflowState = WorkflowState.VERIFYING,
) -> str | None:
    if data.get("schema_version") != VERIFICATION_SCHEMA_VERSION:
        return "Existing verification artifact has an unsupported schema version."
    if data.get("format") != VERIFICATION_ROUND_FORMAT:
        return "Existing verification artifact has an unsupported format."
    if data.get("round_index") != round_index:
        return "Existing verification artifact is for a different round."

    checkpoint = data.get("checkpoint")
    if not isinstance(checkpoint, dict):
        return "Existing verification artifact is missing checkpoint metadata."

    expected = _verification_checkpoint(
        run_record,
        round_index=round_index,
        repository_snapshot=repository_snapshot,
        commands=commands,
        stage=checkpoint_stage,
    )
    for key, expected_value in expected.items():
        if checkpoint.get(key) != expected_value:
            return (
                "Existing verification checkpoint does not match current run "
                f"metadata: {key}."
            )
    return None


def _verification_round_from_dict(
    data: dict[str, Any],
    *,
    json_path: Path,
    log_path: Path,
) -> VerificationRound:
    commands = tuple(
        _verification_command_from_dict(item)
        for item in _required_object_list(data, "commands")
    )
    safety_violations = tuple(
        _verification_safety_violation_from_dict(item)
        for item in _required_object_list(data, "safety_violations")
    )
    correction_reasons = tuple(
        correction_reason_from_dict(item)
        for item in _required_object_list(data, "correction_reasons")
    )
    return VerificationRound(
        round_index=_required_int(data, "round_index"),
        started_at=_required_string(data, "started_at"),
        ended_at=_required_string(data, "ended_at"),
        duration_seconds=_required_number(data, "duration_seconds"),
        status=VerificationStatus(_required_string(data, "status")),
        commands=commands,
        safety_violations=safety_violations,
        correction_reasons=correction_reasons,
        json_path=json_path,
        log_path=log_path,
        checkpoint=data.get("checkpoint")
        if isinstance(data.get("checkpoint"), dict)
        else None,
    )


def _verification_command_from_dict(data: dict[str, Any]) -> VerificationCommandResult:
    error_kind_value = data.get("error_kind")
    error_kind = (
        None if error_kind_value is None else VerificationErrorKind(error_kind_value)
    )
    return VerificationCommandResult(
        name=_required_string(data, "name"),
        argv=_required_string_tuple(data, "argv"),
        cwd=Path(_required_string(data, "cwd")),
        started_at=_required_string(data, "started_at"),
        ended_at=_required_string(data, "ended_at"),
        duration_seconds=_required_number(data, "duration_seconds"),
        status=VerificationStatus(_required_string(data, "status")),
        exit_code=_optional_int(data, "exit_code"),
        stdout=_required_string(data, "stdout", allow_empty=True),
        stderr=_required_string(data, "stderr", allow_empty=True),
        error_kind=error_kind,
        error_message=_optional_string(data, "error_message"),
    )


def _verification_safety_violation_from_dict(
    data: dict[str, Any],
) -> VerificationSafetyViolation:
    return VerificationSafetyViolation(
        name=_required_string(data, "name"),
        expected=_required_string(data, "expected", allow_empty=True),
        actual=_required_string(data, "actual", allow_empty=True),
        message=_required_string(data, "message"),
    )


def _verification_checkpoint(
    run_record: RunRecord,
    *,
    round_index: int,
    repository_snapshot: WorkspaceSnapshot | None,
    commands: tuple[VerificationCommand, ...],
    stage: WorkflowState = WorkflowState.VERIFYING,
) -> dict[str, Any]:
    checkpoint: dict[str, Any] = {
        "schema_version": VERIFICATION_CHECKPOINT_SCHEMA_VERSION,
        "format": VERIFICATION_CHECKPOINT_FORMAT,
        "stage": stage.value,
        "status": "COMPLETE",
        "run_id": run_record.run_id,
        "round_index": round_index,
        "target_repository_path": str(
            Path(run_record.target_repository_path).resolve()
        ),
        "starting_branch": run_record.starting_branch,
        "baseline_sha": run_record.baseline_sha,
        "verification_commands_fingerprint": _verification_commands_fingerprint(
            commands
        ),
    }
    if repository_snapshot is not None:
        checkpoint["source_fingerprint"] = repository_snapshot.fingerprint
    return checkpoint


def _archive_incomplete_verification_artifacts(
    *,
    artifact_directory: Path,
    round_name: str,
    json_path: Path,
    log_path: Path,
) -> None:
    archive_dir = _unique_archive_path(
        artifact_directory / _INCOMPLETE_ARTIFACT_DIR_NAME / round_name
    )
    archive_dir.mkdir(parents=True, exist_ok=False)
    for path in (json_path, log_path):
        if not path.exists():
            continue
        destination = archive_dir / path.name
        if path.is_dir():
            shutil.move(str(path), str(destination))
        else:
            path.replace(destination)


def _unique_archive_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.name}-{index}")
        if not candidate.exists():
            return candidate
    raise VerificationError(f"Could not reserve recovery artifact directory: {path}")


def _required_object_list(data: dict[str, Any], key: str) -> tuple[dict[str, Any], ...]:
    value = data.get(key)
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list.")
    if not all(isinstance(item, dict) for item in value):
        raise TypeError(f"{key} entries must be objects.")
    return tuple(value)


def _required_string(
    data: dict[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = data.get(key)
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{key} must be a string.")
    return value


def _optional_string(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string or null.")
    return value


def _required_string_tuple(data: dict[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{key} must be a string list.")
    return tuple(value)


def _required_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{key} must be an integer.")
    return value


def _optional_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{key} must be an integer or null.")
    return value


def _required_number(data: dict[str, Any], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{key} must be a number.")
    return float(value)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def _write_log(
    path: Path,
    round_result: VerificationRound,
    *,
    include_command_output: bool = True,
) -> None:
    _atomic_write_text(
        path,
        _format_round_log(
            round_result,
            include_command_output=include_command_output,
        ),
    )


def _atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            file_descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as temp_file:
            file_descriptor = -1
            temp_file.write(contents)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    except Exception:
        if file_descriptor != -1:
            os.close(file_descriptor)
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def _format_round_log(
    round_result: VerificationRound,
    *,
    include_command_output: bool = True,
) -> str:
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
        if include_command_output:
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
    now = datetime.now(UTC) if clock is None else clock()
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return now.astimezone(UTC)


def _format_timestamp(value: datetime) -> str:
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _duration_seconds(start: datetime, end: datetime) -> float:
    return max(0.0, (end - start).total_seconds())


def _timestamp(clock: Callable[[], datetime] | None) -> str:
    return _format_timestamp(_utcnow(clock))


def _process_text(value: str | bytes | None) -> str:
    return decode_human_output(value)


__all__ = [
    "VERIFICATION_DIR_NAME",
    "VERIFICATION_ROUND_FORMAT",
    "VERIFICATION_SCHEMA_VERSION",
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
    "run_verification_round",
    "run_verification_stage",
]
