from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from . import executable_resolution
from .config import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING_EFFORT,
    CodexExecutionSettings,
    validate_codex_execution_settings,
)
from .domain.task_results import ResultValidationError, TaskResult
from .models import ATTEMPT_RESULT_ARTIFACT_NAME
from .process_output import decode_human_output
from .task_result_codecs import decode_implementation_result, decode_review_result

DEFAULT_CODEX_EXECUTABLE = "codex"
DEFAULT_TIMEOUT_SECONDS = 60 * 60
PROMPT_ARTIFACT = "prompt.md"
EVENTS_ARTIFACT = "events.jsonl"
STDERR_ARTIFACT = "stderr.log"
_EXECUTION_ARTIFACT = "execution.json"
RESULT_ARTIFACT = ATTEMPT_RESULT_ARTIFACT_NAME


class Sandbox(StrEnum):
    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"


def parse_sandbox(value: str) -> Sandbox:
    try:
        return Sandbox(value)
    except ValueError as error:
        supported = ", ".join(item.value for item in Sandbox)
        raise ValueError(
            f"Unsupported Codex sandbox {value!r}; expected one of: {supported}."
        ) from error


class CodexExecutionStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class CodexFailureKind(StrEnum):
    EXECUTABLE_UNAVAILABLE = "EXECUTABLE_UNAVAILABLE"
    PROCESS_START_FAILED = "PROCESS_START_FAILED"
    TIMEOUT = "TIMEOUT"
    NON_ZERO_EXIT = "NON_ZERO_EXIT"
    MISSING_STRUCTURED_RESULT = "MISSING_STRUCTURED_RESULT"
    INVALID_STRUCTURED_RESULT = "INVALID_STRUCTURED_RESULT"
    AUTHENTICATION_OR_SERVICE = "AUTHENTICATION_OR_SERVICE"
    PROJECT_CONFIGURATION_REJECTED = "PROJECT_CONFIGURATION_REJECTED"


class _CodexResultKind(StrEnum):
    IMPLEMENTATION = "implementation"
    REVIEW = "review"


@dataclass(frozen=True)
class CodexCommand:
    argv: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str] | None = None


@dataclass(frozen=True)
class CodexProcessResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class CodexProcessTimeout:
    stdout: str
    stderr: str
    timeout_seconds: float


class CodexProcessTimedOut(TimeoutError):
    def __init__(self, result: CodexProcessTimeout):
        super().__init__(
            f"Codex process timed out after {result.timeout_seconds:g} seconds."
        )
        self.result = result


class CodexScratchDirectoryError(RuntimeError):
    pass


class CodexProcessRunner(Protocol):
    def run(
        self,
        command: CodexCommand,
        *,
        stdin: str,
        timeout_seconds: float | None,
    ) -> CodexProcessResult: ...


@dataclass(frozen=True)
class CodexExecution:
    argv: tuple[str, ...]
    repo_path: Path
    sandbox: Sandbox
    execution_config: CodexExecutionSettings
    output_schema_path: Path
    artifact_directory: Path
    prompt_path: Path
    events_jsonl_path: Path
    stderr_log_path: Path
    execution_json_path: Path
    result_json_path: Path
    started_at: str
    ended_at: str
    duration_seconds: float
    status: CodexExecutionStatus
    process_started: bool
    process_exit_code: int | None
    timed_out: bool = False
    timeout_seconds: float | None = None
    structured_result_present: bool = False
    structured_result: TaskResult | None = None
    failure_kind: CodexFailureKind | None = None
    failure_message: str | None = None

    @property
    def successful(self) -> bool:
        return self.status == CodexExecutionStatus.SUCCESS


class CodexExecutionFailure(RuntimeError):
    """Raised when the Codex subprocess boundary fails before yielding a valid result."""

    def __init__(
        self,
        message: str,
        *,
        kind: CodexFailureKind,
        execution: CodexExecution,
    ):
        super().__init__(message)
        self.kind = kind
        self.execution = execution


class SubprocessCodexRunner:
    def run(
        self,
        command: CodexCommand,
        *,
        stdin: str,
        timeout_seconds: float | None,
    ) -> CodexProcessResult:
        try:
            completed = subprocess.run(
                command.argv,
                cwd=command.cwd,
                env=_process_environment(command.environment),
                input=stdin.encode("utf-8"),
                capture_output=True,
                text=False,
                check=False,
                timeout=timeout_seconds,
                shell=False,
            )
        except subprocess.TimeoutExpired as error:
            raise CodexProcessTimedOut(
                CodexProcessTimeout(
                    stdout=_process_text(error.stdout),
                    stderr=_process_text(error.stderr),
                    timeout_seconds=float(timeout_seconds or 0),
                )
            ) from error

        return CodexProcessResult(
            returncode=completed.returncode,
            stdout=decode_human_output(completed.stdout),
            stderr=decode_human_output(completed.stderr),
        )


class CodexExecutor:
    def __init__(
        self,
        *,
        executable: str = DEFAULT_CODEX_EXECUTABLE,
        execution_config: CodexExecutionSettings | None = None,
        timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
        runner: CodexProcessRunner | None = None,
    ):
        if not executable:
            raise ValueError("Codex executable must be a non-empty string.")
        self.executable = executable
        self.execution_config = _effective_execution_config(execution_config)
        self.timeout_seconds = timeout_seconds
        self.runner = runner or SubprocessCodexRunner()

    def execute(
        self,
        *,
        prompt: str,
        repo_path: Path,
        sandbox: Sandbox,
        output_schema: Path,
        artifact_directory: Path,
    ) -> CodexExecution:
        return self._execute(
            prompt=prompt,
            repo_path=repo_path,
            sandbox=sandbox,
            output_schema=output_schema,
            artifact_directory=artifact_directory,
            _result_kind=_CodexResultKind.IMPLEMENTATION,
        )

    def _execute(
        self,
        *,
        prompt: str,
        repo_path: Path,
        sandbox: Sandbox,
        output_schema: Path,
        artifact_directory: Path,
        _result_kind: _CodexResultKind,
    ) -> CodexExecution:
        if not isinstance(sandbox, Sandbox):
            raise TypeError("sandbox must be a Sandbox value.")
        if not isinstance(_result_kind, _CodexResultKind):
            raise TypeError("_result_kind must be a _CodexResultKind value.")

        artifact_paths = _artifact_paths(artifact_directory)
        start = _utcnow()
        artifact_paths.directory.mkdir(parents=True, exist_ok=True)
        artifact_paths.result.unlink(missing_ok=True)
        artifact_paths.prompt.write_text(prompt, encoding="utf-8", newline="\n")

        configured_command = _build_codex_command(
            executable=self.executable,
            repo_path=repo_path,
            sandbox=sandbox,
            execution_config=self.execution_config,
            output_schema=output_schema,
            _output_last_message=artifact_paths.result,
        )

        project_config = Path(repo_path) / ".codex" / "config.toml"
        if project_config.is_file():
            artifact_paths.events.write_text("", encoding="utf-8", newline="\n")
            artifact_paths.stderr.write_text(
                f"Target repository Codex configuration rejected: {project_config}\n",
                encoding="utf-8",
                newline="\n",
            )
            return _fail(
                kind=CodexFailureKind.PROJECT_CONFIGURATION_REJECTED,
                message=(
                    "Target repository contains .codex/config.toml; V1 cannot "
                    "reliably suppress project Codex execution configuration."
                ),
                command=configured_command,
                sandbox=sandbox,
                execution_config=self.execution_config,
                output_schema=output_schema,
                artifacts=artifact_paths,
                started_at=start,
                process_started=False,
                exit_code=None,
            )

        resolved_executable = executable_resolution.resolve_executable(
            self.executable,
        )
        if resolved_executable is None:
            artifact_paths.events.write_text("", encoding="utf-8", newline="\n")
            artifact_paths.stderr.write_text(
                f"Executable not found: {self.executable}\n",
                encoding="utf-8",
                newline="\n",
            )
            return _fail(
                kind=CodexFailureKind.EXECUTABLE_UNAVAILABLE,
                message=f"Codex executable is unavailable: {self.executable}",
                command=configured_command,
                sandbox=sandbox,
                execution_config=self.execution_config,
                output_schema=output_schema,
                artifacts=artifact_paths,
                started_at=start,
                process_started=False,
                exit_code=None,
            )

        command = _build_codex_command(
            executable=str(resolved_executable),
            repo_path=repo_path,
            sandbox=sandbox,
            execution_config=self.execution_config,
            output_schema=output_schema,
            _output_last_message=artifact_paths.result,
        )

        try:
            with _external_scratch_environment(repo_path) as scratch_environment:
                command = CodexCommand(
                    argv=command.argv,
                    cwd=command.cwd,
                    environment=scratch_environment,
                )
                process = self.runner.run(
                    command,
                    stdin=prompt,
                    timeout_seconds=self.timeout_seconds,
                )
        except CodexScratchDirectoryError as error:
            artifact_paths.events.write_text("", encoding="utf-8", newline="\n")
            artifact_paths.stderr.write_text(
                f"{error}\n", encoding="utf-8", newline="\n"
            )
            return _fail(
                kind=CodexFailureKind.PROCESS_START_FAILED,
                message=str(error),
                command=command,
                sandbox=sandbox,
                execution_config=self.execution_config,
                output_schema=output_schema,
                artifacts=artifact_paths,
                started_at=start,
                process_started=False,
                exit_code=None,
            )
        except FileNotFoundError as error:
            artifact_paths.events.write_text("", encoding="utf-8", newline="\n")
            artifact_paths.stderr.write_text(
                f"{error}\n", encoding="utf-8", newline="\n"
            )
            return _fail(
                kind=CodexFailureKind.EXECUTABLE_UNAVAILABLE,
                message=f"Codex executable is unavailable: {command.argv[0]}",
                command=command,
                sandbox=sandbox,
                execution_config=self.execution_config,
                output_schema=output_schema,
                artifacts=artifact_paths,
                started_at=start,
                process_started=False,
                exit_code=None,
            )
        except CodexProcessTimedOut as error:
            artifact_paths.events.write_text(
                error.result.stdout,
                encoding="utf-8",
                newline="\n",
            )
            artifact_paths.stderr.write_text(
                error.result.stderr,
                encoding="utf-8",
                newline="\n",
            )
            return _fail(
                kind=CodexFailureKind.TIMEOUT,
                message=str(error),
                command=command,
                sandbox=sandbox,
                execution_config=self.execution_config,
                output_schema=output_schema,
                artifacts=artifact_paths,
                started_at=start,
                process_started=True,
                exit_code=None,
                timed_out=True,
                timeout_seconds=error.result.timeout_seconds,
            )
        except OSError as error:
            artifact_paths.events.write_text("", encoding="utf-8", newline="\n")
            artifact_paths.stderr.write_text(
                f"{error}\n", encoding="utf-8", newline="\n"
            )
            return _fail(
                kind=CodexFailureKind.PROCESS_START_FAILED,
                message=f"Could not start Codex process: {error}",
                command=command,
                sandbox=sandbox,
                execution_config=self.execution_config,
                output_schema=output_schema,
                artifacts=artifact_paths,
                started_at=start,
                process_started=False,
                exit_code=None,
            )

        artifact_paths.events.write_text(process.stdout, encoding="utf-8", newline="\n")
        artifact_paths.stderr.write_text(process.stderr, encoding="utf-8", newline="\n")

        if process.returncode != 0:
            kind = (
                CodexFailureKind.AUTHENTICATION_OR_SERVICE
                if _looks_like_authentication_or_service_failure(process.stderr)
                else CodexFailureKind.NON_ZERO_EXIT
            )
            return _fail(
                kind=kind,
                message=f"Codex exited with code {process.returncode}.",
                command=command,
                sandbox=sandbox,
                execution_config=self.execution_config,
                output_schema=output_schema,
                artifacts=artifact_paths,
                started_at=start,
                process_started=True,
                exit_code=process.returncode,
            )

        try:
            raw_result = artifact_paths.result.read_text(encoding="utf-8")
        except FileNotFoundError:
            return _fail(
                kind=CodexFailureKind.MISSING_STRUCTURED_RESULT,
                message=(
                    "Codex exited successfully but did not write the required "
                    f"typed result: {artifact_paths.result}."
                ),
                command=command,
                sandbox=sandbox,
                execution_config=self.execution_config,
                output_schema=output_schema,
                artifacts=artifact_paths,
                started_at=start,
                process_started=True,
                exit_code=process.returncode,
            )
        except OSError as error:
            return _fail(
                kind=CodexFailureKind.MISSING_STRUCTURED_RESULT,
                message=(
                    "Codex exited successfully but TicketAutomation could not read "
                    f"the typed result: {error}."
                ),
                command=command,
                sandbox=sandbox,
                execution_config=self.execution_config,
                output_schema=output_schema,
                artifacts=artifact_paths,
                started_at=start,
                process_started=True,
                exit_code=process.returncode,
            )

        try:
            result = _parse_codex_result(
                json.loads(raw_result), _result_kind=_result_kind
            )
        except json.JSONDecodeError as error:
            return _fail(
                kind=CodexFailureKind.INVALID_STRUCTURED_RESULT,
                message=(f"Codex typed result was not valid JSON: {error.msg}."),
                command=command,
                sandbox=sandbox,
                execution_config=self.execution_config,
                output_schema=output_schema,
                artifacts=artifact_paths,
                started_at=start,
                process_started=True,
                exit_code=process.returncode,
                structured_result_present=True,
            )
        except ResultValidationError as error:
            return _fail(
                kind=CodexFailureKind.INVALID_STRUCTURED_RESULT,
                message=str(error),
                command=command,
                sandbox=sandbox,
                execution_config=self.execution_config,
                output_schema=output_schema,
                artifacts=artifact_paths,
                started_at=start,
                process_started=True,
                exit_code=process.returncode,
                structured_result_present=True,
            )

        ended = _utcnow()
        execution = CodexExecution(
            argv=command.argv,
            repo_path=command.cwd,
            sandbox=sandbox,
            execution_config=self.execution_config,
            output_schema_path=Path(output_schema).resolve(),
            artifact_directory=artifact_paths.directory,
            prompt_path=artifact_paths.prompt,
            events_jsonl_path=artifact_paths.events,
            stderr_log_path=artifact_paths.stderr,
            execution_json_path=artifact_paths.execution,
            result_json_path=artifact_paths.result,
            started_at=_format_timestamp(start),
            ended_at=_format_timestamp(ended),
            duration_seconds=_duration_seconds(start, ended),
            status=CodexExecutionStatus.SUCCESS,
            process_started=True,
            process_exit_code=process.returncode,
            structured_result_present=True,
            structured_result=result,
        )
        _write_execution_artifact(execution)
        return execution


def execute(
    *,
    prompt: str,
    repo_path: Path,
    sandbox: Sandbox,
    output_schema: Path,
    artifact_directory: Path,
    executable: str = DEFAULT_CODEX_EXECUTABLE,
    execution_config: CodexExecutionSettings | None = None,
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
    runner: CodexProcessRunner | None = None,
) -> CodexExecution:
    return _execute(
        prompt=prompt,
        repo_path=repo_path,
        sandbox=sandbox,
        output_schema=output_schema,
        artifact_directory=artifact_directory,
        executable=executable,
        execution_config=execution_config,
        timeout_seconds=timeout_seconds,
        runner=runner,
        _result_kind=_CodexResultKind.IMPLEMENTATION,
    )


def _execute(
    *,
    prompt: str,
    repo_path: Path,
    sandbox: Sandbox,
    output_schema: Path,
    artifact_directory: Path,
    executable: str = DEFAULT_CODEX_EXECUTABLE,
    execution_config: CodexExecutionSettings | None = None,
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
    runner: CodexProcessRunner | None = None,
    _result_kind: _CodexResultKind,
) -> CodexExecution:
    return CodexExecutor(
        executable=executable,
        execution_config=execution_config,
        timeout_seconds=timeout_seconds,
        runner=runner,
    )._execute(
        prompt=prompt,
        repo_path=repo_path,
        sandbox=sandbox,
        output_schema=output_schema,
        artifact_directory=artifact_directory,
        _result_kind=_result_kind,
    )


def build_codex_command(
    *,
    executable: str,
    repo_path: Path,
    sandbox: Sandbox,
    execution_config: CodexExecutionSettings | None = None,
    output_schema: Path,
) -> CodexCommand:
    """Build a Codex command for callers that only need command inspection."""
    return _build_codex_command(
        executable=executable,
        repo_path=repo_path,
        sandbox=sandbox,
        execution_config=execution_config,
        output_schema=output_schema,
        _output_last_message=Path(output_schema).with_name(RESULT_ARTIFACT),
    )


def _build_codex_command(
    *,
    executable: str,
    repo_path: Path,
    sandbox: Sandbox,
    execution_config: CodexExecutionSettings | None = None,
    output_schema: Path,
    _output_last_message: Path,
) -> CodexCommand:
    if not isinstance(sandbox, Sandbox):
        raise TypeError("sandbox must be a Sandbox value.")
    effective_execution_config = _effective_execution_config(execution_config)
    sandbox_config = (
        ("-c", "sandbox_workspace_write.network_access=true")
        if sandbox is Sandbox.WORKSPACE_WRITE
        else ()
    )
    return CodexCommand(
        argv=(
            executable,
            "exec",
            "--ephemeral",
            "--model",
            effective_execution_config.model,
            "-c",
            (f'model_reasoning_effort="{effective_execution_config.reasoning_effort}"'),
            *sandbox_config,
            "--sandbox",
            sandbox.value,
            "--json",
            "--output-schema",
            str(Path(output_schema).resolve()),
            "--output-last-message",
            str(Path(_output_last_message).resolve()),
            "-",
        ),
        cwd=Path(repo_path),
    )


def _process_environment(
    overrides: Mapping[str, str] | None,
) -> dict[str, str]:
    environment = os.environ.copy()
    if overrides is not None:
        environment.update(overrides)
    return environment


@contextmanager
def _external_scratch_environment(repo_path: Path) -> Iterator[dict[str, str]]:
    """Provide Codex an external temporary root without weakening repository safety."""

    try:
        scratch_path = Path(
            tempfile.mkdtemp(prefix="ticket-automation-codex-")
        ).resolve()
    except OSError as error:
        raise CodexScratchDirectoryError(
            f"Could not create an external Codex scratch directory: {error}"
        ) from error

    try:
        repository_path = Path(repo_path).resolve()
        try:
            scratch_path.relative_to(repository_path)
        except ValueError:
            pass
        else:
            raise CodexScratchDirectoryError(
                "Codex scratch directory must be outside the target repository: "
                f"{scratch_path}"
            )
        scratch = str(scratch_path)
        yield {"TEMP": scratch, "TMP": scratch, "TMPDIR": scratch}
    finally:
        shutil.rmtree(scratch_path, ignore_errors=True)


def _effective_execution_config(
    execution_config: CodexExecutionSettings | None,
) -> CodexExecutionSettings:
    return validate_codex_execution_settings(
        execution_config
        or CodexExecutionSettings(
            model=DEFAULT_CODEX_MODEL,
            reasoning_effort=DEFAULT_CODEX_REASONING_EFFORT,
        )
    )


def _parse_codex_result(
    value: Any,
    *,
    _result_kind: _CodexResultKind,
) -> TaskResult:
    if _result_kind == _CodexResultKind.IMPLEMENTATION:
        return decode_implementation_result(value)
    if _result_kind == _CodexResultKind.REVIEW:
        return decode_review_result(value)
    raise AssertionError(f"Unhandled Codex result kind: {_result_kind!r}.")


@dataclass(frozen=True)
class _ArtifactPaths:
    directory: Path
    prompt: Path
    events: Path
    stderr: Path
    execution: Path
    result: Path


def _artifact_paths(artifact_directory: Path) -> _ArtifactPaths:
    directory = Path(artifact_directory)
    return _ArtifactPaths(
        directory=directory,
        prompt=directory / PROMPT_ARTIFACT,
        events=directory / EVENTS_ARTIFACT,
        stderr=directory / STDERR_ARTIFACT,
        execution=directory / _EXECUTION_ARTIFACT,
        result=directory / RESULT_ARTIFACT,
    )


def _fail(
    *,
    kind: CodexFailureKind,
    message: str,
    command: CodexCommand,
    sandbox: Sandbox,
    execution_config: CodexExecutionSettings,
    output_schema: Path,
    artifacts: _ArtifactPaths,
    started_at: datetime,
    process_started: bool,
    exit_code: int | None,
    timed_out: bool = False,
    timeout_seconds: float | None = None,
    structured_result_present: bool = False,
) -> CodexExecution:
    ended = _utcnow()
    if not artifacts.events.exists():
        artifacts.events.write_text("", encoding="utf-8", newline="\n")
    if not artifacts.stderr.exists():
        artifacts.stderr.write_text("", encoding="utf-8", newline="\n")
    execution = CodexExecution(
        argv=command.argv,
        repo_path=command.cwd,
        sandbox=sandbox,
        execution_config=execution_config,
        output_schema_path=Path(output_schema).resolve(),
        artifact_directory=artifacts.directory,
        prompt_path=artifacts.prompt,
        events_jsonl_path=artifacts.events,
        stderr_log_path=artifacts.stderr,
        execution_json_path=artifacts.execution,
        result_json_path=artifacts.result,
        started_at=_format_timestamp(started_at),
        ended_at=_format_timestamp(ended),
        duration_seconds=_duration_seconds(started_at, ended),
        status=CodexExecutionStatus.FAILED,
        process_started=process_started,
        process_exit_code=exit_code,
        timed_out=timed_out,
        timeout_seconds=timeout_seconds,
        structured_result_present=structured_result_present,
        failure_kind=kind,
        failure_message=message,
    )
    _write_execution_artifact(execution)
    raise CodexExecutionFailure(message, kind=kind, execution=execution)


def _write_json_artifact(path: Path, data: Any) -> None:
    _atomic_write_json(path, data)


def _write_execution_artifact(execution: CodexExecution) -> None:
    _write_json_artifact(execution.execution_json_path, _execution_record(execution))


def _execution_record(execution: CodexExecution) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "format": "ticket_automation.codex_execution",
        "status": execution.status.value,
        "process_started": execution.process_started,
        "process_exit_code": execution.process_exit_code,
        "timed_out": execution.timed_out,
        "timeout_seconds": execution.timeout_seconds,
        "failure_kind": (
            None if execution.failure_kind is None else execution.failure_kind.value
        ),
        "failure_message": execution.failure_message,
        "started_at": execution.started_at,
        "ended_at": execution.ended_at,
        "duration_seconds": execution.duration_seconds,
        "sandbox": execution.sandbox.value,
        "codex": {
            "model": execution.execution_config.model,
            "reasoning_effort": execution.execution_config.reasoning_effort,
        },
        "argv": list(execution.argv),
        "repo_path": str(execution.repo_path),
        "output_schema_path": str(execution.output_schema_path),
        "structured_result_present": execution.structured_result_present,
        "result_json_present": execution.result_json_path.is_file(),
        "artifact_paths": {
            "prompt_md": str(execution.prompt_path),
            "events_jsonl": str(execution.events_jsonl_path),
            "stderr_log": str(execution.stderr_log_path),
            "execution_json": str(execution.execution_json_path),
            "result_json": str(execution.result_json_path),
        },
    }


def _atomic_write_json(path: Path, data: Any) -> None:
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
            file_descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
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


def _looks_like_authentication_or_service_failure(stderr: str) -> bool:
    lowered = stderr.lower()
    markers = (
        "auth",
        "api key",
        "unauthorized",
        "forbidden",
        "rate limit",
        "service unavailable",
        "temporarily unavailable",
        "network",
    )
    return any(marker in lowered for marker in markers)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _duration_seconds(start: datetime, end: datetime) -> float:
    return max(0.0, (end - start).total_seconds())


def _process_text(value: str | bytes | None) -> str:
    return decode_human_output(value)


__all__ = [
    "DEFAULT_CODEX_EXECUTABLE",
    "DEFAULT_TIMEOUT_SECONDS",
    "EVENTS_ARTIFACT",
    "PROMPT_ARTIFACT",
    "RESULT_ARTIFACT",
    "STDERR_ARTIFACT",
    "CodexCommand",
    "CodexExecution",
    "CodexExecutionFailure",
    "CodexExecutionStatus",
    "CodexExecutor",
    "CodexFailureKind",
    "CodexProcessResult",
    "CodexProcessRunner",
    "CodexProcessTimedOut",
    "CodexProcessTimeout",
    "Sandbox",
    "SubprocessCodexRunner",
    "build_codex_command",
    "execute",
    "parse_sandbox",
]
