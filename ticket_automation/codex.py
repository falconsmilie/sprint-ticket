from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
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
from .process_output import (
    ProcessOutputDecodeError,
    decode_human_output,
    decode_protocol_output,
)

DEFAULT_CODEX_EXECUTABLE = "codex"
DEFAULT_TIMEOUT_SECONDS = 60 * 60
PROMPT_ARTIFACT = "prompt.md"
EVENTS_ARTIFACT = "events.jsonl"
STDERR_ARTIFACT = "stderr.log"
_EXECUTION_ARTIFACT = "execution.json"
RESULT_ARTIFACT = "result.json"


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


JSON_SCHEMA_ANNOTATION_KEYS = frozenset(
    {
        "$comment",
        "$id",
        "$schema",
        "default",
        "description",
        "examples",
        "format",
        "title",
    }
)
JSON_SCHEMA_SUPPORTED_KEYS = (
    frozenset(
        {
            "$defs",
            "$ref",
            "additionalProperties",
            "allOf",
            "anyOf",
            "const",
            "definitions",
            "enum",
            "exclusiveMaximum",
            "exclusiveMinimum",
            "items",
            "maxItems",
            "maxLength",
            "maxProperties",
            "maximum",
            "minItems",
            "minLength",
            "minProperties",
            "minimum",
            "multipleOf",
            "not",
            "oneOf",
            "pattern",
            "properties",
            "required",
            "type",
            "uniqueItems",
        }
    )
    | JSON_SCHEMA_ANNOTATION_KEYS
)
JSON_TYPES = frozenset(
    {
        "array",
        "boolean",
        "integer",
        "null",
        "number",
        "object",
        "string",
    }
)
COMBINATOR_KEYS = ("allOf", "anyOf", "oneOf")


class CodexExecutionStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class CodexFailureKind(StrEnum):
    EXECUTABLE_UNAVAILABLE = "EXECUTABLE_UNAVAILABLE"
    PROCESS_START_FAILED = "PROCESS_START_FAILED"
    TIMEOUT = "TIMEOUT"
    NON_ZERO_EXIT = "NON_ZERO_EXIT"
    MALFORMED_EVENT_STREAM = "MALFORMED_EVENT_STREAM"
    MISSING_STRUCTURED_RESULT = "MISSING_STRUCTURED_RESULT"
    INVALID_STRUCTURED_RESULT = "INVALID_STRUCTURED_RESULT"
    AUTHENTICATION_OR_SERVICE = "AUTHENTICATION_OR_SERVICE"
    INVALID_SCHEMA = "INVALID_SCHEMA"


class CodexEventParseError(ValueError):
    """Raised when Codex JSONL output is not a valid stream of JSON objects."""


class CodexResultValidationError(ValueError):
    """Raised when the final structured result does not match the schema."""


class _CodexStructuredResultDecodeError(CodexResultValidationError):
    """Raised when Codex emits a structured result payload that cannot be decoded."""


@dataclass(frozen=True)
class CodexCommand:
    argv: tuple[str, ...]
    cwd: Path


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


class CodexProcessOutputDecodeError(RuntimeError):
    def __init__(self, message: str, result: CodexProcessResult):
        super().__init__(message)
        self.result = result


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
    structured_result: Any | None = None
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

        stderr = decode_human_output(completed.stderr)
        try:
            stdout = decode_protocol_output(
                completed.stdout,
                stream_name="Codex stdout",
            )
        except ProcessOutputDecodeError as error:
            raise CodexProcessOutputDecodeError(
                str(error),
                CodexProcessResult(
                    returncode=completed.returncode,
                    stdout=decode_human_output(completed.stdout),
                    stderr=stderr,
                ),
            ) from error

        return CodexProcessResult(
            returncode=completed.returncode,
            stdout=stdout,
            stderr=stderr,
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
        if not isinstance(sandbox, Sandbox):
            raise TypeError("sandbox must be a Sandbox value.")

        artifact_paths = _artifact_paths(artifact_directory)
        start = _utcnow()
        artifact_paths.directory.mkdir(parents=True, exist_ok=True)
        artifact_paths.prompt.write_text(prompt, encoding="utf-8", newline="\n")

        configured_command = build_codex_command(
            executable=self.executable,
            repo_path=repo_path,
            sandbox=sandbox,
            execution_config=self.execution_config,
            output_schema=output_schema,
        )

        try:
            schema = _load_schema(output_schema)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            return _fail(
                kind=CodexFailureKind.INVALID_SCHEMA,
                message=f"Invalid Codex output schema: {error}",
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
            cwd=repo_path,
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

        command = build_codex_command(
            executable=str(resolved_executable),
            repo_path=repo_path,
            sandbox=sandbox,
            execution_config=self.execution_config,
            output_schema=output_schema,
        )

        try:
            process = self.runner.run(
                command,
                stdin=prompt,
                timeout_seconds=self.timeout_seconds,
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
        except CodexProcessOutputDecodeError as error:
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
                kind=CodexFailureKind.MALFORMED_EVENT_STREAM,
                message=str(error),
                command=command,
                sandbox=sandbox,
                execution_config=self.execution_config,
                output_schema=output_schema,
                artifacts=artifact_paths,
                started_at=start,
                process_started=True,
                exit_code=error.result.returncode,
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
            events = parse_codex_events(process.stdout)
        except CodexEventParseError as error:
            return _fail(
                kind=CodexFailureKind.MALFORMED_EVENT_STREAM,
                message=str(error),
                command=command,
                sandbox=sandbox,
                execution_config=self.execution_config,
                output_schema=output_schema,
                artifacts=artifact_paths,
                started_at=start,
                process_started=True,
                exit_code=process.returncode,
            )

        service_error = _service_error_message(events)
        if service_error:
            return _fail(
                kind=CodexFailureKind.AUTHENTICATION_OR_SERVICE,
                message=service_error,
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
            result = extract_structured_result(events)
        except _CodexStructuredResultDecodeError as error:
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
        except CodexResultValidationError as error:
            return _fail(
                kind=CodexFailureKind.MISSING_STRUCTURED_RESULT,
                message=str(error),
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
            validate_json_schema(result, schema)
        except CodexResultValidationError as error:
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

        _write_json_artifact(artifact_paths.result, result)
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
    return CodexExecutor(
        executable=executable,
        execution_config=execution_config,
        timeout_seconds=timeout_seconds,
        runner=runner,
    ).execute(
        prompt=prompt,
        repo_path=repo_path,
        sandbox=sandbox,
        output_schema=output_schema,
        artifact_directory=artifact_directory,
    )


def build_codex_command(
    *,
    executable: str,
    repo_path: Path,
    sandbox: Sandbox,
    execution_config: CodexExecutionSettings | None = None,
    output_schema: Path,
) -> CodexCommand:
    if not isinstance(sandbox, Sandbox):
        raise TypeError("sandbox must be a Sandbox value.")
    effective_execution_config = _effective_execution_config(execution_config)
    return CodexCommand(
        argv=(
            executable,
            "exec",
            "--model",
            effective_execution_config.model,
            "-c",
            (f'model_reasoning_effort="{effective_execution_config.reasoning_effort}"'),
            "--sandbox",
            sandbox.value,
            "--json",
            "--output-schema",
            str(Path(output_schema).resolve()),
            "-",
        ),
        cwd=Path(repo_path),
    )


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


def parse_codex_events(jsonl: str) -> tuple[dict[str, Any], ...]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(jsonl.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise CodexEventParseError(
                f"Codex JSONL event stream is malformed at line {line_number}: {error.msg}."
            ) from error
        if not isinstance(event, dict):
            raise CodexEventParseError(
                f"Codex JSONL event at line {line_number} must be an object."
            )
        events.append(event)
    return tuple(events)


def extract_structured_result(events: tuple[dict[str, Any], ...]) -> Any:
    if not events or events[-1].get("type") != "turn.completed":
        raise CodexResultValidationError("Codex did not emit a completed final turn.")

    for event in reversed(events):
        result = _structured_result_from_event(event)
        if result is not _MISSING:
            return result
    raise CodexResultValidationError("Codex did not emit a structured final result.")


def validate_json_schema(value: Any, schema: dict[str, Any]) -> None:
    _validate_schema_node(value, schema, path="$")


@dataclass(frozen=True)
class _ArtifactPaths:
    directory: Path
    prompt: Path
    events: Path
    stderr: Path
    execution: Path
    result: Path


class _Missing:
    pass


_MISSING = _Missing()


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


def _load_schema(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as schema_file:
        schema = json.load(schema_file)
    if not isinstance(schema, dict):
        raise ValueError("schema root must be an object")
    _assert_supported_schema(schema, root_schema=schema, path="$")
    return schema


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


def _structured_result_from_event(event: dict[str, Any]) -> Any:
    event_type = event.get("type")
    if event_type == "turn.completed":
        return _structured_result_from_container(event)

    if event_type != "item.completed":
        return _MISSING

    item = event.get("item")
    if not isinstance(item, dict) or item.get("type") != "agent_message":
        return _MISSING

    return _structured_result_from_container(item)


def _structured_result_from_container(container: dict[str, Any]) -> Any:
    for key in ("structured_result", "final_result", "result"):
        if key in container:
            return _decode_structured_result(container[key])

    text = container.get("text")
    if text is not None:
        return _decode_structured_result(text)

    content = container.get("content")
    if isinstance(content, list):
        for part in reversed(content):
            if isinstance(part, dict) and "text" in part:
                return _decode_structured_result(part["text"])
    return _MISSING


def _decode_structured_result(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if not value.strip():
        return _MISSING
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise _CodexStructuredResultDecodeError(
            "Codex final agent message was not valid structured JSON."
        ) from error


def _service_error_message(events: tuple[dict[str, Any], ...]) -> str | None:
    for event in events:
        event_type = event.get("type")
        if event_type not in {"error", "turn.failed"}:
            continue
        message = event.get("message")
        if isinstance(message, str) and message:
            return message
        error = event.get("error")
        if isinstance(error, dict):
            error_message = error.get("message")
            if isinstance(error_message, str) and error_message:
                return error_message
        return f"Codex emitted {event_type}."
    return None


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


def _validate_schema_node(value: Any, schema: dict[str, Any], *, path: str) -> None:
    if schema is True:
        return
    if schema is False:
        raise CodexResultValidationError(f"{path} is not allowed by the schema.")

    root_schema = schema
    _validate_schema_node_against_root(
        value,
        schema,
        path=path,
        root_schema=root_schema,
        ref_stack=(),
    )


def _validate_schema_node_against_root(
    value: Any,
    schema: Any,
    *,
    path: str,
    root_schema: dict[str, Any],
    ref_stack: tuple[str, ...],
) -> None:
    if schema is True:
        return
    if schema is False:
        raise CodexResultValidationError(f"{path} is not allowed by the schema.")
    if not isinstance(schema, dict):
        raise CodexResultValidationError(f"{path} schema must be an object.")

    ref = schema.get("$ref")
    if ref is not None:
        target = _resolve_schema_ref(root_schema, ref, ref_stack=ref_stack)
        _validate_schema_node_against_root(
            value,
            target,
            path=path,
            root_schema=root_schema,
            ref_stack=(*ref_stack, ref),
        )

    for subschema in schema.get("allOf", []):
        _validate_schema_node_against_root(
            value,
            subschema,
            path=path,
            root_schema=root_schema,
            ref_stack=ref_stack,
        )
    if "anyOf" in schema:
        _validate_any_of(
            value,
            schema["anyOf"],
            path=path,
            root_schema=root_schema,
            ref_stack=ref_stack,
        )
    if "oneOf" in schema:
        _validate_one_of(
            value,
            schema["oneOf"],
            path=path,
            root_schema=root_schema,
            ref_stack=ref_stack,
        )
    if "not" in schema and _schema_matches(
        value,
        schema["not"],
        path=path,
        root_schema=root_schema,
        ref_stack=ref_stack,
    ):
        raise CodexResultValidationError(f"{path} must not match the forbidden schema.")

    if "const" in schema and value != schema["const"]:
        raise CodexResultValidationError(f"{path} must equal {schema['const']!r}.")
    if "enum" in schema:
        enum_values = schema["enum"]
        if not isinstance(enum_values, list):
            raise CodexResultValidationError(f"{path} schema enum must be a list.")
        if value not in enum_values:
            raise CodexResultValidationError(f"{path} must be one of {enum_values!r}.")

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        if not any(_matches_json_type(value, item) for item in schema_type):
            raise CodexResultValidationError(
                f"{path} must have one of these JSON types: {schema_type!r}."
            )
    elif isinstance(schema_type, str):
        if not _matches_json_type(value, schema_type):
            raise CodexResultValidationError(
                f"{path} must be JSON type {schema_type!r}."
            )

    if schema_type == "object" or "properties" in schema or "required" in schema:
        _validate_object(
            value,
            schema,
            path=path,
            root_schema=root_schema,
            ref_stack=ref_stack,
        )
    if schema_type == "array" or "items" in schema:
        _validate_array(
            value,
            schema,
            path=path,
            root_schema=root_schema,
            ref_stack=ref_stack,
        )
    _validate_string(value, schema, path=path)
    _validate_number(value, schema, path=path)


def _validate_object(
    value: Any,
    schema: dict[str, Any],
    *,
    path: str,
    root_schema: dict[str, Any],
    ref_stack: tuple[str, ...],
) -> None:
    if not isinstance(value, dict):
        raise CodexResultValidationError(f"{path} must be a JSON object.")

    required = schema.get("required", [])
    if not isinstance(required, list) or not all(
        isinstance(item, str) for item in required
    ):
        raise CodexResultValidationError(
            f"{path} schema required must be a string list."
        )
    for key in required:
        if key not in value:
            raise CodexResultValidationError(f"{path}.{key} is required.")

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise CodexResultValidationError(f"{path} schema properties must be an object.")
    for key, property_schema in properties.items():
        if key not in value:
            continue
        if not isinstance(property_schema, dict | bool):
            raise CodexResultValidationError(
                f"{path}.{key} schema must be an object or boolean."
            )
        _validate_schema_node_against_root(
            value[key],
            property_schema,
            path=f"{path}.{key}",
            root_schema=root_schema,
            ref_stack=ref_stack,
        )

    extra_keys = set(value) - set(properties)
    additional_properties = schema.get("additionalProperties", True)
    if additional_properties is False:
        if extra_keys:
            extra = ", ".join(sorted(extra_keys))
            raise CodexResultValidationError(
                f"{path} contains unsupported properties: {extra}."
            )
    elif isinstance(additional_properties, dict):
        for key in sorted(extra_keys):
            _validate_schema_node_against_root(
                value[key],
                additional_properties,
                path=f"{path}.{key}",
                root_schema=root_schema,
                ref_stack=ref_stack,
            )

    _validate_size(
        len(value),
        minimum=schema.get("minProperties"),
        maximum=schema.get("maxProperties"),
        path=path,
        noun="properties",
    )


def _validate_array(
    value: Any,
    schema: dict[str, Any],
    *,
    path: str,
    root_schema: dict[str, Any],
    ref_stack: tuple[str, ...],
) -> None:
    if not isinstance(value, list):
        raise CodexResultValidationError(f"{path} must be a JSON array.")
    _validate_size(
        len(value),
        minimum=schema.get("minItems"),
        maximum=schema.get("maxItems"),
        path=path,
        noun="items",
    )
    if schema.get("uniqueItems") is True:
        seen: set[str] = set()
        for item in value:
            marker = json.dumps(item, sort_keys=True, separators=(",", ":"))
            if marker in seen:
                raise CodexResultValidationError(f"{path} must contain unique items.")
            seen.add(marker)
    item_schema = schema.get("items")
    if item_schema is None:
        return
    if not isinstance(item_schema, dict | bool):
        raise CodexResultValidationError(
            f"{path} schema items must be an object or boolean."
        )
    for index, item in enumerate(value):
        _validate_schema_node_against_root(
            item,
            item_schema,
            path=f"{path}[{index}]",
            root_schema=root_schema,
            ref_stack=ref_stack,
        )


def _validate_string(value: Any, schema: dict[str, Any], *, path: str) -> None:
    if not isinstance(value, str):
        return
    _validate_size(
        len(value),
        minimum=schema.get("minLength"),
        maximum=schema.get("maxLength"),
        path=path,
        noun="characters",
    )
    pattern = schema.get("pattern")
    if isinstance(pattern, str) and re.search(pattern, value) is None:
        raise CodexResultValidationError(f"{path} must match pattern {pattern!r}.")


def _validate_number(value: Any, schema: dict[str, Any], *, path: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return
    minimum = schema.get("minimum")
    if minimum is not None and value < minimum:
        raise CodexResultValidationError(f"{path} must be at least {minimum!r}.")
    maximum = schema.get("maximum")
    if maximum is not None and value > maximum:
        raise CodexResultValidationError(f"{path} must be at most {maximum!r}.")
    exclusive_minimum = schema.get("exclusiveMinimum")
    if exclusive_minimum is not None and value <= exclusive_minimum:
        raise CodexResultValidationError(
            f"{path} must be greater than {exclusive_minimum!r}."
        )
    exclusive_maximum = schema.get("exclusiveMaximum")
    if exclusive_maximum is not None and value >= exclusive_maximum:
        raise CodexResultValidationError(
            f"{path} must be less than {exclusive_maximum!r}."
        )
    multiple_of = schema.get("multipleOf")
    if multiple_of is not None and value % multiple_of != 0:
        raise CodexResultValidationError(
            f"{path} must be a multiple of {multiple_of!r}."
        )


def _validate_any_of(
    value: Any,
    schemas: list[Any],
    *,
    path: str,
    root_schema: dict[str, Any],
    ref_stack: tuple[str, ...],
) -> None:
    if any(
        _schema_matches(
            value,
            schema,
            path=path,
            root_schema=root_schema,
            ref_stack=ref_stack,
        )
        for schema in schemas
    ):
        return
    raise CodexResultValidationError(f"{path} must match at least one schema in anyOf.")


def _validate_one_of(
    value: Any,
    schemas: list[Any],
    *,
    path: str,
    root_schema: dict[str, Any],
    ref_stack: tuple[str, ...],
) -> None:
    matches = sum(
        1
        for schema in schemas
        if _schema_matches(
            value,
            schema,
            path=path,
            root_schema=root_schema,
            ref_stack=ref_stack,
        )
    )
    if matches != 1:
        raise CodexResultValidationError(
            f"{path} must match exactly one schema in oneOf."
        )


def _schema_matches(
    value: Any,
    schema: Any,
    *,
    path: str,
    root_schema: dict[str, Any],
    ref_stack: tuple[str, ...],
) -> bool:
    try:
        _validate_schema_node_against_root(
            value,
            schema,
            path=path,
            root_schema=root_schema,
            ref_stack=ref_stack,
        )
    except CodexResultValidationError:
        return False
    return True


def _validate_size(
    size: int,
    *,
    minimum: Any,
    maximum: Any,
    path: str,
    noun: str,
) -> None:
    if minimum is not None and size < minimum:
        raise CodexResultValidationError(
            f"{path} must contain at least {minimum} {noun}."
        )
    if maximum is not None and size > maximum:
        raise CodexResultValidationError(
            f"{path} must contain at most {maximum} {noun}."
        )


def _matches_json_type(value: Any, schema_type: str) -> bool:
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return (isinstance(value, int | float)) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "null":
        return value is None
    raise CodexResultValidationError(f"Unsupported JSON Schema type: {schema_type}.")


def _assert_supported_schema(
    schema: Any,
    *,
    root_schema: dict[str, Any],
    path: str,
    ref_stack: tuple[str, ...] = (),
) -> None:
    if schema in (True, False):
        return
    if not isinstance(schema, dict):
        raise ValueError(f"{path} schema must be an object or boolean.")

    unsupported_keys = set(schema) - JSON_SCHEMA_SUPPORTED_KEYS
    if unsupported_keys:
        names = ", ".join(sorted(unsupported_keys))
        raise ValueError(f"{path} uses unsupported JSON Schema keywords: {names}.")

    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        _assert_supported_type(schema_type, path=path)
    elif isinstance(schema_type, list):
        if not schema_type:
            raise ValueError(f"{path}.type must not be empty.")
        for item in schema_type:
            if not isinstance(item, str):
                raise ValueError(f"{path}.type must contain only strings.")
            _assert_supported_type(item, path=path)
    elif schema_type is not None:
        raise ValueError(f"{path}.type must be a string or a list of strings.")

    ref = schema.get("$ref")
    if ref is not None:
        _resolve_schema_ref(root_schema, ref, ref_stack=ref_stack)

    _assert_object_schema(schema, root_schema=root_schema, path=path)
    _assert_array_schema(schema, root_schema=root_schema, path=path)
    _assert_string_schema(schema, path=path)
    _assert_numeric_schema(schema, path=path)

    for key in COMBINATOR_KEYS:
        if key in schema:
            _assert_schema_list(
                schema[key],
                key=key,
                root_schema=root_schema,
                path=path,
                ref_stack=ref_stack,
            )
    if "not" in schema:
        _assert_supported_schema(
            schema["not"],
            root_schema=root_schema,
            path=f"{path}.not",
            ref_stack=ref_stack,
        )

    for definitions_key in ("$defs", "definitions"):
        definitions = schema.get(definitions_key)
        if definitions is None:
            continue
        if not isinstance(definitions, dict):
            raise ValueError(f"{path}.{definitions_key} must be an object.")
        for name, definition in definitions.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"{path}.{definitions_key} keys must be strings.")
            _assert_supported_schema(
                definition,
                root_schema=root_schema,
                path=f"{path}.{definitions_key}.{name}",
                ref_stack=ref_stack,
            )


def _assert_object_schema(
    schema: dict[str, Any],
    *,
    root_schema: dict[str, Any],
    path: str,
) -> None:
    required = schema.get("required")
    if required is not None and (
        not isinstance(required, list)
        or not all(isinstance(item, str) for item in required)
    ):
        raise ValueError(f"{path}.required must be a list of strings.")

    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, dict):
            raise ValueError(f"{path}.properties must be an object.")
        for key, property_schema in properties.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}.properties keys must be strings.")
            _assert_supported_schema(
                property_schema,
                root_schema=root_schema,
                path=f"{path}.properties.{key}",
            )

    additional_properties = schema.get("additionalProperties")
    if additional_properties not in (None, True, False):
        if not isinstance(additional_properties, dict):
            raise ValueError(
                f"{path}.additionalProperties must be a boolean or object."
            )
        _assert_supported_schema(
            additional_properties,
            root_schema=root_schema,
            path=f"{path}.additionalProperties",
        )

    _assert_non_negative_int(schema, "minProperties", path=path)
    _assert_non_negative_int(schema, "maxProperties", path=path)


def _assert_array_schema(
    schema: dict[str, Any],
    *,
    root_schema: dict[str, Any],
    path: str,
) -> None:
    items = schema.get("items")
    if items is not None:
        _assert_supported_schema(items, root_schema=root_schema, path=f"{path}.items")

    unique_items = schema.get("uniqueItems")
    if unique_items is not None and not isinstance(unique_items, bool):
        raise ValueError(f"{path}.uniqueItems must be a boolean.")

    _assert_non_negative_int(schema, "minItems", path=path)
    _assert_non_negative_int(schema, "maxItems", path=path)


def _assert_string_schema(schema: dict[str, Any], *, path: str) -> None:
    _assert_non_negative_int(schema, "minLength", path=path)
    _assert_non_negative_int(schema, "maxLength", path=path)

    pattern = schema.get("pattern")
    if pattern is not None:
        if not isinstance(pattern, str):
            raise ValueError(f"{path}.pattern must be a string.")
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError(f"{path}.pattern is not a valid regex: {error}") from error


def _assert_numeric_schema(schema: dict[str, Any], *, path: str) -> None:
    for key in (
        "exclusiveMaximum",
        "exclusiveMinimum",
        "maximum",
        "minimum",
        "multipleOf",
    ):
        value = schema.get(key)
        if value is not None and (
            not isinstance(value, int | float) or isinstance(value, bool)
        ):
            raise ValueError(f"{path}.{key} must be a number.")
    if schema.get("multipleOf") == 0:
        raise ValueError(f"{path}.multipleOf must not be zero.")


def _assert_schema_list(
    value: Any,
    *,
    key: str,
    root_schema: dict[str, Any],
    path: str,
    ref_stack: tuple[str, ...],
) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}.{key} must be a non-empty list.")
    for index, subschema in enumerate(value):
        _assert_supported_schema(
            subschema,
            root_schema=root_schema,
            path=f"{path}.{key}[{index}]",
            ref_stack=ref_stack,
        )


def _assert_non_negative_int(
    schema: dict[str, Any],
    key: str,
    *,
    path: str,
) -> None:
    value = schema.get(key)
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value < 0
    ):
        raise ValueError(f"{path}.{key} must be a non-negative integer.")


def _assert_supported_type(schema_type: str, *, path: str) -> None:
    if schema_type not in JSON_TYPES:
        supported = ", ".join(sorted(JSON_TYPES))
        raise ValueError(f"{path}.type must be one of: {supported}.")


def _resolve_schema_ref(
    root_schema: dict[str, Any],
    ref: Any,
    *,
    ref_stack: tuple[str, ...],
) -> Any:
    if not isinstance(ref, str) or not ref:
        raise ValueError("$ref must be a non-empty string.")
    if ref in ref_stack:
        raise ValueError(f"Recursive $ref is not supported: {ref}.")
    if ref == "#":
        return root_schema
    if not ref.startswith("#/"):
        raise ValueError(f"Only local JSON Schema refs are supported: {ref}.")

    node: Any = root_schema
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"Unresolved JSON Schema ref: {ref}.")
        node = node[part]
    return node


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
    "CodexEventParseError",
    "CodexExecution",
    "CodexExecutionFailure",
    "CodexExecutionStatus",
    "CodexExecutor",
    "CodexFailureKind",
    "CodexProcessOutputDecodeError",
    "CodexProcessResult",
    "CodexProcessRunner",
    "CodexProcessTimedOut",
    "CodexProcessTimeout",
    "CodexResultValidationError",
    "Sandbox",
    "SubprocessCodexRunner",
    "build_codex_command",
    "execute",
    "extract_structured_result",
    "parse_codex_events",
    "parse_sandbox",
    "validate_json_schema",
]
