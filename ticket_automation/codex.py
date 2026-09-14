from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, TypedDict, cast

from . import executable_resolution
from .config import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING_EFFORT,
    CodexExecutionSettings,
    validate_codex_execution_settings,
)
from .process_output import decode_human_output

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


class CodexResultValidationError(ValueError):
    """Raised when a fixed V1 structured result cannot be parsed."""


class _CodexResultKind(StrEnum):
    IMPLEMENTATION = "implementation"
    REVIEW = "review"


class _ImplementationTestResult(TypedDict):
    command: str
    result: str


class _ImplementationResult(TypedDict):
    status: Literal["COMPLETED", "BLOCKED"]
    summary: str
    tests_run: list[_ImplementationTestResult]
    assumptions: list[str]
    known_issues: list[str]


class _ReviewFindingResult(TypedDict):
    id: str
    disposition: Literal["REQUIRED", "ADVISORY", "FOLLOW_UP"]
    scope_relation: Literal[
        "TICKET",
        "IMPLEMENTATION",
        "REPOSITORY_AUTHORITY",
        "OUT_OF_SCOPE",
        "AMBIGUOUS",
    ]
    title: str
    description: str
    evidence: str
    required_change: str
    acceptance_criteria: list[str]


class _ReviewResult(TypedDict):
    verdict: Literal["PASS", "CORRECTIONS_REQUIRED", "HUMAN_REVIEW_REQUIRED"]
    summary: str
    findings: list[_ReviewFindingResult]


_StructuredResult: TypeAlias = _ImplementationResult | _ReviewResult


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
    structured_result: _StructuredResult | None = None
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
    return CodexCommand(
        argv=(
            executable,
            "exec",
            "--ephemeral",
            "--model",
            effective_execution_config.model,
            "-c",
            (f'model_reasoning_effort="{effective_execution_config.reasoning_effort}"'),
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
) -> _StructuredResult:
    if _result_kind == _CodexResultKind.IMPLEMENTATION:
        return _parse_implementation_result(value)
    if _result_kind == _CodexResultKind.REVIEW:
        return _parse_review_result(value)
    raise AssertionError(f"Unhandled Codex result kind: {_result_kind!r}.")


def _parse_implementation_result(value: Any) -> _ImplementationResult:
    result = _require_exact_object(
        value,
        required=("status", "summary", "tests_run", "assumptions", "known_issues"),
        source="implementation result",
    )
    status = _require_choice(
        result,
        "status",
        values=("COMPLETED", "BLOCKED"),
        source="implementation result",
    )
    return {
        "status": cast(Literal["COMPLETED", "BLOCKED"], status),
        "summary": _require_string(result, "summary", source="implementation result"),
        "tests_run": _parse_implementation_tests(result["tests_run"]),
        "assumptions": _parse_string_list(
            result["assumptions"],
            source="implementation result.assumptions",
        ),
        "known_issues": _parse_string_list(
            result["known_issues"],
            source="implementation result.known_issues",
        ),
    }


def _parse_review_result(value: Any) -> _ReviewResult:
    result = _require_exact_object(
        value,
        required=("verdict", "summary", "findings"),
        source="review result",
    )
    verdict = _require_choice(
        result,
        "verdict",
        values=("PASS", "CORRECTIONS_REQUIRED", "HUMAN_REVIEW_REQUIRED"),
        source="review result",
    )
    findings_value = result["findings"]
    if not isinstance(findings_value, list):
        raise CodexResultValidationError("review result.findings must be an array.")
    return {
        "verdict": cast(
            Literal["PASS", "CORRECTIONS_REQUIRED", "HUMAN_REVIEW_REQUIRED"],
            verdict,
        ),
        "summary": _require_string(result, "summary", source="review result"),
        "findings": [
            _parse_review_finding(item, index=index)
            for index, item in enumerate(findings_value)
        ],
    }


def _parse_implementation_tests(value: Any) -> list[_ImplementationTestResult]:
    if not isinstance(value, list):
        raise CodexResultValidationError(
            "implementation result.tests_run must be an array."
        )
    tests: list[_ImplementationTestResult] = []
    for index, item in enumerate(value):
        test = _require_exact_object(
            item,
            required=("command", "result"),
            source=f"implementation result.tests_run[{index}]",
        )
        tests.append(
            {
                "command": _require_string(
                    test,
                    "command",
                    source=f"implementation result.tests_run[{index}]",
                ),
                "result": _require_string(
                    test,
                    "result",
                    source=f"implementation result.tests_run[{index}]",
                ),
            }
        )
    return tests


def _parse_review_finding(value: Any, *, index: int) -> _ReviewFindingResult:
    source = f"review result.findings[{index}]"
    finding = _require_exact_object(
        value,
        required=(
            "id",
            "disposition",
            "scope_relation",
            "title",
            "description",
            "evidence",
            "required_change",
            "acceptance_criteria",
        ),
        source=source,
    )
    disposition = _require_choice(
        finding,
        "disposition",
        values=("REQUIRED", "ADVISORY", "FOLLOW_UP"),
        source=source,
    )
    scope_relation = _require_choice(
        finding,
        "scope_relation",
        values=(
            "TICKET",
            "IMPLEMENTATION",
            "REPOSITORY_AUTHORITY",
            "OUT_OF_SCOPE",
            "AMBIGUOUS",
        ),
        source=source,
    )
    return {
        "id": _require_string(finding, "id", source=source),
        "disposition": cast(
            Literal["REQUIRED", "ADVISORY", "FOLLOW_UP"],
            disposition,
        ),
        "scope_relation": cast(
            Literal[
                "TICKET",
                "IMPLEMENTATION",
                "REPOSITORY_AUTHORITY",
                "OUT_OF_SCOPE",
                "AMBIGUOUS",
            ],
            scope_relation,
        ),
        "title": _require_string(finding, "title", source=source),
        "description": _require_string(finding, "description", source=source),
        "evidence": _require_string(finding, "evidence", source=source),
        "required_change": _require_string(
            finding,
            "required_change",
            source=source,
        ),
        "acceptance_criteria": _parse_string_list(
            finding["acceptance_criteria"],
            source=f"{source}.acceptance_criteria",
            minimum_items=1,
        ),
    }


def _require_exact_object(
    value: Any,
    *,
    required: tuple[str, ...],
    source: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CodexResultValidationError(f"{source} must be an object.")
    missing = [field for field in required if field not in value]
    if missing:
        raise CodexResultValidationError(
            f"{source} is missing required fields: {', '.join(missing)}."
        )
    extra = sorted(set(value) - set(required))
    if extra:
        raise CodexResultValidationError(
            f"{source} contains unsupported fields: {', '.join(extra)}."
        )
    return value


def _require_string(data: dict[str, Any], field: str, *, source: str) -> str:
    value = data[field]
    if not isinstance(value, str) or not value.strip():
        raise CodexResultValidationError(
            f"{source}.{field} must be a non-empty string."
        )
    return value


def _require_choice(
    data: dict[str, Any],
    field: str,
    *,
    values: tuple[str, ...],
    source: str,
) -> str:
    value = _require_string(data, field, source=source)
    if value not in values:
        raise CodexResultValidationError(
            f"{source}.{field} must be one of: {', '.join(values)}."
        )
    return value


def _parse_string_list(
    value: Any,
    *,
    source: str,
    minimum_items: int = 0,
) -> list[str]:
    if not isinstance(value, list):
        raise CodexResultValidationError(f"{source} must be an array.")
    if len(value) < minimum_items:
        raise CodexResultValidationError(
            f"{source} must contain at least {minimum_items} item(s)."
        )
    strings: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise CodexResultValidationError(
                f"{source}[{index}] must be a non-empty string."
            )
        strings.append(item)
    return strings


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
    "CodexResultValidationError",
    "Sandbox",
    "SubprocessCodexRunner",
    "build_codex_command",
    "execute",
    "parse_sandbox",
]
