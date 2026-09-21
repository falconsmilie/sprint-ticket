"""Codex CLI adapter for the provider-neutral agent execution port."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar, cast

from ...application.agent_execution import (
    AgentCapability,
    AgentExecution,
    AgentExecutionRequest,
    AgentExecutionStatus,
    AgentTaskKind,
    InvocationStart,
    NetworkAccess,
    ProviderMetadataValue,
    RepositoryAccess,
)
from ...domain.task_results import ResultValidationError, TaskResult
from .command import build_command, sandbox_for
from .evidence import (
    CodexArtifactPaths,
    prepare,
    write_execution,
    write_process_output,
)
from .executable import resolve_executable
from .failures import (
    CodexFailureReason,
    invocation_start_for,
    looks_like_authentication_or_service_failure,
    map_failure,
)
from .identity import CAPABILITIES, PROVIDER_ID
from .process import (
    CodexCommand,
    CodexProcessRunner,
    CodexProcessTimedOut,
    CodexScratchDirectoryError,
    SubprocessCodexRunner,
    external_scratch_environment,
    with_environment,
)
from .results import decode_result
from .settings import CodexSettings, validate_codex_settings

ResultT = TypeVar("ResultT", bound=TaskResult)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_IMPLEMENTATION_SCHEMA = _PROJECT_ROOT / "schemas" / "implementation-result.schema.json"
_REVIEW_SCHEMA = _PROJECT_ROOT / "schemas" / "review-result.schema.json"


class CodexCliAgentExecutor:
    """Execute neutral application requests through an isolated Codex CLI call."""

    provider_id = PROVIDER_ID

    @property
    def capabilities(self) -> frozenset[AgentCapability]:
        """Capabilities are provider identity, not mutable executor state."""

        return CAPABILITIES

    def __init__(
        self,
        settings: CodexSettings,
        *,
        configuration_directory: Path | None = None,
        runner: CodexProcessRunner | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = validate_codex_settings(settings)
        self.configuration_directory = configuration_directory
        self.runner = runner or SubprocessCodexRunner()
        self._clock = clock or (lambda: datetime.now(UTC))

    def execute(
        self,
        request: AgentExecutionRequest[ResultT],
        *,
        on_invocation_start: Callable[[], None] | None = None,
    ) -> AgentExecution[ResultT]:
        paths = CodexArtifactPaths.create(request.artifact_directory)
        started = self._clock()
        prepare(paths, request.prompt)

        missing = request.missing_capabilities(self.capabilities)
        if missing:
            names = ", ".join(sorted(item.value for item in missing))
            return self._failure(
                request,
                paths,
                started,
                reason=CodexFailureReason.CAPABILITY_REJECTED,
                message=f"Provider does not support required capabilities: {names}.",
                command=None,
                exit_code=None,
                extra_metadata={
                    "missing_capabilities": tuple(
                        sorted(item.value for item in missing)
                    )
                },
            )

        policy_problem = _policy_problem(request)
        if policy_problem is not None:
            return self._failure(
                request,
                paths,
                started,
                reason=CodexFailureReason.POLICY_REJECTED,
                message=policy_problem,
                command=None,
                exit_code=None,
            )

        schema = _schema_for(request.task_kind)
        configured_command = build_command(
            executable=self.settings.executable,
            repository_path=request.repository_path,
            repository_access=request.repository_access,
            network_access=request.policy.network_access,
            settings=self.settings,
            output_schema=schema,
            output_result=paths.result,
        )

        project_config = request.repository_path / ".codex" / "config.toml"
        if project_config.is_file():
            message = (
                "Target repository contains .codex/config.toml; project provider "
                "configuration cannot be isolated reliably."
            )
            write_process_output(
                paths,
                stdout="",
                stderr=f"Target repository Codex configuration rejected: {project_config}\n",
            )
            return self._failure(
                request,
                paths,
                started,
                reason=CodexFailureReason.PROJECT_CONFIGURATION_REJECTED,
                message=message,
                command=configured_command,
                exit_code=None,
            )

        resolved = resolve_executable(
            self.settings.executable,
            config_dir=self.configuration_directory,
        )
        if resolved is None:
            write_process_output(
                paths,
                stdout="",
                stderr=f"Executable not found: {self.settings.executable}\n",
            )
            return self._failure(
                request,
                paths,
                started,
                reason=CodexFailureReason.EXECUTABLE_UNAVAILABLE,
                message=f"Agent provider executable is unavailable: {self.settings.executable}",
                command=configured_command,
                exit_code=None,
            )

        command = build_command(
            executable=str(resolved),
            repository_path=request.repository_path,
            repository_access=request.repository_access,
            network_access=request.policy.network_access,
            settings=self.settings,
            output_schema=schema,
            output_result=paths.result,
        )
        try:
            with external_scratch_environment(request.repository_path) as environment:
                command = with_environment(command, environment)
                if on_invocation_start is not None:
                    on_invocation_start()
                try:
                    process = self.runner.run(
                        command,
                        stdin=request.prompt,
                        timeout_seconds=request.policy.timeout_seconds,
                    )
                except FileNotFoundError as error:
                    write_process_output(paths, stdout="", stderr=f"{error}\n")
                    return self._failure(
                        request,
                        paths,
                        started,
                        reason=CodexFailureReason.EXECUTABLE_UNAVAILABLE,
                        message=(
                            "Agent provider executable is unavailable: "
                            f"{command.argv[0]}"
                        ),
                        command=command,
                        exit_code=None,
                    )
                except CodexProcessTimedOut as error:
                    write_process_output(
                        paths,
                        stdout=error.result.stdout,
                        stderr=error.result.stderr,
                    )
                    return self._failure(
                        request,
                        paths,
                        started,
                        reason=CodexFailureReason.TIMEOUT,
                        message=str(error),
                        command=command,
                        exit_code=None,
                        timed_out=True,
                        timeout_seconds=error.result.timeout_seconds,
                    )
                except OSError as error:
                    write_process_output(paths, stdout="", stderr=f"{error}\n")
                    return self._failure(
                        request,
                        paths,
                        started,
                        reason=CodexFailureReason.PROCESS_START_FAILED,
                        message=f"Could not start agent provider process: {error}",
                        command=command,
                        exit_code=None,
                    )
        except CodexScratchDirectoryError as error:
            write_process_output(paths, stdout="", stderr=f"{error}\n")
            return self._failure(
                request,
                paths,
                started,
                reason=CodexFailureReason.PROCESS_START_FAILED,
                message=str(error),
                command=command,
                exit_code=None,
            )
        write_process_output(paths, stdout=process.stdout, stderr=process.stderr)
        if process.returncode != 0:
            reason = (
                CodexFailureReason.AUTHENTICATION_OR_SERVICE
                if looks_like_authentication_or_service_failure(process.stderr)
                else CodexFailureReason.NON_ZERO_EXIT
            )
            return self._failure(
                request,
                paths,
                started,
                reason=reason,
                message=f"Agent provider exited with code {process.returncode}.",
                command=command,
                exit_code=process.returncode,
            )

        try:
            raw_result = paths.result.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError) as error:
            detail = (
                "did not write the required typed result"
                if isinstance(error, FileNotFoundError)
                else f"typed result could not be read: {error}"
            )
            return self._failure(
                request,
                paths,
                started,
                reason=CodexFailureReason.MISSING_STRUCTURED_RESULT,
                message=f"Agent provider {detail}.",
                command=command,
                exit_code=process.returncode,
            )

        try:
            decoded = decode_result(
                json.loads(raw_result),
                request,
            )
        except json.JSONDecodeError as error:
            return self._failure(
                request,
                paths,
                started,
                reason=CodexFailureReason.INVALID_STRUCTURED_RESULT,
                message=f"Agent typed result was not valid JSON: {error.msg}.",
                command=command,
                exit_code=process.returncode,
                structured_result_present=True,
            )
        except (ResultValidationError, TypeError) as error:
            return self._failure(
                request,
                paths,
                started,
                reason=CodexFailureReason.INVALID_STRUCTURED_RESULT,
                message=str(error),
                command=command,
                exit_code=process.returncode,
                structured_result_present=True,
            )

        ended = _ended_at(self._clock(), started)
        execution = AgentExecution(
            provider_id=self.provider_id,
            task_kind=request.task_kind,
            status=AgentExecutionStatus.SUCCESS,
            invocation_start=InvocationStart.STARTED,
            started_at=started,
            ended_at=ended,
            duration_seconds=(ended - started).total_seconds(),
            result=decoded,
            artifacts=paths.references(),
            provider_metadata=_metadata(
                request=request,
                settings=self.settings,
                command=command,
                exit_code=process.returncode,
                reason=None,
                timed_out=False,
                timeout_seconds=None,
                structured_result_present=True,
            ),
        )
        write_execution(paths, _execution_record(execution))
        return execution

    def _failure(
        self,
        request: AgentExecutionRequest[ResultT],
        paths: CodexArtifactPaths,
        started: datetime,
        *,
        reason: CodexFailureReason,
        message: str,
        command: CodexCommand | None,
        exit_code: int | None,
        timed_out: bool = False,
        timeout_seconds: float | None = None,
        structured_result_present: bool = False,
        extra_metadata: dict[str, ProviderMetadataValue] | None = None,
    ) -> AgentExecution[ResultT]:
        ended = _ended_at(self._clock(), started)
        metadata = _metadata(
            request=request,
            settings=self.settings,
            command=command,
            exit_code=exit_code,
            reason=reason,
            timed_out=timed_out,
            timeout_seconds=timeout_seconds,
            structured_result_present=structured_result_present,
        )
        if extra_metadata:
            metadata.update(extra_metadata)
        execution = AgentExecution(
            provider_id=self.provider_id,
            task_kind=request.task_kind,
            status=AgentExecutionStatus.FAILED,
            invocation_start=invocation_start_for(reason),
            started_at=started,
            ended_at=ended,
            duration_seconds=(ended - started).total_seconds(),
            failure_category=map_failure(reason),
            failure_message=message,
            artifacts=paths.references(),
            provider_metadata=metadata,
        )
        write_execution(paths, _execution_record(execution))
        return execution


def _policy_problem(request: AgentExecutionRequest[TaskResult]) -> str | None:
    expected = (
        RepositoryAccess.READ_ONLY
        if request.task_kind is AgentTaskKind.REVIEW
        else RepositoryAccess.WORKSPACE_WRITE
    )
    if request.repository_access is not expected:
        return f"{request.task_kind.value} requires {expected.value} repository access."
    if (
        request.repository_access is RepositoryAccess.READ_ONLY
        and request.policy.network_access is NetworkAccess.ALLOWED
    ):
        return "Codex read-only execution cannot guarantee requested network access."
    return None


def _ended_at(candidate: datetime, started: datetime) -> datetime:
    """Keep persisted wall-clock intervals valid when the system clock moves back."""

    return max(candidate, started)


def _schema_for(task_kind: AgentTaskKind) -> Path:
    return (
        _REVIEW_SCHEMA if task_kind is AgentTaskKind.REVIEW else _IMPLEMENTATION_SCHEMA
    )


def _metadata(
    *,
    request: AgentExecutionRequest[TaskResult],
    settings: CodexSettings,
    command: CodexCommand | None,
    exit_code: int | None,
    reason: CodexFailureReason | None,
    timed_out: bool,
    timeout_seconds: float | None,
    structured_result_present: bool,
) -> dict[str, ProviderMetadataValue]:
    return {
        "argv": () if command is None else command.argv,
        "process_exit_code": exit_code,
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "native_failure_reason": None if reason is None else reason.value,
        "sandbox": sandbox_for(request.repository_access).value,
        "model": settings.model,
        "reasoning_effort": settings.reasoning_effort,
        "repository_path": str(request.repository_path.resolve()),
        "output_schema_path": str(_schema_for(request.task_kind).resolve()),
        "structured_result_present": structured_result_present,
    }


def _execution_record(execution: AgentExecution[TaskResult]) -> dict[str, object]:
    metadata = execution.provider_metadata
    return {
        "schema_version": 1,
        "format": "ticket_automation.codex_execution",
        "provider_id": str(execution.provider_id),
        "status": "SUCCESS" if execution.successful else "FAILED",
        "process_started": execution.invocation_start is InvocationStart.STARTED,
        "process_exit_code": metadata.get("process_exit_code"),
        "timed_out": metadata.get("timed_out", False),
        "timeout_seconds": metadata.get("timeout_seconds"),
        "failure_kind": metadata.get("native_failure_reason"),
        "failure_category": (
            None
            if execution.failure_category is None
            else execution.failure_category.value
        ),
        "failure_message": execution.failure_message,
        "started_at": execution.started_at.astimezone(UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "ended_at": execution.ended_at.astimezone(UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "duration_seconds": execution.duration_seconds,
        "sandbox": metadata.get("sandbox"),
        "codex": {
            "model": metadata.get("model"),
            "reasoning_effort": metadata.get("reasoning_effort"),
        },
        "argv": list(cast(tuple[str, ...], metadata.get("argv", ()))),
        "repo_path": metadata.get("repository_path"),
        "output_schema_path": metadata.get("output_schema_path"),
        "structured_result_present": metadata.get("structured_result_present", False),
        "result_json_present": any(
            artifact.name == "structured-result" and artifact.path.is_file()
            for artifact in execution.artifacts
        ),
        "artifact_paths": {
            artifact.name: str(artifact.path) for artifact in execution.artifacts
        },
    }


__all__ = ["CodexCliAgentExecutor"]
