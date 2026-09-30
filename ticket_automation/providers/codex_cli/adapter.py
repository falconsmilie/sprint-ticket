"""Codex CLI adapter for the provider-neutral agent execution port."""

from __future__ import annotations

import json
import time
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
from ...persistence import JsonObject, format_timestamp
from .command import build_command, sandbox_for
from .evidence import (
    RESULT_ARTIFACT,
    CodexArtifactPaths,
    prepare,
    process_capture_environment,
    publish_process_output,
    write_diagnostic_result,
    write_diagnostic_result_bytes,
    write_execution,
    write_process_output,
    write_result,
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
    CLEANUP_SECONDS,
    CodexCommand,
    CodexProcessEvidence,
    CodexProcessRunner,
    CodexProcessTimedOut,
    CodexScratchDirectoryError,
    SubprocessCodexRunner,
    external_scratch_environment,
    with_capture_paths,
    with_environment,
)
from .results import decode_result
from .settings import CodexCliSettings, validate_codex_cli_settings

ResultT = TypeVar("ResultT", bound=TaskResult)


class _InvocationStartObserverError(BaseException):
    def __init__(self, error: BaseException) -> None:
        super().__init__(str(error))
        self.error = error


_DOMAIN_SCHEMAS = Path(__file__).resolve().parents[2] / "domain" / "schemas"
_IMPLEMENTATION_SCHEMA = _DOMAIN_SCHEMAS / "implementation-result.schema.json"
_REVIEW_SCHEMA = _DOMAIN_SCHEMAS / "review-result.schema.json"


class CodexCliAgentExecutor:
    """Execute neutral application requests through an isolated Codex CLI call."""

    provider_id = PROVIDER_ID

    @property
    def capabilities(self) -> frozenset[AgentCapability]:
        return CAPABILITIES

    def __init__(
        self,
        settings: CodexCliSettings,
        *,
        configuration_directory: Path | None = None,
        runner: CodexProcessRunner | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = validate_codex_cli_settings(settings)
        self.configuration_directory = configuration_directory
        self.runner = runner or SubprocessCodexRunner()
        self._clock = clock or (lambda: datetime.now(UTC))

    def execute(
        self,
        request: AgentExecutionRequest[ResultT],
        *,
        on_invocation_start: Callable[[], None] | None = None,
    ) -> AgentExecution[ResultT]:
        assert request.artifact_layout is not None
        paths = CodexArtifactPaths.create(request.artifact_layout)
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

        raw_result: str | None = None
        raw_result_bytes: bytes | None = None
        result_read_error: OSError | UnicodeError | None = None
        output_publication_truncated = False
        command = configured_command
        overall_deadline = (
            None
            if request.policy.timeout_seconds is None
            else time.monotonic() + request.policy.timeout_seconds + CLEANUP_SECONDS
        )
        try:
            with (
                external_scratch_environment(
                    request.repository_path,
                    cleanup_deadline=overall_deadline,
                ) as environment,
                process_capture_environment(paths) as capture_paths,
            ):
                scratch_result = Path(environment["TEMP"]) / RESULT_ARTIFACT
                command = build_command(
                    executable=str(resolved),
                    repository_path=request.repository_path,
                    repository_access=request.repository_access,
                    network_access=request.policy.network_access,
                    settings=self.settings,
                    output_schema=schema,
                    output_result=scratch_result,
                )
                command = with_environment(command, environment)
                command = with_capture_paths(
                    command,
                    stdout=capture_paths[0],
                    stderr=capture_paths[1],
                )
                process_started = [False]
                process_start_observer = _process_start_observer(
                    on_invocation_start,
                    on_started=lambda: process_started.__setitem__(0, True),
                )
                runner_started = time.monotonic()
                try:
                    process = self.runner.run(
                        command,
                        stdin=request.prompt,
                        timeout_seconds=request.policy.timeout_seconds,
                        on_process_start=process_start_observer,
                    )
                except _InvocationStartObserverError as error:
                    raise error.error
                except CodexProcessTimedOut as error:
                    publication = publish_process_output(
                        paths,
                        stdout_source=error.result.stdout_capture,
                        stderr_source=error.result.stderr_capture,
                        stdout_fallback=error.result.stdout,
                        stderr_fallback=error.result.stderr,
                        stdout_capture_complete=(error.result.stdout_capture_complete),
                        stderr_capture_complete=(error.result.stderr_capture_complete),
                    )
                    diagnostic_bytes, diagnostic, _ = _read_scratch_result(
                        scratch_result
                    )
                    _write_scratch_diagnostic(paths, diagnostic_bytes, diagnostic)
                    process_metadata = _process_evidence_metadata(
                        error.result.evidence,
                        configured_timeout=error.result.timeout_seconds,
                    )
                    process_metadata["output_publication_truncated"] = (
                        publication.truncated
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
                        structured_result_present=diagnostic_bytes is not None,
                        extra_metadata=process_metadata,
                    )
                except Exception as error:  # noqa: BLE001 - transport boundary
                    if process_started[0]:
                        publication = publish_process_output(
                            paths,
                            stdout_source=command.stdout_capture,
                            stderr_source=command.stderr_capture,
                            stdout_fallback="",
                            stderr_fallback=f"{error}\n",
                            stdout_capture_complete=False,
                            stderr_capture_complete=False,
                        )
                        diagnostic_bytes, diagnostic, _ = _read_scratch_result(
                            scratch_result
                        )
                        _write_scratch_diagnostic(paths, diagnostic_bytes, diagnostic)
                        elapsed = max(0.0, time.monotonic() - runner_started)
                        process_metadata = _process_evidence_metadata(
                            CodexProcessEvidence(
                                work_timeout_seconds=request.policy.timeout_seconds,
                                work_elapsed_seconds=elapsed,
                                total_elapsed_seconds=elapsed,
                                deadline_outcome="transport-exception",
                                finalization_outcome="failed",
                                cleanup_outcome="unknown",
                                tree_termination_confirmed=False,
                                output_draining_truncated=True,
                            ),
                            configured_timeout=request.policy.timeout_seconds,
                        )
                        process_metadata["output_publication_truncated"] = (
                            publication.truncated
                        )
                        return self._failure(
                            request,
                            paths,
                            started,
                            reason=CodexFailureReason.TRANSPORT_FAILURE,
                            message=f"Agent provider transport failed: {error}",
                            command=command,
                            exit_code=None,
                            timeout_seconds=request.policy.timeout_seconds,
                            structured_result_present=(diagnostic_bytes is not None),
                            extra_metadata=process_metadata,
                        )
                    write_process_output(paths, stdout="", stderr=f"{error}\n")
                    if isinstance(error, FileNotFoundError):
                        reason = CodexFailureReason.EXECUTABLE_UNAVAILABLE
                        message = (
                            "Agent provider executable is unavailable: "
                            f"{command.argv[0]}"
                        )
                    else:
                        reason = CodexFailureReason.PROCESS_START_FAILED
                        message = f"Could not start agent provider process: {error}"
                    return self._failure(
                        request,
                        paths,
                        started,
                        reason=reason,
                        message=message,
                        command=command,
                        exit_code=None,
                    )
                raw_result_bytes, raw_result, result_read_error = _read_scratch_result(
                    scratch_result
                )
                if raw_result_bytes is not None and raw_result is None:
                    write_diagnostic_result_bytes(paths, raw_result_bytes)
                publication = publish_process_output(
                    paths,
                    stdout_source=process.stdout_capture,
                    stderr_source=process.stderr_capture,
                    stdout_fallback=process.stdout,
                    stderr_fallback=process.stderr,
                    stdout_capture_complete=process.stdout_capture_complete,
                    stderr_capture_complete=process.stderr_capture_complete,
                )
                output_publication_truncated = publication.truncated
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
        process_metadata = _process_evidence_metadata(
            process.evidence,
            configured_timeout=request.policy.timeout_seconds,
        )
        process_metadata["output_publication_truncated"] = output_publication_truncated
        if process.transport_failure is not None:
            if raw_result is not None:
                write_diagnostic_result(paths, raw_result)
            reason = (
                CodexFailureReason.FINALIZATION_FAILED
                if process.evidence.finalization_outcome == "expired"
                else CodexFailureReason.TRANSPORT_FAILURE
            )
            return self._failure(
                request,
                paths,
                started,
                reason=reason,
                message=process.transport_failure,
                command=command,
                exit_code=process.returncode,
                timeout_seconds=request.policy.timeout_seconds,
                structured_result_present=raw_result_bytes is not None,
                extra_metadata=process_metadata,
            )
        if process.returncode != 0:
            if raw_result is not None:
                write_diagnostic_result(paths, raw_result)
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
                timeout_seconds=request.policy.timeout_seconds,
                structured_result_present=raw_result_bytes is not None,
                extra_metadata=process_metadata,
            )

        if not (
            process.evidence.completion_before_deadline
            and process.evidence.structured_message_before_deadline
            and process.evidence.structured_message is not None
        ):
            if raw_result is not None:
                write_diagnostic_result(paths, raw_result)
            return self._failure(
                request,
                paths,
                started,
                reason=CodexFailureReason.TRANSPORT_FAILURE,
                message=(
                    "Agent provider exited without a terminal success and final "
                    "structured message observed before the work deadline."
                ),
                command=command,
                exit_code=process.returncode,
                timeout_seconds=request.policy.timeout_seconds,
                structured_result_present=raw_result_bytes is not None,
                extra_metadata=process_metadata,
            )

        if raw_result is None:
            error = result_read_error or FileNotFoundError(paths.result)
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
                timeout_seconds=request.policy.timeout_seconds,
                structured_result_present=raw_result_bytes is not None,
                extra_metadata=process_metadata,
            )

        try:
            raw_value = json.loads(raw_result)
            if process.evidence.structured_message is not None:
                message_value = json.loads(process.evidence.structured_message)
                if message_value != raw_value:
                    raise ResultValidationError(
                        "Canonical result did not match the timely final agent message."
                    )
            decoded = decode_result(raw_value, request)
        except json.JSONDecodeError as error:
            write_diagnostic_result(paths, raw_result)
            return self._failure(
                request,
                paths,
                started,
                reason=CodexFailureReason.INVALID_STRUCTURED_RESULT,
                message=f"Agent typed result was not valid JSON: {error.msg}.",
                command=command,
                exit_code=process.returncode,
                structured_result_present=True,
                timeout_seconds=request.policy.timeout_seconds,
                extra_metadata=process_metadata,
            )
        except (ResultValidationError, TypeError) as error:
            write_diagnostic_result(paths, raw_result)
            return self._failure(
                request,
                paths,
                started,
                reason=CodexFailureReason.INVALID_STRUCTURED_RESULT,
                message=str(error),
                command=command,
                exit_code=process.returncode,
                structured_result_present=True,
                timeout_seconds=request.policy.timeout_seconds,
                extra_metadata=process_metadata,
            )

        write_result(paths, raw_result)
        process_metadata.update(
            {
                "structured_result_validated": True,
                "structured_result_accepted": True,
            }
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
                timeout_seconds=request.policy.timeout_seconds,
                structured_result_present=True,
                extra_metadata=process_metadata,
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


def _process_start_observer(
    observer: Callable[[], None] | None,
    *,
    on_started: Callable[[], None] | None = None,
) -> Callable[[], None] | None:
    if observer is None and on_started is None:
        return None

    def notify() -> None:
        if on_started is not None:
            on_started()
        if observer is None:
            return
        try:
            observer()
        except BaseException as error:
            raise _InvocationStartObserverError(error) from error

    return notify


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


def _read_scratch_result(
    path: Path,
) -> tuple[bytes | None, str | None, OSError | UnicodeError | None]:
    value: bytes | None = None
    try:
        value = path.read_bytes()
    except OSError as error:
        return None, None, error
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        return value, value.decode("utf-8"), None
    except UnicodeError as error:
        return value, None, error


def _write_scratch_diagnostic(
    paths: CodexArtifactPaths,
    value: bytes | None,
    text: str | None,
) -> None:
    if value is None:
        return
    if text is None:
        write_diagnostic_result_bytes(paths, value)
    else:
        write_diagnostic_result(paths, text)


def _process_evidence_metadata(
    evidence: CodexProcessEvidence,
    *,
    configured_timeout: float | None,
) -> dict[str, ProviderMetadataValue]:
    return {
        "work_timeout_seconds": (
            configured_timeout
            if evidence.work_timeout_seconds is None
            else evidence.work_timeout_seconds
        ),
        "work_elapsed_seconds": evidence.work_elapsed_seconds,
        "total_elapsed_seconds": evidence.total_elapsed_seconds,
        "terminal_event_type": evidence.terminal_event_type,
        "terminal_event_elapsed_seconds": evidence.terminal_event_elapsed_seconds,
        "completion_before_deadline": evidence.completion_before_deadline,
        "structured_message_observed": evidence.structured_message is not None,
        "structured_message_elapsed_seconds": (
            evidence.structured_message_elapsed_seconds
        ),
        "structured_message_before_deadline": (
            evidence.structured_message_before_deadline
        ),
        "deadline_outcome": evidence.deadline_outcome,
        "finalization_outcome": evidence.finalization_outcome,
        "cleanup_outcome": evidence.cleanup_outcome,
        "termination_method": evidence.termination_method,
        "tree_termination_confirmed": evidence.tree_termination_confirmed,
        "cleanup_duration_seconds": evidence.cleanup_duration_seconds,
        "output_draining_truncated": evidence.output_draining_truncated,
        "event_stream_problem": evidence.event_stream_problem,
        "structured_result_validated": False,
        "structured_result_accepted": False,
    }


def _metadata(
    *,
    request: AgentExecutionRequest[TaskResult],
    settings: CodexCliSettings,
    command: CodexCommand | None,
    exit_code: int | None,
    reason: CodexFailureReason | None,
    timed_out: bool,
    timeout_seconds: float | None,
    structured_result_present: bool,
    extra_metadata: dict[str, ProviderMetadataValue] | None = None,
) -> dict[str, ProviderMetadataValue]:
    metadata: dict[str, ProviderMetadataValue] = {
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
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata


def _execution_record(execution: AgentExecution[TaskResult]) -> JsonObject:
    metadata = execution.provider_metadata
    return {
        "schema_version": 2,
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
        "started_at": format_timestamp(execution.started_at),
        "ended_at": format_timestamp(execution.ended_at),
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
        "structured_result_validated": metadata.get(
            "structured_result_validated", False
        ),
        "structured_result_accepted": metadata.get("structured_result_accepted", False),
        "work_timeout_seconds": metadata.get("work_timeout_seconds"),
        "work_elapsed_seconds": metadata.get("work_elapsed_seconds"),
        "total_elapsed_seconds": metadata.get("total_elapsed_seconds"),
        "terminal_event_type": metadata.get("terminal_event_type"),
        "terminal_event_elapsed_seconds": metadata.get(
            "terminal_event_elapsed_seconds"
        ),
        "completion_before_deadline": metadata.get("completion_before_deadline", False),
        "structured_message_observed": metadata.get(
            "structured_message_observed", False
        ),
        "structured_message_elapsed_seconds": metadata.get(
            "structured_message_elapsed_seconds"
        ),
        "structured_message_before_deadline": metadata.get(
            "structured_message_before_deadline", False
        ),
        "deadline_outcome": metadata.get("deadline_outcome"),
        "finalization_outcome": metadata.get("finalization_outcome"),
        "cleanup_outcome": metadata.get("cleanup_outcome"),
        "termination_method": metadata.get("termination_method"),
        "tree_termination_confirmed": metadata.get("tree_termination_confirmed"),
        "cleanup_duration_seconds": metadata.get("cleanup_duration_seconds"),
        "output_draining_truncated": metadata.get("output_draining_truncated", False),
        "output_publication_truncated": metadata.get(
            "output_publication_truncated", False
        ),
        "event_stream_problem": metadata.get("event_stream_problem"),
        "result_json_present": any(
            artifact.name == "typed-result" for artifact in execution.artifacts
        ),
        "artifact_paths": {
            artifact.name: artifact.run_relative_path
            for artifact in execution.artifacts
        },
    }


__all__ = ["CodexCliAgentExecutor"]
