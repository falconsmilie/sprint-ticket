from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from tests.helpers import (
    completed_codex_process_result,
    prepend_executable_path,
    write_path_executable,
)
from ticket_automation import executable_resolution
from ticket_automation import persistence as persistence_module
from ticket_automation.application.agent_execution import (
    CORRECTION_RESULT_CONTRACT,
    IMPLEMENTATION_RESULT_CONTRACT,
    REVIEW_RESULT_CONTRACT,
    AgentCapability,
    AgentExecutionPolicy,
    AgentExecutionRequest,
    AgentExecutionStatus,
    AgentFailureCategory,
    AgentTaskKind,
    InvocationStart,
    NetworkAccess,
    RepositoryAccess,
    required_execution_capabilities,
)
from ticket_automation.domain.task_results import ImplementationResult, ReviewResult
from ticket_automation.persistence import PersistenceError
from ticket_automation.providers.codex_cli import (
    CodexCliAgentExecutor,
    CodexCliSettings,
    CodexCliSettingsError,
    CodexCommand,
    CodexProcessEvidence,
    CodexProcessResult,
    CodexProcessTimedOut,
    CodexProcessTimeout,
    SubprocessCodexRunner,
)
from ticket_automation.providers.codex_cli import process as process_module
from ticket_automation.providers.codex_cli.command import build_command

EXISTING_EXECUTABLE = str(Path(sys.executable).resolve())
SETTINGS = CodexCliSettings(EXISTING_EXECUTABLE, "gpt-5.5", "xhigh")


def implementation_payload(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "COMPLETED",
        "summary": "Implemented.",
        "tests_run": [],
        "assumptions": [],
        "known_issues": [],
    }
    payload.update(changes)
    return payload


def review_payload(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "verdict": "PASS",
        "summary": "Approved.",
        "findings": [],
    }
    payload.update(changes)
    return payload


@dataclass
class FakeRunner:
    result: CodexProcessResult | None = None
    typed_result: str | None = None
    error: BaseException | None = None
    command: CodexCommand | None = None
    stdin: str | None = None
    timeout_seconds: float | None = None
    calls: int = 0

    def run(
        self,
        command: CodexCommand,
        *,
        stdin: str,
        timeout_seconds: float | None,
        on_process_start=None,
    ) -> CodexProcessResult:
        self.calls += 1
        self.command = command
        self.stdin = stdin
        self.timeout_seconds = timeout_seconds
        if self.error is not None:
            raise self.error
        if on_process_start is not None:
            on_process_start()
        if self.typed_result is not None:
            output = Path(command.argv[command.argv.index("--output-last-message") + 1])
            output.write_text(self.typed_result, encoding="utf-8")
        assert self.result is not None
        if (
            self.result.returncode == 0
            and self.result.evidence == CodexProcessEvidence()
        ):
            fallback = (
                review_payload()
                if "review-result.schema.json" in " ".join(command.argv)
                else implementation_payload()
            )
            return replace(
                self.result,
                evidence=completed_codex_process_result(
                    self.typed_result or fallback,
                    timeout_seconds=timeout_seconds or 60,
                ).evidence,
            )
        return self.result


class ControlledMonotonic:
    """Clock whose reader observations are independent from monitor polling."""

    def __init__(self, monitor_time: float) -> None:
        self.monitor_time = monitor_time
        self.observed_time = 0.0
        self.launched = False
        self.local = threading.local()
        self.lock = threading.Lock()

    def observe(self, value: float) -> None:
        self.local.value = value
        with self.lock:
            self.observed_time = max(self.observed_time, value)

    def __call__(self) -> float:
        name = threading.current_thread().name
        if name == "codex-stdin-writer":
            return 0.1
        if name == "codex-stdout-reader":
            return getattr(self.local, "value", 0.0)
        with self.lock:
            if not self.launched:
                self.launched = True
                return 0.0
            return max(self.monitor_time, self.observed_time)


class ControlledOutput:
    _codex_close_is_nonblocking = True

    def __init__(
        self,
        clock: ControlledMonotonic,
        chunks: list[tuple[float, bytes]],
        *,
        release: threading.Event | None = None,
        finished_event: threading.Event | None = None,
        eof_error: BaseException | None = None,
    ) -> None:
        self.clock = clock
        self.chunks = list(chunks)
        self.release = release
        self.finished_event = finished_event
        self.eof_error = eof_error
        self.finished = False
        self.closed = False

    def read(self, size: int) -> bytes:
        del size
        if self.release is not None:
            self.release.wait(timeout=2)
        if self.closed or not self.chunks:
            self.finished = True
            if self.finished_event is not None:
                self.finished_event.set()
            if self.eof_error is not None and not self.closed:
                raise self.eof_error
            return b""
        observed, chunk = self.chunks.pop(0)
        self.clock.observe(observed)
        return chunk

    read1 = read

    def close(self) -> None:
        self.closed = True
        if self.release is not None:
            self.release.set()


class ControlledInput:
    _codex_close_is_nonblocking = True

    def __init__(
        self,
        *,
        release: threading.Event,
        blocked: bool = False,
    ) -> None:
        self.release = release
        self.blocked = blocked
        self.closed = False

    def write(self, value: bytes) -> int:
        if self.blocked:
            self.release.wait(timeout=2)
            raise BrokenPipeError("controlled stdin remained blocked")
        return len(value)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True
        self.release.set()


class ControlledProcess:
    def __init__(
        self,
        clock: ControlledMonotonic,
        stdout_chunks: list[tuple[float, bytes]],
        *,
        release_output_on_kill: bool = False,
        release_output_on_poll: bool = False,
        blocked_stdin: bool = False,
        stdout_eof_error: BaseException | None = None,
    ) -> None:
        self.release = threading.Event()
        self.output_finished = threading.Event()
        self.release_output_on_poll = release_output_on_poll
        self.stdin = ControlledInput(release=self.release, blocked=blocked_stdin)
        self.stdout = ControlledOutput(
            clock,
            stdout_chunks,
            release=(
                self.release
                if release_output_on_kill or release_output_on_poll
                else None
            ),
            finished_event=self.output_finished,
            eof_error=stdout_eof_error,
        )
        self.stderr = ControlledOutput(clock, [])
        self.returncode: int | None = None
        self.killed = False

    def poll(self) -> int | None:
        if self.release_output_on_poll:
            self.release.set()
            assert self.output_finished.wait(timeout=2)
            wait_until = time.monotonic() + 2
            while (
                not self.stderr.finished or not self.stdin.closed
            ) and time.monotonic() < wait_until:
                threading.Event().wait(0.001)
        if self.killed:
            self.returncode = -9
        elif self.stdout.finished and self.stderr.finished and self.stdin.closed:
            self.returncode = 0
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.release.set()

    def wait(self, *, timeout: float) -> int:
        del timeout
        returncode = self.poll()
        if returncode is None:
            raise subprocess.TimeoutExpired(("codex",), 0)
        return returncode


def completed_event_chunks(
    payload: dict[str, object],
    *,
    message_at: float,
    terminal_at: float,
) -> list[tuple[float, bytes]]:
    message = (
        json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": json.dumps(payload)},
            }
        ).encode()
        + b"\n"
    )
    return [
        (message_at, message),
        (terminal_at, b'{"type":"turn.completed"}\n'),
    ]


def request(
    tmp_path: Path,
    task_kind: AgentTaskKind = AgentTaskKind.IMPLEMENTATION,
    *,
    access: RepositoryAccess | None = None,
    capabilities: frozenset[AgentCapability] | None = None,
) -> AgentExecutionRequest:
    if task_kind is AgentTaskKind.REVIEW:
        access = access or RepositoryAccess.READ_ONLY
        contract = REVIEW_RESULT_CONTRACT
        network = NetworkAccess.DENIED
    elif task_kind is AgentTaskKind.CORRECTION:
        access = access or RepositoryAccess.WORKSPACE_WRITE
        contract = CORRECTION_RESULT_CONTRACT
        network = NetworkAccess.ALLOWED
    else:
        access = access or RepositoryAccess.WORKSPACE_WRITE
        contract = IMPLEMENTATION_RESULT_CONTRACT
        network = NetworkAccess.ALLOWED
    return AgentExecutionRequest(
        task_kind=task_kind,
        repository_path=tmp_path,
        repository_access=access,
        prompt="Application-owned task prompt.",
        result_contract=contract,
        artifact_directory=tmp_path / "artifacts",
        policy=AgentExecutionPolicy(60, network),
        required_capabilities=(
            capabilities
            if capabilities is not None
            else required_execution_capabilities(access)
        ),
    )


@pytest.mark.parametrize(
    ("access", "network", "sandbox", "network_setting"),
    [
        (RepositoryAccess.READ_ONLY, NetworkAccess.DENIED, "read-only", None),
        (
            RepositoryAccess.WORKSPACE_WRITE,
            NetworkAccess.ALLOWED,
            "workspace-write",
            "sandbox_workspace_write.network_access=true",
        ),
        (
            RepositoryAccess.WORKSPACE_WRITE,
            NetworkAccess.DENIED,
            "workspace-write",
            "sandbox_workspace_write.network_access=false",
        ),
    ],
)
def test_command_construction_enforces_access_and_network(
    tmp_path: Path,
    access: RepositoryAccess,
    network: NetworkAccess,
    sandbox: str,
    network_setting: str | None,
) -> None:
    command = build_command(
        executable="codex",
        repository_path=tmp_path,
        repository_access=access,
        network_access=network,
        settings=SETTINGS,
        output_schema=tmp_path / "schema.json",
        output_result=tmp_path / "result.json",
    )

    assert command.argv[0:3] == ("codex", "exec", "--ephemeral")
    assert command.argv[command.argv.index("--model") + 1] == SETTINGS.model
    assert f'model_reasoning_effort="{SETTINGS.reasoning_effort}"' in command.argv
    assert command.argv[command.argv.index("--sandbox") + 1] == sandbox
    assert "--json" in command.argv
    assert command.argv[command.argv.index("--output-schema") + 1] == str(
        (tmp_path / "schema.json").resolve()
    )
    assert command.argv[command.argv.index("--output-last-message") + 1] == str(
        (tmp_path / "result.json").resolve()
    )
    assert command.argv[-1] == "-"
    assert network_setting is None or network_setting in command.argv


@pytest.mark.parametrize("task_kind", tuple(AgentTaskKind))
@pytest.mark.parametrize("access", tuple(RepositoryAccess))
def test_adapter_accepts_only_the_stage_access_pair(
    tmp_path: Path, task_kind: AgentTaskKind, access: RepositoryAccess
) -> None:
    payload = (
        review_payload()
        if task_kind is AgentTaskKind.REVIEW
        else implementation_payload()
    )
    runner = FakeRunner(CodexProcessResult(0, "events\n", ""), json.dumps(payload))
    execution = CodexCliAgentExecutor(SETTINGS, runner=runner).execute(
        request(tmp_path, task_kind, access=access)
    )
    expected = (
        RepositoryAccess.READ_ONLY
        if task_kind is AgentTaskKind.REVIEW
        else RepositoryAccess.WORKSPACE_WRITE
    )
    if access is expected:
        assert execution.successful
        assert runner.calls == 1
        assert runner.command is not None
        schema = Path(
            runner.command.argv[runner.command.argv.index("--output-schema") + 1]
        )
        expected_schema = (
            "review-result.schema.json"
            if task_kind is AgentTaskKind.REVIEW
            else "implementation-result.schema.json"
        )
        assert schema.name == expected_schema
    else:
        assert (
            execution.failure_category
            is AgentFailureCategory.CAPABILITY_OR_CONFIGURATION_FAILURE
        )
        assert execution.invocation_start is InvocationStart.NOT_STARTED
        assert runner.calls == 0


@pytest.mark.parametrize(
    ("runner", "expected"),
    [
        (
            FakeRunner(error=OSError("permission denied")),
            AgentFailureCategory.INVOCATION_START_FAILURE,
        ),
        (
            FakeRunner(
                error=CodexProcessTimedOut(CodexProcessTimeout("partial", "late", 60))
            ),
            AgentFailureCategory.TIMEOUT,
        ),
        (
            FakeRunner(CodexProcessResult(2, "", "failed")),
            AgentFailureCategory.NON_SUCCESSFUL_EXECUTION,
        ),
        (
            FakeRunner(CodexProcessResult(2, "", "authentication failed")),
            AgentFailureCategory.PROVIDER_REJECTION_OR_SERVICE_FAILURE,
        ),
        (
            FakeRunner(CodexProcessResult(0, "", "")),
            AgentFailureCategory.MISSING_RESULT,
        ),
        (
            FakeRunner(CodexProcessResult(0, "", ""), "{invalid"),
            AgentFailureCategory.INVALID_RESULT,
        ),
    ],
)
def test_failure_mapping_is_deterministic(
    tmp_path: Path, runner: FakeRunner, expected: AgentFailureCategory
) -> None:
    execution = CodexCliAgentExecutor(SETTINGS, runner=runner).execute(
        request(tmp_path)
    )

    assert execution.status is AgentExecutionStatus.FAILED
    assert execution.failure_category is expected
    assert execution.provider_metadata["native_failure_reason"]
    assert (tmp_path / "artifacts" / "codex-execution.json").is_file()


def test_unavailable_executable_is_process_not_started(tmp_path: Path) -> None:
    executor = CodexCliAgentExecutor(
        CodexCliSettings(str(tmp_path / "missing"), "gpt-5.5", "xhigh"),
        runner=FakeRunner(),
    )

    execution = executor.execute(request(tmp_path))

    assert execution.failure_category is AgentFailureCategory.PROVIDER_UNAVAILABLE
    assert execution.invocation_start is InvocationStart.NOT_STARTED


def test_project_configuration_rejection_precedes_process_start(tmp_path: Path) -> None:
    project_config = tmp_path / ".codex" / "config.toml"
    project_config.parent.mkdir()
    project_config.write_text("model = 'untrusted'\n", encoding="utf-8")
    runner = FakeRunner()

    execution = CodexCliAgentExecutor(SETTINGS, runner=runner).execute(
        request(tmp_path)
    )

    assert (
        execution.failure_category
        is AgentFailureCategory.CAPABILITY_OR_CONFIGURATION_FAILURE
    )
    assert execution.invocation_start is InvocationStart.NOT_STARTED
    assert runner.calls == 0


def test_workspace_write_requires_declared_capability(tmp_path: Path) -> None:
    class ReadOnlyCodexAdapter(CodexCliAgentExecutor):
        @property
        def capabilities(self) -> frozenset[AgentCapability]:
            return super().capabilities - {AgentCapability.WORKSPACE_WRITE_EXECUTION}

    runner = FakeRunner()
    executor = ReadOnlyCodexAdapter(SETTINGS, runner=runner)

    execution = executor.execute(request(tmp_path))

    assert (
        execution.failure_category
        is AgentFailureCategory.CAPABILITY_OR_CONFIGURATION_FAILURE
    )
    assert execution.invocation_start is InvocationStart.NOT_STARTED
    assert runner.calls == 0
    assert (tmp_path / "artifacts" / "prompt.md").read_text(encoding="utf-8") == (
        "Application-owned task prompt."
    )
    assert (tmp_path / "artifacts" / "events.jsonl").read_text(encoding="utf-8") == ""
    assert (tmp_path / "artifacts" / "stderr.log").read_text(encoding="utf-8") == ""
    assert (tmp_path / "artifacts" / "codex-execution.json").is_file()


def test_provider_artifact_preparation_never_follows_existing_hardlinks(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    external = {
        "prompt.md": tmp_path / "external-prompt.md",
        "events.jsonl": tmp_path / "external-events.jsonl",
        "stderr.log": tmp_path / "external-stderr.log",
    }
    originals = {name: f"external {name}\n" for name in external}
    for name, target in external.items():
        target.write_text(originals[name], encoding="utf-8")
        os.link(target, artifacts / name)
    runner = FakeRunner(
        CodexProcessResult(0, "", ""), json.dumps(implementation_payload())
    )

    with pytest.raises(PersistenceError, match="exclusively create text"):
        CodexCliAgentExecutor(SETTINGS, runner=runner).execute(request(tmp_path))

    assert runner.calls == 0
    for name, target in external.items():
        assert target.read_text(encoding="utf-8") == originals[name]


def test_process_output_atomically_replaces_hardlinks_created_during_invocation(
    tmp_path: Path,
) -> None:
    external_events = tmp_path / "external-events.jsonl"
    external_stderr = tmp_path / "external-stderr.log"
    external_events.write_text("external events\n", encoding="utf-8")
    external_stderr.write_text("external stderr\n", encoding="utf-8")

    @dataclass
    class ReplacingRunner(FakeRunner):
        def run(self, command, **kwargs):
            artifacts = tmp_path / "artifacts"
            for name, target in (
                ("events.jsonl", external_events),
                ("stderr.log", external_stderr),
            ):
                (artifacts / name).unlink()
                os.link(target, artifacts / name)
            result = super().run(command, **kwargs)
            assert command.stdout_capture is not None
            assert command.stderr_capture is not None
            command.stdout_capture.write_text("local events\n", encoding="utf-8")
            command.stderr_capture.write_text("local stderr\n", encoding="utf-8")
            return replace(
                result,
                stdout="",
                stderr="",
                stdout_capture=command.stdout_capture,
                stderr_capture=command.stderr_capture,
            )

    runner = ReplacingRunner(
        CodexProcessResult(0, "local events\n", "local stderr\n"),
        json.dumps(implementation_payload()),
    )

    execution = CodexCliAgentExecutor(SETTINGS, runner=runner).execute(
        request(tmp_path)
    )

    assert execution.successful
    assert external_events.read_text(encoding="utf-8") == "external events\n"
    assert external_stderr.read_text(encoding="utf-8") == "external stderr\n"
    assert (tmp_path / "artifacts" / "events.jsonl").read_text(encoding="utf-8") == (
        "local events\n"
    )
    assert (tmp_path / "artifacts" / "stderr.log").read_text(encoding="utf-8") == (
        "local stderr\n"
    )
    assert not os.path.samefile(
        external_events, tmp_path / "artifacts" / "events.jsonl"
    )
    assert not os.path.samefile(external_stderr, tmp_path / "artifacts" / "stderr.log")


def test_process_output_rejects_linked_capture_staging_path(tmp_path: Path) -> None:
    external = tmp_path / "external-events.jsonl"
    external.write_text("external events\n", encoding="utf-8")

    class LinkedCaptureRunner:
        def run(self, command, *, stdin, timeout_seconds, on_process_start=None):
            del stdin, timeout_seconds
            if on_process_start is not None:
                on_process_start()
            assert command.stdout_capture is not None
            os.link(external, command.stdout_capture)
            return CodexProcessResult(
                2,
                "fallback must not replace linked staging\n",
                "",
                stdout_capture=command.stdout_capture,
            )

    with pytest.raises(RuntimeError, match="linked or not a regular file"):
        CodexCliAgentExecutor(
            SETTINGS,
            runner=LinkedCaptureRunner(),
        ).execute(request(tmp_path))

    assert external.read_text(encoding="utf-8") == "external events\n"
    assert not os.path.samefile(external, tmp_path / "artifacts" / "events.jsonl")
    assert not list((tmp_path / "artifacts").glob("*.capture"))


@pytest.mark.parametrize(
    ("task_kind", "payload", "result_type"),
    [
        (AgentTaskKind.IMPLEMENTATION, implementation_payload(), ImplementationResult),
        (AgentTaskKind.CORRECTION, implementation_payload(), ImplementationResult),
        (AgentTaskKind.REVIEW, review_payload(), ReviewResult),
    ],
)
def test_typed_results_cross_the_adapter_boundary(
    tmp_path: Path,
    task_kind: AgentTaskKind,
    payload: dict[str, object],
    result_type: type,
) -> None:
    runner = FakeRunner(
        CodexProcessResult(0, "diagnostic event\n", "provider detail\n"),
        json.dumps(payload),
    )

    execution = CodexCliAgentExecutor(SETTINGS, runner=runner).execute(
        request(tmp_path, task_kind)
    )

    assert execution.successful
    assert type(execution.result) is result_type
    assert execution.invocation_start is InvocationStart.STARTED
    assert execution.provider_metadata["work_timeout_seconds"] == 60
    assert execution.provider_metadata["structured_result_accepted"] is True
    assert (tmp_path / "artifacts" / "prompt.md").read_text(encoding="utf-8") == (
        "Application-owned task prompt."
    )
    assert (tmp_path / "artifacts" / "events.jsonl").read_text(encoding="utf-8") == (
        "diagnostic event\n"
    )
    assert (tmp_path / "artifacts" / "stderr.log").read_text(encoding="utf-8") == (
        "provider detail\n"
    )


def test_review_request_is_read_only(tmp_path: Path) -> None:
    runner = FakeRunner(CodexProcessResult(0, "", ""), json.dumps(review_payload()))
    execution = CodexCliAgentExecutor(SETTINGS, runner=runner).execute(
        request(tmp_path, AgentTaskKind.REVIEW)
    )

    assert execution.successful
    assert runner.command is not None
    assert (
        runner.command.argv[runner.command.argv.index("--sandbox") + 1] == "read-only"
    )
    assert "sandbox_workspace_write.network_access=true" not in runner.command.argv


def test_invocation_start_is_reported_after_spawn_before_process_completion(
    tmp_path: Path,
) -> None:
    state = {
        "runner_entered": False,
        "start_observed": False,
        "process_completed": False,
    }

    @dataclass
    class ObservingRunner(FakeRunner):
        def run(
            self,
            command: CodexCommand,
            *,
            stdin: str,
            timeout_seconds: float | None,
            on_process_start=None,
        ) -> CodexProcessResult:
            state["runner_entered"] = True
            if on_process_start is not None:
                on_process_start()
            result = super().run(
                command,
                stdin=stdin,
                timeout_seconds=timeout_seconds,
            )
            state["process_completed"] = True
            return result

    def observe_start() -> None:
        assert state["runner_entered"]
        assert not state["process_completed"]
        state["start_observed"] = True

    tracked = request(tmp_path)
    runner = ObservingRunner(
        CodexProcessResult(0, "", ""),
        json.dumps(implementation_payload()),
    )

    execution = CodexCliAgentExecutor(SETTINGS, runner=runner).execute(
        tracked,
        on_invocation_start=observe_start,
    )

    assert execution.successful
    assert state == {
        "runner_entered": True,
        "start_observed": True,
        "process_completed": True,
    }


@pytest.mark.parametrize(
    "error", [OSError("attempt write failed"), FileNotFoundError("attempt missing")]
)
def test_invocation_start_observer_errors_are_not_mapped_as_provider_failures(
    tmp_path: Path,
    error: OSError,
) -> None:
    runner = FakeRunner(
        CodexProcessResult(0, "", ""),
        json.dumps(implementation_payload()),
    )

    def fail_to_record_start() -> None:
        raise error

    with pytest.raises(type(error), match=str(error)):
        CodexCliAgentExecutor(SETTINGS, runner=runner).execute(
            request(tmp_path),
            on_invocation_start=fail_to_record_start,
        )

    assert runner.calls == 1


def test_post_start_file_error_is_transport_failure_with_uncertain_cleanup(
    tmp_path: Path,
) -> None:
    class StartedThenFailedRunner:
        def run(self, command, *, stdin, timeout_seconds, on_process_start=None):
            del command, stdin, timeout_seconds
            assert on_process_start is not None
            on_process_start()
            raise FileNotFoundError("capture disappeared after launch")

    execution = CodexCliAgentExecutor(
        SETTINGS,
        runner=StartedThenFailedRunner(),
    ).execute(request(tmp_path))

    assert execution.failure_category is AgentFailureCategory.NON_SUCCESSFUL_EXECUTION
    assert execution.invocation_start is InvocationStart.STARTED
    assert execution.provider_metadata["deadline_outcome"] == "transport-exception"
    assert execution.provider_metadata["cleanup_outcome"] == "unknown"
    assert execution.provider_metadata["tree_termination_confirmed"] is False
    assert execution.provider_metadata["structured_result_accepted"] is False


def test_backward_wall_clock_is_recorded_as_a_zero_duration(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    started = datetime(2026, 9, 21, 12, tzinfo=UTC)
    times = iter((started, started - timedelta(seconds=1)))
    runner = FakeRunner(
        CodexProcessResult(0, "", ""),
        json.dumps(implementation_payload()),
    )

    execution = CodexCliAgentExecutor(
        SETTINGS,
        runner=runner,
        clock=lambda: next(times),
    ).execute(request(tmp_path))

    assert execution.started_at == started
    assert execution.ended_at == started
    assert execution.duration_seconds == 0


@pytest.mark.parametrize(
    "payload",
    [
        [],
        "not an object",
        {"status": "COMPLETED"},
        implementation_payload(status="NOT_A_STATUS"),
        implementation_payload(tests_run=[{"command": "pytest"}]),
        implementation_payload(summary=7),
        implementation_payload(unexpected="value"),
    ],
)
def test_adapter_strictly_rejects_invalid_result_fields(
    tmp_path: Path, payload: object
) -> None:
    runner = FakeRunner(CodexProcessResult(0, "", ""), json.dumps(payload))

    execution = CodexCliAgentExecutor(SETTINGS, runner=runner).execute(
        request(tmp_path)
    )

    assert execution.failure_category is AgentFailureCategory.INVALID_RESULT
    assert (tmp_path / "artifacts" / "codex-diagnostic-result.json").is_file()
    assert not (tmp_path / "artifacts" / "codex-result.json").exists()


def test_stale_typed_result_cannot_satisfy_a_new_execution(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    result_path = artifacts / "codex-result.json"
    result_path.write_text(
        json.dumps(implementation_payload(status="BLOCKED")),
        encoding="utf-8",
    )
    runner = FakeRunner(CodexProcessResult(0, "", ""))

    execution = CodexCliAgentExecutor(SETTINGS, runner=runner).execute(
        request(tmp_path)
    )

    assert execution.failure_category is AgentFailureCategory.MISSING_RESULT
    assert not result_path.exists()


def test_canonical_result_wins_over_diagnostic_events(tmp_path: Path) -> None:
    diagnostic = json.dumps(
        {
            "type": "turn.completed",
            "result": implementation_payload(status="BLOCKED"),
        }
    )
    canonical = implementation_payload(status="COMPLETED")
    runner = FakeRunner(
        CodexProcessResult(0, diagnostic + "\n", "progress\n"),
        json.dumps(canonical),
    )

    execution = CodexCliAgentExecutor(SETTINGS, runner=runner).execute(
        request(tmp_path)
    )

    assert execution.successful
    assert isinstance(execution.result, ImplementationResult)
    assert execution.result.status.value == "COMPLETED"
    assert (tmp_path / "artifacts" / "events.jsonl").read_text(
        encoding="utf-8"
    ) == diagnostic + "\n"


def test_adapter_resolves_bare_executable_from_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable = write_path_executable(tmp_path / "tool-dir")
    prepend_executable_path(monkeypatch, executable.parent)
    runner = FakeRunner(
        CodexProcessResult(0, "", ""), json.dumps(implementation_payload())
    )
    settings = CodexCliSettings("codex", "gpt-5.5", "xhigh")

    execution = CodexCliAgentExecutor(settings, runner=runner).execute(
        request(tmp_path)
    )

    assert execution.successful
    assert runner.command is not None
    assert runner.command.argv[0] == str(executable.resolve())


def test_explicit_executable_does_not_use_path_lookup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable = write_path_executable(tmp_path / "tool-dir", name="codex-explicit")
    runner = FakeRunner(
        CodexProcessResult(0, "", ""), json.dumps(implementation_payload())
    )

    def fail_which(_configured: str) -> str | None:
        raise AssertionError("explicit executable paths must not use PATH lookup")

    monkeypatch.setattr(executable_resolution.shutil, "which", fail_which)
    settings = CodexCliSettings(str(executable), "gpt-5.5", "xhigh")

    execution = CodexCliAgentExecutor(settings, runner=runner).execute(
        request(tmp_path)
    )

    assert execution.successful
    assert runner.command is not None
    assert runner.command.argv[0] == str(executable.resolve())


def test_prompt_uses_stdin_repository_cwd_and_external_scratch(tmp_path: Path) -> None:
    repository = tmp_path / "repo with spaces"
    repository.mkdir()
    runner = FakeRunner(
        CodexProcessResult(0, "", ""), json.dumps(implementation_payload())
    )
    execution_request = request(repository)

    execution = CodexCliAgentExecutor(SETTINGS, runner=runner).execute(
        execution_request
    )

    assert execution.successful
    assert runner.stdin == "Application-owned task prompt."
    assert runner.command is not None
    assert runner.command.cwd == repository
    assert runner.command.environment is not None
    scratch = Path(runner.command.environment["TEMP"])
    assert runner.command.environment == {
        "TEMP": str(scratch),
        "TMP": str(scratch),
        "TMPDIR": str(scratch),
    }
    assert runner.command.stdout_capture is not None
    assert runner.command.stderr_capture is not None
    assert runner.command.stdout_capture.parent == repository / "artifacts"
    assert runner.command.stderr_capture.parent == repository / "artifacts"
    assert not scratch.is_relative_to(repository)
    assert not scratch.exists()


def test_timeout_preserves_partial_diagnostic_evidence(tmp_path: Path) -> None:
    runner = FakeRunner(
        error=CodexProcessTimedOut(
            CodexProcessTimeout(
                stdout='{"type":"turn.started"}\n',
                stderr="still working\n",
                timeout_seconds=3,
            )
        )
    )
    execution_request = request(tmp_path)
    execution_request = AgentExecutionRequest(
        task_kind=execution_request.task_kind,
        repository_path=execution_request.repository_path,
        repository_access=execution_request.repository_access,
        prompt=execution_request.prompt,
        result_contract=execution_request.result_contract,
        artifact_directory=execution_request.artifact_directory,
        policy=AgentExecutionPolicy(3, NetworkAccess.ALLOWED),
        required_capabilities=execution_request.required_capabilities,
    )

    execution = CodexCliAgentExecutor(SETTINGS, runner=runner).execute(
        execution_request
    )

    assert execution.failure_category is AgentFailureCategory.TIMEOUT
    assert (tmp_path / "artifacts" / "events.jsonl").read_text(
        encoding="utf-8"
    ) == '{"type":"turn.started"}\n'
    assert (tmp_path / "artifacts" / "stderr.log").read_text(
        encoding="utf-8"
    ) == "still working\n"
    assert execution.provider_metadata["work_timeout_seconds"] == 3


def test_timeout_preserves_raw_result_only_as_native_diagnostic(tmp_path: Path) -> None:
    payload = json.dumps(implementation_payload())

    class ResultThenTimeoutRunner:
        def run(self, command, *, stdin, timeout_seconds, on_process_start=None):
            del stdin
            if on_process_start is not None:
                on_process_start()
            output = Path(command.argv[command.argv.index("--output-last-message") + 1])
            output.write_text(payload, encoding="utf-8")
            raise CodexProcessTimedOut(
                CodexProcessTimeout("partial\n", "late\n", timeout_seconds or 0)
            )

    execution = CodexCliAgentExecutor(
        SETTINGS,
        runner=ResultThenTimeoutRunner(),
    ).execute(request(tmp_path))

    diagnostic = tmp_path / "artifacts" / "codex-diagnostic-result.json"
    assert execution.failure_category is AgentFailureCategory.TIMEOUT
    assert diagnostic.read_text(encoding="utf-8") == payload
    assert not (tmp_path / "artifacts" / "codex-result.json").exists()
    assert execution.provider_metadata["structured_result_present"] is True
    assert execution.provider_metadata["structured_result_accepted"] is False


def test_timeout_preserves_non_utf8_result_as_binary_diagnostic(
    tmp_path: Path,
) -> None:
    payload = b'\xff\xfe{"status":"COMPLETED"}'

    class BinaryResultThenTimeoutRunner:
        def run(self, command, *, stdin, timeout_seconds, on_process_start=None):
            del stdin
            if on_process_start is not None:
                on_process_start()
            output = Path(command.argv[command.argv.index("--output-last-message") + 1])
            output.write_bytes(payload)
            raise CodexProcessTimedOut(
                CodexProcessTimeout("partial\n", "late\n", timeout_seconds or 0)
            )

    execution = CodexCliAgentExecutor(
        SETTINGS,
        runner=BinaryResultThenTimeoutRunner(),
    ).execute(request(tmp_path))

    diagnostic = tmp_path / "artifacts" / "codex-diagnostic-result.bin"
    assert execution.failure_category is AgentFailureCategory.TIMEOUT
    assert execution.result is None
    assert diagnostic.read_bytes() == payload
    assert not (tmp_path / "artifacts" / "codex-result.json").exists()
    assert execution.provider_metadata["structured_result_present"] is True
    assert execution.provider_metadata["structured_result_accepted"] is False
    reference = next(
        item
        for item in execution.artifacts
        if item.run_relative_path.endswith(diagnostic.name)
    )
    assert reference.media_type == "application/octet-stream"


def test_timeout_diagnostic_result_atomically_replaces_invocation_hardlink(
    tmp_path: Path,
) -> None:
    payload = json.dumps(implementation_payload())
    external = tmp_path / "external-diagnostic.json"
    external.write_text("external\n", encoding="utf-8")

    class LinkedDiagnosticThenTimeoutRunner:
        def run(self, command, *, stdin, timeout_seconds, on_process_start=None):
            del stdin
            if on_process_start is not None:
                on_process_start()
            output = Path(command.argv[command.argv.index("--output-last-message") + 1])
            output.write_text(payload, encoding="utf-8")
            os.link(
                external,
                tmp_path / "artifacts" / "codex-diagnostic-result.json",
            )
            raise CodexProcessTimedOut(
                CodexProcessTimeout("partial\n", "late\n", timeout_seconds or 0)
            )

    execution = CodexCliAgentExecutor(
        SETTINGS,
        runner=LinkedDiagnosticThenTimeoutRunner(),
    ).execute(request(tmp_path))

    diagnostic = tmp_path / "artifacts" / "codex-diagnostic-result.json"
    assert execution.failure_category is AgentFailureCategory.TIMEOUT
    assert external.read_text(encoding="utf-8") == "external\n"
    assert diagnostic.read_text(encoding="utf-8") == payload
    assert not os.path.samefile(external, diagnostic)


def test_subprocess_runner_uses_argv_and_disables_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class CompletedProcess:
        returncode = 0

        def communicate(self, *, input=None, timeout=None):
            assert captured["process_started"] is True
            captured["input"] = input
            captured["timeout"] = timeout
            return b"diagnostic output\n", b""

        def kill(self) -> None:
            raise AssertionError("completed process must not be killed")

        def wait(self) -> int:
            return self.returncode

    def fake_popen(*args: object, **kwargs: object) -> object:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return CompletedProcess()

    monkeypatch.setattr(process_module.subprocess, "Popen", fake_popen)
    captured["process_started"] = False

    def record_start() -> None:
        captured["process_started"] = True

    result = SubprocessCodexRunner().run(
        CodexCommand(
            ("codex", "exec", "-"),
            tmp_path,
            {"TEMP": "C:/scratch", "TMP": "C:/scratch", "TMPDIR": "C:/scratch"},
        ),
        stdin="prompt",
        timeout_seconds=10,
        on_process_start=record_start,
    )

    assert result.returncode == 0
    assert captured["args"] == (("codex", "exec", "-"),)
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["cwd"] == tmp_path
    assert captured["input"] == b"prompt"
    assert captured["timeout"] == 10
    assert kwargs["shell"] is False


def test_tracked_subprocess_timeout_kills_process_and_keeps_final_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class TimedOutProcess:
        returncode = None

        def __init__(self) -> None:
            self.communicate_calls = 0
            self.killed = False

        def communicate(self, *, input=None, timeout=None):
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                assert input == b"prompt"
                assert timeout == 5
                raise process_module.subprocess.TimeoutExpired(
                    cmd=("codex", "exec", "-"),
                    timeout=5,
                    output=b"partial stdout\n",
                    stderr=b"partial stderr\n",
                )
            assert self.killed
            return b"partial stdout\nfinal stdout\n", b"partial stderr\nfinal stderr\n"

        def kill(self) -> None:
            self.killed = True

        def wait(self) -> int:
            raise AssertionError("timeout cleanup uses communicate after kill")

    process = TimedOutProcess()
    monkeypatch.setattr(
        process_module.subprocess, "Popen", lambda *args, **kwargs: process
    )
    starts = 0

    def record_start() -> None:
        nonlocal starts
        starts += 1

    with pytest.raises(CodexProcessTimedOut) as raised:
        SubprocessCodexRunner().run(
            CodexCommand(("codex", "exec", "-"), tmp_path),
            stdin="prompt",
            timeout_seconds=5,
            on_process_start=record_start,
        )

    assert starts == 1
    assert process.killed
    assert process.communicate_calls == 2
    assert raised.value.result.stdout == "partial stdout\nfinal stdout\n"
    assert raised.value.result.stderr == "partial stderr\nfinal stderr\n"


def test_tracked_subprocess_stops_child_when_start_observer_is_interrupted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class SpawnedProcess:
        returncode = None

        def __init__(self) -> None:
            self.killed = False
            self.waited = False

        def communicate(self, *, input=None, timeout=None):
            del input, timeout
            raise AssertionError("observer interruption must stop before communicate")

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout=None) -> int:
            del timeout
            self.waited = True
            return 1

    process = SpawnedProcess()
    monkeypatch.setattr(
        process_module.subprocess, "Popen", lambda *args, **kwargs: process
    )

    def interrupt() -> None:
        raise KeyboardInterrupt("interrupted while recording start")

    with pytest.raises(KeyboardInterrupt, match="recording start"):
        SubprocessCodexRunner().run(
            CodexCommand(("codex", "exec", "-"), tmp_path),
            stdin="prompt",
            timeout_seconds=5,
            on_process_start=interrupt,
        )

    assert process.killed
    assert process.waited


def test_real_runner_preserves_start_observer_interruption_and_stops_process(
    tmp_path: Path,
) -> None:
    late = tmp_path / "late-after-interruption"
    script = (
        "import pathlib,time; time.sleep(.5); "
        f"pathlib.Path({str(late)!r}).write_text('late',encoding='utf-8')"
    )

    def interrupt() -> None:
        raise KeyboardInterrupt("interrupted after launch")

    with pytest.raises(KeyboardInterrupt, match="interrupted after launch"):
        SubprocessCodexRunner().run(
            CodexCommand((sys.executable, "-c", script), tmp_path),
            stdin="prompt",
            timeout_seconds=1,
            on_process_start=interrupt,
        )

    time.sleep(0.7)
    assert not late.exists()


def test_blocking_start_observer_is_bounded_and_stops_descendant_tree(
    tmp_path: Path,
) -> None:
    child = tmp_path / "observer-child.py"
    ready = tmp_path / "observer-ready"
    pids_path = tmp_path / "observer-pids.json"
    late = tmp_path / "late-after-blocked-observer"
    child.write_text(
        "import pathlib,sys,time\n"
        "time.sleep(3)\n"
        "pathlib.Path(sys.argv[1]).write_text('late',encoding='utf-8')\n",
        encoding="utf-8",
    )
    parent_script = (
        "import json,os,pathlib,subprocess,sys,time; "
        f"child=subprocess.Popen([sys.executable,{str(child)!r},{str(late)!r}]); "
        f"pathlib.Path({str(pids_path)!r}).write_text(json.dumps([os.getpid(),child.pid]),encoding='utf-8'); "
        f"pathlib.Path({str(ready)!r}).write_text('ready',encoding='utf-8'); "
        "time.sleep(30)"
    )
    release = threading.Event()
    notifications = 0

    def block_after_tree_is_ready() -> None:
        nonlocal notifications
        notifications += 1
        wait_until = time.monotonic() + 5
        while not ready.is_file() and time.monotonic() < wait_until:
            time.sleep(0.01)
        release.wait(timeout=30)

    started = time.monotonic()
    try:
        with pytest.raises(CodexProcessTimedOut) as raised:
            SubprocessCodexRunner().run(
                CodexCommand((sys.executable, "-c", parent_script), tmp_path),
                stdin="prompt",
                timeout_seconds=2,
                on_process_start=block_after_tree_is_ready,
            )
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert elapsed < 4
    assert notifications == 1
    assert ready.is_file()
    assert raised.value.result.evidence.deadline_outcome == "timed-out"
    assert raised.value.result.evidence.tree_termination_confirmed is True
    pids = json.loads(pids_path.read_text(encoding="utf-8"))
    if os.name == "nt":
        assert all(not _windows_pid_is_running(pid) for pid in pids)
    else:
        assert all(not _posix_pid_is_running(pid) for pid in pids)
    time.sleep(1.2)
    assert not late.exists()


def test_subprocess_spawn_failure_does_not_report_process_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    started = False

    def fail_spawn(*args: object, **kwargs: object):
        del args, kwargs
        raise FileNotFoundError("codex unavailable")

    def record_start() -> None:
        nonlocal started
        started = True

    monkeypatch.setattr(process_module.subprocess, "Popen", fail_spawn)

    with pytest.raises(FileNotFoundError, match="codex unavailable"):
        SubprocessCodexRunner().run(
            CodexCommand(("codex", "exec", "-"), tmp_path),
            stdin="prompt",
            timeout_seconds=10,
            on_process_start=record_start,
        )

    assert started is False


def test_real_runner_observes_chunked_jsonl_and_split_utf8(tmp_path: Path) -> None:
    payload = json.dumps(implementation_payload(summary="café"), ensure_ascii=False)
    event = (
        json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": payload},
            },
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
    marker = event.index("é".encode()) + 1
    script = (
        "import os,sys,time; "
        f"data={event!r}; marker={marker}; "
        "os.write(sys.stdout.fileno(),data[:marker]); time.sleep(.02); "
        "os.write(sys.stdout.fileno(),data[marker:]); "
        'os.write(sys.stdout.fileno(),b\'{"type":"turn.completed"}\\n\')'
    )
    stdout = tmp_path / "captured-events.jsonl"
    stderr = tmp_path / "captured-stderr.log"

    result = SubprocessCodexRunner().run(
        CodexCommand(
            (sys.executable, "-c", script),
            tmp_path,
            stdout_capture=stdout,
            stderr_capture=stderr,
        ),
        stdin="prompt",
        timeout_seconds=2,
    )

    assert result.returncode == 0
    assert result.evidence.completion_before_deadline is True
    assert result.evidence.structured_message_before_deadline is True
    assert result.evidence.structured_message == payload
    assert stdout.read_bytes() == event + b'{"type":"turn.completed"}\n'


@pytest.mark.parametrize(
    ("terminal_elapsed", "times_out"),
    [(0.999, False), (1.0, True), (1.001, True)],
)
def test_public_runner_enforces_strict_completion_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    terminal_elapsed: float,
    times_out: bool,
) -> None:
    clock = ControlledMonotonic(monitor_time=terminal_elapsed)
    process = ControlledProcess(
        clock,
        completed_event_chunks(
            implementation_payload(),
            message_at=0.5,
            terminal_at=terminal_elapsed,
        ),
    )
    monkeypatch.setattr(
        process_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    if times_out:
        with pytest.raises(CodexProcessTimedOut) as raised:
            SubprocessCodexRunner(monotonic=clock).run(
                CodexCommand(("codex",), tmp_path),
                stdin="prompt",
                timeout_seconds=1.0,
            )
        evidence = raised.value.result.evidence
        assert evidence.deadline_outcome == "timed-out"
        assert evidence.completion_before_deadline is False
        assert evidence.terminal_event_elapsed_seconds == terminal_elapsed
    else:
        result = SubprocessCodexRunner(monotonic=clock).run(
            CodexCommand(("codex",), tmp_path),
            stdin="prompt",
            timeout_seconds=1.0,
        )
        assert result.returncode == 0
        assert result.evidence.deadline_outcome == "completed-before-deadline"
        assert result.evidence.completion_before_deadline is True


def test_public_runner_resamples_after_late_terminal_and_exit_interleave(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class InterleavingClock(ControlledMonotonic):
        def __init__(self) -> None:
            super().__init__(monitor_time=0.5)
            self.process: ControlledProcess | None = None
            self.monitor_samples = 0

        def __call__(self) -> float:
            name = threading.current_thread().name
            if name in {"codex-stdin-writer", "codex-stdout-reader"}:
                return super().__call__()
            if not self.launched:
                self.launched = True
                return 0.0
            self.monitor_samples += 1
            if self.monitor_samples == 1:
                assert self.process is not None
                self.process.release.set()
                assert self.process.output_finished.wait(timeout=2)
                # This deliberately stale sample reproduces the old polling
                # order: output and exit became final while it was in flight.
                return 0.5
            return max(1.0, self.observed_time)

    clock = InterleavingClock()
    process = ControlledProcess(
        clock,
        completed_event_chunks(
            implementation_payload(),
            message_at=0.5,
            terminal_at=1.0,
        ),
        release_output_on_kill=True,
    )
    clock.process = process
    monkeypatch.setattr(
        process_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    with pytest.raises(CodexProcessTimedOut) as raised:
        SubprocessCodexRunner(monotonic=clock).run(
            CodexCommand(("codex",), tmp_path),
            stdin="prompt",
            timeout_seconds=1.0,
        )

    assert raised.value.result.evidence.deadline_outcome == "timed-out"
    assert raised.value.result.evidence.terminal_event_elapsed_seconds == 1.0
    assert raised.value.result.evidence.completion_before_deadline is False


def test_public_runner_resnapshots_timely_records_delivered_during_exit_poll(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clock = ControlledMonotonic(monitor_time=0.5)
    payload = implementation_payload(summary="observed during EOF interleave")
    process = ControlledProcess(
        clock,
        completed_event_chunks(payload, message_at=0.2, terminal_at=0.3),
        release_output_on_poll=True,
    )
    monkeypatch.setattr(
        process_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    result = SubprocessCodexRunner(monotonic=clock).run(
        CodexCommand(("codex",), tmp_path),
        stdin="prompt",
        timeout_seconds=1.0,
    )

    assert result.transport_failure is None
    assert result.evidence.deadline_outcome == "completed-before-deadline"
    assert result.evidence.structured_message == json.dumps(payload)


def test_stream_failure_published_with_eof_cannot_be_accepted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clock = ControlledMonotonic(monitor_time=0.5)
    process = ControlledProcess(
        clock,
        completed_event_chunks(
            implementation_payload(),
            message_at=0.2,
            terminal_at=0.3,
        ),
        release_output_on_poll=True,
        stdout_eof_error=OSError("reader failed at EOF"),
    )
    monkeypatch.setattr(
        process_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    result = SubprocessCodexRunner(monotonic=clock).run(
        CodexCommand(("codex",), tmp_path),
        stdin="prompt",
        timeout_seconds=1.0,
    )

    assert result.evidence.deadline_outcome == "stream-failure"
    assert result.evidence.finalization_outcome == "failed"
    assert result.transport_failure == (
        "Codex stdout streaming failed: reader failed at EOF"
    )
    assert result.stdout_capture_complete is False
    assert result.stderr_capture_complete is False
    assert result.evidence.output_draining_truncated is True


@pytest.mark.parametrize(
    ("exit_observed_at", "succeeds"),
    [(0.999, True), (1.0, False), (1.001, False)],
)
def test_finalization_exit_boundary_uses_fresh_monotonic_sample(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exit_observed_at: float,
    succeeds: bool,
) -> None:
    monkeypatch.setattr(process_module, "FINALIZATION_SECONDS", 0.6)
    clock = ControlledMonotonic(monitor_time=exit_observed_at)
    process = ControlledProcess(
        clock,
        completed_event_chunks(
            implementation_payload(),
            message_at=0.3,
            terminal_at=0.4,
        ),
    )
    monkeypatch.setattr(
        process_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    result = SubprocessCodexRunner(monotonic=clock).run(
        CodexCommand(("codex",), tmp_path),
        stdin="prompt",
        timeout_seconds=2.0,
    )

    assert (result.transport_failure is None) is succeeds
    assert result.evidence.finalization_outcome == (
        "completed" if succeeds else "expired"
    )


def test_complete_late_success_during_cleanup_remains_diagnostic_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clock = ControlledMonotonic(monitor_time=1.0)
    payload = implementation_payload(summary="late diagnostic only")
    process = ControlledProcess(
        clock,
        completed_event_chunks(payload, message_at=1.1, terminal_at=1.2),
        release_output_on_kill=True,
    )
    monkeypatch.setattr(
        process_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    with pytest.raises(CodexProcessTimedOut) as raised:
        SubprocessCodexRunner(monotonic=clock).run(
            CodexCommand(("codex",), tmp_path),
            stdin="prompt",
            timeout_seconds=1.0,
        )

    evidence = raised.value.result.evidence
    assert evidence.deadline_outcome == "timed-out"
    assert evidence.terminal_event_type == "turn.completed"
    assert evidence.terminal_event_elapsed_seconds == 1.2
    assert evidence.completion_before_deadline is False
    assert evidence.structured_message == json.dumps(payload)
    assert evidence.structured_message_before_deadline is False
    assert evidence.output_draining_truncated is False
    assert not any(
        thread.is_alive()
        and thread.name
        in {"codex-stdout-reader", "codex-stderr-reader", "codex-stdin-writer"}
        for thread in threading.enumerate()
    )


def test_terminal_success_cannot_succeed_while_prompt_delivery_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clock = ControlledMonotonic(monitor_time=1.0)
    process = ControlledProcess(
        clock,
        completed_event_chunks(
            implementation_payload(),
            message_at=0.5,
            terminal_at=0.6,
        ),
        blocked_stdin=True,
    )
    monkeypatch.setattr(
        process_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    with pytest.raises(CodexProcessTimedOut) as raised:
        SubprocessCodexRunner(monotonic=clock).run(
            CodexCommand(("codex",), tmp_path),
            stdin="prompt that never finishes",
            timeout_seconds=1.0,
        )

    assert raised.value.result.evidence.deadline_outcome == "timed-out"
    assert raised.value.result.evidence.completion_before_deadline is True
    assert process.killed is True


@pytest.mark.parametrize(
    ("chunks", "expected_problem"),
    [
        ([(0.5, b'{"type":"turn.completed"}')], None),
        ([(0.5, b'not-json\n{"type":"future.event"}\n')], None),
        (
            [
                *completed_event_chunks(
                    implementation_payload(),
                    message_at=0.3,
                    terminal_at=0.4,
                ),
                (0.5, b'{"type":"turn.failed"}\n'),
            ],
            "contradictory terminal events",
        ),
    ],
)
def test_public_runner_fails_closed_for_ineligible_event_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    chunks: list[tuple[float, bytes]],
    expected_problem: str | None,
) -> None:
    clock = ControlledMonotonic(monitor_time=0.5)
    process = ControlledProcess(clock, chunks)
    monkeypatch.setattr(
        process_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    result = SubprocessCodexRunner(monotonic=clock).run(
        CodexCommand(("codex",), tmp_path),
        stdin="prompt",
        timeout_seconds=1.0,
    )

    assert result.evidence.deadline_outcome != "completed-before-deadline"
    if expected_problem is None:
        assert result.evidence.event_stream_problem is None
    else:
        assert expected_problem in (result.evidence.event_stream_problem or "")
        assert result.transport_failure is not None


@pytest.mark.parametrize(
    ("terminal_elapsed", "completion_before_deadline"),
    [(9.999, True), (10.0, False), (10.001, False)],
)
def test_terminal_observation_uses_a_strict_monotonic_deadline(
    terminal_elapsed: float,
    completion_before_deadline: bool,
) -> None:
    observed_times = iter((1.0, terminal_elapsed))
    observer = process_module._CodexEventObserver(
        started=0.0,
        deadline=10.0,
        clock=lambda: next(observed_times),
    )
    payload = json.dumps(implementation_payload())
    observer.feed(
        (
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": payload},
                }
            )
            + "\n"
        ).encode()
    )
    observer.feed(b'{"type":"turn.completed"}\n')

    evidence = process_module._evidence(
        observer,
        launched_at=0.0,
        clock=lambda: max(10.0, terminal_elapsed),
        timeout_seconds=10.0,
        outcome_elapsed_seconds=10.0,
        deadline_outcome="timed-out",
        finalization_outcome="not-eligible",
        cleanup=process_module._CleanupResult.not_required(),
    )

    assert evidence.completion_before_deadline is completion_before_deadline
    assert evidence.structured_message_before_deadline is True


def test_partial_malformed_unknown_and_contradictory_records_fail_closed() -> None:
    partial = process_module._CodexEventObserver(
        started=0.0,
        deadline=10.0,
        clock=lambda: 1.0,
    )
    partial.feed(b'{"type":"turn.completed"}', final=True)
    assert partial.snapshot()[0] is None

    observer = process_module._CodexEventObserver(
        started=0.0,
        deadline=10.0,
        clock=lambda: 1.0,
    )
    observer.feed(b'not-json\n{"type":"future.event"}\n')
    assert observer.snapshot()[0] is None

    observer.feed(b'{"type":"turn.completed"}\n{"type":"turn.failed"}\n')
    terminal, _, _, _, problem = observer.snapshot()
    assert terminal == "turn.completed"
    assert problem == "contradictory terminal events: turn.completed and turn.failed"


def test_late_contradictory_events_cannot_replace_timeout_classification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observer = process_module._CodexEventObserver(
        started=0.0,
        deadline=10.0,
        clock=lambda: 10.0,
    )
    observer.feed(b'{"type":"turn.completed"}\n{"type":"turn.failed"}\n')

    class Worker:
        def __init__(self, *, completed_at: float | None = None) -> None:
            self.done = process_module.threading.Event()
            self.done.set()
            self.error = None
            self.completed_at = completed_at

    class ExpectedTimeout(Exception):
        pass

    class FinishedProcess:
        @staticmethod
        def poll() -> int:
            return 0

    class InactiveContainment:
        @staticmethod
        def active() -> bool:
            return False

    runner = SubprocessCodexRunner(monotonic=lambda: 10.0)

    def timeout(*args, **kwargs):
        del args, kwargs
        raise ExpectedTimeout

    monkeypatch.setattr(runner, "_timeout", timeout)

    with pytest.raises(ExpectedTimeout):
        runner._monitor(
            FinishedProcess(),
            InactiveContainment(),
            CodexCommand(("codex",), tmp_path),
            observer,
            Worker(),
            Worker(),
            Worker(completed_at=1.0),
            launched_at=0.0,
            deadline=10.0,
            timeout_seconds=10.0,
        )


def test_oversized_incomplete_record_is_bounded_and_protocol_can_continue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(process_module, "_MAX_OBSERVED_RECORD_CHARS", 32)
    observer = process_module._CodexEventObserver(
        started=0.0,
        deadline=10.0,
        clock=lambda: 1.0,
    )

    observer.feed(b"x" * 64)
    assert observer.pending == ""
    assert observer.discarding_oversized_record is True

    observer.feed(b'ignored\n{"type":"turn.completed"}\n')
    assert observer.snapshot()[0] == "turn.completed"


def test_stdout_and_stderr_pressure_are_drained_without_output_loss(
    tmp_path: Path,
) -> None:
    payload = json.dumps(implementation_payload())
    event = (
        json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": payload},
            }
        ).encode()
        + b"\n"
    )
    stdout_bytes = b"x" * (256 * 1024) + b"\n" + event + b'{"type":"turn.completed"}\n'
    stderr_bytes = b"y" * (256 * 1024) + b"\n"
    script = (
        "import sys; "
        f"event={event!r}; "
        "sys.stdout.buffer.write(b'x'*(256*1024)+b'\\n'+event+"
        'b\'{"type":"turn.completed"}\\n\'); '
        "sys.stdout.buffer.flush(); "
        "sys.stderr.buffer.write(b'y'*(256*1024)+b'\\n'); "
        "sys.stderr.buffer.flush()"
    )
    stdout = tmp_path / "pressure-events.jsonl"
    stderr = tmp_path / "pressure-stderr.log"

    result = SubprocessCodexRunner().run(
        CodexCommand(
            (sys.executable, "-c", script),
            tmp_path,
            stdout_capture=stdout,
            stderr_capture=stderr,
        ),
        stdin="prompt",
        timeout_seconds=3,
    )

    assert result.returncode == 0
    assert result.evidence.completion_before_deadline is True
    assert stdout.read_bytes() == stdout_bytes
    assert stderr.read_bytes() == stderr_bytes


def test_no_capture_runner_keeps_only_bounded_diagnostic_tails(tmp_path: Path) -> None:
    stream_size = 2 * 1024 * 1024
    script = (
        "import sys; "
        f"sys.stdout.buffer.write(b'x'*{stream_size}+b'OUT-END'); "
        "sys.stdout.buffer.flush(); "
        f"sys.stderr.buffer.write(b'y'*{stream_size}+b'ERR-END'); "
        "sys.stderr.buffer.flush()"
    )

    result = SubprocessCodexRunner().run(
        CodexCommand((sys.executable, "-c", script), tmp_path),
        stdin="prompt",
        timeout_seconds=5,
    )

    assert result.returncode == 0
    assert len(result.stdout.encode()) <= process_module._DIAGNOSTIC_TAIL_BYTES
    assert len(result.stderr.encode()) <= process_module._DIAGNOSTIC_TAIL_BYTES
    assert result.stdout.endswith("OUT-END")
    assert result.stderr.endswith("ERR-END")
    assert result.stdout_capture is None
    assert result.stderr_capture is None


def test_blocked_stdin_and_silent_process_cannot_bypass_deadline(
    tmp_path: Path,
) -> None:
    stdout = tmp_path / "blocked-events.jsonl"
    stderr = tmp_path / "blocked-stderr.log"
    script = (
        "import sys,time; "
        'print(\'{"type":"turn.started"}\',flush=True); '
        "print('stdin blocked',file=sys.stderr,flush=True); "
        "time.sleep(5)"
    )
    started = time.monotonic()

    with pytest.raises(CodexProcessTimedOut) as raised:
        SubprocessCodexRunner().run(
            CodexCommand(
                (sys.executable, "-c", script),
                tmp_path,
                stdout_capture=stdout,
                stderr_capture=stderr,
            ),
            stdin="p" * (4 * 1024 * 1024),
            timeout_seconds=0.5,
        )

    assert time.monotonic() - started < 3
    assert raised.value.result.evidence.deadline_outcome == "timed-out"
    assert raised.value.result.evidence.tree_termination_confirmed is True
    assert stdout.read_text(encoding="utf-8") == '{"type":"turn.started"}\n'
    assert stderr.read_text(encoding="utf-8") == "stdin blocked\n"
    assert not any(
        thread.is_alive()
        and (
            thread.name.startswith("codex-")
            and (
                thread.name.endswith("-reader")
                or thread.name == "codex-stdin-writer"
                or thread.name.endswith("-closer")
            )
        )
        for thread in threading.enumerate()
    )


def test_stream_capture_failure_terminates_the_invocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FailingCapture:
        def write(self, value: bytes) -> int:
            del value
            raise OSError("capture failed")

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        process_module,
        "_open_capture",
        lambda path: None if path is None else FailingCapture(),
    )
    payload = json.dumps(implementation_payload())
    event = json.dumps(
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": payload},
        }
    )
    script = (
        f"print({event!r},flush=True); "
        'print(\'{"type":"turn.completed"}\',flush=True); '
        "import time; time.sleep(5)"
    )

    result = SubprocessCodexRunner().run(
        CodexCommand(
            (sys.executable, "-c", script),
            tmp_path,
            stdout_capture=tmp_path / "events.jsonl",
        ),
        stdin="prompt",
        timeout_seconds=2,
    )

    assert result.transport_failure == "Codex stdout streaming failed: capture failed"
    assert result.evidence.deadline_outcome == "stream-failure"
    assert result.evidence.tree_termination_confirmed is True
    assert result.stdout_capture_complete is False


def test_capture_failure_tail_reaches_final_provider_artifact_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FailingCapture:
        def write(self, value: bytes) -> int:
            del value
            raise OSError("capture failed")

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    clock = ControlledMonotonic(monitor_time=0.2)
    process = ControlledProcess(clock, [(0.1, b"recoverable byte\n")])
    monkeypatch.setattr(
        process_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        process_module,
        "_open_capture",
        lambda path: None if path is None else FailingCapture(),
    )

    execution = CodexCliAgentExecutor(
        SETTINGS,
        runner=SubprocessCodexRunner(),
    ).execute(request(tmp_path))

    assert not execution.successful
    assert execution.result is None
    assert execution.provider_metadata["deadline_outcome"] == "stream-failure"
    assert execution.provider_metadata["output_publication_truncated"] is True
    assert (tmp_path / "artifacts" / "events.jsonl").read_bytes() == (
        b"recoverable byte\n"
    )
    persisted = json.loads(
        (tmp_path / "artifacts" / "codex-execution.json").read_text(encoding="utf-8")
    )
    assert persisted["output_publication_truncated"] is True


def test_worker_start_failure_returns_typed_cleanup_evidence_and_cleans_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_start(worker: object) -> None:
        del worker
        raise RuntimeError("input worker could not start")

    monkeypatch.setattr(process_module._InputWorker, "start", fail_start)
    late = tmp_path / "late"
    script = (
        "import pathlib,time; time.sleep(.5); "
        f"pathlib.Path({str(late)!r}).write_text('late',encoding='utf-8')"
    )

    result = SubprocessCodexRunner().run(
        CodexCommand((sys.executable, "-c", script), tmp_path),
        stdin="prompt",
        timeout_seconds=1,
    )

    assert result.transport_failure is not None
    assert "input worker could not start" in result.transport_failure
    assert result.evidence.deadline_outcome == "setup-failure"
    assert result.evidence.tree_termination_confirmed is True
    assert result.stdout_capture_complete is True
    assert result.stderr_capture_complete is True
    time.sleep(0.7)
    assert not late.exists()


def test_worker_start_failure_publishes_cleanup_evidence_through_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clock = ControlledMonotonic(monitor_time=0.2)
    process = ControlledProcess(clock, [])

    def spawn(argv, **kwargs):
        del kwargs
        output = Path(argv[argv.index("--output-last-message") + 1])
        output.write_text('{"partial":true}', encoding="utf-8")
        return process

    monkeypatch.setattr(
        process_module.subprocess,
        "Popen",
        spawn,
    )

    def fail_start(worker: object) -> None:
        del worker
        raise RuntimeError("input worker could not start")

    monkeypatch.setattr(process_module._InputWorker, "start", fail_start)

    execution = CodexCliAgentExecutor(
        SETTINGS,
        runner=SubprocessCodexRunner(monotonic=clock),
    ).execute(request(tmp_path))

    assert execution.status is AgentExecutionStatus.FAILED
    assert execution.result is None
    assert execution.failure_category is AgentFailureCategory.NON_SUCCESSFUL_EXECUTION
    assert execution.provider_metadata["deadline_outcome"] == "setup-failure"
    assert execution.provider_metadata["cleanup_outcome"] == "completed"
    assert execution.provider_metadata["tree_termination_confirmed"] is True
    assert execution.provider_metadata["structured_result_present"] is True
    assert execution.provider_metadata["structured_result_accepted"] is False
    assert (tmp_path / "artifacts" / "codex-diagnostic-result.json").read_text(
        encoding="utf-8"
    ) == '{"partial":true}'
    persisted = json.loads(
        (tmp_path / "artifacts" / "codex-execution.json").read_text(encoding="utf-8")
    )
    assert persisted["cleanup_outcome"] == "completed"


def test_containment_setup_failure_is_typed_and_published_by_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clock = ControlledMonotonic(monitor_time=0.2)
    process = ControlledProcess(clock, [])
    monkeypatch.setattr(
        process_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    def fail_containment(spawned: object) -> None:
        assert spawned is process
        raise OSError("containment setup unavailable")

    monkeypatch.setattr(process_module, "_contain_process", fail_containment)

    execution = CodexCliAgentExecutor(
        SETTINGS,
        runner=SubprocessCodexRunner(monotonic=clock),
    ).execute(request(tmp_path))

    assert execution.status is AgentExecutionStatus.FAILED
    assert execution.result is None
    assert execution.failure_category is AgentFailureCategory.NON_SUCCESSFUL_EXECUTION
    assert execution.provider_metadata["deadline_outcome"] == "containment-failure"
    assert execution.provider_metadata["cleanup_outcome"] == "incomplete"
    assert execution.provider_metadata["termination_method"] == (
        "containment-setup-failed"
    )
    assert execution.provider_metadata["tree_termination_confirmed"] is False
    assert execution.provider_metadata["structured_result_accepted"] is False


def test_cleanup_reports_process_reap_failure_with_remaining_budget() -> None:
    observed_waits: list[float] = []

    class Process:
        stdin = None
        stdout = None
        stderr = None

        def poll(self):
            return None

        def wait(self, *, timeout):
            observed_waits.append(timeout)
            raise subprocess.TimeoutExpired(("codex",), timeout)

    class InactiveContainment:
        def terminate(self):
            return "test-containment"

        def active(self):
            return False

    class Worker:
        def __init__(self) -> None:
            self.thread = process_module.threading.Thread(target=lambda: None)

    deadline = time.monotonic() + 0.1
    cleanup = process_module._cleanup(
        Process(),
        InactiveContainment(),
        (Worker(), Worker(), Worker()),
        deadline,
        time.monotonic,
    )

    assert observed_waits and 0 <= observed_waits[0] <= 0.1
    assert cleanup.confirmed is True
    assert cleanup.outcome == "incomplete"


def test_reap_failure_is_explicit_through_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class UnreapableProcess(ControlledProcess):
        def poll(self) -> int | None:
            return -9 if self.killed else None

        def wait(self, *, timeout: float) -> int:
            raise subprocess.TimeoutExpired(("codex",), timeout)

    clock = ControlledMonotonic(monitor_time=1.0)
    process = UnreapableProcess(clock, [])
    monkeypatch.setattr(
        process_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    execution_request = replace(
        request(tmp_path),
        policy=AgentExecutionPolicy(1.0, NetworkAccess.ALLOWED),
    )

    execution = CodexCliAgentExecutor(
        SETTINGS,
        runner=SubprocessCodexRunner(monotonic=clock),
    ).execute(execution_request)

    assert execution.failure_category is AgentFailureCategory.TIMEOUT
    assert execution.result is None
    assert execution.provider_metadata["cleanup_outcome"] == "incomplete"
    assert execution.provider_metadata["tree_termination_confirmed"] is True
    assert execution.provider_metadata["structured_result_accepted"] is False


def test_cleanup_bounds_blocking_stream_close_and_reports_incomplete() -> None:
    class SlowClose:
        def close(self) -> None:
            time.sleep(0.25)

    class Process:
        stdin = SlowClose()
        stdout = SlowClose()
        stderr = SlowClose()

        def poll(self):
            return 1

        def wait(self, *, timeout):
            del timeout
            return 1

    class InactiveContainment:
        def terminate(self):
            return "test-containment"

        def active(self):
            return False

    class Worker:
        error = None

        def __init__(self) -> None:
            self.thread = threading.Thread(target=lambda: None)

    workers = (Worker(), Worker(), Worker())
    started = time.monotonic()
    cleanup = process_module._cleanup(
        Process(),
        InactiveContainment(),
        workers,
        started + 0.02,
        time.monotonic,
    )

    assert time.monotonic() - started < 0.1
    assert cleanup.outcome == "incomplete"
    assert cleanup.drain_truncated is False
    assert not any(worker.thread.is_alive() for worker in workers)
    assert not any(
        thread.is_alive() and thread.name.endswith("-closer")
        for thread in threading.enumerate()
    )


@pytest.mark.skipif(os.name != "nt", reason="requires Windows thread I/O API")
def test_windows_thread_cancellation_uses_required_access_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_access: list[int] = []

    class Kernel32:
        def OpenThread(self, access, inherit, native_id):
            del inherit, native_id
            observed_access.append(access)
            return 42

        def CancelSynchronousIo(self, handle):
            del handle
            return 0

        def CloseHandle(self, handle):
            del handle
            return 1

    monkeypatch.setattr(process_module, "_windows_kernel32", lambda: Kernel32())
    monkeypatch.setattr(process_module.ctypes, "get_last_error", lambda: 5)

    cancelled = process_module._cancel_blocked_windows_thread(
        threading.current_thread()
    )

    assert observed_access == [0x0001]
    assert cancelled is False


def test_cleanup_reader_failure_reports_truncated_output() -> None:
    class Process:
        stdin = None
        stdout = None
        stderr = None

        def poll(self):
            return 1

        def wait(self, *, timeout):
            del timeout
            return 1

    class InactiveContainment:
        def terminate(self):
            return "test-containment"

        def active(self):
            return False

    class Worker:
        def __init__(self, error=None) -> None:
            self.error = error
            self.thread = threading.Thread(target=lambda: None)

    workers = (Worker(OSError("read failed")), Worker(), Worker())
    cleanup = process_module._cleanup(
        Process(),
        InactiveContainment(),
        workers,
        time.monotonic() + 0.1,
        time.monotonic,
    )

    assert cleanup.outcome == "completed"
    assert cleanup.drain_truncated is True
    assert not any(worker.thread.is_alive() for worker in workers)


def test_live_reader_cannot_mutate_attempt_artifacts_after_bounded_return(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    release = threading.Event()

    class LateOutput:
        _codex_close_is_nonblocking = True

        def __init__(self, *, blocked: bool) -> None:
            self.blocked = blocked
            self.sent = False

        def read(self, size: int) -> bytes:
            del size
            if self.blocked:
                release.wait(timeout=5)
            if self.sent or not self.blocked:
                return b""
            self.sent = True
            return b"late-output\n"

        read1 = read

        def close(self) -> None:
            pass

    class Input:
        _codex_close_is_nonblocking = True

        def write(self, value: bytes) -> int:
            return len(value)

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    class Process:
        def __init__(self) -> None:
            self.stdin = Input()
            self.stdout = LateOutput(blocked=True)
            self.stderr = LateOutput(blocked=False)
            self.returncode = None

        def poll(self):
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, *, timeout):
            del timeout
            return self.returncode

    monkeypatch.setattr(process_module, "CLEANUP_SECONDS", 0.2)
    monkeypatch.setattr(
        process_module.subprocess,
        "Popen",
        lambda *args, **kwargs: Process(),
    )
    execution_request = replace(
        request(tmp_path),
        policy=AgentExecutionPolicy(0.05, NetworkAccess.ALLOWED),
    )
    started = time.monotonic()

    execution = CodexCliAgentExecutor(
        SETTINGS,
        runner=SubprocessCodexRunner(),
    ).execute(execution_request)

    assert time.monotonic() - started < 1
    assert execution.failure_category is AgentFailureCategory.TIMEOUT
    assert execution.provider_metadata["cleanup_outcome"] == "incomplete"
    assert execution.provider_metadata["output_draining_truncated"] is True
    assert execution.provider_metadata["output_publication_truncated"] is True
    events = tmp_path / "artifacts" / "events.jsonl"
    before = events.read_bytes()
    release.set()
    time.sleep(0.1)
    assert events.read_bytes() == before
    assert not list((tmp_path / "artifacts").glob("*.capture"))
    assert not any(
        thread.is_alive() and thread.name == "codex-stdout-reader"
        for thread in threading.enumerate()
    )


def test_termination_failure_is_bounded_and_rejected_by_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class RunningProcess(ControlledProcess):
        def poll(self) -> int | None:
            if self.killed:
                return super().poll()
            return None

    class FailingContainment:
        def __init__(self) -> None:
            self.terminate_calls = 0

        def terminate(self) -> str:
            self.terminate_calls += 1
            raise OSError("containment termination failed")

        def active(self) -> bool:
            raise OSError("containment state unavailable")

        def close(self) -> None:
            pass

    clock = ControlledMonotonic(monitor_time=1.0)
    process = RunningProcess(clock, [])
    containment = FailingContainment()
    monkeypatch.setattr(
        process_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        process_module,
        "_contain_process",
        lambda spawned: containment,
    )
    execution_request = replace(
        request(tmp_path),
        policy=AgentExecutionPolicy(1.0, NetworkAccess.ALLOWED),
    )
    started = time.monotonic()

    execution = CodexCliAgentExecutor(
        SETTINGS,
        runner=SubprocessCodexRunner(monotonic=clock),
    ).execute(execution_request)

    assert time.monotonic() - started < 1
    assert containment.terminate_calls == 2
    assert execution.status is AgentExecutionStatus.FAILED
    assert execution.failure_category is AgentFailureCategory.TIMEOUT
    assert execution.result is None
    assert execution.provider_metadata["cleanup_outcome"] == "incomplete"
    assert execution.provider_metadata["termination_method"] == "termination-failed"
    assert execution.provider_metadata["tree_termination_confirmed"] is False
    assert execution.provider_metadata["structured_result_accepted"] is False
    assert not (tmp_path / "artifacts" / "codex-result.json").exists()


def test_termination_confirmation_failure_is_explicit_and_incomplete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class RunningProcess(ControlledProcess):
        def poll(self) -> int | None:
            if self.killed:
                return super().poll()
            return None

    class UnconfirmableContainment:
        def terminate(self) -> str:
            process.kill()
            return "test-containment"

        def active(self) -> bool:
            raise OSError("containment state unavailable")

        def close(self) -> None:
            pass

    clock = ControlledMonotonic(monitor_time=1.0)
    process = RunningProcess(clock, [])
    monkeypatch.setattr(
        process_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        process_module,
        "_contain_process",
        lambda spawned: UnconfirmableContainment(),
    )

    with pytest.raises(CodexProcessTimedOut) as raised:
        SubprocessCodexRunner(monotonic=clock).run(
            CodexCommand(("codex",), tmp_path),
            stdin="prompt",
            timeout_seconds=1.0,
        )

    evidence = raised.value.result.evidence
    assert evidence.cleanup_outcome == "incomplete"
    assert evidence.termination_method == "test-containment"
    assert evidence.tree_termination_confirmed is False


def test_interruption_preserves_original_error_when_termination_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FailingContainment:
        terminate_calls = 0

        def terminate(self) -> str:
            self.terminate_calls += 1
            raise OSError("containment termination failed")

        def active(self) -> bool:
            raise OSError("containment state unavailable")

        def close(self) -> None:
            pass

    clock = ControlledMonotonic(monitor_time=0.0)
    process = ControlledProcess(clock, [])
    containment = FailingContainment()
    monkeypatch.setattr(
        process_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        process_module,
        "_contain_process",
        lambda spawned: containment,
    )

    def interrupt() -> None:
        raise KeyboardInterrupt("original interruption")

    with pytest.raises(KeyboardInterrupt, match="original interruption"):
        SubprocessCodexRunner(monotonic=clock).run(
            CodexCommand(("codex",), tmp_path),
            stdin="prompt",
            timeout_seconds=1.0,
            on_process_start=interrupt,
        )

    assert containment.terminate_calls >= 1
    assert process.killed is True


def test_timely_completion_can_exit_during_finalization_allowance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(process_module, "FINALIZATION_SECONDS", 0.5)
    payload = json.dumps(implementation_payload())
    event = json.dumps(
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": payload},
        }
    )
    script = (
        f"print({event!r},flush=True); "
        'print(\'{"type":"turn.completed"}\',flush=True); '
        "import time; time.sleep(.1)"
    )

    result = SubprocessCodexRunner().run(
        CodexCommand((sys.executable, "-c", script), tmp_path),
        stdin="prompt",
        timeout_seconds=2,
    )

    assert result.returncode == 0
    assert result.evidence.completion_before_deadline is True
    assert result.evidence.finalization_outcome == "completed"


def test_timely_completion_has_only_bounded_finalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(process_module, "FINALIZATION_SECONDS", 0.1)
    late = tmp_path / "late"
    payload = json.dumps(implementation_payload())
    event = json.dumps(
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": payload},
        }
    )
    script = (
        "import pathlib,time; "
        f"print({event!r},flush=True); "
        'print(\'{"type":"turn.completed"}\',flush=True); '
        "time.sleep(1); "
        f"pathlib.Path({str(late)!r}).write_text('late',encoding='utf-8')"
    )
    started = time.monotonic()

    result = SubprocessCodexRunner().run(
        CodexCommand((sys.executable, "-c", script), tmp_path),
        stdin="prompt",
        timeout_seconds=2,
    )

    assert time.monotonic() - started < 1
    assert result.evidence.completion_before_deadline is True
    assert result.evidence.finalization_outcome == "expired"
    assert result.evidence.tree_termination_confirmed is True
    assert result.transport_failure is not None
    time.sleep(1.1)
    assert not late.exists()


@pytest.mark.parametrize(
    ("outcome", "expected_work_elapsed"),
    [
        ("stream-failure", 0.5),
        ("terminal-failure", 0.5),
        ("event-conflict", 0.5),
        ("failed-finalization", 0.7),
    ],
)
def test_provider_metadata_excludes_cleanup_from_failure_work_elapsed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome: str,
    expected_work_elapsed: float,
) -> None:
    monitor_time = 0.7 if outcome == "failed-finalization" else 0.5
    clock = ControlledMonotonic(monitor_time=monitor_time)
    if outcome == "stream-failure":
        process = ControlledProcess(
            clock,
            [(0.1, b'{"type":"turn.started"}\n')],
            release_output_on_poll=True,
            stdout_eof_error=OSError("reader failed early"),
        )
    elif outcome == "terminal-failure":
        process = ControlledProcess(
            clock,
            [(0.2, b'{"type":"turn.failed"}\n')],
        )
    else:
        chunks = completed_event_chunks(
            implementation_payload(),
            message_at=0.1,
            terminal_at=0.3,
        )
        if outcome == "event-conflict":
            chunks.append((0.4, b'{"type":"turn.failed"}\n'))
        process = ControlledProcess(clock, chunks)
    monkeypatch.setattr(
        process_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    if outcome == "failed-finalization":
        monkeypatch.setattr(process_module, "FINALIZATION_SECONDS", 0.4)
    original_cleanup = process_module._cleanup

    def delayed_cleanup(*args, **kwargs):
        cleanup = original_cleanup(*args, **kwargs)
        clock.monitor_time += 0.4
        return replace(cleanup, duration=0.4)

    monkeypatch.setattr(process_module, "_cleanup", delayed_cleanup)
    process_result = SubprocessCodexRunner(monotonic=clock).run(
        CodexCommand(("codex",), tmp_path),
        stdin="prompt",
        timeout_seconds=1.0,
    )

    execution = CodexCliAgentExecutor(
        SETTINGS,
        runner=FakeRunner(process_result),
    ).execute(request(tmp_path))

    metadata = execution.provider_metadata
    assert not execution.successful
    assert metadata["work_elapsed_seconds"] == pytest.approx(expected_work_elapsed)
    assert metadata["cleanup_duration_seconds"] == pytest.approx(0.4)
    assert metadata["total_elapsed_seconds"] + 1e-9 >= (expected_work_elapsed + 0.4)
    assert metadata["deadline_outcome"] == (
        "completed-before-deadline" if outcome == "failed-finalization" else outcome
    )
    persisted = json.loads(
        (tmp_path / "artifacts" / "codex-execution.json").read_text(encoding="utf-8")
    )
    assert persisted["work_elapsed_seconds"] == metadata["work_elapsed_seconds"]
    assert persisted["total_elapsed_seconds"] == metadata["total_elapsed_seconds"]
    assert (
        persisted["cleanup_duration_seconds"] == (metadata["cleanup_duration_seconds"])
    )


def test_timeout_publishes_large_attempt_capture_without_copying(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b"x" * (8 * 1024 * 1024) + b"\npartial diagnostics\n"
    scratch_paths: list[Path] = []

    class LargeTimeoutRunner:
        def run(self, command, *, stdin, timeout_seconds, on_process_start=None):
            del stdin
            if on_process_start is not None:
                on_process_start()
            assert command.stdout_capture is not None
            assert command.environment is not None
            scratch = Path(command.environment["TEMP"])
            scratch_paths.append(scratch)
            (scratch / "provider-leftover.tmp").write_text(
                "leftover",
                encoding="utf-8",
            )
            command.stdout_capture.write_bytes(payload)
            evidence = CodexProcessEvidence(
                work_timeout_seconds=timeout_seconds,
                work_elapsed_seconds=timeout_seconds or 0,
                total_elapsed_seconds=timeout_seconds or 0,
                deadline_outcome="timed-out",
                finalization_outcome="not-eligible",
                cleanup_outcome="completed",
                tree_termination_confirmed=True,
            )
            raise CodexProcessTimedOut(
                CodexProcessTimeout(
                    "bounded tail",
                    "",
                    timeout_seconds or 0,
                    evidence=evidence,
                    stdout_capture=command.stdout_capture,
                    stderr_capture=command.stderr_capture,
                )
            )

    def unbounded_copy_would_block(*args, **kwargs):
        del args, kwargs
        time.sleep(5)
        raise AssertionError("capture publication copied the growing stream")

    monkeypatch.setattr(
        persistence_module,
        "atomic_copy_file",
        unbounded_copy_would_block,
    )
    monkeypatch.setattr(process_module.shutil, "rmtree", unbounded_copy_would_block)
    started = time.monotonic()

    execution = CodexCliAgentExecutor(
        SETTINGS,
        runner=LargeTimeoutRunner(),
    ).execute(request(tmp_path))

    assert time.monotonic() - started < 3
    assert execution.failure_category is AgentFailureCategory.TIMEOUT
    assert execution.provider_metadata["output_publication_truncated"] is False
    assert (tmp_path / "artifacts" / "events.jsonl").read_bytes() == payload
    persisted = json.loads(
        (tmp_path / "artifacts" / "codex-execution.json").read_text(encoding="utf-8")
    )
    assert persisted["output_publication_truncated"] is False
    assert not list((tmp_path / "artifacts").glob("*.capture"))
    assert len(scratch_paths) == 1
    assert not scratch_paths[0].exists()


def test_canonical_result_must_match_timely_structured_message(tmp_path: Path) -> None:
    canonical = implementation_payload(status="COMPLETED")
    conflicting = implementation_payload(status="BLOCKED")
    evidence = CodexProcessEvidence(
        work_timeout_seconds=60,
        terminal_event_type="turn.completed",
        terminal_event_elapsed_seconds=1,
        completion_before_deadline=True,
        structured_message=json.dumps(conflicting),
        structured_message_elapsed_seconds=0.9,
        structured_message_before_deadline=True,
        deadline_outcome="completed-before-deadline",
        finalization_outcome="completed",
        tree_termination_confirmed=True,
    )
    runner = FakeRunner(
        CodexProcessResult(0, "", "", evidence=evidence),
        json.dumps(canonical),
    )

    execution = CodexCliAgentExecutor(SETTINGS, runner=runner).execute(
        request(tmp_path)
    )

    assert execution.failure_category is AgentFailureCategory.INVALID_RESULT
    assert not (tmp_path / "artifacts" / "codex-result.json").exists()
    assert (tmp_path / "artifacts" / "codex-diagnostic-result.json").is_file()


def test_adapter_rejects_success_without_live_timing_evidence(tmp_path: Path) -> None:
    class UntimedRunner:
        def run(self, command, *, stdin, timeout_seconds, on_process_start=None):
            del stdin, timeout_seconds
            if on_process_start is not None:
                on_process_start()
            output = Path(command.argv[command.argv.index("--output-last-message") + 1])
            output.write_text(
                json.dumps(implementation_payload()),
                encoding="utf-8",
            )
            return CodexProcessResult(0, "", "")

    execution = CodexCliAgentExecutor(SETTINGS, runner=UntimedRunner()).execute(
        request(tmp_path)
    )

    assert execution.failure_category is AgentFailureCategory.NON_SUCCESSFUL_EXECUTION
    assert execution.provider_metadata["completion_before_deadline"] is False
    assert execution.provider_metadata["structured_result_accepted"] is False
    assert not (tmp_path / "artifacts" / "codex-result.json").exists()


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows Job Objects")
def test_windows_job_stops_wrapper_descendant_tree_and_preserves_output(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "helper.py"
    child = tmp_path / "child.py"
    grandchild = tmp_path / "grandchild.py"
    wrapper = tmp_path / "codex-wrapper.cmd"
    ready = tmp_path / "ready"
    gate = tmp_path / "gate"
    pids_path = tmp_path / "pids.json"
    late_child = tmp_path / "late-child"
    late_grandchild = tmp_path / "late-grandchild"
    stdout = tmp_path / "events.jsonl"
    stderr = tmp_path / "stderr.log"
    grandchild.write_text(
        "import pathlib,sys,time\n"
        "time.sleep(4)\n"
        "pathlib.Path(sys.argv[1]).write_text('late', encoding='utf-8')\n",
        encoding="utf-8",
    )
    child.write_text(
        "import json,os,pathlib,subprocess,sys,time\n"
        "grand=subprocess.Popen([sys.executable,sys.argv[2],sys.argv[6]])\n"
        "pathlib.Path(sys.argv[3]).write_text(json.dumps([int(sys.argv[1]),os.getpid(),grand.pid]),encoding='utf-8')\n"
        "pathlib.Path(sys.argv[4]).write_text('ready',encoding='utf-8')\n"
        "while not pathlib.Path(sys.argv[5]).is_file(): time.sleep(.01)\n"
        'print(\'{"type":"turn.started"}\',flush=True)\n'
        "print('tree ready',file=sys.stderr,flush=True)\n"
        "time.sleep(4)\n"
        "pathlib.Path(sys.argv[7]).write_text('late',encoding='utf-8')\n",
        encoding="utf-8",
    )
    helper.write_text(
        "import os,subprocess,sys\n"
        "child=subprocess.Popen([sys.executable,sys.argv[1],str(os.getpid()),*sys.argv[2:]])\n"
        "child.wait()\n",
        encoding="utf-8",
    )
    wrapper.write_text(
        "@echo off\n"
        f'start "" /b "{sys.executable}" "{helper}" "{child}" "{grandchild}" '
        f'"{pids_path}" "{ready}" "{gate}" "{late_grandchild}" "{late_child}"\n'
        "exit /b 0\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "watchdog-report.json"
    watchdog_worker = tmp_path / "watchdog-worker.py"
    project_root = Path(process_module.__file__).resolve().parents[3]
    watchdog_worker.write_text(
        "import json,sys,time\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(project_root)!r})\n"
        "from ticket_automation.providers.codex_cli import "
        "CodexCommand,CodexProcessTimedOut,SubprocessCodexRunner\n"
        "wrapper,cwd,stdout,stderr,report,ready,gate=map(Path,sys.argv[1:])\n"
        "def release_ready_tree():\n"
        "    until=time.monotonic()+10\n"
        "    while not ready.is_file() and time.monotonic()<until: time.sleep(.01)\n"
        "    if ready.is_file(): gate.write_text('go',encoding='utf-8')\n"
        "started=time.monotonic()\n"
        "try:\n"
        "    SubprocessCodexRunner().run(CodexCommand((str(wrapper),),cwd,"
        "stdout_capture=stdout,stderr_capture=stderr),stdin='prompt',"
        "timeout_seconds=3,on_process_start=release_ready_tree)\n"
        "except CodexProcessTimedOut as error:\n"
        "    evidence=error.result.evidence\n"
        "    value={'status':'timed-out','elapsed':time.monotonic()-started,"
        "'deadline_outcome':evidence.deadline_outcome,"
        "'termination_method':evidence.termination_method,"
        "'tree_termination_confirmed':evidence.tree_termination_confirmed}\n"
        "else:\n"
        "    value={'status':'unexpected-success','elapsed':time.monotonic()-started}\n"
        "report.write_text(json.dumps(value),encoding='utf-8')\n",
        encoding="utf-8",
    )
    control = subprocess.Popen(
        (sys.executable, "-c", "import time; time.sleep(15)"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pids: list[int] = []
    watchdog_seconds = 16.0
    worker = subprocess.Popen(
        (
            sys.executable,
            str(watchdog_worker),
            str(wrapper),
            str(tmp_path),
            str(stdout),
            str(stderr),
            str(report_path),
            str(ready),
            str(gate),
        ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        try:
            _, worker_stderr = worker.communicate(timeout=watchdog_seconds)
        except subprocess.TimeoutExpired:
            _terminate_windows_process_tree(worker.pid)
            worker.wait(timeout=5)
            pytest.fail(f"native runner exceeded {watchdog_seconds:g}s outer watchdog")
        assert worker.returncode == 0, worker_stderr
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["status"] == "timed-out"
        assert report["elapsed"] <= 13.5
        assert ready.is_file()
        pids = json.loads(pids_path.read_text(encoding="utf-8"))
        assert len(pids) == 3
        assert all(not _windows_pid_is_running(pid) for pid in pids)
        assert control.poll() is None
        assert report["deadline_outcome"] == "timed-out"
        assert report["termination_method"] == "windows-job-object"
        assert report["tree_termination_confirmed"] is True
        assert "turn.started" in stdout.read_text(encoding="utf-8")
        assert "tree ready" in stderr.read_text(encoding="utf-8")
        time.sleep(4.2)
        assert not late_child.exists()
        assert not late_grandchild.exists()
    finally:
        if worker.poll() is None:
            _terminate_windows_process_tree(worker.pid)
            worker.wait(timeout=5)
        control.kill()
        control.wait(timeout=5)
        if pids_path.is_file() and not pids:
            try:
                pids = json.loads(pids_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pids = []
        for pid in pids:
            if _windows_pid_is_running(pid):
                _terminate_windows_process_tree(pid)


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows Job Objects")
def test_windows_interruption_stops_ready_wrapper_descendant_tree(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "interrupt-helper.py"
    child = tmp_path / "interrupt-child.py"
    grandchild = tmp_path / "interrupt-grandchild.py"
    wrapper = tmp_path / "interrupt-wrapper.cmd"
    ready = tmp_path / "interrupt-ready"
    gate = tmp_path / "interrupt-gate"
    pids_path = tmp_path / "interrupt-pids.json"
    late_child = tmp_path / "interrupt-late-child"
    late_grandchild = tmp_path / "interrupt-late-grandchild"
    stdout = tmp_path / "interrupt-events.jsonl"
    stderr = tmp_path / "interrupt-stderr.log"
    grandchild.write_text(
        "import pathlib,sys,time\n"
        "while not pathlib.Path(sys.argv[1]).is_file(): time.sleep(.01)\n"
        "time.sleep(1)\n"
        "pathlib.Path(sys.argv[2]).write_text('late',encoding='utf-8')\n",
        encoding="utf-8",
    )
    child.write_text(
        "import json,os,pathlib,subprocess,sys,time\n"
        "grand=subprocess.Popen([sys.executable,sys.argv[2],sys.argv[5],sys.argv[6]])\n"
        "pathlib.Path(sys.argv[3]).write_text(json.dumps([int(sys.argv[1]),os.getpid(),grand.pid]),encoding='utf-8')\n"
        "pathlib.Path(sys.argv[4]).write_text('ready',encoding='utf-8')\n"
        "while not pathlib.Path(sys.argv[5]).is_file(): time.sleep(.01)\n"
        "time.sleep(1)\n"
        "pathlib.Path(sys.argv[7]).write_text('late',encoding='utf-8')\n",
        encoding="utf-8",
    )
    helper.write_text(
        "import os,subprocess,sys\n"
        "child=subprocess.Popen([sys.executable,sys.argv[1],str(os.getpid()),*sys.argv[2:]])\n"
        "child.wait()\n",
        encoding="utf-8",
    )
    wrapper.write_text(
        "@echo off\n"
        f'start "" /b "{sys.executable}" "{helper}" "{child}" "{grandchild}" '
        f'"{pids_path}" "{ready}" "{gate}" "{late_grandchild}" "{late_child}"\n'
        "exit /b 0\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "interrupt-report.json"
    watchdog_worker = tmp_path / "interrupt-watchdog.py"
    project_root = Path(process_module.__file__).resolve().parents[3]
    watchdog_worker.write_text(
        "import json,sys,time\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(project_root)!r})\n"
        "from ticket_automation.providers.codex_cli import CodexCommand,SubprocessCodexRunner\n"
        "wrapper,cwd,stdout,stderr,report,ready,gate=map(Path,sys.argv[1:])\n"
        "def interrupt():\n"
        "    until=time.monotonic()+10\n"
        "    while not ready.is_file() and time.monotonic()<until: time.sleep(.01)\n"
        "    if not ready.is_file(): raise RuntimeError('tree readiness timed out')\n"
        "    gate.write_text('go',encoding='utf-8')\n"
        "    raise KeyboardInterrupt('interrupt after readiness')\n"
        "started=time.monotonic()\n"
        "try:\n"
        "    SubprocessCodexRunner().run(CodexCommand((str(wrapper),),cwd,stdout_capture=stdout,stderr_capture=stderr),stdin='prompt',timeout_seconds=5,on_process_start=interrupt)\n"
        "except KeyboardInterrupt as error:\n"
        "    value={'status':'interrupted','message':str(error),'elapsed':time.monotonic()-started}\n"
        "except BaseException as error:\n"
        "    value={'status':'wrong-error','message':repr(error),'elapsed':time.monotonic()-started}\n"
        "else:\n"
        "    value={'status':'unexpected-success','elapsed':time.monotonic()-started}\n"
        "report.write_text(json.dumps(value),encoding='utf-8')\n",
        encoding="utf-8",
    )
    control = subprocess.Popen(
        (sys.executable, "-c", "import time; time.sleep(15)"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker = subprocess.Popen(
        (
            sys.executable,
            str(watchdog_worker),
            str(wrapper),
            str(tmp_path),
            str(stdout),
            str(stderr),
            str(report_path),
            str(ready),
            str(gate),
        ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    pids: list[int] = []
    try:
        try:
            _, worker_stderr = worker.communicate(timeout=16)
        except subprocess.TimeoutExpired:
            _terminate_windows_process_tree(worker.pid)
            worker.wait(timeout=5)
            pytest.fail("interruption regression exceeded 16s outer watchdog")
        assert worker.returncode == 0, worker_stderr
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["status"] == "interrupted"
        assert report["message"] == "interrupt after readiness"
        assert report["elapsed"] < 12
        pids = json.loads(pids_path.read_text(encoding="utf-8"))
        assert len(pids) == 3
        assert all(not _windows_pid_is_running(pid) for pid in pids)
        assert control.poll() is None
        time.sleep(1.2)
        assert not late_child.exists()
        assert not late_grandchild.exists()
    finally:
        if worker.poll() is None:
            _terminate_windows_process_tree(worker.pid)
            worker.wait(timeout=5)
        control.kill()
        control.wait(timeout=5)
        if pids_path.is_file() and not pids:
            try:
                pids = json.loads(pids_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pids = []
        for pid in pids:
            if _windows_pid_is_running(pid):
                _terminate_windows_process_tree(pid)


def _windows_pid_is_running(pid: int) -> bool:
    if os.name != "nt":
        return False
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_uint32()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == 259
    finally:
        kernel32.CloseHandle(handle)


def _terminate_windows_process_tree(pid: int) -> None:
    subprocess.run(
        ("taskkill", "/PID", str(pid), "/T", "/F"),
        check=False,
        capture_output=True,
        timeout=5,
    )


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX process groups")
def test_posix_process_group_stops_descendant_tree_and_preserves_output(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "helper.py"
    child = tmp_path / "child.py"
    grandchild = tmp_path / "grandchild.py"
    ready = tmp_path / "ready"
    gate = tmp_path / "gate"
    pids_path = tmp_path / "pids.json"
    late_child = tmp_path / "late-child"
    late_grandchild = tmp_path / "late-grandchild"
    stdout = tmp_path / "events.jsonl"
    stderr = tmp_path / "stderr.log"
    grandchild.write_text(
        "import pathlib,sys,time\n"
        "time.sleep(4)\n"
        "pathlib.Path(sys.argv[1]).write_text('late', encoding='utf-8')\n",
        encoding="utf-8",
    )
    child.write_text(
        "import json,os,pathlib,subprocess,sys,time\n"
        "grand=subprocess.Popen([sys.executable,sys.argv[2],sys.argv[6]])\n"
        "pathlib.Path(sys.argv[3]).write_text(json.dumps([int(sys.argv[1]),os.getpid(),grand.pid]),encoding='utf-8')\n"
        "pathlib.Path(sys.argv[4]).write_text('ready',encoding='utf-8')\n"
        "while not pathlib.Path(sys.argv[5]).is_file(): time.sleep(.01)\n"
        'print(\'{"type":"turn.started"}\',flush=True)\n'
        "print('tree ready',file=sys.stderr,flush=True)\n"
        "time.sleep(4)\n"
        "pathlib.Path(sys.argv[7]).write_text('late',encoding='utf-8')\n",
        encoding="utf-8",
    )
    helper.write_text(
        "import os,subprocess,sys\n"
        "child=subprocess.Popen([sys.executable,sys.argv[1],str(os.getpid()),*sys.argv[2:]])\n"
        "child.wait()\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "watchdog-report.json"
    watchdog_worker = tmp_path / "watchdog-worker.py"
    project_root = Path(process_module.__file__).resolve().parents[3]
    watchdog_worker.write_text(
        "import json,sys,time\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(project_root)!r})\n"
        "from ticket_automation.providers.codex_cli import CodexCommand,CodexProcessTimedOut,SubprocessCodexRunner\n"
        "helper,child,grand,pids,ready,gate,lateg,latec,cwd,stdout,stderr,report=map(Path,sys.argv[1:])\n"
        "def release_ready_tree():\n"
        "    until=time.monotonic()+10\n"
        "    while not ready.is_file() and time.monotonic()<until: time.sleep(.01)\n"
        "    if ready.is_file(): gate.write_text('go',encoding='utf-8')\n"
        "started=time.monotonic()\n"
        "try:\n"
        "    SubprocessCodexRunner().run(CodexCommand((sys.executable,str(helper),str(child),str(grand),str(pids),str(ready),str(gate),str(lateg),str(latec)),cwd,stdout_capture=stdout,stderr_capture=stderr),stdin='prompt',timeout_seconds=3,on_process_start=release_ready_tree)\n"
        "except CodexProcessTimedOut as error:\n"
        "    evidence=error.result.evidence\n"
        "    value={'status':'timed-out','elapsed':time.monotonic()-started,'termination_method':evidence.termination_method,'tree_termination_confirmed':evidence.tree_termination_confirmed}\n"
        "except BaseException as error:\n"
        "    value={'status':'wrong-error','message':repr(error),'elapsed':time.monotonic()-started}\n"
        "else:\n"
        "    value={'status':'unexpected-success','elapsed':time.monotonic()-started}\n"
        "report.write_text(json.dumps(value),encoding='utf-8')\n",
        encoding="utf-8",
    )
    control = subprocess.Popen(
        (sys.executable, "-c", "import time; time.sleep(15)"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    worker = subprocess.Popen(
        (
            sys.executable,
            str(watchdog_worker),
            str(helper),
            str(child),
            str(grandchild),
            str(pids_path),
            str(ready),
            str(gate),
            str(late_grandchild),
            str(late_child),
            str(tmp_path),
            str(stdout),
            str(stderr),
            str(report_path),
        ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    pids: list[int] = []
    try:
        try:
            _, worker_stderr = worker.communicate(timeout=16)
        except subprocess.TimeoutExpired:
            import signal

            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=5)
            pytest.fail("POSIX runner exceeded 16s outer watchdog")
        assert worker.returncode == 0, worker_stderr
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["status"] == "timed-out"
        assert report["elapsed"] < 13
        assert ready.is_file()
        pids = json.loads(pids_path.read_text(encoding="utf-8"))
        assert len(pids) == 3
        assert all(not _posix_pid_is_running(pid) for pid in pids)
        assert control.poll() is None
        assert report["termination_method"] == "posix-process-group"
        assert report["tree_termination_confirmed"] is True
        assert "turn.started" in stdout.read_text(encoding="utf-8")
        assert "tree ready" in stderr.read_text(encoding="utf-8")
        time.sleep(4.2)
        assert not late_child.exists()
        assert not late_grandchild.exists()
    finally:
        if worker.poll() is None:
            import signal

            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=5)
        control.kill()
        control.wait(timeout=5)
        if pids_path.is_file() and not pids:
            try:
                pids = json.loads(pids_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pids = []
        if os.name != "nt":
            import signal

            for pid in pids:
                if _posix_pid_is_running(pid):
                    os.kill(pid, signal.SIGKILL)


def _posix_pid_is_running(pid: int) -> bool:
    if os.name == "nt":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_adapter_rejects_scratch_inside_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    scratch = repository / "provider-scratch"

    def create_repository_scratch(*, prefix: str) -> str:
        del prefix
        scratch.mkdir()
        return str(scratch)

    monkeypatch.setattr(process_module.tempfile, "mkdtemp", create_repository_scratch)
    runner = FakeRunner(
        CodexProcessResult(0, "", ""), json.dumps(implementation_payload())
    )

    execution = CodexCliAgentExecutor(SETTINGS, runner=runner).execute(
        request(repository)
    )

    assert execution.failure_category is AgentFailureCategory.INVOCATION_START_FAILURE
    assert runner.calls == 0
    assert not scratch.exists()


def test_invalid_settings_are_rejected_before_runner_starts(tmp_path: Path) -> None:
    runner = FakeRunner(
        CodexProcessResult(0, "", ""), json.dumps(implementation_payload())
    )

    with pytest.raises(CodexCliSettingsError, match="reasoning_effort"):
        CodexCliAgentExecutor(
            CodexCliSettings(EXISTING_EXECUTABLE, "gpt-5.5", "unsupported"),
            runner=runner,
        )

    assert runner.calls == 0
