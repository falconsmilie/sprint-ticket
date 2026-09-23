from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.helpers import prepend_executable_path, write_path_executable
from ticket_automation import executable_resolution
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
from ticket_automation.providers.codex_cli import (
    CodexCliAgentExecutor,
    CodexCommand,
    CodexProcessResult,
    CodexProcessTimedOut,
    CodexProcessTimeout,
    CodexSettings,
    CodexSettingsError,
    SubprocessCodexRunner,
)
from ticket_automation.providers.codex_cli import process as process_module
from ticket_automation.providers.codex_cli.command import build_command

EXISTING_EXECUTABLE = str(Path(sys.executable).resolve())
SETTINGS = CodexSettings(EXISTING_EXECUTABLE, "gpt-5.5", "xhigh")


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
        return self.result

    def _run_with_start_tracking(
        self,
        command: CodexCommand,
        *,
        stdin: str,
        timeout_seconds: float | None,
        on_process_start,
    ) -> CodexProcessResult:
        return self.run(
            command,
            stdin=stdin,
            timeout_seconds=timeout_seconds,
            on_process_start=on_process_start,
        )


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
    assert (tmp_path / "artifacts" / "execution.json").is_file()


def test_unavailable_executable_is_process_not_started(tmp_path: Path) -> None:
    executor = CodexCliAgentExecutor(
        CodexSettings(str(tmp_path / "missing"), "gpt-5.5", "xhigh"),
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


def test_legacy_runner_remains_read_only_compatible_and_cannot_claim_write(
    tmp_path: Path,
) -> None:
    @dataclass
    class LegacyRunner:
        calls: int = 0

        def run(self, command, *, stdin, timeout_seconds):
            del stdin, timeout_seconds
            self.calls += 1
            output = Path(command.argv[command.argv.index("--output-last-message") + 1])
            output.write_text(json.dumps(review_payload()), encoding="utf-8")
            return CodexProcessResult(0, "", "")

    review_repository = tmp_path / "review"
    review_repository.mkdir()
    runner = LegacyRunner()
    executor = CodexCliAgentExecutor(SETTINGS, runner=runner)

    review = executor.execute(request(review_repository, AgentTaskKind.REVIEW))

    assert review.successful
    assert runner.calls == 1
    assert AgentCapability.WORKSPACE_WRITE_EXECUTION not in executor.capabilities

    write_repository = tmp_path / "write"
    write_repository.mkdir()
    rejected = executor.execute(request(write_repository))

    assert not rejected.successful
    assert (
        rejected.failure_category
        is AgentFailureCategory.CAPABILITY_OR_CONFIGURATION_FAILURE
    )
    assert rejected.invocation_start is InvocationStart.NOT_STARTED
    assert runner.calls == 1


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
    assert (tmp_path / "artifacts" / "codex-result.json").is_file()


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
    settings = CodexSettings("codex", "gpt-5.5", "xhigh")

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
    settings = CodexSettings(str(executable), "gpt-5.5", "xhigh")

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


def test_subprocess_timeout_decodes_partial_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(*args: object, **kwargs: object) -> object:
        raise process_module.subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=kwargs["timeout"],
            output=b'{"type":"turn.started"}\n',
            stderr=b"partial stderr\n",
        )

    monkeypatch.setattr(process_module.subprocess, "run", fake_run)

    with pytest.raises(CodexProcessTimedOut) as raised:
        SubprocessCodexRunner().run(
            CodexCommand(("codex", "exec", "-"), tmp_path),
            stdin="prompt",
            timeout_seconds=5,
        )

    assert raised.value.result.timeout_seconds == 5
    assert raised.value.result.stdout == '{"type":"turn.started"}\n'
    assert raised.value.result.stderr == "partial stderr\n"


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

    result = SubprocessCodexRunner()._run_with_start_tracking(
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
        SubprocessCodexRunner()._run_with_start_tracking(
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

        def wait(self) -> int:
            self.waited = True
            return 1

    process = SpawnedProcess()
    monkeypatch.setattr(
        process_module.subprocess, "Popen", lambda *args, **kwargs: process
    )

    def interrupt() -> None:
        raise KeyboardInterrupt("interrupted while recording start")

    with pytest.raises(KeyboardInterrupt, match="recording start"):
        SubprocessCodexRunner()._run_with_start_tracking(
            CodexCommand(("codex", "exec", "-"), tmp_path),
            stdin="prompt",
            timeout_seconds=5,
            on_process_start=interrupt,
        )

    assert process.killed
    assert process.waited


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
        SubprocessCodexRunner()._run_with_start_tracking(
            CodexCommand(("codex", "exec", "-"), tmp_path),
            stdin="prompt",
            timeout_seconds=10,
            on_process_start=record_start,
        )

    assert started is False


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

    with pytest.raises(CodexSettingsError, match="reasoning_effort"):
        CodexCliAgentExecutor(
            CodexSettings(EXISTING_EXECUTABLE, "gpt-5.5", "unsupported"),
            runner=runner,
        )

    assert runner.calls == 0
