from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.helpers import prepend_executable_path, write_path_executable
from ticket_automation import codex as codex_module
from ticket_automation import executable_resolution
from ticket_automation.codex import (
    CodexCommand,
    CodexExecutionFailure,
    CodexExecutionStatus,
    CodexExecutor,
    CodexFailureKind,
    CodexProcessResult,
    CodexProcessTimedOut,
    CodexProcessTimeout,
    CodexResultValidationError,
    Sandbox,
    SubprocessCodexRunner,
    _CodexResultKind,
    _execute,
    _parse_implementation_result,
    _parse_review_result,
    build_codex_command,
    execute,
)
from ticket_automation.config import CodexExecutionSettings, ConfigError

EXISTING_EXECUTABLE = str(Path(sys.executable).resolve())


@dataclass
class FakeRunner:
    result: CodexProcessResult | None = None
    typed_result: str | None = None
    error: Exception | None = None
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
    ) -> CodexProcessResult:
        self.calls += 1
        self.command = command
        self.stdin = stdin
        self.timeout_seconds = timeout_seconds
        if self.error is not None:
            raise self.error
        if self.typed_result is not None:
            _output_result_path(command).write_text(self.typed_result, encoding="utf-8")
        assert self.result is not None
        return self.result


class StartFailureRunner:
    def run(
        self,
        command: CodexCommand,
        *,
        stdin: str,
        timeout_seconds: float | None,
    ) -> CodexProcessResult:
        del command, stdin, timeout_seconds
        raise OSError("permission denied")


def _output_result_path(command: CodexCommand) -> Path:
    return Path(command.argv[command.argv.index("--output-last-message") + 1])


def implementation_result(**changes: object) -> dict[str, object]:
    result: dict[str, object] = {
        "status": "COMPLETED",
        "summary": "Implemented the ticket.",
        "tests_run": [{"command": "pytest", "result": "passed"}],
        "assumptions": [],
        "known_issues": [],
    }
    result.update(changes)
    return result


def review_result(**changes: object) -> dict[str, object]:
    result: dict[str, object] = {
        "verdict": "PASS",
        "summary": "No issues found.",
        "findings": [],
    }
    result.update(changes)
    return result


def successful_process(*, stdout: str = "") -> CodexProcessResult:
    return CodexProcessResult(returncode=0, stdout=stdout, stderr="progress\n")


def test_command_writes_typed_result_to_the_canonical_result_artifact(tmp_path):
    schema = tmp_path / "schema.json"
    result = tmp_path / "result.json"

    command = build_codex_command(
        executable="codex",
        repo_path=tmp_path,
        sandbox=Sandbox.WORKSPACE_WRITE,
        output_schema=schema,
    )

    assert command.argv[command.argv.index("--output-schema") + 1] == str(
        schema.resolve()
    )
    assert command.argv[command.argv.index("--output-last-message") + 1] == str(
        result.resolve()
    )
    assert "--json" in command.argv


def test_implementation_result_is_parsed_without_a_duplicate_raw_result(tmp_path):
    payload = implementation_result()
    runner = FakeRunner(successful_process(), json.dumps(payload))

    execution = execute(
        prompt="implement",
        repo_path=tmp_path,
        sandbox=Sandbox.WORKSPACE_WRITE,
        output_schema=tmp_path / "implementation.schema.json",
        artifact_directory=tmp_path / "artifacts",
        executable=EXISTING_EXECUTABLE,
        runner=runner,
    )

    assert execution.status == CodexExecutionStatus.SUCCESS
    assert execution.structured_result == payload
    assert json.loads(execution.result_json_path.read_text(encoding="utf-8")) == payload
    assert not (execution.artifact_directory / "last-message.json").exists()
    assert execution.process_started is True
    assert execution.process_exit_code == 0


def test_review_result_uses_review_parser(tmp_path):
    payload = review_result(
        verdict="CORRECTIONS_REQUIRED",
        findings=[
            {
                "id": "R1",
                "disposition": "REQUIRED",
                "scope_relation": "IMPLEMENTATION",
                "title": "Fix it",
                "description": "A required change.",
                "evidence": "A failing test.",
                "required_change": "Make it pass.",
                "acceptance_criteria": ["Test passes."],
            }
        ],
    )
    runner = FakeRunner(successful_process(), json.dumps(payload))

    execution = _execute(
        prompt="review",
        repo_path=tmp_path,
        sandbox=Sandbox.READ_ONLY,
        output_schema=tmp_path / "review.schema.json",
        artifact_directory=tmp_path / "artifacts",
        executable=EXISTING_EXECUTABLE,
        runner=runner,
        _result_kind=_CodexResultKind.REVIEW,
    )

    assert execution.structured_result == payload


def test_correction_uses_implementation_parser(tmp_path):
    payload = implementation_result(status="BLOCKED")
    runner = FakeRunner(successful_process(), json.dumps(payload))

    execution = execute(
        prompt="correct",
        repo_path=tmp_path,
        sandbox=Sandbox.WORKSPACE_WRITE,
        output_schema=tmp_path / "implementation.schema.json",
        artifact_directory=tmp_path / "artifacts",
        executable=EXISTING_EXECUTABLE,
        runner=runner,
    )

    assert execution.structured_result == payload


def test_missing_typed_result_fails_clearly(tmp_path):
    runner = FakeRunner(successful_process())

    with pytest.raises(CodexExecutionFailure, match="did not write") as raised:
        execute(
            prompt="implement",
            repo_path=tmp_path,
            sandbox=Sandbox.WORKSPACE_WRITE,
            output_schema=tmp_path / "implementation.schema.json",
            artifact_directory=tmp_path / "artifacts",
            executable=EXISTING_EXECUTABLE,
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.MISSING_STRUCTURED_RESULT
    assert raised.value.execution.process_started is True
    assert raised.value.execution.process_exit_code == 0


def test_malformed_typed_result_json_fails_clearly(tmp_path):
    runner = FakeRunner(successful_process(), "{not json")

    with pytest.raises(CodexExecutionFailure, match="not valid JSON") as raised:
        execute(
            prompt="implement",
            repo_path=tmp_path,
            sandbox=Sandbox.WORKSPACE_WRITE,
            output_schema=tmp_path / "implementation.schema.json",
            artifact_directory=tmp_path / "artifacts",
            executable=EXISTING_EXECUTABLE,
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.INVALID_STRUCTURED_RESULT
    assert raised.value.execution.structured_result_present is True


@pytest.mark.parametrize(
    "payload",
    [
        [],
        "not an object",
        {"status": "COMPLETED"},
        implementation_result(status="NOT_A_STATUS"),
        implementation_result(tests_run=[{"command": "pytest"}]),
        implementation_result(summary=7),
        implementation_result(unexpected="value"),
    ],
)
def test_invalid_implementation_result_fields_fail(tmp_path, payload):
    runner = FakeRunner(successful_process(), json.dumps(payload))

    with pytest.raises(CodexExecutionFailure) as raised:
        execute(
            prompt="implement",
            repo_path=tmp_path,
            sandbox=Sandbox.WORKSPACE_WRITE,
            output_schema=tmp_path / "implementation.schema.json",
            artifact_directory=tmp_path / "artifacts",
            executable=EXISTING_EXECUTABLE,
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.INVALID_STRUCTURED_RESULT
    assert raised.value.execution.result_json_path.is_file()


def test_diagnostic_jsonl_is_persisted_without_affecting_result(tmp_path):
    diagnostics = (
        "not-json\n" + json.dumps({"type": "future.event", "shape": []}) + "\n"
    )
    payload = implementation_result()
    runner = FakeRunner(successful_process(stdout=diagnostics), json.dumps(payload))

    execution = execute(
        prompt="implement",
        repo_path=tmp_path,
        sandbox=Sandbox.WORKSPACE_WRITE,
        output_schema=tmp_path / "implementation.schema.json",
        artifact_directory=tmp_path / "artifacts",
        executable=EXISTING_EXECUTABLE,
        runner=runner,
    )

    assert execution.successful
    assert execution.events_jsonl_path.read_text(encoding="utf-8") == diagnostics
    assert execution.stderr_log_path.read_text(encoding="utf-8") == "progress\n"


def test_canonical_typed_result_wins_over_plausible_jsonl_result(tmp_path):
    diagnostics = (
        json.dumps(
            {
                "type": "turn.completed",
                "result": implementation_result(status="BLOCKED"),
            }
        )
        + "\n"
    )
    canonical_result = implementation_result(status="COMPLETED")
    runner = FakeRunner(
        successful_process(stdout=diagnostics), json.dumps(canonical_result)
    )

    execution = execute(
        prompt="implement",
        repo_path=tmp_path,
        sandbox=Sandbox.WORKSPACE_WRITE,
        output_schema=tmp_path / "implementation.schema.json",
        artifact_directory=tmp_path / "artifacts",
        executable=EXISTING_EXECUTABLE,
        runner=runner,
    )

    assert execution.structured_result == canonical_result
    assert json.loads(execution.result_json_path.read_text(encoding="utf-8")) == (
        canonical_result
    )


def test_stale_typed_result_cannot_satisfy_a_new_execution(tmp_path):
    artifact_directory = tmp_path / "artifacts"
    artifact_directory.mkdir()
    (artifact_directory / "result.json").write_text(
        json.dumps(implementation_result(status="BLOCKED")),
        encoding="utf-8",
    )
    runner = FakeRunner(successful_process())

    with pytest.raises(CodexExecutionFailure, match="did not write") as raised:
        execute(
            prompt="implement",
            repo_path=tmp_path,
            sandbox=Sandbox.WORKSPACE_WRITE,
            output_schema=tmp_path / "implementation.schema.json",
            artifact_directory=artifact_directory,
            executable=EXISTING_EXECUTABLE,
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.MISSING_STRUCTURED_RESULT
    assert not (artifact_directory / "result.json").exists()


def test_non_zero_exit_still_fails_from_process_evidence(tmp_path):
    runner = FakeRunner(
        CodexProcessResult(returncode=2, stdout="diagnostic\n", stderr="auth failed\n"),
        json.dumps(implementation_result()),
    )

    with pytest.raises(CodexExecutionFailure) as raised:
        execute(
            prompt="implement",
            repo_path=tmp_path,
            sandbox=Sandbox.WORKSPACE_WRITE,
            output_schema=tmp_path / "implementation.schema.json",
            artifact_directory=tmp_path / "artifacts",
            executable=EXISTING_EXECUTABLE,
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.AUTHENTICATION_OR_SERVICE
    assert raised.value.execution.process_started is True
    assert raised.value.execution.process_exit_code == 2


def test_process_start_failure_keeps_process_started_false(tmp_path):
    with pytest.raises(CodexExecutionFailure) as raised:
        execute(
            prompt="implement",
            repo_path=tmp_path,
            sandbox=Sandbox.WORKSPACE_WRITE,
            output_schema=tmp_path / "implementation.schema.json",
            artifact_directory=tmp_path / "artifacts",
            executable=EXISTING_EXECUTABLE,
            runner=StartFailureRunner(),
        )

    assert raised.value.kind == CodexFailureKind.PROCESS_START_FAILED
    assert raised.value.execution.process_started is False
    assert raised.value.execution.process_exit_code is None


def test_execution_rejects_target_codex_project_configuration(tmp_path):
    repository = tmp_path / "repo"
    repository.joinpath(".codex").mkdir(parents=True)
    repository.joinpath(".codex", "config.toml").write_text(
        "model = 'untrusted'\n",
        encoding="utf-8",
    )
    runner = FakeRunner(successful_process(), json.dumps(implementation_result()))

    with pytest.raises(CodexExecutionFailure) as raised:
        execute(
            prompt="implement",
            repo_path=repository,
            sandbox=Sandbox.WORKSPACE_WRITE,
            output_schema=tmp_path / "implementation.schema.json",
            artifact_directory=tmp_path / "artifacts",
            executable=EXISTING_EXECUTABLE,
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.PROJECT_CONFIGURATION_REJECTED
    assert runner.calls == 0


def test_execute_resolves_bare_executable_from_path(monkeypatch, tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    executable = write_path_executable(tmp_path / "tool-dir")
    prepend_executable_path(monkeypatch, executable.parent)
    runner = FakeRunner(successful_process(), json.dumps(implementation_result()))

    execute(
        prompt="implement",
        repo_path=repository,
        sandbox=Sandbox.WORKSPACE_WRITE,
        output_schema=tmp_path / "implementation.schema.json",
        artifact_directory=tmp_path / "artifacts",
        executable="codex",
        runner=runner,
    )

    assert runner.command is not None
    assert runner.command.argv[0] == str(executable.resolve())


def test_explicit_executable_does_not_use_path_lookup(monkeypatch, tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    executable = write_path_executable(tmp_path / "tool-dir", name="codex-explicit")
    runner = FakeRunner(successful_process(), json.dumps(implementation_result()))

    def fail_which(_configured: str) -> str | None:
        raise AssertionError("explicit executable paths must not use PATH lookup")

    monkeypatch.setattr(executable_resolution.shutil, "which", fail_which)

    execute(
        prompt="implement",
        repo_path=repository,
        sandbox=Sandbox.WORKSPACE_WRITE,
        output_schema=tmp_path / "implementation.schema.json",
        artifact_directory=tmp_path / "artifacts",
        executable=str(executable),
        runner=runner,
    )

    assert runner.command is not None
    assert runner.command.argv[0] == str(executable.resolve())


def test_prompt_is_sent_over_stdin_and_uses_repository_cwd(tmp_path):
    repository = tmp_path / "repo with spaces"
    repository.mkdir()
    runner = FakeRunner(successful_process(), json.dumps(implementation_result()))

    execution = execute(
        prompt="# Ticket\n\nImplement it.",
        repo_path=repository,
        sandbox=Sandbox.WORKSPACE_WRITE,
        output_schema=tmp_path / "implementation.schema.json",
        artifact_directory=tmp_path / "artifacts",
        executable=EXISTING_EXECUTABLE,
        runner=runner,
    )

    assert runner.stdin == "# Ticket\n\nImplement it."
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
    assert execution.successful


def test_timeout_preserves_available_diagnostic_evidence(tmp_path):
    runner = FakeRunner(
        error=CodexProcessTimedOut(
            CodexProcessTimeout(
                stdout='{"type":"turn.started"}\n',
                stderr="still working\n",
                timeout_seconds=3,
            )
        )
    )

    with pytest.raises(CodexExecutionFailure) as raised:
        CodexExecutor(
            executable=EXISTING_EXECUTABLE,
            timeout_seconds=3,
            runner=runner,
        ).execute(
            prompt="implement",
            repo_path=tmp_path,
            sandbox=Sandbox.WORKSPACE_WRITE,
            output_schema=tmp_path / "implementation.schema.json",
            artifact_directory=tmp_path / "artifacts",
        )

    assert raised.value.kind == CodexFailureKind.TIMEOUT
    assert raised.value.execution.process_started is True
    assert raised.value.execution.events_jsonl_path.read_text(encoding="utf-8") == (
        '{"type":"turn.started"}\n'
    )
    assert raised.value.execution.stderr_log_path.read_text(encoding="utf-8") == (
        "still working\n"
    )


def test_subprocess_timeout_preserves_partial_output(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        raise codex_module.subprocess.TimeoutExpired(
            cmd=args[0],
            timeout=kwargs["timeout"],
            output=b'{"type":"turn.started"}\n',
            stderr=b"partial stderr\n",
        )

    monkeypatch.setattr(codex_module.subprocess, "run", fake_run)

    with pytest.raises(CodexProcessTimedOut) as raised:
        SubprocessCodexRunner().run(
            CodexCommand(argv=("codex", "exec", "-"), cwd=tmp_path),
            stdin="prompt",
            timeout_seconds=5,
        )

    assert raised.value.result.timeout_seconds == 5
    assert raised.value.result.stdout == '{"type":"turn.started"}\n'
    assert raised.value.result.stderr == "partial stderr\n"


def test_subprocess_runner_disables_shell_execution(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

        class Completed:
            returncode = 0
            stdout = b"diagnostic output\n"
            stderr = b""

        return Completed()

    monkeypatch.setattr(codex_module.subprocess, "run", fake_run)

    result = SubprocessCodexRunner().run(
        CodexCommand(
            argv=("codex", "exec", "-"),
            cwd=tmp_path,
            environment={
                "TEMP": "C:/scratch",
                "TMP": "C:/scratch",
                "TMPDIR": "C:/scratch",
            },
        ),
        stdin="prompt",
        timeout_seconds=10,
    )

    assert result.returncode == 0
    assert captured["args"] == (("codex", "exec", "-"),)
    assert isinstance(captured["kwargs"], dict)
    assert captured["kwargs"]["cwd"] == tmp_path
    assert captured["kwargs"]["input"] == b"prompt"
    environment = captured["kwargs"]["env"]
    assert isinstance(environment, dict)
    assert environment["TEMP"] == "C:/scratch"
    assert environment["TMP"] == "C:/scratch"
    assert environment["TMPDIR"] == "C:/scratch"
    assert captured["kwargs"]["shell"] is False


def test_executor_rejects_a_scratch_directory_inside_the_repository(
    monkeypatch, tmp_path
):
    repository = tmp_path / "repo"
    repository.mkdir()
    scratch = repository / "pytest-tmp"

    def create_repository_scratch(*, prefix):
        del prefix
        scratch.mkdir()
        return str(scratch)

    monkeypatch.setattr(codex_module.tempfile, "mkdtemp", create_repository_scratch)
    runner = FakeRunner(successful_process(), json.dumps(implementation_result()))

    with pytest.raises(CodexExecutionFailure, match="outside the target repository"):
        execute(
            prompt="implement",
            repo_path=repository,
            sandbox=Sandbox.WORKSPACE_WRITE,
            output_schema=tmp_path / "implementation.schema.json",
            artifact_directory=tmp_path / "artifacts",
            executable=EXISTING_EXECUTABLE,
            runner=runner,
        )

    assert runner.calls == 0
    assert not scratch.exists()


def test_invalid_execution_config_fails_before_runner_starts(tmp_path):
    runner = FakeRunner(successful_process(), json.dumps(implementation_result()))

    with pytest.raises(ConfigError, match="codex.reasoning_effort"):
        execute(
            prompt="implement",
            repo_path=tmp_path,
            sandbox=Sandbox.READ_ONLY,
            output_schema=tmp_path / "implementation.schema.json",
            artifact_directory=tmp_path / "artifacts",
            executable=EXISTING_EXECUTABLE,
            execution_config=CodexExecutionSettings(
                model="gpt-5.5",
                reasoning_effort="unsupported",
            ),
            runner=runner,
        )

    assert runner.calls == 0


def test_missing_executable_fails_before_runner_starts(monkeypatch, tmp_path):
    monkeypatch.setattr(executable_resolution.shutil, "which", lambda _name: None)
    runner = FakeRunner(successful_process(), json.dumps(implementation_result()))

    with pytest.raises(CodexExecutionFailure) as raised:
        execute(
            prompt="implement",
            repo_path=tmp_path,
            sandbox=Sandbox.READ_ONLY,
            output_schema=tmp_path / "implementation.schema.json",
            artifact_directory=tmp_path / "artifacts",
            executable="ticket-automation-missing-codex",
            runner=runner,
        )

    assert runner.calls == 0
    assert raised.value.kind == CodexFailureKind.EXECUTABLE_UNAVAILABLE


def test_fixed_parsers_reject_unsupported_fields():
    with pytest.raises(CodexResultValidationError, match="unsupported fields"):
        _parse_implementation_result(implementation_result(extra="no"))
    with pytest.raises(CodexResultValidationError, match="unsupported fields"):
        _parse_review_result(review_result(extra="no"))
