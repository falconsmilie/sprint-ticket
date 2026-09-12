from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.helpers import (
    GIT,
    create_git_repo,
    make_config,
    prepend_executable_path,
    write_path_executable,
)
from ticket_automation import codex as codex_module
from ticket_automation import executable_resolution
from ticket_automation.codex import (
    CodexCommand,
    CodexExecutionFailure,
    CodexExecutionStatus,
    CodexExecutor,
    CodexFailureKind,
    CodexProcessOutputDecodeError,
    CodexProcessResult,
    CodexProcessTimedOut,
    CodexProcessTimeout,
    Sandbox,
    SubprocessCodexRunner,
    build_codex_command,
    execute,
    parse_sandbox,
)
from ticket_automation.preflight import run_preflight

EXISTING_EXECUTABLE = str(Path(sys.executable).resolve())


@dataclass
class FakeRunner:
    result: CodexProcessResult | None = None
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
        assert self.result is not None
        return self.result


def test_builds_workspace_write_command_arguments(tmp_path):
    schema = write_schema(tmp_path / "schema.json")

    command = build_codex_command(
        executable="codex",
        repo_path=tmp_path / "repo",
        sandbox=Sandbox.WORKSPACE_WRITE,
        output_schema=schema,
    )

    assert command.argv == (
        "codex",
        "exec",
        "-",
        "--sandbox",
        "workspace-write",
        "--json",
        "--output-schema",
        str(schema.resolve()),
    )
    assert command.cwd == tmp_path / "repo"


def test_builds_read_only_command_arguments(tmp_path):
    schema = write_schema(tmp_path / "schema.json")

    command = build_codex_command(
        executable="codex",
        repo_path=tmp_path / "repo",
        sandbox=Sandbox.READ_ONLY,
        output_schema=schema,
    )

    assert command.argv[4] == "read-only"


def test_rejects_unvalidated_sandbox_strings(tmp_path):
    schema = write_schema(tmp_path / "schema.json")

    with pytest.raises(TypeError, match="Sandbox"):
        build_codex_command(
            executable="codex",
            repo_path=tmp_path / "repo",
            sandbox="workspace-write",  # type: ignore[arg-type]
            output_schema=schema,
        )


def test_parse_sandbox_converts_validated_config_strings():
    assert parse_sandbox("workspace-write") == Sandbox.WORKSPACE_WRITE
    with pytest.raises(ValueError, match="Unsupported Codex sandbox"):
        parse_sandbox("danger-full-access")


def test_execute_resolves_bare_executable_from_path(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    schema = write_schema(tmp_path / "schema.json")
    executable = write_path_executable(tmp_path / "tool dir")
    prepend_executable_path(monkeypatch, executable.parent)
    runner = FakeRunner(result=successful_process({"status": "ok"}))

    execute(
        prompt="prompt",
        repo_path=repo,
        sandbox=Sandbox.WORKSPACE_WRITE,
        output_schema=schema,
        artifact_directory=tmp_path / "artifacts",
        executable="codex",
        runner=runner,
    )

    assert runner.command is not None
    assert runner.command.argv == (
        str(executable.resolve()),
        "exec",
        "-",
        "--sandbox",
        "workspace-write",
        "--json",
        "--output-schema",
        str(schema.resolve()),
    )


def test_execute_uses_windows_cmd_launcher_returned_by_which(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    schema = write_schema(tmp_path / "schema.json")
    executable = tmp_path / "nodejs" / "codex.CMD"
    executable.parent.mkdir(parents=True)
    executable.write_text("@echo off\nexit /b 0\n", encoding="utf-8")
    runner = FakeRunner(result=successful_process({"status": "ok"}))

    def fake_which(configured: str) -> str | None:
        if configured == "codex":
            return str(executable)
        return None

    monkeypatch.setattr(executable_resolution.shutil, "which", fake_which)

    execute(
        prompt="prompt",
        repo_path=repo,
        sandbox=Sandbox.READ_ONLY,
        output_schema=schema,
        artifact_directory=tmp_path / "artifacts",
        executable="codex",
        runner=runner,
    )

    assert runner.command is not None
    assert runner.command.argv[0] == str(executable.resolve())
    assert runner.command.argv[0] != "codex"
    assert runner.command.argv[1:3] == ("exec", "-")


def test_execute_uses_explicit_absolute_executable_without_path_lookup(
    monkeypatch,
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    schema = write_schema(tmp_path / "schema.json")
    executable = write_path_executable(tmp_path / "tools", name="codex-explicit")
    runner = FakeRunner(result=successful_process({"status": "ok"}))

    def fail_which(_configured: str) -> str | None:
        raise AssertionError("explicit executable paths must not use PATH lookup")

    monkeypatch.setattr(executable_resolution.shutil, "which", fail_which)

    execute(
        prompt="prompt",
        repo_path=repo,
        sandbox=Sandbox.READ_ONLY,
        output_schema=schema,
        artifact_directory=tmp_path / "artifacts",
        executable=str(executable),
        runner=runner,
    )

    assert runner.command is not None
    assert runner.command.argv[0] == str(executable.resolve())


@pytest.mark.skipif(GIT is None, reason="git executable is required for shared test")
def test_preflight_and_execution_use_shared_resolver(monkeypatch, tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    schema = write_schema(tmp_path / "schema.json")
    resolved_codex = tmp_path / "shared resolver" / "codex.CMD"
    resolved_codex.parent.mkdir(parents=True)
    resolved_codex.write_text("@echo off\nexit /b 0\n", encoding="utf-8")
    resolved_codex = resolved_codex.resolve()
    calls: list[tuple[str, Path | None]] = []
    original_resolve_executable = executable_resolution.resolve_executable

    def fake_resolve_executable(
        configured: str,
        *,
        cwd: Path | str | None = None,
    ) -> Path | None:
        calls.append((configured, None if cwd is None else Path(cwd)))
        if configured == "codex":
            return resolved_codex
        return original_resolve_executable(configured, cwd=cwd)

    monkeypatch.setattr(
        executable_resolution,
        "resolve_executable",
        fake_resolve_executable,
    )
    config = make_config(repo, codex_executable="codex")
    runner = FakeRunner(result=successful_process({"status": "ok"}))

    preflight = run_preflight(config)
    execute(
        prompt="prompt",
        repo_path=repo,
        sandbox=Sandbox.READ_ONLY,
        output_schema=schema,
        artifact_directory=tmp_path / "artifacts",
        executable=config.codex.executable,
        runner=runner,
    )

    assert preflight.passed
    assert runner.command is not None
    assert runner.command.argv[0] == str(resolved_codex)
    assert [call for call in calls if call[0] == "codex"] == [
        ("codex", repo),
        ("codex", repo),
    ]


def test_supplies_prompt_over_stdin_and_uses_repository_as_cwd(tmp_path):
    repo = tmp_path / "repo with spaces"
    repo.mkdir()
    schema = write_schema(tmp_path / "schema.json")
    runner = FakeRunner(result=successful_process({"status": "ok"}))

    result = execute(
        prompt="# Ticket\n\nDo the work.",
        repo_path=repo,
        sandbox=Sandbox.WORKSPACE_WRITE,
        output_schema=schema,
        artifact_directory=tmp_path / "artifacts",
        executable=EXISTING_EXECUTABLE,
        runner=runner,
    )

    assert runner.stdin == "# Ticket\n\nDo the work."
    assert runner.command is not None
    assert runner.command.cwd == repo
    assert result.status == CodexExecutionStatus.SUCCESS


def test_persists_jsonl_stderr_prompt_and_structured_result(tmp_path):
    schema = write_schema(tmp_path / "schema.json")
    events = event_stream({"status": "ok", "notes": ["done"]})
    runner = FakeRunner(
        result=CodexProcessResult(returncode=0, stdout=events, stderr="progress\n")
    )
    prompt = "generated prompt\nSECRET_TOKEN=do-not-persist"

    result = execute(
        prompt=prompt,
        repo_path=tmp_path,
        sandbox=Sandbox.READ_ONLY,
        output_schema=schema,
        artifact_directory=tmp_path / "artifacts",
        executable=EXISTING_EXECUTABLE,
        runner=runner,
    )

    assert result.prompt_path.read_text(encoding="utf-8") == prompt
    assert result.events_jsonl_path.read_text(encoding="utf-8") == events
    assert result.stderr_log_path.read_text(encoding="utf-8") == "progress\n"
    assert json.loads(result.result_json_path.read_text(encoding="utf-8")) == {
        "notes": ["done"],
        "status": "ok",
    }
    execution_record = read_execution_record(result)
    assert execution_record["status"] == "SUCCESS"
    assert execution_record["process_started"] is True
    assert execution_record["process_exit_code"] == 0
    assert execution_record["timed_out"] is False
    assert execution_record["failure_kind"] is None
    assert execution_record["structured_result_present"] is True
    assert execution_record["result_json_present"] is True
    assert execution_record["artifact_paths"] == {
        "events_jsonl": str(result.events_jsonl_path),
        "execution_json": str(result.execution_json_path),
        "prompt_md": str(result.prompt_path),
        "result_json": str(result.result_json_path),
        "stderr_log": str(result.stderr_log_path),
    }
    assert "structured_result" not in execution_record
    assert "SECRET_TOKEN" not in json.dumps(execution_record)
    assert result.structured_result == {"status": "ok", "notes": ["done"]}
    assert result.process_exit_code == 0
    assert result.successful


def test_accepts_trusted_turn_completed_result_shape(tmp_path):
    schema = write_schema(tmp_path / "schema.json")
    events = (
        "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "thread"}),
                json.dumps({"type": "turn.started"}),
                json.dumps({"type": "turn.completed", "result": {"status": "ok"}}),
            ]
        )
        + "\n"
    )
    runner = FakeRunner(
        result=CodexProcessResult(returncode=0, stdout=events, stderr="")
    )

    result = execute(
        prompt="prompt",
        repo_path=tmp_path,
        sandbox=Sandbox.READ_ONLY,
        output_schema=schema,
        artifact_directory=tmp_path / "artifacts",
        executable=EXISTING_EXECUTABLE,
        runner=runner,
    )

    assert result.structured_result == {"status": "ok"}


def test_accepts_trusted_agent_message_content_shape(tmp_path):
    schema = write_schema(tmp_path / "schema.json")
    events = (
        "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "thread"}),
                json.dumps({"type": "turn.started"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "item_1",
                            "type": "agent_message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": json.dumps({"status": "ok"}),
                                }
                            ],
                        },
                    }
                ),
                json.dumps({"type": "turn.completed"}),
            ]
        )
        + "\n"
    )
    runner = FakeRunner(
        result=CodexProcessResult(returncode=0, stdout=events, stderr="")
    )

    result = execute(
        prompt="prompt",
        repo_path=tmp_path,
        sandbox=Sandbox.READ_ONLY,
        output_schema=schema,
        artifact_directory=tmp_path / "artifacts",
        executable=EXISTING_EXECUTABLE,
        runner=runner,
    )

    assert result.structured_result == {"status": "ok"}


def test_nonzero_exit_is_typed_failure_and_not_fake_result(tmp_path):
    schema = write_schema(tmp_path / "schema.json")
    runner = FakeRunner(
        result=CodexProcessResult(
            returncode=2,
            stdout=event_stream({"status": "ok"}),
            stderr="boom\n",
        )
    )

    with pytest.raises(CodexExecutionFailure) as raised:
        execute(
            prompt="prompt",
            repo_path=tmp_path,
            sandbox=Sandbox.READ_ONLY,
            output_schema=schema,
            artifact_directory=tmp_path / "artifacts",
            executable=EXISTING_EXECUTABLE,
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.NON_ZERO_EXIT
    assert raised.value.execution.status == CodexExecutionStatus.FAILED
    assert raised.value.execution.process_exit_code == 2
    assert raised.value.execution.events_jsonl_path.read_text(encoding="utf-8")
    assert (
        raised.value.execution.stderr_log_path.read_text(encoding="utf-8") == "boom\n"
    )
    assert_failure_execution_record(
        raised.value.execution,
        failure_kind=CodexFailureKind.NON_ZERO_EXIT,
        process_started=True,
        exit_code=2,
    )
    assert not raised.value.execution.result_json_path.exists()


def test_incomplete_stream_with_agent_message_is_typed_failure(tmp_path):
    schema = write_schema(tmp_path / "schema.json")
    runner = FakeRunner(
        result=CodexProcessResult(
            returncode=0,
            stdout="\n".join(
                [
                    json.dumps({"type": "thread.started", "thread_id": "thread"}),
                    json.dumps({"type": "turn.started"}),
                    json.dumps(
                        {
                            "type": "item.completed",
                            "item": {
                                "id": "item_1",
                                "type": "agent_message",
                                "text": json.dumps({"status": "ok"}),
                            },
                        }
                    ),
                ]
            )
            + "\n",
            stderr="",
        )
    )

    with pytest.raises(CodexExecutionFailure) as raised:
        execute(
            prompt="prompt",
            repo_path=tmp_path,
            sandbox=Sandbox.READ_ONLY,
            output_schema=schema,
            artifact_directory=tmp_path / "artifacts",
            executable=EXISTING_EXECUTABLE,
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.MISSING_STRUCTURED_RESULT
    assert not raised.value.execution.result_json_path.exists()


def test_non_agent_result_event_does_not_count_as_final_result(tmp_path):
    schema = write_schema(tmp_path / "schema.json")
    runner = FakeRunner(
        result=CodexProcessResult(
            returncode=0,
            stdout="\n".join(
                [
                    json.dumps({"type": "thread.started", "thread_id": "thread"}),
                    json.dumps({"type": "turn.started"}),
                    json.dumps(
                        {
                            "type": "item.completed",
                            "result": {"status": "ok"},
                            "item": {
                                "id": "item_1",
                                "type": "command_execution",
                                "status": "completed",
                            },
                        }
                    ),
                    json.dumps({"type": "turn.completed"}),
                ]
            )
            + "\n",
            stderr="",
        )
    )

    with pytest.raises(CodexExecutionFailure) as raised:
        execute(
            prompt="prompt",
            repo_path=tmp_path,
            sandbox=Sandbox.READ_ONLY,
            output_schema=schema,
            artifact_directory=tmp_path / "artifacts",
            executable=EXISTING_EXECUTABLE,
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.MISSING_STRUCTURED_RESULT
    assert not raised.value.execution.result_json_path.exists()


def test_present_but_malformed_structured_result_is_invalid_result(tmp_path):
    schema = write_schema(tmp_path / "schema.json")
    runner = FakeRunner(
        result=CodexProcessResult(
            returncode=0,
            stdout="\n".join(
                [
                    json.dumps({"type": "thread.started", "thread_id": "thread"}),
                    json.dumps({"type": "turn.started"}),
                    json.dumps(
                        {
                            "type": "item.completed",
                            "item": {
                                "id": "item_1",
                                "type": "agent_message",
                                "text": "{not valid json",
                            },
                        }
                    ),
                    json.dumps({"type": "turn.completed"}),
                ]
            )
            + "\n",
            stderr="",
        )
    )

    with pytest.raises(CodexExecutionFailure) as raised:
        execute(
            prompt="prompt",
            repo_path=tmp_path,
            sandbox=Sandbox.READ_ONLY,
            output_schema=schema,
            artifact_directory=tmp_path / "artifacts",
            executable=EXISTING_EXECUTABLE,
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.INVALID_STRUCTURED_RESULT
    assert_failure_execution_record(
        raised.value.execution,
        failure_kind=CodexFailureKind.INVALID_STRUCTURED_RESULT,
        process_started=True,
        exit_code=0,
        structured_result_present=True,
    )


def test_timeout_preserves_available_logs(tmp_path):
    schema = write_schema(tmp_path / "schema.json")
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
            prompt="prompt",
            repo_path=tmp_path,
            sandbox=Sandbox.WORKSPACE_WRITE,
            output_schema=schema,
            artifact_directory=tmp_path / "artifacts",
        )

    assert raised.value.kind == CodexFailureKind.TIMEOUT
    assert raised.value.execution.process_exit_code is None
    assert (
        raised.value.execution.events_jsonl_path.read_text(encoding="utf-8")
        == '{"type":"turn.started"}\n'
    )
    assert (
        raised.value.execution.stderr_log_path.read_text(encoding="utf-8")
        == "still working\n"
    )
    execution_record = assert_failure_execution_record(
        raised.value.execution,
        failure_kind=CodexFailureKind.TIMEOUT,
        process_started=True,
        exit_code=None,
        timed_out=True,
    )
    assert execution_record["timeout_seconds"] == 3
    assert runner.timeout_seconds == 3


def test_subprocess_timeout_converts_partial_output(monkeypatch, tmp_path):
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


def test_codex_output_decode_failure_is_typed_execution_failure(tmp_path):
    schema = write_schema(tmp_path / "schema.json")
    runner = FakeRunner(
        error=CodexProcessOutputDecodeError(
            "Codex stdout was not valid utf-8.",
            CodexProcessResult(
                returncode=0,
                stdout="replacement � output\n",
                stderr="progress\n",
            ),
        )
    )

    with pytest.raises(CodexExecutionFailure) as raised:
        execute(
            prompt="prompt",
            repo_path=tmp_path,
            sandbox=Sandbox.READ_ONLY,
            output_schema=schema,
            artifact_directory=tmp_path / "artifacts",
            executable=EXISTING_EXECUTABLE,
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.MALFORMED_EVENT_STREAM
    assert raised.value.execution.events_jsonl_path.read_text(encoding="utf-8") == (
        "replacement � output\n"
    )
    assert raised.value.execution.stderr_log_path.read_text(encoding="utf-8") == (
        "progress\n"
    )
    assert_failure_execution_record(
        raised.value.execution,
        failure_kind=CodexFailureKind.MALFORMED_EVENT_STREAM,
        process_started=True,
        exit_code=0,
    )


def test_malformed_jsonl_is_typed_failure(tmp_path):
    schema = write_schema(tmp_path / "schema.json")
    runner = FakeRunner(
        result=CodexProcessResult(
            returncode=0,
            stdout='{"type":"turn.started"}\nnope\n',
            stderr="",
        )
    )

    with pytest.raises(CodexExecutionFailure) as raised:
        execute(
            prompt="prompt",
            repo_path=tmp_path,
            sandbox=Sandbox.READ_ONLY,
            output_schema=schema,
            artifact_directory=tmp_path / "artifacts",
            executable=EXISTING_EXECUTABLE,
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.MALFORMED_EVENT_STREAM
    assert raised.value.execution.events_jsonl_path.read_text(
        encoding="utf-8"
    ).endswith("nope\n")
    assert_failure_execution_record(
        raised.value.execution,
        failure_kind=CodexFailureKind.MALFORMED_EVENT_STREAM,
        process_started=True,
        exit_code=0,
    )
    assert not raised.value.execution.result_json_path.exists()


def test_missing_structured_result_is_typed_failure(tmp_path):
    schema = write_schema(tmp_path / "schema.json")
    runner = FakeRunner(
        result=CodexProcessResult(
            returncode=0,
            stdout='{"type":"turn.started"}\n{"type":"turn.completed"}\n',
            stderr="",
        )
    )

    with pytest.raises(CodexExecutionFailure) as raised:
        execute(
            prompt="prompt",
            repo_path=tmp_path,
            sandbox=Sandbox.READ_ONLY,
            output_schema=schema,
            artifact_directory=tmp_path / "artifacts",
            executable=EXISTING_EXECUTABLE,
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.MISSING_STRUCTURED_RESULT
    assert_failure_execution_record(
        raised.value.execution,
        failure_kind=CodexFailureKind.MISSING_STRUCTURED_RESULT,
        process_started=True,
        exit_code=0,
    )


def test_schema_incompatible_result_is_typed_failure(tmp_path):
    schema = write_schema(tmp_path / "schema.json")
    runner = FakeRunner(result=successful_process({"status": 100}))

    with pytest.raises(CodexExecutionFailure) as raised:
        execute(
            prompt="prompt",
            repo_path=tmp_path,
            sandbox=Sandbox.READ_ONLY,
            output_schema=schema,
            artifact_directory=tmp_path / "artifacts",
            executable=EXISTING_EXECUTABLE,
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.INVALID_STRUCTURED_RESULT
    assert_failure_execution_record(
        raised.value.execution,
        failure_kind=CodexFailureKind.INVALID_STRUCTURED_RESULT,
        process_started=True,
        exit_code=0,
        structured_result_present=True,
    )
    assert not raised.value.execution.result_json_path.exists()


def test_supported_json_schema_contract_keywords_accept_valid_result(tmp_path):
    schema = write_contract_schema(tmp_path / "schema.json")
    runner = FakeRunner(
        result=successful_process(
            {
                "ticket": "TA-004",
                "attempts": 1,
                "notes": ["ok"],
                "mode": "implement",
                "score": None,
                "summary": "ready",
                "metadata": "test",
            }
        )
    )

    result = execute(
        prompt="prompt",
        repo_path=tmp_path,
        sandbox=Sandbox.READ_ONLY,
        output_schema=schema,
        artifact_directory=tmp_path / "artifacts",
        executable=EXISTING_EXECUTABLE,
        runner=runner,
    )

    assert result.successful
    assert result.structured_result["metadata"] == "test"


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("ticket", "004"),
        ("attempts", 0),
        ("notes", []),
        ("mode", "correct"),
        ("score", 5),
        ("summary", "no"),
    ],
)
def test_supported_json_schema_contract_keywords_reject_bad_result(
    tmp_path,
    field,
    bad_value,
):
    schema = write_contract_schema(tmp_path / "schema.json")
    result_payload = {
        "ticket": "TA-004",
        "attempts": 1,
        "notes": ["ok"],
        "mode": "implement",
        "score": None,
        "summary": "ready",
    }
    result_payload[field] = bad_value
    runner = FakeRunner(result=successful_process(result_payload))

    with pytest.raises(CodexExecutionFailure) as raised:
        execute(
            prompt="prompt",
            repo_path=tmp_path,
            sandbox=Sandbox.READ_ONLY,
            output_schema=schema,
            artifact_directory=tmp_path / "artifacts",
            executable=EXISTING_EXECUTABLE,
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.INVALID_STRUCTURED_RESULT
    assert not raised.value.execution.result_json_path.exists()


def test_unsupported_schema_keywords_fail_without_running_codex(tmp_path):
    schema = write_custom_schema(
        tmp_path / "schema.json",
        {
            "type": "object",
            "dependentRequired": {"status": ["notes"]},
        },
    )
    runner = FakeRunner(result=successful_process({"status": "ok"}))

    with pytest.raises(CodexExecutionFailure) as raised:
        execute(
            prompt="prompt",
            repo_path=tmp_path,
            sandbox=Sandbox.READ_ONLY,
            output_schema=schema,
            artifact_directory=tmp_path / "artifacts",
            executable=EXISTING_EXECUTABLE,
            runner=runner,
        )

    assert runner.calls == 0
    assert raised.value.kind == CodexFailureKind.INVALID_SCHEMA
    assert_failure_execution_record(
        raised.value.execution,
        failure_kind=CodexFailureKind.INVALID_SCHEMA,
        process_started=False,
        exit_code=None,
    )


def test_paths_containing_spaces_retain_argument_boundaries(tmp_path):
    repo = tmp_path / "target repo"
    repo.mkdir()
    schema = write_schema(tmp_path / "schema dir" / "result schema.json")
    artifact_dir = tmp_path / "run artifacts"
    executable = write_path_executable(tmp_path / "tool dir", name="codex cli")
    runner = FakeRunner(result=successful_process({"status": "ok"}))

    result = execute(
        prompt="prompt",
        repo_path=repo,
        sandbox=Sandbox.WORKSPACE_WRITE,
        output_schema=schema,
        artifact_directory=artifact_dir,
        executable=str(executable),
        runner=runner,
    )

    assert runner.command is not None
    assert runner.command.argv[0] == str(executable.resolve())
    assert runner.command.argv[-1] == str(schema.resolve())
    assert runner.command.cwd == repo
    assert result.artifact_directory == artifact_dir
    assert result.result_json_path.is_file()


def test_process_start_failure_is_typed(tmp_path):
    schema = write_schema(tmp_path / "schema.json")
    runner = FakeRunner(error=FileNotFoundError("missing codex"))

    with pytest.raises(CodexExecutionFailure) as raised:
        execute(
            prompt="prompt",
            repo_path=tmp_path,
            sandbox=Sandbox.READ_ONLY,
            output_schema=schema,
            artifact_directory=tmp_path / "artifacts",
            executable=EXISTING_EXECUTABLE,
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.EXECUTABLE_UNAVAILABLE
    assert raised.value.execution.prompt_path.read_text(encoding="utf-8") == "prompt"
    assert raised.value.execution.stderr_log_path.read_text(encoding="utf-8")
    assert_failure_execution_record(
        raised.value.execution,
        failure_kind=CodexFailureKind.EXECUTABLE_UNAVAILABLE,
        process_started=False,
        exit_code=None,
    )


def test_missing_executable_fails_before_runner_starts(monkeypatch, tmp_path):
    monkeypatch.setattr(executable_resolution.shutil, "which", lambda _name: None)
    schema = write_schema(tmp_path / "schema.json")
    runner = FakeRunner(result=successful_process({"status": "ok"}))

    with pytest.raises(CodexExecutionFailure) as raised:
        execute(
            prompt="prompt",
            repo_path=tmp_path,
            sandbox=Sandbox.READ_ONLY,
            output_schema=schema,
            artifact_directory=tmp_path / "artifacts",
            executable="ticket-automation-missing-codex",
            runner=runner,
        )

    assert runner.calls == 0
    assert raised.value.kind == CodexFailureKind.EXECUTABLE_UNAVAILABLE
    assert "ticket-automation-missing-codex" in str(raised.value)


def test_subprocess_creation_failure_is_typed_and_persisted(tmp_path):
    schema = write_schema(tmp_path / "schema.json")
    runner = FakeRunner(error=OSError("permission denied"))

    with pytest.raises(CodexExecutionFailure) as raised:
        execute(
            prompt="prompt",
            repo_path=tmp_path,
            sandbox=Sandbox.READ_ONLY,
            output_schema=schema,
            artifact_directory=tmp_path / "artifacts",
            executable=EXISTING_EXECUTABLE,
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.PROCESS_START_FAILED
    assert_failure_execution_record(
        raised.value.execution,
        failure_kind=CodexFailureKind.PROCESS_START_FAILED,
        process_started=False,
        exit_code=None,
    )


def test_authentication_or_service_failure_event_is_typed(tmp_path):
    schema = write_schema(tmp_path / "schema.json")
    runner = FakeRunner(
        result=CodexProcessResult(
            returncode=0,
            stdout=json.dumps({"type": "error", "message": "authentication failed"})
            + "\n",
            stderr="",
        )
    )

    with pytest.raises(CodexExecutionFailure) as raised:
        execute(
            prompt="prompt",
            repo_path=tmp_path,
            sandbox=Sandbox.READ_ONLY,
            output_schema=schema,
            artifact_directory=tmp_path / "artifacts",
            executable=EXISTING_EXECUTABLE,
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.AUTHENTICATION_OR_SERVICE
    assert_failure_execution_record(
        raised.value.execution,
        failure_kind=CodexFailureKind.AUTHENTICATION_OR_SERVICE,
        process_started=True,
        exit_code=0,
    )


def test_subprocess_runner_does_not_use_shell(monkeypatch, tmp_path):
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

        class Completed:
            returncode = 0
            stdout = event_stream({"status": "ok"})
            stderr = ""

        return Completed()

    monkeypatch.setattr(codex_module.subprocess, "run", fake_run)

    result = SubprocessCodexRunner().run(
        CodexCommand(argv=("codex", "exec", "-"), cwd=tmp_path),
        stdin="prompt",
        timeout_seconds=10,
    )

    assert result.returncode == 0
    assert captured["args"] == (("codex", "exec", "-"),)
    assert captured["kwargs"]["cwd"] == tmp_path
    assert captured["kwargs"]["input"] == b"prompt"
    assert captured["kwargs"]["shell"] is False


def read_execution_record(execution) -> dict[str, object]:
    assert execution.execution_json_path.is_file()
    return json.loads(execution.execution_json_path.read_text(encoding="utf-8"))


def assert_failure_execution_record(
    execution,
    *,
    failure_kind: CodexFailureKind,
    process_started: bool,
    exit_code: int | None,
    timed_out: bool = False,
    structured_result_present: bool = False,
) -> dict[str, object]:
    assert execution.prompt_path.is_file()
    assert execution.events_jsonl_path.is_file()
    assert execution.stderr_log_path.is_file()
    assert execution.execution_json_path.is_file()
    assert not execution.result_json_path.exists()
    record = read_execution_record(execution)
    assert record["status"] == "FAILED"
    assert record["process_started"] is process_started
    assert record["process_exit_code"] == exit_code
    assert record["timed_out"] is timed_out
    assert record["failure_kind"] == failure_kind.value
    assert isinstance(record["failure_message"], str)
    assert record["structured_result_present"] is structured_result_present
    assert record["result_json_present"] is False
    assert record["artifact_paths"] == {
        "events_jsonl": str(execution.events_jsonl_path),
        "execution_json": str(execution.execution_json_path),
        "prompt_md": str(execution.prompt_path),
        "result_json": str(execution.result_json_path),
        "stderr_log": str(execution.stderr_log_path),
    }
    assert "structured_result" not in record
    return record


def successful_process(result: dict[str, object]) -> CodexProcessResult:
    return CodexProcessResult(returncode=0, stdout=event_stream(result), stderr="")


def event_stream(result: dict[str, object]) -> str:
    return (
        "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "thread"}),
                json.dumps({"type": "turn.started"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "item_1",
                            "type": "agent_message",
                            "text": json.dumps(result),
                        },
                    }
                ),
                json.dumps({"type": "turn.completed"}),
            ]
        )
        + "\n"
    )


def write_schema(path):
    return write_custom_schema(
        path,
        {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "notes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["status"],
            "additionalProperties": False,
        },
    )


def write_contract_schema(path):
    return write_custom_schema(
        path,
        {
            "$defs": {
                "metadataValue": {"type": "string", "minLength": 1},
            },
            "type": "object",
            "properties": {
                "ticket": {"type": "string", "pattern": "^TA-[0-9]{3}$"},
                "attempts": {"type": "integer", "minimum": 1},
                "notes": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 2},
                },
                "mode": {
                    "oneOf": [
                        {"const": "implement"},
                        {"const": "review"},
                    ]
                },
                "score": {
                    "anyOf": [
                        {"type": "integer", "minimum": 10},
                        {"type": "null"},
                    ]
                },
                "summary": {
                    "allOf": [
                        {"type": "string"},
                        {"minLength": 5},
                    ]
                },
            },
            "required": [
                "attempts",
                "mode",
                "notes",
                "score",
                "summary",
                "ticket",
            ],
            "additionalProperties": {"$ref": "#/$defs/metadataValue"},
        },
    )


def write_custom_schema(path, schema):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema), encoding="utf-8")
    return path
