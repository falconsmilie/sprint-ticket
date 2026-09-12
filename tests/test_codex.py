from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from ticket_automation import codex as codex_module
from ticket_automation.codex import (
    CodexCommand,
    CodexExecutionFailure,
    CodexExecutionStatus,
    CodexExecutor,
    CodexFailureKind,
    CodexProcessResult,
    CodexProcessTimedOut,
    CodexProcessTimeout,
    Sandbox,
    SubprocessCodexRunner,
    build_codex_command,
    execute,
    parse_sandbox,
)


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
        executable="codex",
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

    result = execute(
        prompt="generated prompt",
        repo_path=tmp_path,
        sandbox=Sandbox.READ_ONLY,
        output_schema=schema,
        artifact_directory=tmp_path / "artifacts",
        executable="codex",
        runner=runner,
    )

    assert result.prompt_path.read_text(encoding="utf-8") == "generated prompt"
    assert result.events_jsonl_path.read_text(encoding="utf-8") == events
    assert result.stderr_log_path.read_text(encoding="utf-8") == "progress\n"
    assert json.loads(result.result_json_path.read_text(encoding="utf-8")) == {
        "notes": ["done"],
        "status": "ok",
    }
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
        executable="codex",
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
        executable="codex",
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
            executable="codex",
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.NON_ZERO_EXIT
    assert raised.value.execution.status == CodexExecutionStatus.FAILED
    assert raised.value.execution.process_exit_code == 2
    assert raised.value.execution.events_jsonl_path.read_text(encoding="utf-8")
    assert (
        raised.value.execution.stderr_log_path.read_text(encoding="utf-8") == "boom\n"
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
            executable="codex",
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
            executable="codex",
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.MISSING_STRUCTURED_RESULT
    assert not raised.value.execution.result_json_path.exists()


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
        CodexExecutor(executable="codex", timeout_seconds=3, runner=runner).execute(
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
            executable="codex",
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.MALFORMED_EVENT_STREAM
    assert raised.value.execution.events_jsonl_path.read_text(
        encoding="utf-8"
    ).endswith("nope\n")
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
            executable="codex",
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.MISSING_STRUCTURED_RESULT


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
            executable="codex",
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.INVALID_STRUCTURED_RESULT
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
        executable="codex",
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
            executable="codex",
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
            executable="codex",
            runner=runner,
        )

    assert runner.calls == 0
    assert raised.value.kind == CodexFailureKind.INVALID_SCHEMA


def test_paths_containing_spaces_retain_argument_boundaries(tmp_path):
    repo = tmp_path / "target repo"
    repo.mkdir()
    schema = write_schema(tmp_path / "schema dir" / "result schema.json")
    artifact_dir = tmp_path / "run artifacts"
    runner = FakeRunner(result=successful_process({"status": "ok"}))

    result = execute(
        prompt="prompt",
        repo_path=repo,
        sandbox=Sandbox.WORKSPACE_WRITE,
        output_schema=schema,
        artifact_directory=artifact_dir,
        executable="codex cli",
        runner=runner,
    )

    assert runner.command is not None
    assert runner.command.argv[0] == "codex cli"
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
            executable="missing-codex",
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.EXECUTABLE_UNAVAILABLE
    assert raised.value.execution.prompt_path.read_text(encoding="utf-8") == "prompt"
    assert raised.value.execution.stderr_log_path.read_text(encoding="utf-8")


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
            executable="codex",
            runner=runner,
        )

    assert raised.value.kind == CodexFailureKind.AUTHENTICATION_OR_SERVICE


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
    assert captured["kwargs"]["input"] == "prompt"
    assert captured["kwargs"]["shell"] is False


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
