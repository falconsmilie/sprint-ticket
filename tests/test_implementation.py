from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pytest

from tests.helpers import GIT, create_git_repo, make_config, run_git
from ticket_automation.codex import (
    CodexCommand,
    CodexProcessResult,
    CodexResultValidationError,
    Sandbox,
    validate_json_schema,
)
from ticket_automation.implementation import (
    ImplementationError,
    run_implementation_stage,
)
from ticket_automation.models import WorkflowState
from ticket_automation.runs import create_run_snapshot, load_run_record, save_run_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_DIR = "implementation"
DIFFS_DIR = "diffs"
AFTER_IMPLEMENTATION_PATCH = "after-implementation.patch"
AFTER_IMPLEMENTATION_STATS = "after-implementation.stat"


def fixed_clock() -> datetime:
    return datetime(2026, 9, 11, 13, 5, 13, tzinfo=timezone.utc)


@dataclass
class MutatingRunner:
    result: dict[str, object] | None = None
    mutation: Callable[[Path], None] | None = None
    returncode: int = 0
    stderr: str = ""
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
        if self.mutation is not None:
            self.mutation(command.cwd)
        if self.returncode != 0:
            return CodexProcessResult(
                returncode=self.returncode,
                stdout="",
                stderr=self.stderr,
            )
        assert self.result is not None
        return CodexProcessResult(
            returncode=0,
            stdout=event_stream(self.result),
            stderr=self.stderr,
        )


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_prompt_contains_full_original_ticket_verbatim(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "TA-005.md"
    ticket_text = (
        "# TA-005\r\n\r\n"
        "Keep every requirement.\r\n\r\n"
        "```text\r\n"
        "$HOME stays literal\r\n"
        "```\r\n"
    )
    ticket.write_bytes(ticket_text.encode("utf-8"))
    config = make_config(repo)
    snapshot_result = create_run_snapshot(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    runner = MutatingRunner(
        result=implementation_result(),
        mutation=lambda cwd: cwd.joinpath("file.txt").write_text(
            "implemented\n",
            encoding="utf-8",
        ),
    )

    run_implementation_stage(config, snapshot_result.run_dir, codex_runner=runner)

    assert runner.stdin is not None
    prompt = runner.stdin
    assert ticket_text in prompt
    assert "Read and obey repository `AGENTS.md` files where present." in prompt
    assert "Do not create or switch branches." in prompt
    assert "Do not stage files." in prompt
    assert "Do not commit." in prompt
    assert "Do not reset or revert existing work." in prompt


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_completed_implementation_is_accepted_and_artifacts_are_stored(tmp_path):
    repo, run_dir, config = snapshot(tmp_path)
    runner = MutatingRunner(
        result=implementation_result(),
        mutation=lambda cwd: cwd.joinpath("file.txt").write_text(
            "implemented\n",
            encoding="utf-8",
        ),
    )

    result = run_implementation_stage(
        config,
        run_dir,
        codex_runner=runner,
        clock=fixed_clock,
    )

    assert result.successful
    assert result.run_record.state == WorkflowState.IMPLEMENT
    assert load_run_record(run_dir / "run.json").state == WorkflowState.IMPLEMENT
    assert runner.command is not None
    assert runner.command.cwd == repo
    assert sandbox_value(runner.command.argv) == Sandbox.WORKSPACE_WRITE.value
    assert runner.stdin is not None
    assert "# TA-005" in runner.stdin
    implementation_dir = run_dir / IMPLEMENTATION_DIR
    assert implementation_dir.joinpath("prompt.md").is_file()
    assert implementation_dir.joinpath("events.jsonl").is_file()
    assert implementation_dir.joinpath("stderr.log").is_file()
    assert (
        json.loads(implementation_dir.joinpath("result.json").read_text())["status"]
        == "COMPLETED"
    )
    assert result.changed_files == ("file.txt",)


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_snapshotted_ticket_bytes_are_preserved_in_prompt_artifact(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "TA-005.md"
    ticket_bytes = b"# TA-005\r\n\r\nPreserve CRLF ticket bytes.\r\n"
    ticket.write_bytes(ticket_bytes)
    config = make_config(repo)
    snapshot_result = create_run_snapshot(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )

    run_implementation_stage(
        config,
        snapshot_result.run_dir,
        codex_runner=MutatingRunner(
            result=implementation_result(),
            mutation=lambda cwd: cwd.joinpath("file.txt").write_text(
                "implemented\n",
                encoding="utf-8",
            ),
        ),
    )

    prompt_bytes = snapshot_result.run_dir.joinpath(
        IMPLEMENTATION_DIR,
        "prompt.md",
    ).read_bytes()
    assert ticket_bytes in prompt_bytes


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_dirty_worktree_before_implementation_blocks_without_invoking_codex(tmp_path):
    repo, run_dir, config = snapshot(tmp_path)
    repo.joinpath("file.txt").write_text("preexisting drift\n", encoding="utf-8")
    runner = MutatingRunner(result=implementation_result())

    result = run_implementation_stage(config, run_dir, codex_runner=runner)

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert runner.calls == 0
    assert result.codex_execution is None
    assert "worktree" in {violation.name for violation in result.safety_violations}
    assert not run_dir.joinpath(IMPLEMENTATION_DIR, "prompt.md").exists()
    assert run_git(repo, "diff", "--name-only") == "file.txt"


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_changed_branch_before_implementation_blocks_without_invoking_codex(tmp_path):
    repo, run_dir, config = snapshot(tmp_path)
    run_git(repo, "checkout", "-b", "preexisting-branch")
    runner = MutatingRunner(result=implementation_result())

    result = run_implementation_stage(config, run_dir, codex_runner=runner)

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert runner.calls == 0
    assert result.codex_execution is None
    assert "branch" in {violation.name for violation in result.safety_violations}
    assert run_git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "preexisting-branch"


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_wrong_run_state_is_rejected_before_invocation(tmp_path):
    _repo, run_dir, config = snapshot(tmp_path)
    run_record_path = run_dir / "run.json"
    run_record = load_run_record(run_record_path).with_state(
        WorkflowState.PREFLIGHT,
        updated_timestamp="2026-09-11T13:05:14Z",
    )
    save_run_record(run_record, run_record_path)

    with pytest.raises(ImplementationError, match="requires run state SNAPSHOT"):
        run_implementation_stage(
            config,
            run_dir,
            codex_runner=MutatingRunner(result=implementation_result()),
        )


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_baseline_mismatch_is_rejected_before_invocation(tmp_path):
    _repo, run_dir, config = snapshot(tmp_path)
    baseline_path = run_dir / "baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["head_sha"] = "abc123"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    with pytest.raises(ImplementationError, match="baseline HEAD"):
        run_implementation_stage(
            config,
            run_dir,
            codex_runner=MutatingRunner(result=implementation_result()),
        )


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_blocked_result_becomes_human_required_and_preserves_explanation(tmp_path):
    _repo, run_dir, config = snapshot(tmp_path)
    runner = MutatingRunner(result=implementation_result(status="BLOCKED"))

    result = run_implementation_stage(config, run_dir, codex_runner=runner)

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    saved_result = json.loads(
        run_dir.joinpath(IMPLEMENTATION_DIR, "result.json").read_text()
    )
    assert saved_result["status"] == "BLOCKED"
    assert saved_result["known_issues"] == ["design judgment required"]
    assert "BLOCKED" in result.controller_message


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_changed_head_is_human_required(tmp_path):
    _repo, run_dir, config = snapshot(tmp_path)

    def commit_change(cwd: Path) -> None:
        cwd.joinpath("file.txt").write_text("committed\n", encoding="utf-8")
        run_git(cwd, "add", "file.txt")
        run_git(cwd, "commit", "-m", "agent changed head")

    result = run_implementation_stage(
        config,
        run_dir,
        codex_runner=MutatingRunner(
            result=implementation_result(), mutation=commit_change
        ),
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert "HEAD" in {violation.name for violation in result.safety_violations}
    assert run_git(
        Path(result.run_record.target_repository_path), "rev-parse", "HEAD"
    ) != (result.run_record.baseline_sha)


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_changed_branch_is_human_required(tmp_path):
    _repo, run_dir, config = snapshot(tmp_path)

    result = run_implementation_stage(
        config,
        run_dir,
        codex_runner=MutatingRunner(
            result=implementation_result(),
            mutation=lambda cwd: run_git(cwd, "checkout", "-b", "agent-branch"),
        ),
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert "branch" in {violation.name for violation in result.safety_violations}
    assert run_git(
        Path(result.run_record.target_repository_path),
        "rev-parse",
        "--abbrev-ref",
        "HEAD",
    ) == ("agent-branch")


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_staged_files_are_human_required(tmp_path):
    _repo, run_dir, config = snapshot(tmp_path)

    def stage_change(cwd: Path) -> None:
        cwd.joinpath("file.txt").write_text("staged\n", encoding="utf-8")
        run_git(cwd, "add", "file.txt")

    result = run_implementation_stage(
        config,
        run_dir,
        codex_runner=MutatingRunner(
            result=implementation_result(), mutation=stage_change
        ),
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert "staging" in {violation.name for violation in result.safety_violations}


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_unstaged_source_changes_are_accepted(tmp_path):
    _repo, run_dir, config = snapshot(tmp_path)

    result = run_implementation_stage(
        config,
        run_dir,
        codex_runner=MutatingRunner(
            result=implementation_result(),
            mutation=lambda cwd: cwd.joinpath("file.txt").write_text(
                "ordinary worktree change\n",
                encoding="utf-8",
            ),
        ),
    )

    assert result.run_record.state == WorkflowState.IMPLEMENT
    assert result.safety_violations == ()


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_no_change_completed_implementation_is_human_required(tmp_path):
    _repo, run_dir, config = snapshot(tmp_path)

    result = run_implementation_stage(
        config,
        run_dir,
        codex_runner=MutatingRunner(result=implementation_result()),
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert result.patch_path is None
    assert "without repository changes" in result.controller_message


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_patch_and_diff_statistics_are_captured_from_git(tmp_path):
    _repo, run_dir, config = snapshot(tmp_path)

    result = run_implementation_stage(
        config,
        run_dir,
        codex_runner=MutatingRunner(
            result=implementation_result(),
            mutation=lambda cwd: cwd.joinpath("file.txt").write_text(
                "patch evidence\n",
                encoding="utf-8",
            ),
        ),
    )

    assert result.patch_path == run_dir / DIFFS_DIR / AFTER_IMPLEMENTATION_PATCH
    assert result.diff_stats_path == run_dir / DIFFS_DIR / AFTER_IMPLEMENTATION_STATS
    assert "diff --git a/file.txt b/file.txt" in result.patch_path.read_text()
    assert "+patch evidence" in result.patch_path.read_text()
    assert "file.txt" in result.diff_stats_path.read_text()


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_untracked_files_are_accepted_and_included_in_patch(tmp_path):
    _repo, run_dir, config = snapshot(tmp_path)

    result = run_implementation_stage(
        config,
        run_dir,
        codex_runner=MutatingRunner(
            result=implementation_result(),
            mutation=lambda cwd: cwd.joinpath("added.txt").write_text(
                "new file evidence\n",
                encoding="utf-8",
            ),
        ),
    )

    assert result.run_record.state == WorkflowState.IMPLEMENT
    assert result.changed_files == ("added.txt",)
    assert result.patch_path is not None
    assert "diff --git a/added.txt b/added.txt" in result.patch_path.read_text()
    assert "+new file evidence" in result.patch_path.read_text()
    assert result.diff_stats_path is not None
    assert "added.txt" in result.diff_stats_path.read_text()


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_process_failure_becomes_failed_state(tmp_path):
    _repo, run_dir, config = snapshot(tmp_path)
    runner = MutatingRunner(returncode=2, stderr="boom\n")

    result = run_implementation_stage(config, run_dir, codex_runner=runner)

    assert result.run_record.state == WorkflowState.FAILED
    assert result.codex_execution.failure_message == "Codex exited with code 2."
    assert (
        result.codex_execution.stderr_log_path.read_text(encoding="utf-8") == "boom\n"
    )
    assert not result.codex_execution.result_json_path.exists()


def test_implementation_result_schema_accepts_trusted_statuses():
    schema = implementation_schema()

    validate_json_schema(implementation_result("COMPLETED"), schema)
    validate_json_schema(implementation_result("BLOCKED"), schema)


@pytest.mark.parametrize(
    "patch",
    [
        {"status": "DONE"},
        {"changed_files": ["file.txt"]},
        {"tests_run": [{"command": "pytest"}]},
        {"summary": ""},
    ],
)
def test_implementation_result_schema_rejects_unsupported_results(patch):
    schema = implementation_schema()
    result = implementation_result()
    result.update(patch)

    with pytest.raises(CodexResultValidationError):
        validate_json_schema(result, schema)


@pytest.mark.parametrize("missing_field", ["status", "summary", "tests_run"])
def test_implementation_result_schema_rejects_missing_required_fields(missing_field):
    schema = implementation_schema()
    result = implementation_result()
    del result[missing_field]

    with pytest.raises(CodexResultValidationError):
        validate_json_schema(result, schema)


def snapshot(tmp_path, ticket_text: str = "# TA-005\n\nImplement the ticket.\n"):
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "TA-005.md"
    ticket.write_text(ticket_text, encoding="utf-8")
    config = make_config(repo)
    result = create_run_snapshot(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    return repo, result.run_dir, config


def implementation_result(status: str = "COMPLETED") -> dict[str, object]:
    return {
        "status": status,
        "summary": "implemented" if status == "COMPLETED" else "blocked",
        "tests_run": [{"command": "pytest tests/test_example.py", "result": "PASS"}],
        "assumptions": [],
        "known_issues": [] if status == "COMPLETED" else ["design judgment required"],
    }


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


def sandbox_value(argv: tuple[str, ...]) -> str:
    sandbox_index = argv.index("--sandbox")
    return argv[sandbox_index + 1]


def implementation_schema() -> dict[str, object]:
    schema_path = PROJECT_ROOT / "schemas" / "implementation-result.schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))
