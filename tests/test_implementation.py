from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.helpers import (
    GIT,
    create_git_repo,
    make_config,
    run_git,
)
from tests.helpers import create_trusted_prepared_run as create_run_snapshot
from ticket_automation.codex import (
    CodexCommand,
    CodexFailureKind,
    CodexProcessResult,
    CodexProcessTimedOut,
    CodexProcessTimeout,
    CodexResultValidationError,
    Sandbox,
    validate_json_schema,
)
from ticket_automation.implementation import (
    ImplementationError,
    run_implementation_stage,
)
from ticket_automation.models import (
    StageOutcome,
    StopCategory,
    StopReason,
    WorkflowState,
)
from ticket_automation.runs import load_run_record, save_run_record

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_DIR = "implementation"
DIFFS_DIR = "diffs"
AFTER_IMPLEMENTATION_PATCH = "after-implementation.patch"
AFTER_IMPLEMENTATION_STATS = "after-implementation.stat"


def fixed_clock() -> datetime:
    return datetime(2026, 9, 11, 13, 5, 13, tzinfo=UTC)


@dataclass
class MutatingRunner:
    result: dict[str, object] | None = None
    mutation: Callable[[Path], None] | None = None
    returncode: int = 0
    stdout: str | None = None
    stderr: str = ""
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
    ) -> CodexProcessResult:
        self.calls += 1
        self.command = command
        self.stdin = stdin
        self.timeout_seconds = timeout_seconds
        if self.mutation is not None:
            self.mutation(command.cwd)
        if self.error is not None:
            raise self.error
        if self.stdout is not None:
            return CodexProcessResult(
                returncode=self.returncode,
                stdout=self.stdout,
                stderr=self.stderr,
            )
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
    mark_implementing(snapshot_result.run_dir)
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
def test_implementation_prompt_contains_environment_tooling_policy(tmp_path):
    _repo, run_dir, config = snapshot(tmp_path)
    runner = MutatingRunner(
        result=implementation_result(),
        mutation=lambda cwd: cwd.joinpath("file.txt").write_text(
            "implemented\n",
            encoding="utf-8",
        ),
    )

    run_implementation_stage(config, run_dir, codex_runner=runner)

    assert runner.stdin is not None
    prompt = runner.stdin.lower()
    assert "existing configured development environment" in prompt
    assert "project tooling" in prompt
    assert "do not create a new virtual environment" in prompt
    assert "conda environment" in prompt
    assert "inside the target repository" in prompt
    assert "do not install project dependencies globally" in prompt
    assert "do not modify `.gitignore`" in prompt
    assert "report the limitation" in prompt
    assert "return `blocked`" in prompt


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
    assert result.outcome == StageOutcome.COMPLETED
    assert load_run_record(run_dir / "run.json").state == WorkflowState.IMPLEMENTING
    assert runner.command is not None
    assert runner.command.cwd == repo
    assert sandbox_value(runner.command.argv) == Sandbox.WORKSPACE_WRITE.value
    assert runner.stdin is not None
    assert "# TA-005" in runner.stdin
    implementation_dir = run_dir / IMPLEMENTATION_DIR
    assert implementation_dir.joinpath("prompt.md").is_file()
    assert implementation_dir.joinpath("events.jsonl").is_file()
    assert implementation_dir.joinpath("stderr.log").is_file()
    assert implementation_dir.joinpath("execution.json").is_file()
    execution_record = json.loads(
        implementation_dir.joinpath("execution.json").read_text()
    )
    assert execution_record["status"] == "SUCCESS"
    assert execution_record["structured_result_present"] is True
    assert execution_record["result_json_present"] is True
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
    mark_implementing(snapshot_result.run_dir)

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

    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert runner.calls == 0
    assert result.codex_execution is None
    assert "tracked-diff" in {violation.name for violation in result.safety_violations}
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

    assert result.outcome == StageOutcome.HUMAN_REQUIRED
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
    run_record = load_run_record(run_record_path).transition_to(
        WorkflowState.HUMAN_REQUIRED,
        updated_timestamp="2026-09-11T13:05:14Z",
        terminal_reason="Synthetic human stop.",
        stop_reason=StopReason(
            category=StopCategory.HUMAN_JUDGMENT_REQUIRED,
            message="Synthetic human stop.",
            retryable=False,
        ),
    )
    save_run_record(run_record, run_record_path)

    with pytest.raises(ImplementationError, match="requires run state IMPLEMENTING"):
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

    assert result.outcome == StageOutcome.HUMAN_REQUIRED
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

    assert result.outcome == StageOutcome.HUMAN_REQUIRED
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

    assert result.outcome == StageOutcome.HUMAN_REQUIRED
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

    assert result.outcome == StageOutcome.HUMAN_REQUIRED
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

    assert result.outcome == StageOutcome.COMPLETED
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

    assert result.outcome == StageOutcome.HUMAN_REQUIRED
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

    assert result.outcome == StageOutcome.COMPLETED
    assert result.changed_files == ("added.txt",)
    assert result.patch_path is not None
    assert "diff --git a/added.txt b/added.txt" in result.patch_path.read_text()
    assert "+new file evidence" in result.patch_path.read_text()
    assert result.diff_stats_path is not None
    assert "added.txt" in result.diff_stats_path.read_text()


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_new_environment_after_completed_implementation_is_human_required(tmp_path):
    repo, run_dir, config = snapshot(tmp_path)

    def create_environment_and_source_change(cwd: Path) -> None:
        cwd.joinpath("file.txt").write_text("implemented\n", encoding="utf-8")
        create_pyvenv(cwd / ".venv-correction")

    result = run_implementation_stage(
        config,
        run_dir,
        codex_runner=MutatingRunner(
            result=implementation_result(),
            mutation=create_environment_and_source_change,
        ),
        clock=fixed_clock,
    )

    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert result.workspace_guard is not None
    assert result.workspace_guard.has_violation
    assert result.patch_path is None
    assert "Workspace hygiene violation" in result.controller_message
    assert ".venv-correction/" in result.controller_message
    assert ".venv-correction/pyvenv.cfg" in result.controller_message
    assert "did not exist before" in result.controller_message
    assert "No files were deleted automatically" in result.controller_message
    assert (
        "branch unchanged; HEAD unchanged; staging empty" in result.controller_message
    )
    assert repo.joinpath(".venv-correction", "pyvenv.cfg").is_file()
    assert repo.joinpath("file.txt").read_text(encoding="utf-8") == "implemented\n"
    assert run_git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "feature/example"
    assert run_git(repo, "rev-parse", "HEAD") == result.run_record.baseline_sha
    assert run_git(repo, "diff", "--cached", "--name-only") == ""
    guard_path = run_dir / "workspace-guard" / "implementation.json"
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    assert guard["phase"] == "IMPLEMENTING"
    assert guard["environments_before"] == []
    assert guard["new_environments"][0]["root"] == ".venv-correction"
    assert guard["new_environments"][0]["markers"] == [".venv-correction/pyvenv.cfg"]


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_preexisting_ignored_environment_does_not_block_implementation(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    repo.joinpath(".gitignore").write_text(".venv/\n", encoding="utf-8")
    run_git(repo, "add", ".gitignore")
    run_git(repo, "commit", "-m", "ignore project environment")
    create_pyvenv(repo / ".venv")
    ticket = tmp_path / "TA-005.md"
    ticket.write_text("# TA-005\n\nImplement the ticket.\n", encoding="utf-8")
    config = make_config(repo)
    snapshot_result = create_run_snapshot(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    mark_implementing(snapshot_result.run_dir)

    result = run_implementation_stage(
        config,
        snapshot_result.run_dir,
        codex_runner=MutatingRunner(
            result=implementation_result(),
            mutation=lambda cwd: cwd.joinpath("file.txt").write_text(
                "implemented with existing env\n",
                encoding="utf-8",
            ),
        ),
        clock=fixed_clock,
    )

    assert result.outcome == StageOutcome.COMPLETED
    assert result.workspace_guard is not None
    assert not result.workspace_guard.has_violation
    guard = json.loads(
        snapshot_result.run_dir.joinpath(
            "workspace-guard",
            "implementation.json",
        ).read_text(encoding="utf-8")
    )
    assert guard["environments_before"][0]["root"] == ".venv"
    assert guard["environments_after"][0]["root"] == ".venv"
    assert guard["new_environments"] == []


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_unchanged_started_process_failure_is_safely_failed(tmp_path):
    _repo, run_dir, config = snapshot(tmp_path)
    runner = MutatingRunner(returncode=2, stderr="boom\n")

    result = run_implementation_stage(config, run_dir, codex_runner=runner)

    assert result.outcome == StageOutcome.FAILED
    assert result.codex_execution.failure_message == "Codex exited with code 2."
    assert result.codex_execution.process_started is True
    assert (
        result.codex_execution.stderr_log_path.read_text(encoding="utf-8") == "boom\n"
    )
    assert result.controller_message == "Codex exited with code 2."
    assert result.patch_path == run_dir / DIFFS_DIR / "failed-implementation.patch"
    assert result.patch_path.is_file()
    execution_record = json.loads(
        result.codex_execution.execution_json_path.read_text(encoding="utf-8")
    )
    assert execution_record["status"] == "FAILED"
    assert execution_record["failure_kind"] == "NON_ZERO_EXIT"
    assert execution_record["process_exit_code"] == 2
    assert execution_record["result_json_present"] is False
    assert not result.codex_execution.result_json_path.exists()


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_failed_writable_implementation_with_partial_changes_is_human_required(
    tmp_path,
):
    repo, run_dir, config = snapshot(tmp_path)
    runner = MutatingRunner(
        mutation=lambda cwd: cwd.joinpath("file.txt").write_text(
            "partial implementation\n",
            encoding="utf-8",
        ),
        returncode=2,
        stderr="boom\n",
    )

    result = run_implementation_stage(config, run_dir, codex_runner=runner)

    assert runner.calls == 1
    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert result.changed_files == ("file.txt",)
    assert "worktree" in {violation.name for violation in result.safety_violations}
    assert run_git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "feature/example"
    assert run_git(repo, "rev-parse", "HEAD") == result.run_record.baseline_sha
    assert run_git(repo, "diff", "--cached", "--name-only") == ""
    assert run_git(repo, "diff", "--name-only") == "file.txt"
    assert repo.joinpath("file.txt").read_text(encoding="utf-8") == (
        "partial implementation\n"
    )
    assert result.patch_path == run_dir / DIFFS_DIR / "failed-implementation.patch"
    assert "+partial implementation" in result.patch_path.read_text(encoding="utf-8")
    assert result.diff_stats_path == run_dir / DIFFS_DIR / "failed-implementation.stat"
    assert result.codex_execution.execution_json_path.is_file()
    assert result.codex_execution.events_jsonl_path.is_file()
    assert result.codex_execution.stderr_log_path.is_file()
    assert "Execution metadata:" in result.controller_message
    assert "Baseline-relative failure patch:" in result.controller_message
    assert not run_dir.joinpath(DIFFS_DIR, AFTER_IMPLEMENTATION_PATCH).exists()


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_started_untrusted_completion_without_changes_is_human_required(tmp_path):
    _repo, run_dir, config = snapshot(tmp_path)
    runner = MutatingRunner(stdout='{"type":"turn.started"}\n', stderr="lost\n")

    result = run_implementation_stage(config, run_dir, codex_runner=runner)

    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert result.changed_files == ()
    assert result.safety_violations == ()
    assert result.patch_path == run_dir / DIFFS_DIR / "failed-implementation.patch"
    assert result.patch_path.read_text(encoding="utf-8") == ""
    assert result.codex_execution.process_started is True
    assert (
        result.codex_execution.failure_kind
        == CodexFailureKind.MISSING_STRUCTURED_RESULT
    )
    assert "Process started: yes" in result.controller_message


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_writable_timeout_after_process_start_requires_human_even_if_unchanged(
    tmp_path,
):
    _repo, run_dir, config = snapshot(tmp_path)

    result = run_implementation_stage(
        config,
        run_dir,
        codex_runner=MutatingRunner(
            error=CodexProcessTimedOut(
                CodexProcessTimeout(
                    stdout="",
                    stderr="timed out",
                    timeout_seconds=1,
                )
            )
        ),
    )

    attempt = json.loads(
        run_dir.joinpath("writable-attempts", "implementation.json").read_text(
            encoding="utf-8"
        )
    )
    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert result.codex_execution.process_started is True
    assert attempt["process_started"] is True
    assert attempt["before_fingerprint"] == attempt["after_fingerprint"]


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_proven_process_start_failure_without_changes_remains_failed(tmp_path):
    _repo, run_dir, config = snapshot(tmp_path)
    runner = MutatingRunner(error=FileNotFoundError("missing codex"))

    result = run_implementation_stage(config, run_dir, codex_runner=runner)

    assert result.outcome == StageOutcome.FAILED
    assert result.changed_files == ()
    assert result.patch_path is None
    assert result.diff_stats_path is None
    assert result.codex_execution.process_started is False
    assert (
        result.codex_execution.failure_kind == CodexFailureKind.EXECUTABLE_UNAVAILABLE
    )
    assert not run_dir.joinpath(DIFFS_DIR, "failed-implementation.patch").exists()


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_failed_implementation_with_observed_changes_overrides_nonstart_metadata(
    tmp_path,
):
    repo, run_dir, config = snapshot(tmp_path)
    runner = MutatingRunner(
        mutation=lambda cwd: cwd.joinpath("file.txt").write_text(
            "nonstart partial implementation\n",
            encoding="utf-8",
        ),
        error=FileNotFoundError("missing codex"),
    )

    result = run_implementation_stage(config, run_dir, codex_runner=runner)

    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert result.changed_files == ("file.txt",)
    assert result.codex_execution.process_started is False
    assert "worktree" in {violation.name for violation in result.safety_violations}
    assert result.patch_path == run_dir / DIFFS_DIR / "failed-implementation.patch"
    assert "+nonstart partial implementation" in result.patch_path.read_text(
        encoding="utf-8"
    )
    assert run_git(repo, "rev-parse", "HEAD") == result.run_record.baseline_sha
    assert run_git(repo, "diff", "--cached", "--name-only") == ""
    assert "Process started: no" in result.controller_message


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_failed_writable_implementation_records_environment_guard_finding(tmp_path):
    repo, run_dir, config = snapshot(tmp_path)

    def create_environment_and_partial_change(cwd: Path) -> None:
        cwd.joinpath("file.txt").write_text(
            "partial implementation\n",
            encoding="utf-8",
        )
        create_pyvenv(cwd / ".venv-correction")

    result = run_implementation_stage(
        config,
        run_dir,
        codex_runner=MutatingRunner(
            mutation=create_environment_and_partial_change,
            returncode=2,
            stderr="boom\n",
        ),
        clock=fixed_clock,
    )

    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert not hasattr(result.run_record, "last_completed_state")
    assert "Codex exited with code 2" in result.controller_message
    assert "Workspace hygiene violation" in result.controller_message
    assert ".venv-correction/pyvenv.cfg" in result.controller_message
    assert result.workspace_guard is not None
    assert result.workspace_guard.has_violation
    assert repo.joinpath(".venv-correction", "pyvenv.cfg").is_file()
    assert repo.joinpath("file.txt").read_text(encoding="utf-8") == (
        "partial implementation\n"
    )
    assert result.patch_path == run_dir / DIFFS_DIR / "failed-implementation.patch"
    guard = json.loads(
        run_dir.joinpath("workspace-guard", "implementation.json").read_text(
            encoding="utf-8"
        )
    )
    assert guard["new_environments"][0]["root"] == ".venv-correction"


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for implementation tests"
)
def test_failed_implementation_patch_capture_error_stays_human_required(tmp_path):
    _repo, run_dir, config = snapshot(tmp_path)
    run_dir.joinpath(DIFFS_DIR).write_text("not a directory\n", encoding="utf-8")

    result = run_implementation_stage(
        config,
        run_dir,
        codex_runner=MutatingRunner(
            mutation=lambda cwd: cwd.joinpath("file.txt").write_text(
                "partial implementation\n",
                encoding="utf-8",
            ),
            returncode=2,
        ),
    )

    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert result.patch_path is None
    assert "Failure patch capture error:" in result.controller_message
    assert "may have left partial source changes" in result.controller_message


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
    mark_implementing(result.run_dir)
    return repo, result.run_dir, config


def mark_implementing(run_dir: Path) -> None:
    record_path = run_dir / "run.json"
    record = load_run_record(record_path).transition_to(
        WorkflowState.IMPLEMENTING,
        updated_timestamp="2026-09-11T13:05:13Z",
    )
    save_run_record(record, record_path)


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


def create_pyvenv(path: Path) -> None:
    path.mkdir(parents=True)
    path.joinpath("pyvenv.cfg").write_text("home = python\n", encoding="utf-8")
