from __future__ import annotations

import json
import sys
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
from ticket_automation import corrections as corrections_module
from ticket_automation.codex import (
    CodexCommand,
    CodexFailureKind,
    CodexProcessResult,
    Sandbox,
)
from ticket_automation.config import AppConfig, VerificationCommand
from ticket_automation.corrections import (
    CORRECTION_EXECUTIONS_DIR_NAME,
    CORRECTIONS_DIR_NAME,
    CorrectionError,
    ReviewFinding,
    render_correction_ticket,
    run_correction_stage,
)
from ticket_automation.git import GitCommandError
from ticket_automation.models import StageOutcome, WorkflowState
from ticket_automation.reporting import generate_terminal_report_best_effort
from ticket_automation.review import ReviewVerdict, run_review_stage
from ticket_automation.runs import load_run_record, save_run_record
from ticket_automation.verification import (
    VerificationProcessCommand,
    VerificationProcessResult,
    run_verification_stage,
)

ROUND_1 = "round-1"


def fixed_clock() -> datetime:
    return datetime(2026, 9, 11, 13, 5, 16, tzinfo=UTC)


@dataclass
class CodexRunner:
    result: dict[str, object]
    mutation: Callable[[Path], None] | None = None
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
        return CodexProcessResult(
            returncode=0,
            stdout=event_stream(self.result),
            stderr="correction progress\n",
        )


@dataclass
class ProcessFailureRunner:
    mutation: Callable[[Path], None] | None = None
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
        return CodexProcessResult(
            returncode=2,
            stdout="",
            stderr="synthetic codex failure\n",
        )


class PassingVerificationRunner:
    def run(
        self,
        command: VerificationProcessCommand,
        *,
        timeout_seconds: float | None,
    ) -> VerificationProcessResult:
        del command, timeout_seconds
        return VerificationProcessResult(
            returncode=0,
            stdout="deterministic passed\n",
            stderr="",
        )


def test_required_review_finding_renders_markdown_correction():
    markdown = render_correction_ticket(
        ticket_id="QDEB-003",
        round_number=1,
        reasons=(
            ReviewFinding(
                finding_id="R1-F1",
                summary="Validation incorrectly includes untested sites",
                details="The validation table includes rows that were never tested.",
                severity="MEDIUM",
                category="CORRECTNESS",
                evidence="tests/test_validation.py::test_sites fails.",
                required_change="Filter validation rows to tested sites only.",
                acceptance_criteria=("Untested sites are excluded.",),
            ),
        ),
    )

    assert "# QDEB-003 - Corrective Round 1" in markdown
    assert "independent review of QDEB-003" in markdown
    assert "### R1-F1 - Validation incorrectly includes untested sites" in markdown
    assert "Severity: Medium" in markdown
    assert "Category: Correctness" in markdown
    assert "Filter validation rows to tested sites only." in markdown
    assert "- Untested sites are excluded." in markdown


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_verification_failure_renders_markdown_correction(tmp_path):
    _repo, run_dir, config = verification_correct_run(tmp_path)
    runner = completed_runner()

    result = run_correction_stage(config, run_dir, codex_runner=runner)

    markdown = result.ticket_path.read_text(encoding="utf-8")
    assert result.ticket_path == run_dir / CORRECTIONS_DIR_NAME / "QDEB-003-CORR-R1.md"
    assert "deterministic verification failures for QDEB-003" in markdown
    assert "independent review" not in markdown
    assert "### Verification failure - tests" in markdown
    assert "Command:" in markdown
    assert "Exit code:\n7" in markdown
    assert "failing stdout" in markdown
    assert "failing stderr" in markdown
    assert "Full log:\nverification\\round-0.log" in markdown or (
        "Full log:\nverification/round-0.log" in markdown
    )


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_advisory_and_follow_up_findings_are_excluded(tmp_path):
    _repo, run_dir, config = review_correct_run(
        tmp_path,
        findings=[
            finding("R1-F1", disposition="REQUIRED", title="Required fix"),
            finding("R1-F2", disposition="ADVISORY", title="Useful note"),
            finding("R1-F3", disposition="FOLLOW_UP", title="Later cleanup"),
        ],
    )

    result = run_correction_stage(config, run_dir, codex_runner=completed_runner())

    markdown = result.ticket_path.read_text(encoding="utf-8")
    assert "Required fix" in markdown
    assert "Useful note" not in markdown
    assert "Later cleanup" not in markdown


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_multiple_required_findings_are_combined(tmp_path):
    _repo, run_dir, config = review_correct_run(
        tmp_path,
        findings=[
            finding("R1-F1", disposition="REQUIRED", title="First required fix"),
            finding("R1-F2", disposition="REQUIRED", title="Second required fix"),
        ],
    )

    result = run_correction_stage(config, run_dir, codex_runner=completed_runner())

    markdown = result.ticket_path.read_text(encoding="utf-8")
    assert result.ticket_path == run_dir / CORRECTIONS_DIR_NAME / "QDEB-003-CORR-R1.md"
    assert "### R1-F1 - First required fix" in markdown
    assert "### R1-F2 - Second required fix" in markdown
    assert len(tuple((run_dir / CORRECTIONS_DIR_NAME).glob("*.md"))) == 1


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_correction_prompt_retains_original_ticket_and_preserve_instruction(tmp_path):
    ticket_text = (
        "# QDEB-003\r\n\r\n"
        "Implement the original behavior.\r\n\r\n"
        "```text\r\n"
        "$HOME remains literal\r\n"
        "```\r\n"
    )
    _repo, run_dir, config = review_correct_run(tmp_path, ticket_text=ticket_text)
    runner = completed_runner()

    run_correction_stage(config, run_dir, codex_runner=runner)

    assert runner.stdin is not None
    assert "BEGIN ORIGINAL TICKET" in runner.stdin
    assert ticket_text in runner.stdin
    assert "BEGIN CORRECTIVE TICKET" in runner.stdin
    assert (
        "Existing uncommitted changes are the implementation of the original ticket. "
        "Preserve valid existing work and make only the changes necessary to "
        "satisfy this corrective round."
    ) in runner.stdin


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_correction_prompt_contains_environment_tooling_policy(tmp_path):
    _repo, run_dir, config = review_correct_run(tmp_path)
    runner = completed_runner()

    run_correction_stage(config, run_dir, codex_runner=runner)

    assert runner.stdin is not None
    prompt = runner.stdin.lower()
    assert (
        "existing uncommitted source changes are the implementation being corrected"
        in prompt
    )
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
    GIT is None, reason="git executable is required for correction tests"
)
def test_correction_runs_as_fresh_workspace_write_invocation(tmp_path):
    _repo, run_dir, config = review_correct_run(tmp_path)
    runner = completed_runner()

    result = run_correction_stage(config, run_dir, codex_runner=runner)

    assert runner.calls == 1
    assert runner.command is not None
    assert sandbox_value(runner.command.argv) == Sandbox.WORKSPACE_WRITE.value
    assert "resume" not in runner.command.argv
    correction_dir = run_dir / CORRECTION_EXECUTIONS_DIR_NAME / ROUND_1
    assert result.artifact_directory == correction_dir
    assert correction_dir.joinpath("prompt.md").is_file()
    assert correction_dir.joinpath("events.jsonl").is_file()
    assert correction_dir.joinpath("stderr.log").is_file()
    assert correction_dir.joinpath("execution.json").is_file()
    assert correction_dir.joinpath("result.json").is_file()
    execution_record = json.loads(correction_dir.joinpath("execution.json").read_text())
    assert execution_record["status"] == "SUCCESS"
    assert execution_record["sandbox"] == Sandbox.WORKSPACE_WRITE.value
    assert execution_record["structured_result_present"] is True
    assert execution_record["result_json_present"] is True
    assert correction_dir.joinpath("stderr.log").read_text() == "correction progress\n"


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_completed_correction_is_accepted_and_returns_to_verify(tmp_path):
    _repo, run_dir, config = review_correct_run(tmp_path)

    result = run_correction_stage(config, run_dir, codex_runner=completed_runner())

    assert result.successful
    assert result.outcome == StageOutcome.COMPLETED
    assert result.advance_correction_round
    assert result.correction_round == 1
    assert result.run_record.current_correction_round == 0
    assert load_run_record(run_dir / "run.json").state == WorkflowState.CORRECTING
    assert load_run_record(run_dir / "run.json").current_correction_round == 0
    assert "verification must run next" in result.controller_message
    assert result.patch_path == run_dir / "diffs" / "after-correction-1.patch"


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_blocked_correction_becomes_human_required_and_preserves_explanation(tmp_path):
    _repo, run_dir, config = review_correct_run(tmp_path)
    runner = CodexRunner(result=correction_result(status="BLOCKED"))

    result = run_correction_stage(config, run_dir, codex_runner=runner)

    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert result.agent_result["status"] == "BLOCKED"
    assert result.agent_result["known_issues"] == ["blocked by synthetic ambiguity"]
    assert "BLOCKED" in result.controller_message
    saved = json.loads(
        run_dir.joinpath(
            CORRECTION_EXECUTIONS_DIR_NAME, ROUND_1, "result.json"
        ).read_text()
    )
    assert saved["status"] == "BLOCKED"


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_branch_mutation_is_human_required(tmp_path):
    repo, run_dir, config = review_correct_run(tmp_path)

    result = run_correction_stage(
        config,
        run_dir,
        codex_runner=CodexRunner(
            result=correction_result(),
            mutation=lambda cwd: run_git(cwd, "checkout", "-b", "correction-branch"),
        ),
    )

    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert [violation.name for violation in result.safety_violations] == ["branch"]
    assert run_git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "correction-branch"


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_head_mutation_is_human_required(tmp_path):
    repo, run_dir, config = review_correct_run(tmp_path)

    def commit_change(cwd: Path) -> None:
        cwd.joinpath("file.txt").write_text("committed correction\n", encoding="utf-8")
        run_git(cwd, "add", "file.txt")
        run_git(cwd, "commit", "-m", "correction changed head")

    result = run_correction_stage(
        config,
        run_dir,
        codex_runner=CodexRunner(result=correction_result(), mutation=commit_change),
    )

    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert [violation.name for violation in result.safety_violations] == ["HEAD"]
    assert run_git(repo, "rev-parse", "HEAD") != result.run_record.baseline_sha


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_staged_files_are_human_required(tmp_path):
    _repo, run_dir, config = review_correct_run(tmp_path)

    def stage_change(cwd: Path) -> None:
        cwd.joinpath("file.txt").write_text("staged correction\n", encoding="utf-8")
        run_git(cwd, "add", "file.txt")

    result = run_correction_stage(
        config,
        run_dir,
        codex_runner=CodexRunner(result=correction_result(), mutation=stage_change),
    )

    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert [violation.name for violation in result.safety_violations] == ["staging"]


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_starting_staged_files_do_not_invoke_or_consume_round(tmp_path):
    _repo, run_dir, config = review_correct_run(tmp_path)
    run_git(Path(config.project.repo), "add", "file.txt")
    runner = completed_runner()

    result = run_correction_stage(config, run_dir, codex_runner=runner)

    assert runner.calls == 0
    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert result.run_record.current_correction_round == 0
    assert load_run_record(run_dir / "run.json").current_correction_round == 0
    assert result.ticket_path is None
    assert not (run_dir / CORRECTIONS_DIR_NAME).exists()
    assert not (run_dir / CORRECTION_EXECUTIONS_DIR_NAME / ROUND_1).exists()
    assert {violation.name for violation in result.safety_violations} == {
        "staging",
        "workspace-fingerprint",
    }


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_correction_rejects_untrusted_verification_checkpoint_metadata(tmp_path):
    _repo, run_dir, config = verification_correct_run(tmp_path)
    verification_path = run_dir / "verification" / "round-0.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verification["checkpoint"]["baseline_sha"] = "0" * 40
    verification_path.write_text(json.dumps(verification), encoding="utf-8")
    runner = completed_runner()

    result = run_correction_stage(config, run_dir, codex_runner=runner)

    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert runner.calls == 0
    assert [violation.name for violation in result.safety_violations] == [
        "workspace-checkpoint"
    ]


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_after_correction_patch_is_complete_baseline_relative_diff(tmp_path):
    _repo, run_dir, config = review_correct_run(tmp_path, add_extra_work=True)

    result = run_correction_stage(config, run_dir, codex_runner=completed_runner())

    patch = result.patch_path.read_text(encoding="utf-8")
    assert "diff --git a/file.txt b/file.txt" in patch
    assert "+corrected by fake correction" in patch
    assert "diff --git a/kept.txt b/kept.txt" in patch
    assert "+original implementation work" in patch


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_new_environment_after_completed_correction_is_human_required(tmp_path):
    repo, run_dir, config = review_correct_run(tmp_path)

    def create_environment_and_correction(cwd: Path) -> None:
        cwd.joinpath("file.txt").write_text("corrected\n", encoding="utf-8")
        create_pyvenv(cwd / ".venv-correction")

    result = run_correction_stage(
        config,
        run_dir,
        codex_runner=CodexRunner(
            result=correction_result(),
            mutation=create_environment_and_correction,
        ),
        clock=fixed_clock,
    )

    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert result.advance_correction_round
    assert result.correction_round == 1
    assert result.workspace_guard is not None
    assert result.workspace_guard.has_violation
    assert "Workspace hygiene violation" in result.controller_message
    assert ".venv-correction/" in result.controller_message
    assert ".venv-correction/pyvenv.cfg" in result.controller_message
    assert "did not exist before" in result.controller_message
    assert "No files were deleted automatically" in result.controller_message
    assert (
        "branch unchanged; HEAD unchanged; staging empty" in result.controller_message
    )
    assert repo.joinpath(".venv-correction", "pyvenv.cfg").is_file()
    assert repo.joinpath("file.txt").read_text(encoding="utf-8") == "corrected\n"
    assert run_git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "feature/example"
    assert run_git(repo, "rev-parse", "HEAD") == result.run_record.baseline_sha
    assert run_git(repo, "diff", "--cached", "--name-only") == ""
    guard_path = run_dir / "workspace-guard" / "correction-round-1.json"
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    assert guard["phase"] == "CORRECTING"
    assert guard["new_environments"][0]["root"] == ".venv-correction"
    assert guard["new_environments"][0]["markers"] == [".venv-correction/pyvenv.cfg"]


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_oversized_logs_are_not_blindly_embedded(tmp_path):
    huge_stdout = "A" * 10_000
    _repo, run_dir, config = verification_correct_run(tmp_path, stdout=huge_stdout)

    result = run_correction_stage(config, run_dir, codex_runner=completed_runner())

    markdown = result.ticket_path.read_text(encoding="utf-8")
    assert len(markdown) < 4_000
    assert "truncated" in markdown
    assert "see full log" in markdown
    assert huge_stdout not in markdown


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_unchanged_codex_boundary_failure_is_safely_failed_with_canonical_artifact(
    tmp_path,
):
    _repo, run_dir, config = review_correct_run(tmp_path)
    runner = ProcessFailureRunner()

    result = run_correction_stage(config, run_dir, codex_runner=runner)

    artifact_dir = run_dir / CORRECTION_EXECUTIONS_DIR_NAME / ROUND_1
    execution_record = json.loads(
        artifact_dir.joinpath("execution.json").read_text(encoding="utf-8")
    )
    assert runner.calls == 1
    assert result.outcome == StageOutcome.FAILED
    assert artifact_dir.joinpath("prompt.md").is_file()
    assert artifact_dir.joinpath("events.jsonl").is_file()
    assert artifact_dir.joinpath("stderr.log").is_file()
    assert not artifact_dir.joinpath("result.json").exists()
    assert result.patch_path == run_dir / "diffs" / "failed-correction-1.patch"
    assert result.patch_path.is_file()
    assert result.safety_violations == ()
    assert result.controller_message == "Codex exited with code 2."
    assert execution_record["status"] == "FAILED"
    assert execution_record["failure_kind"] == CodexFailureKind.NON_ZERO_EXIT.value
    assert execution_record["process_exit_code"] == 2
    assert execution_record["structured_result_present"] is False
    assert execution_record["result_json_present"] is False


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_failed_writable_correction_with_partial_changes_is_human_required(tmp_path):
    repo, run_dir, config = review_correct_run(tmp_path)
    runner = ProcessFailureRunner(
        mutation=lambda cwd: cwd.joinpath("file.txt").write_text(
            "partial correction\n",
            encoding="utf-8",
        ),
    )

    result = run_correction_stage(config, run_dir, codex_runner=runner)

    assert runner.calls == 1
    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert result.run_record.current_correction_round == 0
    assert run_git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "feature/example"
    assert run_git(repo, "rev-parse", "HEAD") == result.run_record.baseline_sha
    assert run_git(repo, "diff", "--cached", "--name-only") == ""
    assert run_git(repo, "diff", "--name-only") == "file.txt"
    assert repo.joinpath("file.txt").read_text(encoding="utf-8") == (
        "partial correction\n"
    )
    assert result.patch_path == run_dir / "diffs" / "failed-correction-1.patch"
    assert "+partial correction" in result.patch_path.read_text(encoding="utf-8")
    assert result.codex_execution.execution_json_path.is_file()
    assert result.codex_execution.events_jsonl_path.is_file()
    assert result.codex_execution.stderr_log_path.is_file()
    assert "worktree" in {violation.name for violation in result.safety_violations}
    assert "Baseline-relative failure patch:" in result.controller_message
    assert not run_dir.joinpath("diffs", "after-correction-1.patch").exists()


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_failed_correction_with_observed_changes_overrides_nonstart_metadata(tmp_path):
    repo, run_dir, config = review_correct_run(tmp_path)
    runner = ProcessFailureRunner(
        mutation=lambda cwd: cwd.joinpath("file.txt").write_text(
            "nonstart partial correction\n",
            encoding="utf-8",
        ),
        error=FileNotFoundError("missing codex"),
    )

    result = run_correction_stage(config, run_dir, codex_runner=runner)

    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert result.run_record.current_correction_round == 0
    assert result.codex_execution.process_started is False
    assert "worktree" in {violation.name for violation in result.safety_violations}
    assert result.patch_path == run_dir / "diffs" / "failed-correction-1.patch"
    assert "+nonstart partial correction" in result.patch_path.read_text(
        encoding="utf-8"
    )
    assert run_git(repo, "rev-parse", "HEAD") == result.run_record.baseline_sha
    assert run_git(repo, "diff", "--cached", "--name-only") == ""
    assert "Process started: no" in result.controller_message


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_failed_writable_correction_records_environment_guard_finding(tmp_path):
    repo, run_dir, config = review_correct_run(tmp_path)

    def create_environment_and_partial_correction(cwd: Path) -> None:
        cwd.joinpath("file.txt").write_text("partial correction\n", encoding="utf-8")
        create_pyvenv(cwd / ".venv-correction")

    result = run_correction_stage(
        config,
        run_dir,
        codex_runner=ProcessFailureRunner(
            mutation=create_environment_and_partial_correction,
        ),
        clock=fixed_clock,
    )

    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert result.run_record.current_correction_round == 0
    assert "Codex exited with code 2" in result.controller_message
    assert "Workspace hygiene violation" in result.controller_message
    assert ".venv-correction/pyvenv.cfg" in result.controller_message
    assert result.workspace_guard is not None
    assert result.workspace_guard.has_violation
    assert repo.joinpath(".venv-correction", "pyvenv.cfg").is_file()
    assert repo.joinpath("file.txt").read_text(encoding="utf-8") == (
        "partial correction\n"
    )
    assert result.patch_path == run_dir / "diffs" / "failed-correction-1.patch"
    guard = json.loads(
        run_dir.joinpath("workspace-guard", "correction-round-1.json").read_text(
            encoding="utf-8"
        )
    )
    assert guard["new_environments"][0]["root"] == ".venv-correction"


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_failed_correction_patch_capture_error_stays_human_required(tmp_path):
    _repo, run_dir, config = review_correct_run(tmp_path)
    run_dir.joinpath("diffs").write_text("not a directory\n", encoding="utf-8")

    result = run_correction_stage(
        config,
        run_dir,
        codex_runner=ProcessFailureRunner(
            mutation=lambda cwd: cwd.joinpath("file.txt").write_text(
                "partial correction\n",
                encoding="utf-8",
            ),
        ),
    )

    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert result.run_record.current_correction_round == 0
    assert result.patch_path is None
    assert "Failure patch capture error:" in result.controller_message
    assert "may have left partial source changes" in result.controller_message


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_correction_diff_capture_failure_after_mutation_is_human_required(
    monkeypatch,
    tmp_path,
):
    repo, run_dir, config = review_correct_run(tmp_path)

    def fail_diff(repository, baseline_sha):
        del repository, baseline_sha
        raise GitCommandError("synthetic diff decode failure")

    monkeypatch.setattr(corrections_module, "_diff_including_untracked", fail_diff)
    runner = completed_runner()

    result = run_correction_stage(config, run_dir, codex_runner=runner)

    assert runner.calls == 1
    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert result.patch_path is None
    assert "Could not capture correction diff" in result.controller_message
    assert "synthetic diff decode failure" in result.controller_message
    assert run_git(repo, "diff", "--name-only") == "file.txt"
    assert repo.joinpath("file.txt").read_text(encoding="utf-8") == (
        "corrected by fake correction\n"
    )
    assert not run_dir.joinpath("diffs", "after-correction-1.patch").exists()
    assert result.codex_execution is not None
    assert result.codex_execution.execution_json_path.is_file()


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_untracked_unicode_file_is_handled_by_correction_and_reporting(tmp_path):
    _repo, run_dir, config = review_correct_run(tmp_path)
    unicode_text = "phospho α/β č - en dash – em dash — Søren 😀\n"

    result = run_correction_stage(
        config,
        run_dir,
        codex_runner=CodexRunner(
            result=correction_result(),
            mutation=lambda cwd: cwd.joinpath("unicode.txt").write_text(
                unicode_text,
                encoding="utf-8",
            ),
        ),
    )

    assert result.outcome == StageOutcome.COMPLETED
    assert result.patch_path is not None
    correction_patch = result.patch_path.read_text(encoding="utf-8")
    assert "diff --git a/unicode.txt b/unicode.txt" in correction_patch
    assert f"+{unicode_text.rstrip()}" in correction_patch

    report_path = generate_terminal_report_best_effort(run_dir)

    assert report_path is not None
    final_patch = run_dir.joinpath("diffs", "final.patch").read_text(encoding="utf-8")
    assert f"+{unicode_text.rstrip()}" in final_patch
    report = report_path.read_text(encoding="utf-8")
    assert "unicode.txt" in report


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_malformed_required_review_finding_is_rejected_before_invocation(tmp_path):
    _repo, run_dir, config = review_correct_run(tmp_path)
    result_path = run_dir / "reviews" / ROUND_1 / "result.json"
    result_data = json.loads(result_path.read_text(encoding="utf-8"))
    result_data["findings"] = [
        {
            "id": "R1-F1",
            "severity": "MEDIUM",
            "category": "CORRECTNESS",
            "disposition": "REQUIRED",
            "scope_relation": "TICKET",
            "title": "Missing evidence",
            "description": "This finding is malformed.",
            "required_change": "Reject malformed required findings.",
            "acceptance_criteria": ["No correction is invoked."],
        }
    ]
    result_path.write_text(json.dumps(result_data), encoding="utf-8")
    runner = completed_runner()

    with pytest.raises(CorrectionError, match="field 'evidence'"):
        run_correction_stage(config, run_dir, codex_runner=runner)

    assert runner.calls == 0
    assert not (run_dir / CORRECTION_EXECUTIONS_DIR_NAME / ROUND_1).exists()


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_malformed_verification_reason_is_rejected_before_invocation(tmp_path):
    _repo, run_dir, config = verification_correct_run(tmp_path)
    result_path = run_dir / "verification" / "round-0.json"
    result_data = json.loads(result_path.read_text(encoding="utf-8"))
    result_data["correction_reasons"] = [
        {
            "kind": "VerificationFailure",
            "gate_name": "tests",
            "failure_summary": "Missing command.",
            "stdout_excerpt": "",
            "stderr_excerpt": "",
            "exit_code": 1,
            "log_path": str(run_dir / "verification" / "round-0.log"),
        }
    ]
    result_path.write_text(json.dumps(result_data), encoding="utf-8")
    runner = completed_runner()

    with pytest.raises(CorrectionError, match="field 'command'"):
        run_correction_stage(config, run_dir, codex_runner=runner)

    assert runner.calls == 0
    assert not (run_dir / CORRECTION_EXECUTIONS_DIR_NAME / ROUND_1).exists()


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_no_real_outstanding_ta_ticket_is_required_for_correction_behavior(tmp_path):
    ticket_text = "# Synthetic Correction\n\nNo real TA review finding exists.\n"
    _repo, run_dir, config = review_correct_run(tmp_path, ticket_text=ticket_text)

    result = run_correction_stage(config, run_dir, codex_runner=completed_runner())

    assert result.outcome == StageOutcome.COMPLETED
    assert "Synthetic Correction" in result.codex_execution.prompt_path.read_text(
        encoding="utf-8"
    )


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for correction tests"
)
def test_verification_can_run_after_completed_correction(tmp_path):
    _repo, run_dir, config = review_correct_run(
        tmp_path,
        verification_commands=(passing_gate("tests"),),
    )
    correction = run_correction_stage(config, run_dir, codex_runner=completed_runner())
    assert correction.outcome == StageOutcome.COMPLETED
    record_path = run_dir / "run.json"
    save_run_record(
        load_run_record(record_path).transition_to(
            WorkflowState.VERIFYING,
            updated_timestamp="2026-09-11T13:05:17Z",
            current_correction_round=correction.correction_round,
        ),
        record_path,
    )

    result = run_verification_stage(config, run_dir)

    assert result.outcome == StageOutcome.COMPLETED
    assert result.round_result.round_index == 1
    assert result.round_result.json_path == run_dir / "verification" / "round-1.json"


def completed_runner() -> CodexRunner:
    return CodexRunner(
        result=correction_result(),
        mutation=lambda cwd: cwd.joinpath("file.txt").write_text(
            "corrected by fake correction\n",
            encoding="utf-8",
        ),
    )


def correction_result(status: str = "COMPLETED") -> dict[str, object]:
    return {
        "status": status,
        "summary": "corrected" if status == "COMPLETED" else "blocked",
        "tests_run": [{"command": "targeted synthetic test", "result": "PASS"}],
        "assumptions": [],
        "known_issues": []
        if status == "COMPLETED"
        else ["blocked by synthetic ambiguity"],
    }


def verification_correct_run(
    tmp_path,
    *,
    stdout: str = "failing stdout\n",
    stderr: str = "failing stderr\n",
) -> tuple[Path, Path, AppConfig]:
    return prepared_verification_run(
        tmp_path,
        verification_commands=(failing_gate("tests", stdout=stdout, stderr=stderr),),
    )


def prepared_verification_run(
    tmp_path,
    *,
    verification_commands: tuple[VerificationCommand, ...],
) -> tuple[Path, Path, AppConfig]:
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "QDEB-003.md"
    ticket.write_text("# QDEB-003\n\nImplement the ticket.\n", encoding="utf-8")
    config = make_config(repo, verification_commands=verification_commands)
    snapshot = create_run_snapshot(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        verification_runner=PassingVerificationRunner(),
        clock=fixed_clock,
    )
    repo.joinpath("file.txt").write_text("implemented\n", encoding="utf-8")
    mark_run_state(snapshot.run_dir, WorkflowState.VERIFYING)
    verification_result = run_verification_stage(config, snapshot.run_dir)
    assert verification_result.outcome == StageOutcome.CORRECTION_REQUIRED
    mark_run_state(
        snapshot.run_dir,
        WorkflowState.CORRECTING,
    )
    return repo, snapshot.run_dir, config


def review_correct_run(
    tmp_path,
    *,
    ticket_text: str = "# QDEB-003\n\nImplement the ticket.\n",
    findings: list[dict[str, object]] | None = None,
    add_extra_work: bool = False,
    verification_commands: tuple[VerificationCommand, ...] | None = None,
) -> tuple[Path, Path, AppConfig]:
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "QDEB-003.md"
    ticket.write_bytes(ticket_text.encode("utf-8"))
    config = make_config(
        repo,
        verification_commands=verification_commands or (passing_gate("tests"),),
    )
    snapshot = create_run_snapshot(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    repo.joinpath("file.txt").write_text("implemented\n", encoding="utf-8")
    if add_extra_work:
        repo.joinpath("kept.txt").write_text(
            "original implementation work\n",
            encoding="utf-8",
        )
    write_implementation_result(snapshot.run_dir)
    write_passing_verification(snapshot.run_dir, config)
    mark_run_state(snapshot.run_dir, WorkflowState.REVIEWING)
    review_stage_result = run_review_stage(
        config,
        snapshot.run_dir,
        codex_runner=CodexRunner(
            result=review_result(
                findings=findings
                or [finding("R1-F1", disposition="REQUIRED", title="Synthetic fix")]
            )
        ),
    )
    assert review_stage_result.outcome == StageOutcome.CORRECTION_REQUIRED
    mark_run_state(
        snapshot.run_dir,
        WorkflowState.CORRECTING,
        current_review_round=1,
    )
    return repo, snapshot.run_dir, config


def write_implementation_result(run_dir: Path) -> None:
    implementation_dir = run_dir / "implementation"
    implementation_dir.mkdir()
    implementation_dir.joinpath("result.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "summary": "implemented synthetic ticket",
                "tests_run": [{"command": "fake", "result": "PASS"}],
                "assumptions": [],
                "known_issues": [],
            }
        ),
        encoding="utf-8",
    )


def write_passing_verification(run_dir: Path, config: AppConfig) -> None:
    record_path = run_dir / "run.json"
    save_run_record(
        load_run_record(record_path)
        .transition_to(
            WorkflowState.IMPLEMENTING,
            updated_timestamp="2026-09-11T13:05:15Z",
        )
        .transition_to(
            WorkflowState.VERIFYING,
            updated_timestamp="2026-09-11T13:05:16Z",
        ),
        record_path,
    )
    result = run_verification_stage(
        config,
        run_dir,
        process_runner=PassingVerificationRunner(),
        clock=fixed_clock,
    )
    assert result.outcome == StageOutcome.COMPLETED


def mark_run_state(
    run_dir: Path,
    state: WorkflowState,
    *,
    current_correction_round: int | None = None,
    current_review_round: int | None = None,
) -> None:
    run_record_path = run_dir / "run.json"
    run_record = load_run_record(run_record_path)
    if state == WorkflowState.REVIEWING and run_record.state == WorkflowState.VERIFYING:
        transition_path = (WorkflowState.REVIEWING,)
    elif state == WorkflowState.CORRECTING and run_record.state in {
        WorkflowState.VERIFYING,
        WorkflowState.REVIEWING,
    }:
        transition_path = (
            WorkflowState.CORRECTION_PENDING,
            WorkflowState.CORRECTING,
        )
    else:
        transition_path = {
            WorkflowState.VERIFYING: (
                WorkflowState.IMPLEMENTING,
                WorkflowState.VERIFYING,
            ),
            WorkflowState.REVIEWING: (
                WorkflowState.IMPLEMENTING,
                WorkflowState.VERIFYING,
                WorkflowState.REVIEWING,
            ),
            WorkflowState.CORRECTING: (
                WorkflowState.IMPLEMENTING,
                WorkflowState.VERIFYING,
                WorkflowState.CORRECTION_PENDING,
                WorkflowState.CORRECTING,
            ),
        }[state]
    for next_state in transition_path:
        run_record = run_record.transition_to(
            next_state,
            updated_timestamp="2026-09-11T13:05:16Z",
            current_correction_round=current_correction_round,
            current_review_round=current_review_round,
        )
    save_run_record(run_record, run_record_path)


def review_result(
    *,
    findings: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "verdict": ReviewVerdict.CORRECTIONS_REQUIRED.value,
        "summary": "review requires correction",
        "confidence": "HIGH",
        "findings": findings,
    }


def finding(
    finding_id: str,
    *,
    disposition: str,
    title: str,
) -> dict[str, object]:
    return {
        "id": finding_id,
        "severity": "MEDIUM",
        "category": "CORRECTNESS",
        "disposition": disposition,
        "scope_relation": "TICKET",
        "title": title,
        "description": f"{title} description.",
        "evidence": f"{title} evidence.",
        "required_change": f"{title} required change.",
        "acceptance_criteria": [f"{title} acceptance criterion."],
    }


def passing_gate(name: str) -> VerificationCommand:
    return VerificationCommand(
        name=name,
        argv=(sys.executable, "-c", "raise SystemExit(0)"),
        timeout_seconds=1800,
    )


def failing_gate(
    name: str,
    *,
    stdout: str = "failing stdout\n",
    stderr: str = "failing stderr\n",
    exit_code: int = 7,
) -> VerificationCommand:
    code = (
        "import sys; "
        f"sys.stdout.write({json.dumps(stdout)}); "
        f"sys.stderr.write({json.dumps(stderr)}); "
        f"raise SystemExit({exit_code})"
    )
    return VerificationCommand(
        name=name,
        argv=(sys.executable, "-c", code),
        timeout_seconds=1800,
    )


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


def create_pyvenv(path: Path) -> None:
    path.mkdir(parents=True)
    path.joinpath("pyvenv.cfg").write_text("home = python\n", encoding="utf-8")
