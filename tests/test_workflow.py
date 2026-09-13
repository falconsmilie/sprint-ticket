from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
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
from ticket_automation.codex import CodexCommand, CodexProcessResult, Sandbox
from ticket_automation.config import AppConfig, VerificationCommand
from ticket_automation.git import GitCommandError
from ticket_automation.implementation import (
    ImplementationStageResult,
)
from ticket_automation.implementation import (
    run_implementation_stage as execute_implementation_stage,
)
from ticket_automation.models import StageOutcome, WorkflowState
from ticket_automation.review import (
    ReviewStageResult,
    ReviewVerdict,
)
from ticket_automation.review import (
    run_review_stage as execute_review_stage,
)
from ticket_automation.runs import load_run_record, save_run_record
from ticket_automation.verification import (
    VerificationProcessCommand,
    VerificationProcessResult,
    VerificationStageResult,
)
from ticket_automation.verification import (
    run_verification_stage as execute_verification_stage,
)
from ticket_automation.workflow import (
    format_lifecycle_result,
    resume_ticket_lifecycle,
    run_ticket_lifecycle,
)


def fixed_clock() -> datetime:
    return datetime(2026, 9, 11, 13, 5, 17, tzinfo=UTC)


def run_implementation_stage(
    config: AppConfig,
    run_dir: Path,
    *,
    codex_runner=None,
    clock=None,
) -> ImplementationStageResult:
    record_path = run_dir / "run.json"
    record = load_run_record(record_path).transition_to(
        WorkflowState.IMPLEMENTING,
        updated_timestamp="2026-09-11T13:05:17Z",
    )
    save_run_record(record, record_path)
    result = execute_implementation_stage(
        config,
        run_dir,
        codex_runner=codex_runner,
        clock=clock,
    )
    return persist_stage_result(result)


def run_verification_stage(
    config: AppConfig,
    run_dir: Path,
    *,
    process_runner=None,
    clock=None,
) -> VerificationStageResult:
    result = execute_verification_stage(
        config,
        run_dir,
        process_runner=process_runner,
        clock=clock,
    )
    return persist_stage_result(result)


def run_review_stage(
    config: AppConfig,
    run_dir: Path,
    *,
    codex_runner=None,
    clock=None,
) -> ReviewStageResult:
    result = execute_review_stage(
        config,
        run_dir,
        codex_runner=codex_runner,
        clock=clock,
    )
    return persist_stage_result(result)


def persist_stage_result(result):
    run_record = result.run_record
    if result.outcome == StageOutcome.HUMAN_REQUIRED:
        state = WorkflowState.HUMAN_REQUIRED
    elif result.outcome == StageOutcome.FAILED:
        state = WorkflowState.FAILED
    elif result.outcome == StageOutcome.CORRECTION_REQUIRED:
        state = WorkflowState.CORRECTION_PENDING
    else:
        state = {
            WorkflowState.IMPLEMENTING: WorkflowState.VERIFYING,
            WorkflowState.VERIFYING: WorkflowState.REVIEWING,
            WorkflowState.REVIEWING: WorkflowState.REPORTING,
        }[run_record.state]
    updated_record = run_record.transition_to(
        state,
        updated_timestamp="2026-09-11T13:05:17Z",
        current_review_round=(
            run_record.current_review_round + 1
            if run_record.state == WorkflowState.REVIEWING
            else None
        ),
        terminal_reason=(
            result.controller_message
            if state in {WorkflowState.HUMAN_REQUIRED, WorkflowState.FAILED}
            else None
        ),
    )
    save_run_record(updated_record, Path(result.run_dir) / "run.json")
    return replace(result, run_record=updated_record)


@dataclass(frozen=True)
class CodexStep:
    result: dict[str, object] | None = None
    mutation: Callable[[Path], None] | None = None
    returncode: int = 0
    stderr: str = "fake codex progress\n"
    error: BaseException | None = None


@dataclass
class SequencedCodexRunner:
    steps: list[CodexStep]
    calls: list[tuple[CodexCommand, str]]

    def run(
        self,
        command: CodexCommand,
        *,
        stdin: str,
        timeout_seconds: float | None,
    ) -> CodexProcessResult:
        del timeout_seconds
        self.calls.append((command, stdin))
        assert self.steps, "Unexpected Codex invocation."
        step = self.steps.pop(0)
        if step.error is not None:
            raise step.error
        if step.mutation is not None:
            step.mutation(command.cwd)
        if step.returncode != 0:
            return CodexProcessResult(
                returncode=step.returncode,
                stdout="",
                stderr=step.stderr,
            )
        assert step.result is not None
        return CodexProcessResult(
            returncode=0,
            stdout=event_stream(step.result),
            stderr=step.stderr,
        )


@dataclass(frozen=True)
class VerificationStep:
    returncode: int = 0
    stdout: str = "verification passed\n"
    stderr: str = ""
    error: BaseException | None = None


@dataclass
class SequencedVerificationRunner:
    steps: list[VerificationStep]
    calls: list[VerificationProcessCommand]

    def run(
        self,
        command: VerificationProcessCommand,
        *,
        timeout_seconds: float | None,
    ) -> VerificationProcessResult:
        del timeout_seconds
        self.calls.append(command)
        assert self.steps, "Unexpected verification invocation."
        step = self.steps.pop(0)
        if step.error is not None:
            raise step.error
        return VerificationProcessResult(
            returncode=step.returncode,
            stdout=step.stdout,
            stderr=step.stderr,
        )


@dataclass
class PassingBaselineVerificationRunner:
    subsequent: SequencedVerificationRunner
    baseline_pending: bool = True

    def run(
        self,
        command: VerificationProcessCommand,
        *,
        timeout_seconds: float | None,
    ) -> VerificationProcessResult:
        if self.baseline_pending:
            self.baseline_pending = False
            return VerificationProcessResult(
                returncode=0,
                stdout="baseline verification passed\n",
                stderr="",
            )
        return self.subsequent.run(command, timeout_seconds=timeout_seconds)


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_lifecycle_first_pass_success_reaches_ready_for_human(tmp_path):
    repo, ticket, config = workflow_inputs(tmp_path)
    codex = SequencedCodexRunner(
        steps=[
            CodexStep(
                result=implementation_result(), mutation=write_file("implemented\n")
            ),
            CodexStep(result=review_result()),
        ],
        calls=[],
    )
    verification = SequencedVerificationRunner(steps=[VerificationStep()], calls=[])

    result = run_ticket_lifecycle(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        codex_runner=codex,
        verification_runner=PassingBaselineVerificationRunner(verification),
        clock=fixed_clock,
    )

    assert result.successful
    assert result.run_record.state == WorkflowState.READY_FOR_HUMAN
    assert not hasattr(result.run_record, "last_completed_state")
    assert result.run_record.current_correction_round == 0
    assert result.run_record.current_review_round == 1
    assert [sandbox_value(command.argv) for command, _stdin in codex.calls] == [
        Sandbox.WORKSPACE_WRITE.value,
        Sandbox.READ_ONLY.value,
    ]
    assert len(result.verification_results) == 1
    assert len(result.review_results) == 1
    assert result.correction_results == ()
    assert result.run_dir.joinpath("diffs", "final.patch").is_file()
    assert result.run_dir.joinpath("final-report.md").is_file()
    assert run_git(repo, "diff", "--name-only") == "file.txt"
    assert "QDEB-003 - READY FOR HUMAN REVIEW" in format_lifecycle_result(result)
    assert "No files have been staged or committed." in format_lifecycle_result(result)


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_lifecycle_review_correction_is_verified_and_rereviewed(tmp_path):
    _repo, ticket, config = workflow_inputs(tmp_path)
    codex = SequencedCodexRunner(
        steps=[
            CodexStep(
                result=implementation_result(), mutation=write_file("implemented\n")
            ),
            CodexStep(
                result=review_result(verdict=ReviewVerdict.CORRECTIONS_REQUIRED.value)
            ),
            CodexStep(
                result=implementation_result(), mutation=write_file("corrected\n")
            ),
            CodexStep(result=review_result()),
        ],
        calls=[],
    )
    verification = SequencedVerificationRunner(
        steps=[VerificationStep(), VerificationStep()],
        calls=[],
    )

    result = run_ticket_lifecycle(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        codex_runner=codex,
        verification_runner=PassingBaselineVerificationRunner(verification),
        clock=fixed_clock,
    )

    assert result.successful
    assert result.run_record.current_correction_round == 1
    assert result.run_record.current_review_round == 2
    assert [item.round_result.round_index for item in result.verification_results] == [
        0,
        1,
    ]
    assert [item.review_result["verdict"] for item in result.review_results] == [
        ReviewVerdict.CORRECTIONS_REQUIRED.value,
        ReviewVerdict.PASS.value,
    ]
    assert [sandbox_value(command.argv) for command, _stdin in codex.calls] == [
        Sandbox.WORKSPACE_WRITE.value,
        Sandbox.READ_ONLY.value,
        Sandbox.WORKSPACE_WRITE.value,
        Sandbox.READ_ONLY.value,
    ]
    rereview_prompt = result.review_results[-1].codex_execution.prompt_path.read_text(
        encoding="utf-8"
    )
    assert "baseline SHA -> complete current working tree" in rereview_prompt
    assert result.run_dir.joinpath("reviews", "round-2", "result.json").is_file()


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_lifecycle_verification_failure_drives_correction_without_review(tmp_path):
    _repo, ticket, config = workflow_inputs(tmp_path)
    codex = SequencedCodexRunner(
        steps=[
            CodexStep(
                result=implementation_result(), mutation=write_file("implemented\n")
            ),
            CodexStep(
                result=implementation_result(), mutation=write_file("corrected\n")
            ),
            CodexStep(result=review_result()),
        ],
        calls=[],
    )
    verification = SequencedVerificationRunner(
        steps=[
            VerificationStep(
                returncode=7,
                stdout="runner failed\n",
                stderr="failure detail\n",
            ),
            VerificationStep(),
        ],
        calls=[],
    )

    result = run_ticket_lifecycle(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        codex_runner=codex,
        verification_runner=PassingBaselineVerificationRunner(verification),
        clock=fixed_clock,
    )

    assert result.successful
    assert result.run_record.current_correction_round == 1
    assert result.run_record.current_review_round == 1
    assert result.verification_results[0].round_result.status.value == "FAIL"
    assert (
        result.verification_results[0]
        .round_result.correction_reasons[0]
        .to_dict()["kind"]
        == "VerificationFailure"
    )
    assert len(result.review_results) == 1
    assert [sandbox_value(command.argv) for command, _stdin in codex.calls] == [
        Sandbox.WORKSPACE_WRITE.value,
        Sandbox.WORKSPACE_WRITE.value,
        Sandbox.READ_ONLY.value,
    ]
    assert result.run_dir.joinpath("reviews", "round-1", "result.json").is_file()


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_lifecycle_human_review_required_stops_at_human_required(tmp_path):
    _repo, ticket, config = workflow_inputs(tmp_path)
    codex = SequencedCodexRunner(
        steps=[
            CodexStep(
                result=implementation_result(), mutation=write_file("implemented\n")
            ),
            CodexStep(
                result=review_result(
                    verdict=ReviewVerdict.HUMAN_REVIEW_REQUIRED.value,
                    findings=[],
                )
            ),
        ],
        calls=[],
    )
    verification = SequencedVerificationRunner(steps=[VerificationStep()], calls=[])

    result = run_ticket_lifecycle(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        codex_runner=codex,
        verification_runner=PassingBaselineVerificationRunner(verification),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert result.run_record.current_correction_round == 0
    assert result.run_record.current_review_round == 1
    assert result.correction_results == ()
    assert codex.steps == []


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_lifecycle_exhausts_shared_correction_limit_before_next_correction(tmp_path):
    _repo, ticket, config = workflow_inputs(tmp_path, max_correction_rounds=3)
    codex = SequencedCodexRunner(
        steps=[
            CodexStep(
                result=implementation_result(), mutation=write_file("implemented\n")
            ),
            CodexStep(
                result=review_result(verdict=ReviewVerdict.CORRECTIONS_REQUIRED.value)
            ),
            CodexStep(
                result=implementation_result(), mutation=write_file("corrected 1\n")
            ),
            CodexStep(
                result=review_result(verdict=ReviewVerdict.CORRECTIONS_REQUIRED.value)
            ),
            CodexStep(
                result=implementation_result(), mutation=write_file("corrected 2\n")
            ),
            CodexStep(
                result=review_result(verdict=ReviewVerdict.CORRECTIONS_REQUIRED.value)
            ),
            CodexStep(
                result=implementation_result(), mutation=write_file("corrected 3\n")
            ),
            CodexStep(
                result=review_result(verdict=ReviewVerdict.CORRECTIONS_REQUIRED.value)
            ),
        ],
        calls=[],
    )
    verification = SequencedVerificationRunner(
        steps=[
            VerificationStep(),
            VerificationStep(),
            VerificationStep(),
            VerificationStep(),
        ],
        calls=[],
    )

    result = run_ticket_lifecycle(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        codex_runner=codex,
        verification_runner=PassingBaselineVerificationRunner(verification),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert result.run_record.current_correction_round == 3
    assert result.run_record.current_review_round == 4
    assert len(result.correction_results) == 3
    assert len(result.review_results) == 4
    assert codex.steps == []
    assert not result.run_dir.joinpath("correction-executions", "round-4").exists()
    assert "Maximum corrective rounds exhausted" in result.run_record.terminal_reason


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
@pytest.mark.parametrize(
    ("mutation", "violation"),
    [
        (lambda cwd: run_git(cwd, "checkout", "-b", "agent-branch"), "branch"),
        (lambda cwd: commit_file(cwd), "HEAD"),
        (lambda cwd: stage_file(cwd), "staging"),
    ],
)
def test_lifecycle_git_safety_violation_stops_without_repair(
    tmp_path,
    mutation,
    violation,
):
    repo, ticket, config = workflow_inputs(tmp_path)
    codex = SequencedCodexRunner(
        steps=[CodexStep(result=implementation_result(), mutation=mutation)],
        calls=[],
    )
    verification = SequencedVerificationRunner(steps=[], calls=[])

    result = run_ticket_lifecycle(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        codex_runner=codex,
        verification_runner=PassingBaselineVerificationRunner(verification),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert violation in {
        item.name for item in result.implementation_result.safety_violations
    }
    assert f"  {violation}:" in format_lifecycle_result(result)
    assert verification.calls == []
    if violation == "branch":
        assert run_git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "agent-branch"
    if violation == "HEAD":
        assert run_git(repo, "rev-parse", "HEAD") != result.run_record.baseline_sha
    if violation == "staging":
        assert run_git(repo, "diff", "--cached", "--name-only") == "file.txt"


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_lifecycle_implementation_environment_pollution_stops_before_verification(
    tmp_path,
):
    repo, ticket, config = workflow_inputs(tmp_path)

    def create_environment_and_source_change(cwd: Path) -> None:
        cwd.joinpath("file.txt").write_text("implemented\n", encoding="utf-8")
        create_pyvenv(cwd / ".venv-correction")

    codex = SequencedCodexRunner(
        steps=[
            CodexStep(
                result=implementation_result(),
                mutation=create_environment_and_source_change,
            )
        ],
        calls=[],
    )
    verification = SequencedVerificationRunner(steps=[VerificationStep()], calls=[])

    result = run_ticket_lifecycle(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        codex_runner=codex,
        verification_runner=PassingBaselineVerificationRunner(verification),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert result.implementation_result is not None
    assert result.implementation_result.workspace_guard is not None
    assert result.implementation_result.workspace_guard.has_violation
    assert verification.calls == []
    assert result.review_results == ()
    assert result.correction_results == ()
    assert len(codex.calls) == 1
    assert repo.joinpath(".venv-correction", "pyvenv.cfg").is_file()
    assert repo.joinpath("file.txt").read_text(encoding="utf-8") == "implemented\n"
    assert run_git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "feature/example"
    assert run_git(repo, "rev-parse", "HEAD") == result.run_record.baseline_sha
    assert run_git(repo, "diff", "--cached", "--name-only") == ""
    guard_path = result.run_dir / "workspace-guard" / "implementation.json"
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    assert guard["new_environments"][0]["root"] == ".venv-correction"
    report = result.run_dir.joinpath("final-report.md").read_text(encoding="utf-8")
    assert "### Workspace Hygiene" in report
    assert ".venv-correction" in report
    assert "did not exist before this writable operation" in report
    assert "TicketAutomation did not delete detected environments." in report


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_lifecycle_started_writable_codex_failure_is_human_required(tmp_path):
    repo, ticket, config = workflow_inputs(tmp_path)
    codex = SequencedCodexRunner(
        steps=[
            CodexStep(
                mutation=write_file("partial implementation\n"),
                returncode=2,
                stderr="service unavailable\n",
            ),
            CodexStep(result=review_result()),
        ],
        calls=[],
    )

    result = run_ticket_lifecycle(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        codex_runner=codex,
        verification_runner=PassingBaselineVerificationRunner(
            SequencedVerificationRunner(steps=[], calls=[])
        ),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert "Codex exited with code 2" in result.run_record.terminal_reason
    assert "may have left partial source changes" in result.run_record.terminal_reason
    assert result.verification_results == ()
    assert result.review_results == ()
    assert result.correction_results == ()
    assert len(codex.calls) == 1
    assert run_git(repo, "diff", "--name-only") == "file.txt"
    assert result.run_dir.joinpath("diffs", "failed-implementation.patch").is_file()
    report = result.run_dir.joinpath("final-report.md").read_text(encoding="utf-8")
    assert "Terminal outcome: HUMAN_REQUIRED" in report
    assert "Codex failure" in report
    assert "failed-implementation.patch" in report
    assert "file.txt" in report


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_lifecycle_correction_environment_pollution_stops_before_reverification(
    tmp_path,
):
    repo, ticket, config = workflow_inputs(tmp_path)

    def create_environment_and_correction(cwd: Path) -> None:
        cwd.joinpath("file.txt").write_text("corrected\n", encoding="utf-8")
        create_pyvenv(cwd / ".venv-correction")

    codex = SequencedCodexRunner(
        steps=[
            CodexStep(
                result=implementation_result(),
                mutation=write_file("implemented\n"),
            ),
            CodexStep(
                result=review_result(verdict=ReviewVerdict.CORRECTIONS_REQUIRED.value)
            ),
            CodexStep(
                result=implementation_result(),
                mutation=create_environment_and_correction,
            ),
            CodexStep(result=review_result()),
        ],
        calls=[],
    )
    verification = SequencedVerificationRunner(
        steps=[VerificationStep(), VerificationStep()],
        calls=[],
    )

    result = run_ticket_lifecycle(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        codex_runner=codex,
        verification_runner=PassingBaselineVerificationRunner(verification),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert result.run_record.current_correction_round == 1
    assert len(result.correction_results) == 1
    assert result.correction_results[0].workspace_guard is not None
    assert result.correction_results[0].workspace_guard.has_violation
    assert len(verification.calls) == 1
    assert len(result.verification_results) == 1
    assert len(result.review_results) == 1
    assert len(codex.calls) == 3
    assert repo.joinpath(".venv-correction", "pyvenv.cfg").is_file()
    assert repo.joinpath("file.txt").read_text(encoding="utf-8") == "corrected\n"
    assert run_git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "feature/example"
    assert run_git(repo, "rev-parse", "HEAD") == result.run_record.baseline_sha
    assert run_git(repo, "diff", "--cached", "--name-only") == ""
    guard_path = result.run_dir / "workspace-guard" / "correction-round-1.json"
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    assert guard["phase"] == "CORRECTING"
    assert guard["new_environments"][0]["markers"] == [".venv-correction/pyvenv.cfg"]
    report = result.run_dir.joinpath("final-report.md").read_text(encoding="utf-8")
    assert "correction-round-1.json" in report
    assert "TicketAutomation did not delete detected environments." in report


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_lifecycle_failed_correction_does_not_count_as_completed_round(tmp_path):
    repo, ticket, config = workflow_inputs(tmp_path)
    codex = SequencedCodexRunner(
        steps=[
            CodexStep(
                result=implementation_result(),
                mutation=write_file("implemented\n"),
            ),
            CodexStep(
                mutation=write_file("partial correction\n"),
                returncode=2,
                stderr="correction failed\n",
            ),
            CodexStep(result=review_result()),
        ],
        calls=[],
    )
    verification = SequencedVerificationRunner(
        steps=[
            VerificationStep(
                returncode=7,
                stdout="runner failed\n",
                stderr="failure detail\n",
            ),
        ],
        calls=[],
    )

    result = run_ticket_lifecycle(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        codex_runner=codex,
        verification_runner=PassingBaselineVerificationRunner(verification),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert result.run_record.current_correction_round == 0
    assert result.run_record.current_review_round == 0
    assert result.review_results == ()
    assert len(result.correction_results) == 1
    assert len(codex.calls) == 2
    assert run_git(repo, "diff", "--name-only") == "file.txt"
    assert result.run_dir.joinpath("diffs", "failed-correction-1.patch").is_file()
    report = result.run_dir.joinpath("final-report.md").read_text(encoding="utf-8")
    assert "Corrective rounds completed: 0" in report
    assert "failed-correction-1.patch" in report
    assert "Correction completed" not in result.run_record.terminal_reason


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_lifecycle_proven_codex_start_failure_without_changes_is_failed(tmp_path):
    _repo, ticket, config = workflow_inputs(tmp_path)
    codex = SequencedCodexRunner(
        steps=[CodexStep(error=FileNotFoundError("missing codex"))],
        calls=[],
    )

    result = run_ticket_lifecycle(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        codex_runner=codex,
        verification_runner=PassingBaselineVerificationRunner(
            SequencedVerificationRunner(steps=[], calls=[])
        ),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.FAILED
    assert "Codex executable is unavailable" in result.run_record.terminal_reason
    assert result.verification_results == ()
    assert result.review_results == ()
    assert result.correction_results == ()


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_lifecycle_internal_exception_after_snapshot_is_failed(tmp_path):
    _repo, ticket, config = workflow_inputs(tmp_path)
    codex = SequencedCodexRunner(
        steps=[CodexStep(error=RuntimeError("synthetic internal failure"))],
        calls=[],
    )

    result = run_ticket_lifecycle(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        codex_runner=codex,
        verification_runner=PassingBaselineVerificationRunner(
            SequencedVerificationRunner(steps=[], calls=[])
        ),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.FAILED
    assert result.controller_error == result.run_record.terminal_reason
    assert "Internal TicketAutomation exception" in result.run_record.terminal_reason
    assert "synthetic internal failure" in result.run_record.terminal_reason


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_lifecycle_broken_verification_environment_is_human_required(tmp_path):
    _repo, ticket, config = workflow_inputs(tmp_path)
    codex = SequencedCodexRunner(
        steps=[
            CodexStep(
                result=implementation_result(), mutation=write_file("implemented\n")
            ),
        ],
        calls=[],
    )
    verification = SequencedVerificationRunner(
        steps=[VerificationStep(error=FileNotFoundError("missing verifier"))],
        calls=[],
    )

    result = run_ticket_lifecycle(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        codex_runner=codex,
        verification_runner=PassingBaselineVerificationRunner(verification),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert not hasattr(result.run_record, "last_completed_state")
    assert result.correction_results == ()
    assert result.review_results == ()


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_lifecycle_pass_with_advisory_findings_does_not_correct(tmp_path):
    _repo, ticket, config = workflow_inputs(tmp_path)
    codex = SequencedCodexRunner(
        steps=[
            CodexStep(
                result=implementation_result(), mutation=write_file("implemented\n")
            ),
            CodexStep(
                result=review_result(
                    findings=[
                        finding("R1-F1", disposition="ADVISORY"),
                        finding("R1-F2", disposition="FOLLOW_UP"),
                    ]
                )
            ),
        ],
        calls=[],
    )
    verification = SequencedVerificationRunner(steps=[VerificationStep()], calls=[])

    result = run_ticket_lifecycle(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        codex_runner=codex,
        verification_runner=PassingBaselineVerificationRunner(verification),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.READY_FOR_HUMAN
    assert result.correction_results == ()
    assert result.review_results[0].review_result["verdict"] == "PASS"
    assert [
        item["disposition"]
        for item in result.review_results[0].review_result["findings"]
    ] == ["ADVISORY", "FOLLOW_UP"]


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_final_report_distinguishes_controller_facts_from_agent_claims(tmp_path):
    _repo, ticket, config = workflow_inputs(tmp_path)
    codex = SequencedCodexRunner(
        steps=[
            CodexStep(
                result=implementation_result(), mutation=write_file("implemented\n")
            ),
            CodexStep(
                result=review_result(
                    findings=[
                        finding("R1-F1", disposition="ADVISORY"),
                        finding("R1-F2", disposition="FOLLOW_UP"),
                    ]
                )
            ),
        ],
        calls=[],
    )

    result = run_ticket_lifecycle(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        codex_runner=codex,
        verification_runner=PassingBaselineVerificationRunner(
            SequencedVerificationRunner(
                steps=[VerificationStep()],
                calls=[],
            )
        ),
        clock=fixed_clock,
    )

    report = result.run_dir.joinpath("final-report.md").read_text(encoding="utf-8")
    assert "Authoritative controller evidence" in report
    assert "Agent-reported information" in report
    assert (
        "Implementation-agent targeted validation is reported here as an agent claim"
        in report
    )
    assert "Final review verdict: PASS" in report
    assert "Advisory findings: 1" in report
    assert "Follow-up findings: 1" in report
    assert "TicketAutomation did not intentionally stage" in report
    assert "TicketAutomation did not intentionally commit" in report


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_final_patch_is_complete_baseline_relative_diff_after_correction(tmp_path):
    _repo, ticket, config = workflow_inputs(tmp_path)
    codex = SequencedCodexRunner(
        steps=[
            CodexStep(
                result=implementation_result(),
                mutation=lambda cwd: (
                    cwd.joinpath("file.txt").write_text(
                        "implemented\n",
                        encoding="utf-8",
                    ),
                    cwd.joinpath("kept.txt").write_text(
                        "original implementation work\n",
                        encoding="utf-8",
                    ),
                ),
            ),
            CodexStep(
                result=review_result(verdict=ReviewVerdict.CORRECTIONS_REQUIRED.value)
            ),
            CodexStep(
                result=implementation_result(), mutation=write_file("corrected\n")
            ),
            CodexStep(result=review_result()),
        ],
        calls=[],
    )

    result = run_ticket_lifecycle(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        codex_runner=codex,
        verification_runner=PassingBaselineVerificationRunner(
            SequencedVerificationRunner(
                steps=[VerificationStep(), VerificationStep()],
                calls=[],
            )
        ),
        clock=fixed_clock,
    )

    patch = result.run_dir.joinpath("diffs", "final.patch").read_text(encoding="utf-8")
    assert "diff --git a/file.txt b/file.txt" in patch
    assert "+corrected" in patch
    assert "diff --git a/kept.txt b/kept.txt" in patch
    assert "+original implementation work" in patch


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_resume_from_prepared_starts_implementation(tmp_path):
    _repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    snapshot = create_run_snapshot(config, ticket, runs_dir=runs_dir, clock=fixed_clock)
    codex = SequencedCodexRunner(
        steps=[
            CodexStep(
                result=implementation_result(),
                mutation=write_file("implemented\n"),
            ),
            CodexStep(result=review_result()),
        ],
        calls=[],
    )
    verification = SequencedVerificationRunner(steps=[VerificationStep()], calls=[])

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=runs_dir,
        codex_runner=codex,
        verification_runner=verification,
        clock=fixed_clock,
    )

    assert result.successful
    assert [sandbox_value(command.argv) for command, _stdin in codex.calls] == [
        Sandbox.WORKSPACE_WRITE.value,
        Sandbox.READ_ONLY.value,
    ]
    assert len(verification.calls) == 1


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_resume_from_completed_implementation_runs_verification_review_and_report(
    tmp_path,
):
    _repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    snapshot = create_run_snapshot(config, ticket, runs_dir=runs_dir, clock=fixed_clock)
    run_implementation_stage(
        config,
        snapshot.run_dir,
        codex_runner=SequencedCodexRunner(
            steps=[
                CodexStep(
                    result=implementation_result(),
                    mutation=write_file("implemented\n"),
                )
            ],
            calls=[],
        ),
        clock=fixed_clock,
    )
    codex = SequencedCodexRunner(steps=[CodexStep(result=review_result())], calls=[])
    verification = SequencedVerificationRunner(steps=[VerificationStep()], calls=[])

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=runs_dir,
        codex_runner=codex,
        verification_runner=verification,
        clock=fixed_clock,
    )

    assert result.successful
    assert len(verification.calls) == 1
    assert [sandbox_value(command.argv) for command, _stdin in codex.calls] == [
        Sandbox.READ_ONLY.value
    ]
    assert result.run_dir.joinpath("final-report.md").is_file()


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
@pytest.mark.parametrize("patch_evidence", ["changed", "missing"])
def test_resume_uses_workspace_fingerprint_instead_of_patch_evidence(
    tmp_path,
    patch_evidence,
):
    _repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    snapshot = create_run_snapshot(config, ticket, runs_dir=runs_dir, clock=fixed_clock)
    run_implementation_stage(
        config,
        snapshot.run_dir,
        codex_runner=SequencedCodexRunner(
            steps=[
                CodexStep(
                    result=implementation_result(),
                    mutation=write_file("implemented\n"),
                )
            ],
            calls=[],
        ),
        clock=fixed_clock,
    )
    patch_path = snapshot.run_dir / "diffs" / "after-implementation.patch"
    if patch_evidence == "changed":
        patch_path.write_text(
            "human evidence changed independently\n",
            encoding="utf-8",
        )
    else:
        patch_path.unlink()

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=runs_dir,
        codex_runner=SequencedCodexRunner(
            steps=[CodexStep(result=review_result())],
            calls=[],
        ),
        verification_runner=SequencedVerificationRunner(
            steps=[VerificationStep()],
            calls=[],
        ),
        clock=fixed_clock,
    )

    assert result.successful


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_resume_fails_closed_without_canonical_workspace_checkpoint(tmp_path):
    _repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    snapshot = create_run_snapshot(config, ticket, runs_dir=runs_dir, clock=fixed_clock)
    run_implementation_stage(
        config,
        snapshot.run_dir,
        codex_runner=SequencedCodexRunner(
            steps=[
                CodexStep(
                    result=implementation_result(),
                    mutation=write_file("implemented\n"),
                )
            ],
            calls=[],
        ),
        clock=fixed_clock,
    )
    snapshot.run_dir.joinpath(
        "diffs",
        "after-implementation.workspace.sha256",
    ).unlink()
    codex = SequencedCodexRunner(steps=[], calls=[])
    verification = SequencedVerificationRunner(steps=[], calls=[])

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=runs_dir,
        codex_runner=codex,
        verification_runner=verification,
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert "workspace fingerprint checkpoint is missing" in (
        result.run_record.terminal_reason or ""
    )
    assert codex.calls == []
    assert verification.calls == []


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_resume_from_reporting_regenerates_report_without_rerunning_stages(tmp_path):
    _repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    snapshot = create_run_snapshot(config, ticket, runs_dir=runs_dir, clock=fixed_clock)
    run_implementation_stage(
        config,
        snapshot.run_dir,
        codex_runner=SequencedCodexRunner(
            steps=[
                CodexStep(
                    result=implementation_result(),
                    mutation=write_file("implemented\n"),
                )
            ],
            calls=[],
        ),
        clock=fixed_clock,
    )
    run_verification_stage(
        config,
        snapshot.run_dir,
        process_runner=SequencedVerificationRunner(
            steps=[VerificationStep()],
            calls=[],
        ),
        clock=fixed_clock,
    )
    run_review_stage(
        config,
        snapshot.run_dir,
        codex_runner=SequencedCodexRunner(
            steps=[CodexStep(result=review_result())],
            calls=[],
        ),
        clock=fixed_clock,
    )
    assert load_run_record(snapshot.run_dir / "run.json").state == (
        WorkflowState.REPORTING
    )
    codex = SequencedCodexRunner(steps=[], calls=[])
    verification = SequencedVerificationRunner(steps=[], calls=[])

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=runs_dir,
        codex_runner=codex,
        verification_runner=verification,
        clock=fixed_clock,
    )

    assert result.successful
    assert result.report_result is not None
    assert result.run_dir.joinpath("final-report.md").is_file()
    assert codex.calls == []
    assert verification.calls == []


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_resume_adopts_completed_verification_artifact_when_run_state_is_stale(
    tmp_path,
):
    _repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    snapshot = create_run_snapshot(config, ticket, runs_dir=runs_dir, clock=fixed_clock)
    run_implementation_stage(
        config,
        snapshot.run_dir,
        codex_runner=SequencedCodexRunner(
            steps=[
                CodexStep(
                    result=implementation_result(),
                    mutation=write_file("implemented\n"),
                )
            ],
            calls=[],
        ),
        clock=fixed_clock,
    )
    record_path = snapshot.run_dir / "run.json"
    stale_record = load_run_record(record_path)
    completed_verification = SequencedVerificationRunner(
        steps=[VerificationStep(stdout="persisted verification\n")],
        calls=[],
    )
    run_verification_stage(
        config,
        snapshot.run_dir,
        process_runner=completed_verification,
        clock=fixed_clock,
    )
    save_run_record(stale_record, record_path)
    codex = SequencedCodexRunner(steps=[CodexStep(result=review_result())], calls=[])
    verification = SequencedVerificationRunner(steps=[VerificationStep()], calls=[])

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=runs_dir,
        codex_runner=codex,
        verification_runner=verification,
        clock=fixed_clock,
    )

    assert result.successful
    assert completed_verification.calls
    assert verification.calls == []
    assert len(result.verification_results) == 1
    assert (
        result.verification_results[0].round_result.commands[0].stdout
        == "persisted verification\n"
    )
    assert [sandbox_value(command.argv) for command, _stdin in codex.calls] == [
        Sandbox.READ_ONLY.value
    ]


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_resume_adopts_completed_review_result_when_run_state_is_stale(tmp_path):
    _repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    snapshot = create_run_snapshot(config, ticket, runs_dir=runs_dir, clock=fixed_clock)
    run_implementation_stage(
        config,
        snapshot.run_dir,
        codex_runner=SequencedCodexRunner(
            steps=[
                CodexStep(
                    result=implementation_result(),
                    mutation=write_file("implemented\n"),
                )
            ],
            calls=[],
        ),
        clock=fixed_clock,
    )
    run_verification_stage(
        config,
        snapshot.run_dir,
        process_runner=SequencedVerificationRunner(
            steps=[VerificationStep()],
            calls=[],
        ),
        clock=fixed_clock,
    )
    record_path = snapshot.run_dir / "run.json"
    stale_record = load_run_record(record_path)
    completed_review = SequencedCodexRunner(
        steps=[CodexStep(result=review_result())],
        calls=[],
    )
    run_review_stage(
        config,
        snapshot.run_dir,
        codex_runner=completed_review,
        clock=fixed_clock,
    )
    save_run_record(stale_record, record_path)
    codex = SequencedCodexRunner(steps=[CodexStep(result=review_result())], calls=[])

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=runs_dir,
        codex_runner=codex,
        verification_runner=SequencedVerificationRunner(steps=[], calls=[]),
        clock=fixed_clock,
    )

    assert result.successful
    assert completed_review.calls
    assert codex.calls == []
    assert len(result.review_results) == 1
    assert result.review_results[0].review_result["verdict"] == ReviewVerdict.PASS.value


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_resume_adopts_review_result_without_execution_metadata_when_checkpoint_matches(
    tmp_path,
):
    _repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    snapshot = create_run_snapshot(config, ticket, runs_dir=runs_dir, clock=fixed_clock)
    run_implementation_stage(
        config,
        snapshot.run_dir,
        codex_runner=SequencedCodexRunner(
            steps=[
                CodexStep(
                    result=implementation_result(),
                    mutation=write_file("implemented\n"),
                )
            ],
            calls=[],
        ),
        clock=fixed_clock,
    )
    run_verification_stage(
        config,
        snapshot.run_dir,
        process_runner=SequencedVerificationRunner(
            steps=[VerificationStep()],
            calls=[],
        ),
        clock=fixed_clock,
    )
    record_path = snapshot.run_dir / "run.json"
    stale_record = load_run_record(record_path)
    run_review_stage(
        config,
        snapshot.run_dir,
        codex_runner=SequencedCodexRunner(
            steps=[CodexStep(result=review_result())],
            calls=[],
        ),
        clock=fixed_clock,
    )
    snapshot.run_dir.joinpath("reviews", "round-1", "execution.json").unlink()
    save_run_record(stale_record, record_path)
    codex = SequencedCodexRunner(steps=[CodexStep(result=review_result())], calls=[])

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=runs_dir,
        codex_runner=codex,
        verification_runner=SequencedVerificationRunner(steps=[], calls=[]),
        clock=fixed_clock,
    )

    assert result.successful
    assert codex.calls == []
    assert len(result.review_results) == 1
    assert result.review_results[0].review_result["verdict"] == ReviewVerdict.PASS.value


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_resume_reruns_review_result_without_checkpoint_metadata(tmp_path):
    _repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    snapshot = create_run_snapshot(config, ticket, runs_dir=runs_dir, clock=fixed_clock)
    run_implementation_stage(
        config,
        snapshot.run_dir,
        codex_runner=SequencedCodexRunner(
            steps=[
                CodexStep(
                    result=implementation_result(),
                    mutation=write_file("implemented\n"),
                )
            ],
            calls=[],
        ),
        clock=fixed_clock,
    )
    run_verification_stage(
        config,
        snapshot.run_dir,
        process_runner=SequencedVerificationRunner(
            steps=[VerificationStep()],
            calls=[],
        ),
        clock=fixed_clock,
    )
    record_path = snapshot.run_dir / "run.json"
    stale_record = load_run_record(record_path)
    run_review_stage(
        config,
        snapshot.run_dir,
        codex_runner=SequencedCodexRunner(
            steps=[CodexStep(result=review_result(summary="old review"))],
            calls=[],
        ),
        clock=fixed_clock,
    )
    snapshot.run_dir.joinpath("reviews", "round-1", "checkpoint.json").unlink()
    save_run_record(stale_record, record_path)
    codex = SequencedCodexRunner(
        steps=[CodexStep(result=review_result(summary="fresh review"))],
        calls=[],
    )

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=runs_dir,
        codex_runner=codex,
        verification_runner=SequencedVerificationRunner(steps=[], calls=[]),
        clock=fixed_clock,
    )

    assert result.successful
    assert [sandbox_value(command.argv) for command, _stdin in codex.calls] == [
        Sandbox.READ_ONLY.value
    ]
    assert (
        result.run_dir.joinpath("reviews", "_incomplete", "round-1", "result.json")
        .read_text(encoding="utf-8")
        .find("old review")
        != -1
    )
    assert result.review_results[0].review_result["summary"] == "fresh review"


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_resume_does_not_adopt_verification_artifact_for_changed_gate_config(
    tmp_path,
):
    repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    snapshot = create_run_snapshot(config, ticket, runs_dir=runs_dir, clock=fixed_clock)
    run_implementation_stage(
        config,
        snapshot.run_dir,
        codex_runner=SequencedCodexRunner(
            steps=[
                CodexStep(
                    result=implementation_result(),
                    mutation=write_file("implemented\n"),
                )
            ],
            calls=[],
        ),
        clock=fixed_clock,
    )
    record_path = snapshot.run_dir / "run.json"
    stale_record = load_run_record(record_path)
    run_verification_stage(
        config,
        snapshot.run_dir,
        process_runner=SequencedVerificationRunner(
            steps=[VerificationStep()],
            calls=[],
        ),
        clock=fixed_clock,
    )
    save_run_record(stale_record, record_path)
    changed_config = make_config(
        repo,
        verification_commands=(
            VerificationCommand(
                name="changed-tests",
                argv=(Path(sys.executable).as_posix(), "-c", "raise SystemExit(0)"),
                timeout_seconds=1800,
            ),
        ),
    )
    verification = SequencedVerificationRunner(steps=[VerificationStep()], calls=[])
    codex = SequencedCodexRunner(steps=[CodexStep(result=review_result())], calls=[])

    result = resume_ticket_lifecycle(
        changed_config,
        snapshot.run_record.run_id,
        runs_dir=runs_dir,
        codex_runner=codex,
        verification_runner=verification,
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert "verification_commands_fingerprint" in result.run_record.terminal_reason
    assert verification.calls == []
    assert codex.calls == []


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_resume_does_not_reuse_review_result_after_repository_changed(tmp_path):
    repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    snapshot = create_run_snapshot(config, ticket, runs_dir=runs_dir, clock=fixed_clock)
    run_implementation_stage(
        config,
        snapshot.run_dir,
        codex_runner=SequencedCodexRunner(
            steps=[
                CodexStep(
                    result=implementation_result(),
                    mutation=write_file("implemented\n"),
                )
            ],
            calls=[],
        ),
        clock=fixed_clock,
    )
    run_verification_stage(
        config,
        snapshot.run_dir,
        process_runner=SequencedVerificationRunner(
            steps=[VerificationStep()],
            calls=[],
        ),
        clock=fixed_clock,
    )
    record_path = snapshot.run_dir / "run.json"
    stale_record = load_run_record(record_path)
    run_review_stage(
        config,
        snapshot.run_dir,
        codex_runner=SequencedCodexRunner(
            steps=[CodexStep(result=review_result())],
            calls=[],
        ),
        clock=fixed_clock,
    )
    save_run_record(stale_record, record_path)
    repo.joinpath("file.txt").write_text("changed after review\n", encoding="utf-8")
    codex = SequencedCodexRunner(steps=[CodexStep(result=review_result())], calls=[])

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=runs_dir,
        codex_runner=codex,
        verification_runner=SequencedVerificationRunner(steps=[], calls=[]),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert "Current workspace no longer matches" in result.run_record.terminal_reason
    assert codex.calls == []


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_resume_archives_partial_verification_artifact_and_reruns(tmp_path):
    _repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    snapshot = create_run_snapshot(config, ticket, runs_dir=runs_dir, clock=fixed_clock)
    run_implementation_stage(
        config,
        snapshot.run_dir,
        codex_runner=SequencedCodexRunner(
            steps=[
                CodexStep(
                    result=implementation_result(),
                    mutation=write_file("implemented\n"),
                )
            ],
            calls=[],
        ),
        clock=fixed_clock,
    )
    verification_dir = snapshot.run_dir / "verification"
    verification_dir.mkdir()
    verification_dir.joinpath("round-0.json").write_text(
        '{"status": "PASS"',
        encoding="utf-8",
    )
    codex = SequencedCodexRunner(steps=[CodexStep(result=review_result())], calls=[])
    verification = SequencedVerificationRunner(
        steps=[VerificationStep(stdout="fresh verification\n")],
        calls=[],
    )

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=runs_dir,
        codex_runner=codex,
        verification_runner=verification,
        clock=fixed_clock,
    )

    assert result.successful
    assert len(verification.calls) == 1
    assert (
        result.verification_results[0].round_result.commands[0].stdout
        == "fresh verification\n"
    )
    assert result.run_dir.joinpath("verification", "_incomplete", "round-0").is_dir()
    assert result.run_dir.joinpath("verification", "round-0.json").is_file()


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_resume_does_not_reuse_verification_artifact_after_repository_changed(
    tmp_path,
):
    repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    snapshot = create_run_snapshot(config, ticket, runs_dir=runs_dir, clock=fixed_clock)
    run_implementation_stage(
        config,
        snapshot.run_dir,
        codex_runner=SequencedCodexRunner(
            steps=[
                CodexStep(
                    result=implementation_result(),
                    mutation=write_file("implemented\n"),
                )
            ],
            calls=[],
        ),
        clock=fixed_clock,
    )
    record_path = snapshot.run_dir / "run.json"
    stale_record = load_run_record(record_path)
    run_verification_stage(
        config,
        snapshot.run_dir,
        process_runner=SequencedVerificationRunner(
            steps=[VerificationStep()],
            calls=[],
        ),
        clock=fixed_clock,
    )
    save_run_record(stale_record, record_path)
    repo.joinpath("file.txt").write_text(
        "changed after verification\n", encoding="utf-8"
    )
    codex = SequencedCodexRunner(steps=[CodexStep(result=review_result())], calls=[])
    verification = SequencedVerificationRunner(steps=[VerificationStep()], calls=[])

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=runs_dir,
        codex_runner=codex,
        verification_runner=verification,
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert "Current workspace no longer matches" in result.run_record.terminal_reason
    assert verification.calls == []
    assert codex.calls == []


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
@pytest.mark.parametrize(
    "terminal_state",
    [
        WorkflowState.READY_FOR_HUMAN,
        WorkflowState.HUMAN_REQUIRED,
        WorkflowState.FAILED,
    ],
)
def test_resume_terminal_run_does_not_regenerate_report_artifacts(
    tmp_path,
    terminal_state,
):
    _repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    snapshot = create_run_snapshot(config, ticket, runs_dir=runs_dir, clock=fixed_clock)
    transition_path = {
        WorkflowState.READY_FOR_HUMAN: (
            WorkflowState.IMPLEMENTING,
            WorkflowState.VERIFYING,
            WorkflowState.REVIEWING,
            WorkflowState.REPORTING,
            WorkflowState.READY_FOR_HUMAN,
        ),
        WorkflowState.HUMAN_REQUIRED: (WorkflowState.HUMAN_REQUIRED,),
        WorkflowState.FAILED: (WorkflowState.FAILED,),
    }[terminal_state]
    run_record = snapshot.run_record
    for state in transition_path:
        run_record = run_record.transition_to(
            state,
            updated_timestamp="2026-09-11T13:05:17Z",
            terminal_reason=(
                "synthetic terminal reason"
                if state in {WorkflowState.HUMAN_REQUIRED, WorkflowState.FAILED}
                else None
            ),
        )
    save_run_record(run_record, snapshot.run_dir / "run.json")
    final_patch = snapshot.run_dir / "diffs" / "final.patch"
    final_patch.parent.mkdir(parents=True)
    final_report = snapshot.run_dir / "final-report.md"
    final_patch.write_text("preserved patch\n", encoding="utf-8")
    final_report.write_text("preserved report\n", encoding="utf-8")
    codex = SequencedCodexRunner(steps=[CodexStep(result=review_result())], calls=[])
    verification = SequencedVerificationRunner(steps=[VerificationStep()], calls=[])

    resumed = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=runs_dir,
        codex_runner=codex,
        verification_runner=verification,
        clock=fixed_clock,
    )

    assert resumed.run_record.state == terminal_state
    assert final_patch.read_text(encoding="utf-8") == "preserved patch\n"
    assert final_report.read_text(encoding="utf-8") == "preserved report\n"
    assert codex.calls == []
    assert verification.calls == []


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_resume_from_review_correction_checkpoint_uses_persisted_findings(tmp_path):
    _repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    snapshot = create_run_snapshot(config, ticket, runs_dir=runs_dir, clock=fixed_clock)
    run_implementation_stage(
        config,
        snapshot.run_dir,
        codex_runner=SequencedCodexRunner(
            steps=[
                CodexStep(
                    result=implementation_result(),
                    mutation=write_file("implemented\n"),
                )
            ],
            calls=[],
        ),
        clock=fixed_clock,
    )
    run_verification_stage(
        config,
        snapshot.run_dir,
        process_runner=SequencedVerificationRunner(
            steps=[VerificationStep()],
            calls=[],
        ),
        clock=fixed_clock,
    )
    run_review_stage(
        config,
        snapshot.run_dir,
        codex_runner=SequencedCodexRunner(
            steps=[
                CodexStep(
                    result=review_result(
                        verdict=ReviewVerdict.CORRECTIONS_REQUIRED.value
                    )
                )
            ],
            calls=[],
        ),
        clock=fixed_clock,
    )
    codex = SequencedCodexRunner(
        steps=[
            CodexStep(result=implementation_result(), mutation=write_file("fixed\n")),
            CodexStep(result=review_result()),
        ],
        calls=[],
    )

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=runs_dir,
        codex_runner=codex,
        verification_runner=SequencedVerificationRunner(
            steps=[VerificationStep()],
            calls=[],
        ),
        clock=fixed_clock,
    )

    assert result.successful
    assert result.run_record.current_correction_round == 1
    assert [sandbox_value(command.argv) for command, _stdin in codex.calls] == [
        Sandbox.WORKSPACE_WRITE.value,
        Sandbox.READ_ONLY.value,
    ]


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_resume_missing_review_correction_source_becomes_human_required(tmp_path):
    _repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    snapshot = create_run_snapshot(config, ticket, runs_dir=runs_dir, clock=fixed_clock)
    run_implementation_stage(
        config,
        snapshot.run_dir,
        codex_runner=SequencedCodexRunner(
            steps=[
                CodexStep(
                    result=implementation_result(),
                    mutation=write_file("implemented\n"),
                )
            ],
            calls=[],
        ),
        clock=fixed_clock,
    )
    run_verification_stage(
        config,
        snapshot.run_dir,
        process_runner=SequencedVerificationRunner(
            steps=[VerificationStep()],
            calls=[],
        ),
        clock=fixed_clock,
    )
    run_review_stage(
        config,
        snapshot.run_dir,
        codex_runner=SequencedCodexRunner(
            steps=[
                CodexStep(
                    result=review_result(
                        verdict=ReviewVerdict.CORRECTIONS_REQUIRED.value
                    )
                )
            ],
            calls=[],
        ),
        clock=fixed_clock,
    )
    snapshot.run_dir.joinpath("reviews", "round-1", "result.json").unlink()
    codex = SequencedCodexRunner(
        steps=[CodexStep(result=implementation_result())],
        calls=[],
    )

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=runs_dir,
        codex_runner=codex,
        verification_runner=SequencedVerificationRunner(steps=[], calls=[]),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert "Correction source" in result.run_record.terminal_reason
    assert codex.calls == []


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_resume_missing_verification_correction_source_becomes_human_required(tmp_path):
    _repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    snapshot = create_run_snapshot(config, ticket, runs_dir=runs_dir, clock=fixed_clock)
    run_implementation_stage(
        config,
        snapshot.run_dir,
        codex_runner=SequencedCodexRunner(
            steps=[
                CodexStep(
                    result=implementation_result(),
                    mutation=write_file("implemented\n"),
                )
            ],
            calls=[],
        ),
        clock=fixed_clock,
    )
    run_verification_stage(
        config,
        snapshot.run_dir,
        process_runner=SequencedVerificationRunner(
            steps=[
                VerificationStep(
                    returncode=7,
                    stdout="runner failed\n",
                    stderr="failure detail\n",
                )
            ],
            calls=[],
        ),
        clock=fixed_clock,
    )
    snapshot.run_dir.joinpath("verification", "round-0.json").unlink()
    codex = SequencedCodexRunner(
        steps=[CodexStep(result=implementation_result())],
        calls=[],
    )

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=runs_dir,
        codex_runner=codex,
        verification_runner=SequencedVerificationRunner(steps=[], calls=[]),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert "Correction source" in result.run_record.terminal_reason
    assert codex.calls == []


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_resume_malformed_verification_correction_source_becomes_human_required(
    tmp_path,
):
    _repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    snapshot = create_run_snapshot(config, ticket, runs_dir=runs_dir, clock=fixed_clock)
    run_implementation_stage(
        config,
        snapshot.run_dir,
        codex_runner=SequencedCodexRunner(
            steps=[
                CodexStep(
                    result=implementation_result(),
                    mutation=write_file("implemented\n"),
                )
            ],
            calls=[],
        ),
        clock=fixed_clock,
    )
    run_verification_stage(
        config,
        snapshot.run_dir,
        process_runner=SequencedVerificationRunner(
            steps=[
                VerificationStep(
                    returncode=7,
                    stdout="runner failed\n",
                    stderr="failure detail\n",
                )
            ],
            calls=[],
        ),
        clock=fixed_clock,
    )
    verification_path = snapshot.run_dir / "verification" / "round-0.json"
    verification_result = json.loads(verification_path.read_text(encoding="utf-8"))
    verification_result["correction_reasons"] = "not a list"
    verification_path.write_text(
        json.dumps(verification_result),
        encoding="utf-8",
    )
    codex = SequencedCodexRunner(
        steps=[CodexStep(result=implementation_result())],
        calls=[],
    )

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=runs_dir,
        codex_runner=codex,
        verification_runner=SequencedVerificationRunner(steps=[], calls=[]),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert "correction_reasons must be a list" in result.run_record.terminal_reason
    assert codex.calls == []


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_resume_interrupted_implementation_becomes_human_required(tmp_path):
    repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    snapshot = create_run_snapshot(config, ticket, runs_dir=runs_dir, clock=fixed_clock)
    repo.joinpath("file.txt").write_text("partial implementation\n", encoding="utf-8")
    record_path = snapshot.run_dir / "run.json"
    save_run_record(
        load_run_record(record_path).transition_to(
            WorkflowState.IMPLEMENTING,
            updated_timestamp="2026-09-11T13:05:18Z",
        ),
        record_path,
    )
    codex = SequencedCodexRunner(steps=[CodexStep(result=review_result())], calls=[])

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=runs_dir,
        codex_runner=codex,
        verification_runner=SequencedVerificationRunner(steps=[], calls=[]),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert (
        "Writable implementation was interrupted" in result.run_record.terminal_reason
    )
    assert codex.calls == []
    assert result.run_dir.joinpath("final-report.md").is_file()
    report = result.run_dir.joinpath("final-report.md").read_text(encoding="utf-8")
    assert "Workflow state: HUMAN_REQUIRED" in report
    assert "Source changes may be incomplete" in report
    assert "What requires human inspection: Writable implementation" in report
    assert "### Relevant artifact/log paths" in report
    assert "- run.json" in report


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_resume_interrupted_correction_becomes_human_required(tmp_path):
    repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    snapshot = create_run_snapshot(config, ticket, runs_dir=runs_dir, clock=fixed_clock)
    repo.joinpath("file.txt").write_text("partial correction\n", encoding="utf-8")
    record_path = snapshot.run_dir / "run.json"
    interrupted_record = (
        load_run_record(record_path)
        .transition_to(
            WorkflowState.IMPLEMENTING,
            updated_timestamp="2026-09-11T13:05:18Z",
        )
        .transition_to(
            WorkflowState.VERIFYING,
            updated_timestamp="2026-09-11T13:05:18Z",
        )
        .transition_to(
            WorkflowState.CORRECTION_PENDING,
            updated_timestamp="2026-09-11T13:05:18Z",
        )
        .transition_to(
            WorkflowState.CORRECTING,
            updated_timestamp="2026-09-11T13:05:18Z",
        )
    )
    save_run_record(interrupted_record, record_path)

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=runs_dir,
        codex_runner=SequencedCodexRunner(
            steps=[CodexStep(result=implementation_result())],
            calls=[],
        ),
        verification_runner=SequencedVerificationRunner(steps=[], calls=[]),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert "Writable correction was interrupted" in result.run_record.terminal_reason
    assert result.run_dir.joinpath("final-report.md").is_file()


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_best_effort_report_preserves_primary_failure_when_final_patch_fails(
    monkeypatch,
    tmp_path,
):
    repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    snapshot = create_run_snapshot(config, ticket, runs_dir=runs_dir, clock=fixed_clock)
    repo.joinpath("file.txt").write_text("partial correction\n", encoding="utf-8")
    record_path = snapshot.run_dir / "run.json"
    interrupted_record = (
        load_run_record(record_path)
        .transition_to(
            WorkflowState.IMPLEMENTING,
            updated_timestamp="2026-09-11T13:05:18Z",
        )
        .transition_to(
            WorkflowState.VERIFYING,
            updated_timestamp="2026-09-11T13:05:18Z",
        )
        .transition_to(
            WorkflowState.CORRECTION_PENDING,
            updated_timestamp="2026-09-11T13:05:18Z",
        )
        .transition_to(
            WorkflowState.CORRECTING,
            updated_timestamp="2026-09-11T13:05:18Z",
        )
    )
    save_run_record(interrupted_record, record_path)

    def fail_diff(repository, baseline_sha):
        del repository, baseline_sha
        raise GitCommandError("synthetic final patch failure")

    monkeypatch.setattr(
        "ticket_automation.reporting.diff_including_untracked",
        fail_diff,
    )

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=runs_dir,
        codex_runner=SequencedCodexRunner(
            steps=[CodexStep(result=implementation_result())],
            calls=[],
        ),
        verification_runner=SequencedVerificationRunner(steps=[], calls=[]),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert "Writable correction was interrupted" in result.run_record.terminal_reason
    report = result.run_dir.joinpath("final-report.md").read_text(encoding="utf-8")
    assert "Terminal outcome: HUMAN_REQUIRED" in report
    assert (
        "What requires human inspection: Writable correction was interrupted" in report
    )
    assert "Final patch unavailable" in report
    assert "synthetic final patch failure" in report
    assert run_git(repo, "diff", "--name-only") == "file.txt"


@pytest.mark.skipif(GIT is None, reason="git executable is required for workflow tests")
def test_resume_restarts_interrupted_read_only_review_with_fresh_review(tmp_path):
    _repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    snapshot = create_run_snapshot(config, ticket, runs_dir=runs_dir, clock=fixed_clock)
    run_implementation_stage(
        config,
        snapshot.run_dir,
        codex_runner=SequencedCodexRunner(
            steps=[
                CodexStep(
                    result=implementation_result(),
                    mutation=write_file("implemented\n"),
                )
            ],
            calls=[],
        ),
        clock=fixed_clock,
    )
    run_verification_stage(
        config,
        snapshot.run_dir,
        process_runner=SequencedVerificationRunner(
            steps=[VerificationStep()],
            calls=[],
        ),
        clock=fixed_clock,
    )
    partial_review_dir = snapshot.run_dir / "reviews" / "round-1"
    partial_review_dir.mkdir(parents=True)
    partial_review_dir.joinpath("prompt.md").write_text(
        "partial read-only review\n",
        encoding="utf-8",
    )
    codex = SequencedCodexRunner(steps=[CodexStep(result=review_result())], calls=[])

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=runs_dir,
        codex_runner=codex,
        verification_runner=SequencedVerificationRunner(steps=[], calls=[]),
        clock=fixed_clock,
    )

    assert result.successful
    assert [sandbox_value(command.argv) for command, _stdin in codex.calls] == [
        Sandbox.READ_ONLY.value
    ]
    assert result.run_dir.joinpath("reviews", "round-1", "result.json").is_file()


def workflow_inputs(
    tmp_path,
    *,
    max_correction_rounds: int = 3,
) -> tuple[Path, Path, object]:
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "QDEB-003.md"
    ticket.write_text("# QDEB-003\n\nImplement the ticket.\n", encoding="utf-8")
    config = make_config(
        repo,
        max_correction_rounds=max_correction_rounds,
        verification_commands=(
            VerificationCommand(
                name="tests",
                argv=(Path(sys.executable).as_posix(), "-c", "raise SystemExit(0)"),
                timeout_seconds=1800,
            ),
        ),
    )
    return repo, ticket, config


def implementation_result(status: str = "COMPLETED") -> dict[str, object]:
    return {
        "status": status,
        "summary": "implemented" if status == "COMPLETED" else "blocked",
        "tests_run": [{"command": "synthetic", "result": "PASS"}],
        "assumptions": [],
        "known_issues": [] if status == "COMPLETED" else ["blocked"],
    }


def review_result(
    *,
    verdict: str = ReviewVerdict.PASS.value,
    summary: str = "review result",
    findings: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if findings is None and verdict == ReviewVerdict.CORRECTIONS_REQUIRED.value:
        findings = [finding("R1-F1", disposition="REQUIRED")]
    return {
        "verdict": verdict,
        "summary": summary,
        "confidence": "HIGH",
        "findings": [] if findings is None else findings,
    }


def finding(finding_id: str, *, disposition: str) -> dict[str, object]:
    return {
        "id": finding_id,
        "severity": "MEDIUM",
        "category": "CORRECTNESS",
        "disposition": disposition,
        "scope_relation": "TICKET",
        "title": "Synthetic finding",
        "description": "Synthetic finding description.",
        "evidence": "Synthetic evidence.",
        "required_change": "Make the synthetic correction.",
        "acceptance_criteria": ["Synthetic correction is present."],
    }


def write_file(contents: str) -> Callable[[Path], None]:
    def mutate(cwd: Path) -> None:
        cwd.joinpath("file.txt").write_text(contents, encoding="utf-8")

    return mutate


def stage_file(cwd: Path) -> None:
    cwd.joinpath("file.txt").write_text("staged\n", encoding="utf-8")
    run_git(cwd, "add", "file.txt")


def commit_file(cwd: Path) -> None:
    cwd.joinpath("file.txt").write_text("committed\n", encoding="utf-8")
    run_git(cwd, "add", "file.txt")
    run_git(cwd, "commit", "-m", "agent changed head")


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
