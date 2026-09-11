from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.helpers import GIT, create_git_repo, make_config, run_git
from ticket_automation.codex import CodexCommand, CodexProcessResult, Sandbox
from ticket_automation.config import VerificationCommand
from ticket_automation.models import WorkflowState
from ticket_automation.review import ReviewVerdict
from ticket_automation.verification import (
    VerificationProcessCommand,
    VerificationProcessResult,
)
from ticket_automation.workflow import format_lifecycle_result, run_ticket_lifecycle


def fixed_clock() -> datetime:
    return datetime(2026, 9, 11, 13, 5, 17, tzinfo=UTC)


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
        verification_runner=verification,
        clock=fixed_clock,
    )

    assert result.successful
    assert result.run_record.state == WorkflowState.READY_FOR_HUMAN
    assert result.run_record.last_completed_state == WorkflowState.REVIEW
    assert result.run_record.current_correction_round == 0
    assert result.run_record.current_review_round == 1
    assert [sandbox_value(command.argv) for command, _stdin in codex.calls] == [
        Sandbox.WORKSPACE_WRITE.value,
        Sandbox.READ_ONLY.value,
    ]
    assert len(result.verification_results) == 1
    assert len(result.review_results) == 1
    assert result.correction_results == ()
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
        verification_runner=verification,
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
        verification_runner=verification,
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
        verification_runner=verification,
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
        verification_runner=verification,
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
        verification_runner=verification,
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
def test_lifecycle_codex_infrastructure_failure_is_failed(tmp_path):
    _repo, ticket, config = workflow_inputs(tmp_path)
    codex = SequencedCodexRunner(
        steps=[CodexStep(returncode=2, stderr="service unavailable\n")],
        calls=[],
    )

    result = run_ticket_lifecycle(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        codex_runner=codex,
        verification_runner=SequencedVerificationRunner(steps=[], calls=[]),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.FAILED
    assert "Codex exited with code 2" in result.run_record.terminal_reason
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
        verification_runner=SequencedVerificationRunner(steps=[], calls=[]),
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
        verification_runner=verification,
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert result.run_record.last_completed_state == WorkflowState.VERIFY
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
        verification_runner=verification,
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.READY_FOR_HUMAN
    assert result.correction_results == ()
    assert result.review_results[0].review_result["verdict"] == "PASS"
    assert [
        item["disposition"]
        for item in result.review_results[0].review_result["findings"]
    ] == ["ADVISORY", "FOLLOW_UP"]


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
    findings: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if findings is None and verdict == ReviewVerdict.CORRECTIONS_REQUIRED.value:
        findings = [finding("R1-F1", disposition="REQUIRED")]
    return {
        "verdict": verdict,
        "summary": "review result",
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
