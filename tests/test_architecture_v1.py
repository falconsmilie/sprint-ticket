from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

import ticket_automation.workflow as workflow_module
from tests.helpers import GIT, create_git_repo, create_trusted_prepared_run, make_config
from ticket_automation.attempts import load_attempt_records, start_attempt
from ticket_automation.codex import CodexProcessResult
from ticket_automation.implementation import run_implementation_stage
from ticket_automation.models import WorkflowState
from ticket_automation.reporting import ReportError, run_report_stage
from ticket_automation.review import run_review_stage
from ticket_automation.runs import create_run_snapshot, load_run_record, save_run_record
from ticket_automation.verification import (
    VerificationProcessResult,
    run_verification_stage,
)
from ticket_automation.workflow import resume_ticket_lifecycle, run_ticket_lifecycle


def fixed_clock() -> datetime:
    return datetime(2026, 9, 14, 10, 15, tzinfo=UTC)


@dataclass
class PassingVerificationRunner:
    calls: int = 0

    def run(self, command, *, timeout_seconds):
        del command, timeout_seconds
        self.calls += 1
        return VerificationProcessResult(
            returncode=0,
            stdout="verification passed\n",
            stderr="",
        )


@dataclass
class FailOnceVerificationRunner:
    calls: int = 0

    def run(self, command, *, timeout_seconds):
        del command, timeout_seconds
        self.calls += 1
        return VerificationProcessResult(
            returncode=1 if self.calls == 2 else 0,
            stdout="verification output\n",
            stderr="",
        )


@dataclass
class CompletingCodexRunner:
    calls: int = 0

    def run(self, command, *, stdin, timeout_seconds):
        del stdin, timeout_seconds
        self.calls += 1
        is_review = command.argv[command.argv.index("--sandbox") + 1] == "read-only"
        if is_review:
            result = {
                "verdict": "PASS",
                "summary": "review passed",
                "findings": [],
            }
        else:
            command.cwd.joinpath("file.txt").write_text(
                "implemented\n", encoding="utf-8"
            )
            result = {
                "status": "COMPLETED",
                "summary": "implementation passed",
                "tests_run": [],
                "assumptions": [],
                "known_issues": [],
            }
        output = Path(command.argv[command.argv.index("--output-last-message") + 1])
        output.write_text(json.dumps(result), encoding="utf-8")
        return CodexProcessResult(returncode=0, stdout="", stderr="")


def _ticket(tmp_path: Path) -> Path:
    ticket = tmp_path / "TA-ARCH-009.md"
    ticket.write_text("# Implement the requested change\n", encoding="utf-8")
    return ticket


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_lifecycle_writes_only_numbered_attempt_evidence(tmp_path, monkeypatch):
    repo = create_git_repo(tmp_path / "target")
    runner = PassingVerificationRunner()

    result = run_ticket_lifecycle(
        make_config(repo),
        _ticket(tmp_path),
        runs_dir=tmp_path / "runs",
        verification_runner=runner,
        codex_runner=CompletingCodexRunner(),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.READY_FOR_HUMAN
    assert runner.calls == 2
    attempts = load_attempt_records(result.run_dir)
    assert [(item.sequence, item.phase, item.status) for item in attempts] == [
        (1, "PREPARING", "COMPLETED"),
        (2, "IMPLEMENTING", "COMPLETED"),
        (3, "VERIFYING", "COMPLETED"),
        (4, "REVIEWING", "COMPLETED"),
        (5, "REPORTING", "COMPLETED"),
    ]
    assert [item.artifact_directory.name for item in attempts] == [
        "001-preparation",
        "002-implementation",
        "003-verification",
        "004-review",
        "005-reporting",
    ]
    implementation = attempts[1].artifact_directory
    assert (implementation / "prompt.md").is_file()
    assert (implementation / "events.jsonl").is_file()
    assert (implementation / "stderr.log").is_file()
    assert (implementation / "result.json").is_file()
    execution_metadata = json.loads((implementation / "execution.json").read_text())
    assert execution_metadata["workspace_guard"]["new_environments"] == []
    assert "workspace_guard" not in attempts[1].metadata
    verification = attempts[2].artifact_directory
    verification_result = json.loads((verification / "result.json").read_text())
    assert verification_result["commands"][0]["stdout"] == "verification passed\n"
    assert not list(verification.glob("*.log"))
    assert (result.run_dir / "final.patch").is_file()
    assert (result.run_dir / "report.md").is_file()
    assert not any(
        (result.run_dir / name).exists()
        for name in (
            "baseline-verification",
            "implementation",
            "verification",
            "reviews",
            "correction-executions",
            "corrections",
            "diffs",
            "writable-attempts",
        )
    )


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_resume_reruns_safe_verification_as_a_new_attempt(tmp_path, monkeypatch):
    repo = create_git_repo(tmp_path / "target")
    config = make_config(repo)
    snapshot = create_trusted_prepared_run(
        config,
        _ticket(tmp_path),
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    implementing = snapshot.run_record.transition_to(
        WorkflowState.IMPLEMENTING,
        updated_timestamp="2026-09-14T10:16:00Z",
    )
    save_run_record(implementing, snapshot.run_dir / "run.json")
    implementation = run_implementation_stage(
        config,
        snapshot.run_dir,
        codex_runner=CompletingCodexRunner(),
        clock=fixed_clock,
    )
    assert implementation.successful
    verifying = implementation.run_record.transition_to(
        WorkflowState.VERIFYING,
        updated_timestamp="2026-09-14T10:17:00Z",
    )
    save_run_record(verifying, snapshot.run_dir / "run.json")
    first_verification = PassingVerificationRunner()
    assert run_verification_stage(
        config,
        snapshot.run_dir,
        process_runner=first_verification,
        clock=fixed_clock,
    ).successful

    resumed = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=tmp_path / "runs",
        verification_runner=PassingVerificationRunner(),
        codex_runner=CompletingCodexRunner(),
        clock=fixed_clock,
    )

    assert resumed.run_record.state == WorkflowState.READY_FOR_HUMAN
    verification_attempts = [
        item
        for item in load_attempt_records(snapshot.run_dir)
        if item.phase == "VERIFYING"
    ]
    assert len(verification_attempts) == 2
    assert all(item.status == "COMPLETED" for item in verification_attempts)


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_interrupted_writable_state_requires_human_inspection(tmp_path):
    repo = create_git_repo(tmp_path / "target")
    config = make_config(repo)
    snapshot = create_trusted_prepared_run(
        config,
        _ticket(tmp_path),
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    interrupted = snapshot.run_record.transition_to(
        WorkflowState.IMPLEMENTING,
        updated_timestamp="2026-09-14T10:16:00Z",
    )
    save_run_record(interrupted, snapshot.run_dir / "run.json")

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert (
        "Writable implementation was interrupted" in result.run_record.terminal_reason
    )
    assert (result.run_dir / "report.md").is_file()


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_reporting_renders_without_changing_controller_state(tmp_path, monkeypatch):
    repo = create_git_repo(tmp_path / "target")
    result = run_ticket_lifecycle(
        make_config(repo),
        _ticket(tmp_path),
        runs_dir=tmp_path / "runs",
        verification_runner=PassingVerificationRunner(),
        codex_runner=CompletingCodexRunner(),
        clock=fixed_clock,
    )
    before = load_run_record(result.run_dir / "run.json")

    report = run_report_stage(result.run_dir)

    after = load_run_record(result.run_dir / "run.json")
    assert report.run_record == before
    assert after == before


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_resume_restarts_preparation_in_a_new_attempt(tmp_path):
    repository = create_git_repo(tmp_path / "target")
    ticket = _ticket(tmp_path)
    snapshot = create_run_snapshot(
        make_config(repository),
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    start_attempt(
        snapshot.run_dir,
        phase=WorkflowState.PREPARING.value,
        before_workspace_fingerprint=snapshot.run_record.baseline_sha,
        clock=fixed_clock,
    )

    result = resume_ticket_lifecycle(
        make_config(repository),
        snapshot.run_record.run_id,
        runs_dir=tmp_path / "runs",
        verification_runner=PassingVerificationRunner(),
        codex_runner=CompletingCodexRunner(),
        clock=fixed_clock,
    )

    preparation_attempts = [
        item
        for item in load_attempt_records(snapshot.run_dir)
        if item.phase == WorkflowState.PREPARING.value
    ]
    assert result.run_record.state == WorkflowState.READY_FOR_HUMAN
    assert [item.status for item in preparation_attempts] == ["STARTED", "COMPLETED"]


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_resume_restarts_review_in_a_new_attempt(tmp_path):
    repository = create_git_repo(tmp_path / "target")
    config = make_config(repository)
    snapshot = create_trusted_prepared_run(
        config,
        _ticket(tmp_path),
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    implementing = snapshot.run_record.transition_to(
        WorkflowState.IMPLEMENTING,
        updated_timestamp="2026-09-14T10:16:00Z",
    )
    save_run_record(implementing, snapshot.run_dir / "run.json")
    implementation = run_implementation_stage(
        config,
        snapshot.run_dir,
        codex_runner=CompletingCodexRunner(),
        clock=fixed_clock,
    )
    verifying = implementation.run_record.transition_to(
        WorkflowState.VERIFYING,
        updated_timestamp="2026-09-14T10:17:00Z",
    )
    save_run_record(verifying, snapshot.run_dir / "run.json")
    verification = run_verification_stage(
        config,
        snapshot.run_dir,
        process_runner=PassingVerificationRunner(),
        clock=fixed_clock,
    )
    reviewing = verification.run_record.transition_to(
        WorkflowState.REVIEWING,
        updated_timestamp="2026-09-14T10:18:00Z",
    )
    save_run_record(reviewing, snapshot.run_dir / "run.json")
    start_attempt(
        snapshot.run_dir,
        phase=WorkflowState.REVIEWING.value,
        before_workspace_fingerprint=None,
        clock=fixed_clock,
    )

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=tmp_path / "runs",
        verification_runner=PassingVerificationRunner(),
        codex_runner=CompletingCodexRunner(),
        clock=fixed_clock,
    )

    review_attempts = [
        item
        for item in load_attempt_records(snapshot.run_dir)
        if item.phase == WorkflowState.REVIEWING.value
    ]
    assert result.run_record.state == WorkflowState.READY_FOR_HUMAN
    assert [item.status for item in review_attempts] == ["STARTED", "COMPLETED"]


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_resume_restarts_reporting_in_a_new_attempt(tmp_path):
    repository = create_git_repo(tmp_path / "target")
    config = make_config(repository)
    snapshot = create_trusted_prepared_run(
        config,
        _ticket(tmp_path),
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    implementing = snapshot.run_record.transition_to(
        WorkflowState.IMPLEMENTING,
        updated_timestamp="2026-09-14T10:16:00Z",
    )
    save_run_record(implementing, snapshot.run_dir / "run.json")
    implementation = run_implementation_stage(
        config,
        snapshot.run_dir,
        codex_runner=CompletingCodexRunner(),
        clock=fixed_clock,
    )
    verifying = implementation.run_record.transition_to(
        WorkflowState.VERIFYING,
        updated_timestamp="2026-09-14T10:17:00Z",
    )
    save_run_record(verifying, snapshot.run_dir / "run.json")
    verification = run_verification_stage(
        config,
        snapshot.run_dir,
        process_runner=PassingVerificationRunner(),
        clock=fixed_clock,
    )
    reviewing = verification.run_record.transition_to(
        WorkflowState.REVIEWING,
        updated_timestamp="2026-09-14T10:18:00Z",
    )
    save_run_record(reviewing, snapshot.run_dir / "run.json")
    review = run_review_stage(
        config,
        snapshot.run_dir,
        codex_runner=CompletingCodexRunner(),
        clock=fixed_clock,
    )
    reporting = review.run_record.transition_to(
        WorkflowState.REPORTING,
        updated_timestamp="2026-09-14T10:19:00Z",
    )
    save_run_record(reporting, snapshot.run_dir / "run.json")
    start_attempt(
        snapshot.run_dir,
        phase=WorkflowState.REPORTING.value,
        before_workspace_fingerprint=None,
        clock=fixed_clock,
    )

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=tmp_path / "runs",
        verification_runner=PassingVerificationRunner(),
        codex_runner=CompletingCodexRunner(),
        clock=fixed_clock,
    )

    reporting_attempts = [
        item
        for item in load_attempt_records(snapshot.run_dir)
        if item.phase == WorkflowState.REPORTING.value
    ]
    assert result.run_record.state == WorkflowState.READY_FOR_HUMAN
    assert [item.status for item in reporting_attempts] == ["STARTED", "COMPLETED"]


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_rendering_failure_cannot_reclassify_an_accepted_handoff(tmp_path, monkeypatch):
    repository = create_git_repo(tmp_path / "target")

    def fail_renderer(run_dir):
        del run_dir
        raise ReportError("disk unavailable")

    monkeypatch.setattr(workflow_module, "run_report_stage", fail_renderer)
    result = run_ticket_lifecycle(
        make_config(repository),
        _ticket(tmp_path),
        runs_dir=tmp_path / "runs",
        verification_runner=PassingVerificationRunner(),
        codex_runner=CompletingCodexRunner(),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.READY_FOR_HUMAN
    assert (result.run_dir / "final.patch").is_file()


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_terminal_report_describes_a_failed_baseline_attempt(tmp_path):
    repository = create_git_repo(tmp_path / "target")

    class FailingBaselineRunner:
        def run(self, command, *, timeout_seconds):
            del command, timeout_seconds
            return VerificationProcessResult(returncode=1, stdout="failed", stderr="")

    result = run_ticket_lifecycle(
        make_config(repository),
        _ticket(tmp_path),
        runs_dir=tmp_path / "runs",
        verification_runner=FailingBaselineRunner(),
        clock=fixed_clock,
    )

    report = (result.run_dir / "report.md").read_text(encoding="utf-8")
    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert "- Baseline verification: FAIL" in report


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_one_correction_round_preserves_the_ordinary_lifecycle(tmp_path):
    repository = create_git_repo(tmp_path / "target")
    verification = FailOnceVerificationRunner()

    result = run_ticket_lifecycle(
        make_config(repository),
        _ticket(tmp_path),
        runs_dir=tmp_path / "runs",
        verification_runner=verification,
        codex_runner=CompletingCodexRunner(),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.READY_FOR_HUMAN
    assert result.run_record.current_correction_round == 1
    assert verification.calls == 3
    assert [item.phase for item in load_attempt_records(result.run_dir)] == [
        "PREPARING",
        "IMPLEMENTING",
        "VERIFYING",
        "CORRECTING",
        "VERIFYING",
        "REVIEWING",
        "REPORTING",
    ]
