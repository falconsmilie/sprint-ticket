from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import ticket_automation.review as review_module
from tests.helpers import (
    create_git_repo,
    create_trusted_prepared_run,
    make_config,
)
from ticket_automation.attempts import latest_attempt
from ticket_automation.codex import CodexProcessResult
from ticket_automation.implementation import run_implementation_stage
from ticket_automation.models import WorkflowState
from ticket_automation.review import run_review_stage
from ticket_automation.runs import save_run_record
from ticket_automation.verification import (
    VerificationProcessResult,
    run_verification_stage,
)


def fixed_clock() -> datetime:
    return datetime(2026, 9, 14, 10, 15, tzinfo=UTC)


class CompletingCodexRunner:
    def run(self, command, *, stdin, timeout_seconds):
        del stdin, timeout_seconds
        if command.argv[command.argv.index("--sandbox") + 1] == "read-only":
            result = {"verdict": "PASS", "summary": "passed", "findings": []}
        else:
            command.cwd.joinpath("implemented.txt").write_text(
                "done\n", encoding="utf-8"
            )
            result = {
                "status": "COMPLETED",
                "summary": "implemented",
                "tests_run": [],
                "assumptions": [],
                "known_issues": [],
            }
        Path(command.argv[command.argv.index("--output-last-message") + 1]).write_text(
            json.dumps(result),
            encoding="utf-8",
        )
        return CodexProcessResult(returncode=0, stdout="", stderr="")


class PassingVerificationRunner:
    def run(self, command, *, timeout_seconds):
        del command, timeout_seconds
        return VerificationProcessResult(returncode=0, stdout="", stderr="")


def test_review_prompt_requires_requirement_to_evidence_traceability(
    tmp_path: Path,
) -> None:
    repository = create_git_repo(tmp_path / "target")
    config = make_config(repository)
    ticket = tmp_path / "TA-ARCH-009.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")
    snapshot = create_trusted_prepared_run(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )

    prompt = review_module._render_review_prompt(
        ticket_text="# Ticket\n",
        run_record=snapshot.run_record,
        current_branch="main",
        verification_results="{}",
        implementation_summary="implemented",
    )

    assert "## Evidence-Based Requirement Review" in prompt
    assert "trace every testable ticket requirement and acceptance" in prompt
    normalized_prompt = " ".join(prompt.split())
    assert "trusted or internal direct construction" in normalized_prompt
    assert "leaves input data and observable state unchanged" in normalized_prompt
    assert "supported provenance-assembly boundary" in normalized_prompt


def test_review_snapshot_failure_is_recorded_as_human_required(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = create_git_repo(tmp_path / "target")
    config = make_config(repository)
    ticket = tmp_path / "TA-ARCH-009.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")
    snapshot = create_trusted_prepared_run(
        config,
        ticket,
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

    def fail_capture(repository):
        del repository
        raise OSError("snapshot unavailable")

    monkeypatch.setattr(
        review_module.WorkspaceSnapshot,
        "capture",
        staticmethod(fail_capture),
    )
    result = run_review_stage(config, snapshot.run_dir, clock=fixed_clock)

    attempt = latest_attempt(snapshot.run_dir, phases=(WorkflowState.REVIEWING.value,))
    assert result.outcome.value == "HUMAN_REQUIRED"
    assert attempt is not None
    assert attempt.process_started is False
    assert attempt.status == "HUMAN_REQUIRED"
