from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import ticket_automation.verification as verification_module
from tests.helpers import (
    create_git_repo,
    create_trusted_prepared_run,
    make_config,
)
from ticket_automation.attempts import latest_attempt
from ticket_automation.config import VerificationCommand
from ticket_automation.models import WorkflowState
from ticket_automation.runs import create_run_snapshot, save_run_record
from ticket_automation.verification import (
    VerificationProcessResult,
    _run_baseline_verification_stage,
    run_verification_stage,
)


def fixed_clock() -> datetime:
    return datetime(2026, 9, 14, 10, 15, tzinfo=UTC)


@dataclass
class RecordingVerificationRunner:
    commands: list[tuple[str, ...]]

    def run(self, command, *, timeout_seconds):
        del timeout_seconds
        self.commands.append(command.argv)
        return VerificationProcessResult(returncode=0, stdout="", stderr="")


def test_baseline_verification_uses_the_persisted_resolved_commands(
    tmp_path: Path,
) -> None:
    repository = create_git_repo(tmp_path / "target")
    ticket = tmp_path / "TA-ARCH-009.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")
    original = make_config(repository)
    snapshot = create_run_snapshot(
        original,
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    changed_local_config = make_config(
        repository,
        verification_commands=(
            VerificationCommand(
                name="different",
                argv=("different-tool", "--unexpected"),
                timeout_seconds=30,
            ),
        ),
    )
    runner = RecordingVerificationRunner([])

    result = _run_baseline_verification_stage(
        changed_local_config,
        snapshot.run_dir,
        process_runner=runner,
        clock=fixed_clock,
    )

    assert result.successful
    assert runner.commands == [original.verification.commands[0].argv]


def test_verification_snapshot_failure_is_recorded_without_starting_a_process(
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
    verifying = implementing.transition_to(
        WorkflowState.VERIFYING,
        updated_timestamp="2026-09-14T10:17:00Z",
    )
    save_run_record(verifying, snapshot.run_dir / "run.json")

    def fail_capture(repository):
        del repository
        raise OSError("snapshot unavailable")

    monkeypatch.setattr(
        verification_module.WorkspaceSnapshot,
        "capture",
        staticmethod(fail_capture),
    )
    result = run_verification_stage(config, snapshot.run_dir, clock=fixed_clock)

    attempt = latest_attempt(snapshot.run_dir, phases=(WorkflowState.VERIFYING.value,))
    assert result.outcome.value == "HUMAN_REQUIRED"
    assert attempt is not None
    assert attempt.process_started is False
    assert attempt.status == "HUMAN_REQUIRED"


def test_verification_process_start_failure_is_not_recorded_as_a_started_process(
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
    implementing = snapshot.run_record.transition_to(
        WorkflowState.IMPLEMENTING,
        updated_timestamp="2026-09-14T10:16:00Z",
    )
    verifying = implementing.transition_to(
        WorkflowState.VERIFYING,
        updated_timestamp="2026-09-14T10:17:00Z",
    )
    save_run_record(verifying, snapshot.run_dir / "run.json")

    class UnavailableRunner:
        def run(self, command, *, timeout_seconds):
            del command, timeout_seconds
            raise FileNotFoundError("missing verifier")

    result = run_verification_stage(
        config,
        snapshot.run_dir,
        process_runner=UnavailableRunner(),
        clock=fixed_clock,
    )

    attempt = latest_attempt(snapshot.run_dir, phases=(WorkflowState.VERIFYING.value,))
    assert result.outcome.value == "HUMAN_REQUIRED"
    assert attempt is not None
    assert attempt.process_started is False
