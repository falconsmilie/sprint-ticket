from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.helpers import (
    create_git_repo,
    create_trusted_prepared_run,
    make_config,
)
from ticket_automation.attempts import (
    AttemptError,
    attempt_result_path,
    complete_attempt,
    load_attempt_records,
    start_attempt,
)
from ticket_automation.models import WorkflowState
from ticket_automation.workflow import resume_ticket_lifecycle


def fixed_clock() -> datetime:
    return datetime(2026, 9, 14, 10, 15, tzinfo=UTC)


def test_attempt_creation_skips_an_orphaned_crash_directory(tmp_path: Path) -> None:
    orphan = tmp_path / "attempts" / "001-verification"
    orphan.mkdir(parents=True)

    record = start_attempt(
        tmp_path,
        phase="VERIFYING",
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )

    assert record.sequence == 2
    assert record.artifact_directory.name == "002-verification"
    assert [item.sequence for item in load_attempt_records(tmp_path)] == [2]


def test_trusted_attempt_record_resolves_only_its_own_artifacts(tmp_path: Path) -> None:
    started = start_attempt(
        tmp_path,
        phase="VERIFYING",
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )
    completed = complete_attempt(
        started,
        status="COMPLETED",
        after_workspace_fingerprint="after",
        clock=fixed_clock,
    )

    loaded = load_attempt_records(tmp_path)

    assert loaded == (completed,)
    assert attempt_result_path(tmp_path, loaded[0]) == (
        completed.artifact_directory / "result.json"
    )


def test_tampered_attempt_path_is_rejected_before_result_can_escape(
    tmp_path: Path,
) -> None:
    record = start_attempt(
        tmp_path,
        phase="VERIFYING",
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )
    data = json.loads(record.path.read_text(encoding="utf-8"))
    data["result_path"] = "../run.json"
    record.path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(AttemptError, match="result_path"):
        load_attempt_records(tmp_path)


def test_resume_requires_human_inspection_for_invalid_attempt_evidence(
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
    baseline_attempt = load_attempt_records(snapshot.run_dir)[0]
    data = json.loads(baseline_attempt.path.read_text(encoding="utf-8"))
    data["execution_path"] = "/outside-run.json"
    baseline_attempt.path.write_text(json.dumps(data), encoding="utf-8")

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert "Attempt evidence is invalid" in result.run_record.terminal_reason
