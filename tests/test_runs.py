from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import ticket_automation.runs as runs_module
from tests.helpers import GIT, create_git_repo, make_config, run_git
from ticket_automation.models import StopCategory, StopReason, WorkflowState
from ticket_automation.runs import (
    BASELINE_RECORD_FORMAT,
    RUN_RECORD_FORMAT,
    RUN_SCHEMA_VERSION,
    RunError,
    RunPreflightError,
    RunRecord,
    TicketInputError,
    create_run_snapshot,
    list_run_records,
    load_baseline_record,
    load_run_record,
    sanitize_ticket_id,
    save_run_record,
)


def fixed_clock() -> datetime:
    return datetime(2026, 9, 11, 13, 5, 12, tzinfo=UTC)


_ACTIVE_STATES = (
    WorkflowState.PREPARING,
    WorkflowState.PREPARED,
    WorkflowState.IMPLEMENTING,
    WorkflowState.VERIFYING,
    WorkflowState.REVIEWING,
    WorkflowState.CORRECTION_PENDING,
    WorkflowState.CORRECTING,
    WorkflowState.REPORTING,
)
_NORMAL_AND_CORRECTION_TRANSITIONS = (
    (WorkflowState.PREPARING, WorkflowState.PREPARED),
    (WorkflowState.PREPARED, WorkflowState.IMPLEMENTING),
    (WorkflowState.IMPLEMENTING, WorkflowState.VERIFYING),
    (WorkflowState.VERIFYING, WorkflowState.REVIEWING),
    (WorkflowState.VERIFYING, WorkflowState.CORRECTION_PENDING),
    (WorkflowState.REVIEWING, WorkflowState.REPORTING),
    (WorkflowState.REVIEWING, WorkflowState.CORRECTION_PENDING),
    (WorkflowState.CORRECTION_PENDING, WorkflowState.CORRECTING),
    (WorkflowState.CORRECTING, WorkflowState.VERIFYING),
    (WorkflowState.REPORTING, WorkflowState.READY_FOR_HUMAN),
)
_VALID_TRANSITIONS = (
    *_NORMAL_AND_CORRECTION_TRANSITIONS,
    *((state, WorkflowState.HUMAN_REQUIRED) for state in _ACTIVE_STATES),
    *((state, WorkflowState.FAILED) for state in _ACTIVE_STATES),
)


@pytest.mark.skipif(GIT is None, reason="git executable is required for run tests")
def test_valid_run_directory_creation(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "QDEB-003.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")

    result = create_run_snapshot(
        make_config(repo),
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )

    assert result.run_dir.name == "20260911-130512_QDEB-003"
    assert result.run_dir.joinpath("run.json").is_file()
    assert result.run_dir.joinpath("ticket.md").is_file()
    assert result.run_dir.joinpath("baseline.json").is_file()
    assert result.run_record.state == WorkflowState.PREPARING
    assert "last_completed_state" not in json.loads(
        result.run_dir.joinpath("run.json").read_text(encoding="utf-8")
    )


@pytest.mark.skipif(GIT is None, reason="git executable is required for run tests")
def test_ticket_is_copied_byte_for_byte(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    contents = b"# QDEB-003\r\n\r\nPreserve these bytes.  \r\n"
    ticket = tmp_path / "QDEB-003.md"
    ticket.write_bytes(contents)

    result = create_run_snapshot(
        make_config(repo),
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )

    assert result.run_dir.joinpath("ticket.md").read_bytes() == contents


@pytest.mark.skipif(GIT is None, reason="git executable is required for run tests")
def test_branch_and_sha_are_recorded(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "QDEB-003.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")
    expected_branch = run_git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    expected_sha = run_git(repo, "rev-parse", "--verify", "HEAD")
    expected_status = run_git(repo, "status", "--short")

    result = create_run_snapshot(
        make_config(repo),
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    run_record = load_run_record(result.run_dir / "run.json")
    baseline_record = load_baseline_record(result.run_dir / "baseline.json")

    assert run_record.starting_branch == expected_branch
    assert run_record.baseline_sha == expected_sha
    assert baseline_record.branch == expected_branch
    assert baseline_record.head_sha == expected_sha
    assert baseline_record.clean_worktree is True
    assert baseline_record.has_staged_files is False
    assert baseline_record.staging_status == "clean"
    assert run_git(repo, "status", "--short") == expected_status


@pytest.mark.skipif(GIT is None, reason="git executable is required for run tests")
def test_run_json_survives_load_save_round_trip(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "QDEB-003.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")
    result = create_run_snapshot(
        make_config(repo),
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    round_trip_path = tmp_path / "round-trip.json"

    save_run_record(result.run_record, round_trip_path)

    assert load_run_record(round_trip_path) == result.run_record


@pytest.mark.parametrize(("source", "target"), _VALID_TRANSITIONS)
def test_run_record_accepts_every_legal_transition(source, target):
    run_record = trusted_run_record(source)
    stop_reason = terminal_stop_reason(target)

    transitioned = run_record.transition_to(
        target,
        updated_timestamp="2026-09-11T13:05:13Z",
        terminal_reason=None if stop_reason is None else stop_reason.message,
        stop_reason=stop_reason,
    )

    assert transitioned.state == target
    assert run_record.state == source


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (WorkflowState.PREPARING, WorkflowState.IMPLEMENTING),
        (WorkflowState.PREPARED, WorkflowState.VERIFYING),
        (WorkflowState.VERIFYING, WorkflowState.REPORTING),
        (WorkflowState.CORRECTION_PENDING, WorkflowState.VERIFYING),
        (WorkflowState.READY_FOR_HUMAN, WorkflowState.PREPARING),
        (WorkflowState.HUMAN_REQUIRED, WorkflowState.FAILED),
        (WorkflowState.FAILED, WorkflowState.HUMAN_REQUIRED),
    ],
)
def test_run_record_rejects_representative_invalid_transitions(source, target):
    with pytest.raises(ValueError, match=rf"{source.value} -> {target.value}"):
        trusted_run_record(source).transition_to(
            target,
            updated_timestamp="2026-09-11T13:05:13Z",
        )


@pytest.mark.parametrize("state", list(WorkflowState))
def test_run_record_accepts_each_state_at_trusted_deserialization_boundary(state):
    assert trusted_run_record(state).state == state


@pytest.mark.parametrize(
    "state",
    (WorkflowState.HUMAN_REQUIRED, WorkflowState.FAILED),
)
def test_terminal_stop_requires_typed_reason(state):
    with pytest.raises(ValueError, match="require stop_reason"):
        trusted_run_record(WorkflowState.PREPARING).transition_to(
            state,
            updated_timestamp="2026-09-11T13:05:13Z",
        )


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"stop_reason": None}, "require stop_reason"),
        (
            {
                "stop_reason": {
                    "category": "UNSUPPORTED",
                    "message": "stop",
                    "retryable": False,
                }
            },
            "unsupported stop category",
        ),
        (
            {
                "stop_reason": {
                    "category": "CONTROLLER_FAILURE",
                    "message": "stop",
                    "retryable": "no",
                }
            },
            "boolean: stop_reason.retryable",
        ),
        (
            {
                "terminal_reason": "different message",
            },
            "must exactly match",
        ),
    ],
)
def test_terminal_stop_record_rejects_missing_or_inconsistent_evidence(patch, message):
    data = trusted_run_record(WorkflowState.FAILED).to_dict()
    data.update(patch)

    with pytest.raises(RunError, match=message):
        RunRecord.from_dict(data)


@pytest.mark.skipif(GIT is None, reason="git executable is required for run tests")
def test_old_run_schema_instructs_operator_to_start_a_new_run(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "QDEB-003.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")
    result = create_run_snapshot(
        make_config(repo),
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    record_path = result.run_dir / "run.json"
    data = json.loads(record_path.read_text(encoding="utf-8"))
    data["schema_version"] = 1
    data["state"] = "SNAPSHOT"
    data["last_completed_state"] = "SNAPSHOT"
    record_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(RunError, match="Start a new run"):
        load_run_record(record_path)


@pytest.mark.skipif(GIT is None, reason="git executable is required for run tests")
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 999, "Unsupported run record schema version"),
        ("format", "ticket_automation.future_run", "Unsupported format"),
    ],
)
def test_run_record_rejects_unsupported_schema_metadata(
    tmp_path,
    field,
    value,
    message,
):
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "QDEB-003.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")
    result = create_run_snapshot(
        make_config(repo),
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    record_path = result.run_dir / "run.json"
    data = json.loads(record_path.read_text(encoding="utf-8"))
    data[field] = value
    record_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(RunError, match=message):
        load_run_record(record_path)


@pytest.mark.skipif(GIT is None, reason="git executable is required for run tests")
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 999, "Unsupported schema_version"),
        ("format", "ticket_automation.future_baseline", "Unsupported format"),
    ],
)
def test_baseline_record_rejects_unsupported_schema_metadata(
    tmp_path,
    field,
    value,
    message,
):
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "QDEB-003.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")
    result = create_run_snapshot(
        make_config(repo),
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    record_path = result.run_dir / "baseline.json"
    data = json.loads(record_path.read_text(encoding="utf-8"))
    data[field] = value
    record_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(RunError, match=message):
        load_baseline_record(record_path)


@pytest.mark.skipif(GIT is None, reason="git executable is required for run tests")
@pytest.mark.parametrize(
    "field",
    [
        "ticket_sha256",
        "verification_commands_fingerprint",
        "workspace_fingerprint",
    ],
)
@pytest.mark.parametrize("value", ["too-short", "A" * 64, "g" * 64])
def test_baseline_record_rejects_invalid_sha256_evidence(tmp_path, field, value):
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "QDEB-003.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")
    result = create_run_snapshot(
        make_config(repo),
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    record_path = result.run_dir / "baseline.json"
    data = json.loads(record_path.read_text(encoding="utf-8"))
    data[field] = value
    record_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(RunError, match="lowercase SHA-256 digest"):
        load_baseline_record(record_path)


@pytest.mark.skipif(GIT is None, reason="git executable is required for run tests")
def test_record_schema_metadata_uses_current_supported_formats(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "QDEB-003.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")

    result = create_run_snapshot(
        make_config(repo),
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    run_data = json.loads(
        result.run_dir.joinpath("run.json").read_text(encoding="utf-8")
    )
    baseline_data = json.loads(
        result.run_dir.joinpath("baseline.json").read_text(encoding="utf-8")
    )

    assert run_data["schema_version"] == 3
    assert run_data["format"] == RUN_RECORD_FORMAT
    assert baseline_data["schema_version"] == 2
    assert baseline_data["format"] == BASELINE_RECORD_FORMAT


@pytest.mark.skipif(GIT is None, reason="git executable is required for run tests")
def test_correction_round_and_maximum_are_persisted(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "QDEB-003.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")

    result = create_run_snapshot(
        make_config(repo, max_correction_rounds=7),
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    data = json.loads(result.run_dir.joinpath("run.json").read_text(encoding="utf-8"))

    assert data["current_correction_round"] == 0
    assert data["max_correction_rounds"] == 7
    assert data["current_review_round"] == 0
    assert "last_completed_state" not in data
    assert data["terminal_reason"] is None
    assert data["stop_reason"] is None


@pytest.mark.skipif(GIT is None, reason="git executable is required for run tests")
def test_codex_execution_config_is_persisted_in_run_record(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "QDEB-003.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")

    result = create_run_snapshot(
        make_config(
            repo,
            codex_model="configured-model",
            codex_reasoning_effort="high",
        ),
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    data = json.loads(result.run_dir.joinpath("run.json").read_text(encoding="utf-8"))

    assert data["codex"] == {
        "model": "configured-model",
        "reasoning_effort": "high",
    }
    assert load_run_record(result.run_dir / "run.json").codex.model == (
        "configured-model"
    )
    assert load_run_record(result.run_dir / "run.json").codex.reasoning_effort == (
        "high"
    )


@pytest.mark.skipif(GIT is None, reason="git executable is required for run tests")
def test_run_record_loads_optional_lifecycle_fields(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "QDEB-003.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")
    result = create_run_snapshot(
        make_config(repo),
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    record_path = result.run_dir / "run.json"
    data = json.loads(record_path.read_text(encoding="utf-8"))
    del data["current_review_round"]
    del data["terminal_reason"]
    record_path.write_text(json.dumps(data), encoding="utf-8")

    record = load_run_record(record_path)

    assert not hasattr(record, "last_completed_state")
    assert record.current_review_round == 0
    assert record.terminal_reason is None


@pytest.mark.skipif(GIT is None, reason="git executable is required for run tests")
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "state",
            "UNSUPPORTED",
            "unsupported workflow state: state",
        ),
        (
            "current_correction_round",
            -1,
            "non-negative integer: current_correction_round",
        ),
        (
            "current_review_round",
            -1,
            "non-negative integer: current_review_round",
        ),
    ],
)
def test_run_record_rejects_malformed_or_negative_lifecycle_fields(
    tmp_path,
    field,
    value,
    message,
):
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "QDEB-003.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")
    result = create_run_snapshot(
        make_config(repo),
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    record_path = result.run_dir / "run.json"
    data = json.loads(record_path.read_text(encoding="utf-8"))
    data[field] = value
    record_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(RunError, match=message):
        load_run_record(record_path)


@pytest.mark.skipif(GIT is None, reason="git executable is required for run tests")
def test_invalid_ticket_input_fails_before_creating_misleading_run(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    runs_dir = tmp_path / "runs"

    with pytest.raises(TicketInputError):
        create_run_snapshot(
            make_config(repo),
            tmp_path / "missing.md",
            runs_dir=runs_dir,
            clock=fixed_clock,
        )

    assert not runs_dir.exists()


@pytest.mark.skipif(GIT is None, reason="git executable is required for run tests")
def test_empty_ticket_input_fails_before_creating_misleading_run(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "empty.md"
    ticket.write_bytes(b"")
    runs_dir = tmp_path / "runs"

    with pytest.raises(TicketInputError):
        create_run_snapshot(
            make_config(repo),
            ticket,
            runs_dir=runs_dir,
            clock=fixed_clock,
        )

    assert not runs_dir.exists()


@pytest.mark.skipif(GIT is None, reason="git executable is required for run tests")
def test_dirty_repository_fails_during_preflight_without_run_directory(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "QDEB-003.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")
    repo.joinpath("file.txt").write_text("dirty\n", encoding="utf-8")
    runs_dir = tmp_path / "runs"

    with pytest.raises(RunPreflightError):
        create_run_snapshot(
            make_config(repo),
            ticket,
            runs_dir=runs_dir,
            clock=fixed_clock,
        )

    assert not runs_dir.exists()


@pytest.mark.skipif(GIT is None, reason="git executable is required for run tests")
def test_snapshot_persistence_failure_does_not_leave_listed_partial_run(
    tmp_path,
    monkeypatch,
):
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "QDEB-003.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")
    runs_dir = tmp_path / "runs"

    def fail_save_baseline(*args, **kwargs) -> None:
        raise OSError("cannot persist baseline")

    monkeypatch.setattr(runs_module, "save_baseline_record", fail_save_baseline)

    with pytest.raises(OSError, match="cannot persist baseline"):
        create_run_snapshot(
            make_config(repo),
            ticket,
            runs_dir=runs_dir,
            clock=fixed_clock,
        )

    assert runs_dir.is_dir()
    assert not any(runs_dir.iterdir())
    assert list_run_records(runs_dir) == ()


@pytest.mark.skipif(GIT is None, reason="git executable is required for run tests")
def test_existing_run_directories_are_not_silently_clobbered(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "QDEB-003.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")
    runs_dir = tmp_path / "runs"
    existing = runs_dir / "20260911-130512_QDEB-003"
    existing.mkdir(parents=True)
    existing.joinpath("run.json").write_text('{"sentinel": true}\n', encoding="utf-8")

    result = create_run_snapshot(
        make_config(repo),
        ticket,
        runs_dir=runs_dir,
        clock=fixed_clock,
    )

    assert result.run_dir.name == "20260911-130512_QDEB-003_2"
    assert json.loads(existing.joinpath("run.json").read_text(encoding="utf-8")) == {
        "sentinel": True
    }


@pytest.mark.skipif(GIT is None, reason="git executable is required for run tests")
def test_atomic_run_persistence_keeps_previous_record_when_replace_fails(
    tmp_path,
    monkeypatch,
):
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "QDEB-003.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")
    result = create_run_snapshot(
        make_config(repo),
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    record_path = result.run_dir / "run.json"
    original_record = result.run_record
    changed_record = original_record.transition_to(
        WorkflowState.PREPARED,
        updated_timestamp="2026-09-11T13:05:13Z",
    )

    def fail_replace(source: Path | str, destination: Path | str) -> None:
        raise OSError(f"cannot replace {source} -> {destination}")

    monkeypatch.setattr(runs_module.os, "replace", fail_replace)

    with pytest.raises(OSError):
        save_run_record(changed_record, record_path)

    assert load_run_record(record_path) == original_record
    assert not tuple(result.run_dir.glob("*.tmp"))


@pytest.mark.skipif(GIT is None, reason="git executable is required for run tests")
def test_status_can_read_multiple_run_records(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    first_ticket = tmp_path / "QDEB-003.md"
    second_ticket = tmp_path / "QDEB-004.md"
    first_ticket.write_text("# First\n", encoding="utf-8")
    second_ticket.write_text("# Second\n", encoding="utf-8")
    runs_dir = tmp_path / "runs"

    create_run_snapshot(
        make_config(repo), first_ticket, runs_dir=runs_dir, clock=fixed_clock
    )
    create_run_snapshot(
        make_config(repo), second_ticket, runs_dir=runs_dir, clock=fixed_clock
    )

    records = list_run_records(runs_dir)

    assert {record.ticket_id for record in records} == {"QDEB-003", "QDEB-004"}
    assert {record.state for record in records} == {WorkflowState.PREPARING}


def test_ticket_id_is_sanitized_for_paths():
    assert sanitize_ticket_id(" QDEB 003: first pass ") == "QDEB-003-first-pass"
    assert sanitize_ticket_id("...") == "ticket"


def trusted_run_record(state: WorkflowState) -> RunRecord:
    stop_reason = terminal_stop_reason(state)
    return RunRecord.from_dict(
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "format": RUN_RECORD_FORMAT,
            "run_id": "trusted-run",
            "ticket_id": "TA-ARCH-001",
            "original_ticket_path": "ticket.md",
            "run_ticket_copy_path": "runs/trusted-run/ticket.md",
            "target_repository_path": ".",
            "state": state.value,
            "starting_branch": "feature/state-machine",
            "baseline_sha": "abc123",
            "current_correction_round": 0,
            "max_correction_rounds": 3,
            "current_review_round": 0,
            "codex": {"model": "test-model", "reasoning_effort": "high"},
            "terminal_reason": None if stop_reason is None else stop_reason.message,
            "stop_reason": (
                None
                if stop_reason is None
                else {
                    "category": stop_reason.category.value,
                    "message": stop_reason.message,
                    "retryable": stop_reason.retryable,
                }
            ),
            "created_timestamp": "2026-09-11T13:05:12Z",
            "updated_timestamp": "2026-09-11T13:05:12Z",
        }
    )


def terminal_stop_reason(state: WorkflowState) -> StopReason | None:
    if state not in {WorkflowState.HUMAN_REQUIRED, WorkflowState.FAILED}:
        return None
    return StopReason(
        category=StopCategory.CONTROLLER_FAILURE,
        message=f"Trusted terminal stop: {state.value}.",
        retryable=False,
    )
