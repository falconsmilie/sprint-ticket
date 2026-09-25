from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.fake_agent_executor import InMemoryAgentExecutor
from tests.helpers import (
    GIT,
    create_directory_link,
    create_git_repo,
    create_trusted_prepared_run,
    fail_stat_for_path,
    make_agent_executors,
    make_config,
    make_final_patch_capture,
    make_report_publisher,
    make_resume_agent_executor_factory,
    remove_directory_link,
)
from ticket_automation.application.agent_execution import (
    IMPLEMENTATION_RESULT_CONTRACT,
    AgentExecutionPolicy,
    AgentExecutionRequest,
    AgentTaskKind,
    ArtifactReference,
    ArtifactRole,
    NetworkAccess,
    RepositoryAccess,
    required_execution_capabilities,
)
from ticket_automation.application.guarded_writable_operation import (
    GuardedWritableOperation,
    GuardedWritableRequest,
    WritableBaseline,
    WritableSucceeded,
)
from ticket_automation.application.lifecycle.resume import resume_preflight_problem
from ticket_automation.attempts import (
    ATTEMPT_RECORD_SCHEMA_VERSION,
    AttemptError,
    StageAttempt,
    attempt_result_path,
    complete_attempt,
    complete_stage_attempt,
    latest_attempt,
    load_attempt_records,
    start_attempt,
    update_attempt,
)
from ticket_automation.corrections import (
    CorrectionCauseSet,
    VerificationCorrectionCause,
    run_correction_stage,
)
from ticket_automation.domain.task_results import (
    ImplementationResult,
    ImplementationStatus,
)
from ticket_automation.git import GitRepository
from ticket_automation.git_safety import WorkspaceSnapshot
from ticket_automation.implementation import run_implementation_stage
from ticket_automation.models import (
    AttemptPhase,
    AttemptStatus,
    StageOutcome,
    WorkflowState,
)
from ticket_automation.persistence_codecs import (
    PersistenceCodecError,
    read_review_result,
    read_stage_message_result,
    write_stage_message_result,
)
from ticket_automation.run_ownership import RunOwnership, RunOwnershipError
from ticket_automation.runs import RunError, save_run_record
from ticket_automation.verification_evidence import read_verification_evidence
from ticket_automation.workflow import resume_ticket_lifecycle


def fixed_clock() -> datetime:
    return datetime(2026, 9, 14, 10, 15, tzinfo=UTC)


def test_start_attempt_rejects_a_linked_attempts_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    external = tmp_path / "external-attempts"
    run_dir.mkdir()
    external.mkdir()
    attempts_root = run_dir / "attempts"
    link_kind = create_directory_link(attempts_root, external)
    if link_kind is None:
        pytest.skip("directory links are unavailable on this platform")
    try:
        with pytest.raises(AttemptError, match="owning run|safely"):
            start_attempt(
                run_dir,
                phase=AttemptPhase.VERIFYING,
                before_workspace_fingerprint=None,
                clock=fixed_clock,
            )
        assert tuple(external.iterdir()) == ()
    finally:
        remove_directory_link(attempts_root)


def test_attempt_update_rejects_a_retargeted_attempts_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    record = start_attempt(
        run_dir,
        phase=AttemptPhase.REVIEWING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )
    attempts_root = record.artifact_directory.parent
    external = tmp_path / "external-attempts"
    attempts_root.rename(external)
    external_record = external / record.artifact_directory.name / "attempt.json"
    original = external_record.read_bytes()
    link_kind = create_directory_link(attempts_root, external)
    if link_kind is None:
        pytest.skip("directory links are unavailable on this platform")
    try:
        with pytest.raises(AttemptError, match="owning run|confinement"):
            update_attempt(record, process_started=True)
        assert external_record.read_bytes() == original
    finally:
        remove_directory_link(attempts_root)


def test_attempt_creation_and_update_reject_a_retargeted_owned_run(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "owned-run"
    run_dir.mkdir(parents=True)
    ownership = RunOwnership.acquire(runs_dir, run_dir.name)
    record = start_attempt(
        run_dir,
        phase=AttemptPhase.REVIEWING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
        run_ownership=ownership,
    )
    moved_run = tmp_path / "moved-run"
    run_dir.rename(moved_run)
    moved_record = (
        moved_run / "attempts" / record.artifact_directory.name / "attempt.json"
    )
    original = moved_record.read_bytes()
    link_kind = create_directory_link(run_dir, moved_run)
    if link_kind is None:
        moved_run.rename(run_dir)
        pytest.skip("directory links are unavailable on this platform")
    try:
        with pytest.raises(RunOwnershipError, match="ownership was lost"):
            start_attempt(
                run_dir,
                phase=AttemptPhase.VERIFYING,
                before_workspace_fingerprint="before",
                clock=fixed_clock,
                run_ownership=ownership,
            )
        with pytest.raises(RunOwnershipError, match="ownership was lost"):
            update_attempt(
                record,
                process_started=True,
                run_ownership=ownership,
            )
        assert moved_record.read_bytes() == original
        assert [item.name for item in (moved_run / "attempts").iterdir()] == [
            record.artifact_directory.name
        ]
    finally:
        remove_directory_link(run_dir)


@pytest.mark.skipif(GIT is None, reason="git executable is required")
@pytest.mark.parametrize("run_id_shape", ["absolute", "traversal"])
def test_resume_rejects_a_run_outside_the_configured_runs_root(
    tmp_path: Path,
    run_id_shape: str,
) -> None:
    repository = create_git_repo(tmp_path / "target")
    config = make_config(repository)
    ticket = tmp_path / "TA-REV-001.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")
    external_runs = tmp_path / "external-runs"
    snapshot = create_trusted_prepared_run(
        config,
        ticket,
        runs_dir=external_runs,
        clock=fixed_clock,
    )
    configured_runs = tmp_path / "configured-runs"
    configured_runs.mkdir()
    run_id = (
        str(snapshot.run_dir)
        if run_id_shape == "absolute"
        else os.path.relpath(snapshot.run_dir, configured_runs)
    )
    attempts_before = tuple((snapshot.run_dir / "attempts").iterdir())

    with pytest.raises(RunError, match="Run ID"):
        resume_ticket_lifecycle(
            run_id,
            runs_dir=configured_runs,
            agent_executor_factory=object(),  # type: ignore[arg-type]
            final_patch_capture=make_final_patch_capture(),
            report_publisher=make_report_publisher(),
            clock=fixed_clock,
        )

    assert tuple((snapshot.run_dir / "attempts").iterdir()) == attempts_before


def test_complete_stage_attempt_rejects_wrong_run_and_recompletion(
    tmp_path: Path,
) -> None:
    first_run = tmp_path / "first-run"
    second_run = tmp_path / "second-run"
    started = start_attempt(
        first_run,
        phase=AttemptPhase.IMPLEMENTING,
        before_workspace_fingerprint=None,
        clock=fixed_clock,
    )

    with pytest.raises(AttemptError, match="does not belong"):
        complete_stage_attempt(
            second_run,
            started,
            stage_outcome=StageOutcome.COMPLETED,
            after_workspace_fingerprint="after",
            process_started=False,
            clock=fixed_clock,
        )

    completed = complete_stage_attempt(
        first_run,
        started,
        stage_outcome=StageOutcome.COMPLETED,
        after_workspace_fingerprint="after",
        process_started=False,
        clock=fixed_clock,
    )
    assert completed.status is AttemptStatus.COMPLETED
    with pytest.raises(AttemptError, match="active started attempt"):
        complete_stage_attempt(
            first_run,
            started,
            stage_outcome=StageOutcome.COMPLETED,
            after_workspace_fingerprint="after",
            process_started=False,
            clock=fixed_clock,
        )


def test_complete_stage_attempt_accepts_a_relative_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = Path("relative-run")
    started = start_attempt(
        run_dir,
        phase=AttemptPhase.IMPLEMENTING,
        before_workspace_fingerprint=None,
        clock=fixed_clock,
    )

    completed = complete_stage_attempt(
        run_dir,
        started,
        stage_outcome=StageOutcome.COMPLETED,
        after_workspace_fingerprint="after",
        process_started=False,
        clock=fixed_clock,
    )

    assert completed.status is AttemptStatus.COMPLETED
    assert completed.artifact_directory == started.artifact_directory.resolve()


def test_stage_message_codec_rejects_an_attempt_owned_by_another_run(
    tmp_path: Path,
) -> None:
    first_run = tmp_path / "first-run"
    second_run = tmp_path / "second-run"
    started = start_attempt(
        first_run,
        phase=AttemptPhase.REPORTING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )

    with pytest.raises(AttemptError, match="does not belong"):
        write_stage_message_result(
            second_run,
            started,
            status=StageOutcome.COMPLETED,
            message="must remain in the owning run",
        )
    assert not (second_run / "attempts").exists()


@pytest.mark.parametrize("status", list(StageOutcome))
def test_stage_message_status_round_trips_as_a_typed_outcome(
    tmp_path: Path,
    status: StageOutcome,
) -> None:
    record = start_attempt(
        tmp_path,
        phase=AttemptPhase.REPORTING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )

    path = write_stage_message_result(
        tmp_path,
        record,
        status=status,
        message="typed stage result",
    )
    loaded = read_stage_message_result(tmp_path, record)

    assert loaded is not None
    assert loaded.status is status
    expected_token = "PASS" if status is StageOutcome.COMPLETED else status.value
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == expected_token


def test_stage_message_codec_rejects_an_unknown_status_token(tmp_path: Path) -> None:
    record = start_attempt(
        tmp_path,
        phase=AttemptPhase.REPORTING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )
    record.artifact_directory.joinpath("result.json").write_text(
        json.dumps({"status": "UNKNOWN", "message": "tampered"}),
        encoding="utf-8",
    )

    with pytest.raises(PersistenceCodecError, match="unsupported"):
        read_stage_message_result(tmp_path, record)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ({"sequence": 0}, "positive integer"),
        ({"phase": "IMPLEMENTING"}, "AttemptPhase"),
        ({"result_path": "../result.json"}, "inside the attempt directory"),
        ({"artifact_directory": Path("wrong")}, "does not match"),
    ],
)
def test_stage_attempt_validates_public_identity_fields(
    tmp_path: Path,
    replacement: dict[str, object],
    message: str,
) -> None:
    record = start_attempt(
        tmp_path,
        phase=AttemptPhase.IMPLEMENTING,
        before_workspace_fingerprint=None,
        clock=fixed_clock,
    )
    trusted = StageAttempt.from_record(record)

    with pytest.raises(AttemptError, match=message):
        replace(trusted, **replacement)


def test_implementation_rejects_stage_attempt_from_an_unowned_directory(
    tmp_path: Path,
) -> None:
    repository = create_git_repo(tmp_path / "target")
    config = make_config(repository)
    ticket = tmp_path / "TA-LIFE-002.md"
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
    record = start_attempt(
        snapshot.run_dir,
        phase=AttemptPhase.IMPLEMENTING,
        before_workspace_fingerprint=None,
        clock=fixed_clock,
    )
    forged = replace(
        StageAttempt.from_record(record),
        artifact_directory=tmp_path / "unowned" / record.artifact_directory.name,
    )

    with pytest.raises(AttemptError, match="does not belong"):
        run_implementation_stage(
            config,
            snapshot.run_dir,
            agent_executor=make_agent_executors(config).implementation,
            attempt_record=forged,
            clock=fixed_clock,
        )


def test_correction_stage_keeps_the_established_direct_call_shape(
    tmp_path: Path,
) -> None:
    repository = create_git_repo(tmp_path / "target")
    config = make_config(repository)
    ticket = tmp_path / "TA-LIFE-002.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")
    snapshot = create_trusted_prepared_run(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    active = snapshot.run_record
    for state in (
        WorkflowState.IMPLEMENTING,
        WorkflowState.VERIFYING,
        WorkflowState.REVIEWING,
        WorkflowState.CORRECTION_PENDING,
        WorkflowState.CORRECTING,
    ):
        active = active.transition_to(
            state,
            updated_timestamp="2026-09-14T10:16:00Z",
            current_correction_round=(
                active.max_correction_rounds
                if state is WorkflowState.CORRECTING
                else None
            ),
        )
    save_run_record(active, snapshot.run_dir / "run.json")
    causes = CorrectionCauseSet(
        (
            VerificationCorrectionCause(
                gate_name="tests",
                command=("python", "-m", "pytest"),
                failure_summary="tests failed",
                stdout_excerpt="",
                stderr_excerpt="failure",
                exit_code=1,
                result_path=snapshot.run_dir / "verification.json",
            ),
        )
    )

    result = run_correction_stage(
        config,
        snapshot.run_dir,
        cause_set=causes,
        agent_executor=make_agent_executors(config).correction,
        clock=fixed_clock,
    )

    assert result.outcome is StageOutcome.HUMAN_REQUIRED
    attempt = latest_attempt(
        snapshot.run_dir,
        phases=(AttemptPhase.CORRECTING,),
    )
    assert attempt is not None
    assert attempt.status is AttemptStatus.HUMAN_REQUIRED


def test_attempt_creation_skips_an_orphaned_crash_directory(tmp_path: Path) -> None:
    orphan = tmp_path / "attempts" / "001-verification"
    orphan.mkdir(parents=True)

    record = start_attempt(
        tmp_path,
        phase=AttemptPhase.VERIFYING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )

    assert record.sequence == 2
    assert record.artifact_directory.name == "002-verification"
    assert [item.sequence for item in load_attempt_records(tmp_path)] == [2]


def test_missing_attempts_root_loads_as_an_empty_ledger(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    assert load_attempt_records(run_dir) == ()


@pytest.mark.parametrize(
    "blocked_path",
    ["attempts-root", "attempt-root", "attempt-record"],
)
def test_attempt_loading_rejects_filesystem_inspection_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blocked_path: str,
) -> None:
    record = start_attempt(
        tmp_path / "run",
        phase=AttemptPhase.VERIFYING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )
    blocked = {
        "attempts-root": record.artifact_directory.parent,
        "attempt-root": record.artifact_directory,
        "attempt-record": record.path,
    }[blocked_path]
    fail_stat_for_path(monkeypatch, blocked)

    with pytest.raises(AttemptError, match="simulated filesystem inspection failure"):
        load_attempt_records(tmp_path / "run")


def test_attempt_loading_uses_strict_resolution_for_the_attempts_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    record = start_attempt(
        run_dir,
        phase=AttemptPhase.VERIFYING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )
    attempts_root = record.artifact_directory.parent
    original_resolve = Path.resolve

    def fail_strict_resolution(path: Path, strict: bool = False) -> Path:
        if path == attempts_root:
            if strict:
                raise PermissionError("simulated resolution failure")
            return path.absolute()
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", fail_strict_resolution)

    with pytest.raises(AttemptError, match="simulated resolution failure"):
        load_attempt_records(run_dir)


def test_optional_attempt_result_rejects_inspection_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = start_attempt(
        tmp_path / "run",
        phase=AttemptPhase.REVIEWING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )
    result_path = record.artifact_directory / record.result_path
    result_path.write_text("{}", encoding="utf-8")
    fail_stat_for_path(monkeypatch, result_path)

    with pytest.raises(AttemptError, match="simulated filesystem inspection failure"):
        read_review_result(tmp_path / "run", record)


def test_trusted_attempt_record_resolves_only_its_own_artifacts(tmp_path: Path) -> None:
    started = start_attempt(
        tmp_path,
        phase=AttemptPhase.VERIFYING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )
    completed = complete_attempt(
        started,
        status=AttemptStatus.COMPLETED,
        after_workspace_fingerprint="after",
        clock=fixed_clock,
    )

    loaded = load_attempt_records(tmp_path)

    assert loaded == (completed,)
    assert attempt_result_path(tmp_path, loaded[0]) == (
        completed.artifact_directory / "result.json"
    )


def test_linked_attempt_directory_is_rejected_before_external_evidence_is_parsed(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    record = start_attempt(
        run_dir,
        phase=AttemptPhase.REVIEWING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )
    external = tmp_path / "external-attempt"
    record.artifact_directory.rename(external)
    external.joinpath("attempt.json").write_text("not json", encoding="utf-8")
    link_kind = create_directory_link(record.artifact_directory, external)
    if link_kind is None:
        pytest.skip("directory links are unavailable on this platform")
    try:
        with pytest.raises(AttemptError, match="confinement|owning run"):
            load_attempt_records(run_dir)
        with pytest.raises(AttemptError, match="confinement|owning run"):
            attempt_result_path(run_dir, record)
        with pytest.raises(AttemptError, match="confinement|owning run"):
            read_review_result(run_dir, record)
        with pytest.raises(AttemptError, match="confinement|owning run"):
            read_verification_evidence(run_dir, record)
    finally:
        remove_directory_link(record.artifact_directory)


def test_nested_artifact_directory_link_cannot_escape_its_attempt(
    tmp_path: Path,
) -> None:
    record = start_attempt(
        tmp_path / "run",
        phase=AttemptPhase.REVIEWING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )
    external = tmp_path / "external-result"
    external.mkdir()
    external.joinpath("untrusted.json").write_text("{}", encoding="utf-8")
    result_link = record.artifact_directory / record.result_path
    link_kind = create_directory_link(result_link, external)
    if link_kind is None:
        pytest.skip("directory links are unavailable on this platform")
    try:
        with pytest.raises(AttemptError, match="confinement|owning attempt"):
            attempt_result_path(tmp_path / "run", record)
        with pytest.raises(AttemptError, match="confinement|owning attempt"):
            read_review_result(tmp_path / "run", record)
    finally:
        remove_directory_link(result_link)


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_resume_preflight_rejects_a_linked_attempt_directory(
    tmp_path: Path,
) -> None:
    repository = create_git_repo(tmp_path / "target")
    config = make_config(repository)
    ticket = tmp_path / "TA-CORR-001.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")
    snapshot = create_trusted_prepared_run(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    record = load_attempt_records(snapshot.run_dir)[0]
    external = tmp_path / "external-baseline-attempt"
    record.artifact_directory.rename(external)
    link_kind = create_directory_link(record.artifact_directory, external)
    if link_kind is None:
        pytest.skip("directory links are unavailable on this platform")
    try:
        problem = resume_preflight_problem(
            config,
            snapshot.run_dir,
            snapshot.run_record,
        )
        assert problem is not None
        assert "Attempt evidence is invalid" in problem
    finally:
        remove_directory_link(record.artifact_directory)


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_resume_preflight_cannot_fall_back_past_an_unreadable_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = create_git_repo(tmp_path / "target")
    config = make_config(repository)
    ticket = tmp_path / "TA-CORR-001.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")
    snapshot = create_trusted_prepared_run(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    inaccessible = start_attempt(
        snapshot.run_dir,
        phase=AttemptPhase.PREPARING,
        before_workspace_fingerprint=None,
        clock=fixed_clock,
    )
    fail_stat_for_path(monkeypatch, inaccessible.path)

    problem = resume_preflight_problem(
        config,
        snapshot.run_dir,
        snapshot.run_record,
    )

    assert problem is not None
    assert "Attempt evidence is invalid" in problem
    assert "simulated filesystem inspection failure" in problem


def test_trusted_attempt_record_cannot_resolve_artifacts_for_another_run(
    tmp_path: Path,
) -> None:
    first_run = tmp_path / "first-run"
    second_run = tmp_path / "second-run"
    record = start_attempt(
        first_run,
        phase=AttemptPhase.VERIFYING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )

    with pytest.raises(AttemptError, match="does not belong"):
        attempt_result_path(second_run, record)


def test_tampered_attempt_path_is_rejected_before_result_can_escape(
    tmp_path: Path,
) -> None:
    record = start_attempt(
        tmp_path,
        phase=AttemptPhase.VERIFYING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )
    data = json.loads(record.path.read_text(encoding="utf-8"))
    data["result_path"] = "../run.json"
    record.path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(AttemptError, match="result_path"):
        load_attempt_records(tmp_path)


@pytest.mark.parametrize(
    ("field", "unknown"),
    [("phase", "verification"), ("status", "DONE")],
)
def test_unknown_persisted_lifecycle_values_are_rejected(
    tmp_path: Path,
    field: str,
    unknown: str,
) -> None:
    record = start_attempt(
        tmp_path,
        phase=AttemptPhase.VERIFYING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )
    data = json.loads(record.path.read_text(encoding="utf-8"))
    data[field] = unknown
    record.path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(AttemptError, match=rf"{field} is unsupported"):
        load_attempt_records(tmp_path)


@pytest.mark.parametrize("result_path", [None, "other-result.json"])
def test_persisted_result_path_must_match_the_phase_catalog(
    tmp_path: Path,
    result_path: object,
) -> None:
    record = start_attempt(
        tmp_path,
        phase=AttemptPhase.VERIFYING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )
    data = json.loads(record.path.read_text(encoding="utf-8"))
    data["result_path"] = result_path
    record.path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(AttemptError, match="result_path"):
        load_attempt_records(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [("phase", "VERIFYING"), ("status", "STARTED"), ("result_path", None)],
)
def test_attempt_record_rejects_invalid_trusted_construction(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    record = start_attempt(
        tmp_path,
        phase=AttemptPhase.VERIFYING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )

    with pytest.raises(AttemptError):
        replace(record, **{field: value})


def test_attempt_queries_reject_raw_phase_and_status_filters(tmp_path: Path) -> None:
    start_attempt(
        tmp_path,
        phase=AttemptPhase.VERIFYING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )

    with pytest.raises(AttemptError, match="phase filter"):
        latest_attempt(tmp_path, phases=("VERIFYING",))  # type: ignore[arg-type]
    with pytest.raises(AttemptError, match="status filter"):
        latest_attempt(tmp_path, statuses=("STARTED",))  # type: ignore[arg-type]


def test_attempt_commands_reject_raw_phase_and_status_values(tmp_path: Path) -> None:
    invalid_run_dir = tmp_path / "invalid"
    with pytest.raises(AttemptError, match="AttemptPhase"):
        start_attempt(
            invalid_run_dir,
            phase="VERIFYING",  # type: ignore[arg-type]
            before_workspace_fingerprint="before",
            clock=fixed_clock,
        )
    assert not invalid_run_dir.exists()

    record = start_attempt(
        tmp_path / "valid",
        phase=AttemptPhase.VERIFYING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )
    with pytest.raises(AttemptError, match="status"):
        complete_attempt(
            record,
            status="COMPLETED",  # type: ignore[arg-type]
            after_workspace_fingerprint="after",
            clock=fixed_clock,
        )
    assert load_attempt_records(tmp_path / "valid") == (record,)


def test_guarded_writable_operation_rejects_a_non_writable_phase(
    tmp_path: Path,
) -> None:
    repository = GitRepository(create_git_repo(tmp_path / "target"))
    run_dir = tmp_path / "run"
    record = start_attempt(
        run_dir,
        phase=AttemptPhase.VERIFYING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )

    with pytest.raises(ValueError, match="is not writable"):
        request = AgentExecutionRequest(
            task_kind=AgentTaskKind.IMPLEMENTATION,
            repository_path=repository.path,
            repository_access=RepositoryAccess.WORKSPACE_WRITE,
            prompt="unused",
            result_contract=IMPLEMENTATION_RESULT_CONTRACT,
            artifact_directory=record.artifact_directory,
            policy=AgentExecutionPolicy(60, NetworkAccess.ALLOWED),
            required_capabilities=required_execution_capabilities(
                RepositoryAccess.WORKSPACE_WRITE
            ),
        )
        executor = InMemoryAgentExecutor(
            {
                AgentTaskKind.IMPLEMENTATION: ImplementationResult(
                    ImplementationStatus.COMPLETED, "unused", (), (), ()
                )
            },
            capabilities=required_execution_capabilities(
                RepositoryAccess.WORKSPACE_WRITE
            ),
        )
        snapshot = WorkspaceSnapshot.capture(repository)
        guarded_request = GuardedWritableRequest(
            phase=AttemptPhase.VERIFYING,
            execution_request=request,
            baseline=WritableBaseline(
                repository.path,
                snapshot.branch,
                snapshot.head_sha or "",
            ),
        )
        GuardedWritableOperation(executor).execute(guarded_request)


def test_guarded_writable_operation_does_not_modify_provider_native_artifacts(
    tmp_path: Path,
) -> None:
    repository = GitRepository(create_git_repo(tmp_path / "target"))
    run_dir = tmp_path / "run"
    record = start_attempt(
        run_dir,
        phase=AttemptPhase.IMPLEMENTING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )
    request = AgentExecutionRequest(
        task_kind=AgentTaskKind.IMPLEMENTATION,
        repository_path=repository.path,
        repository_access=RepositoryAccess.WORKSPACE_WRITE,
        prompt="Implement the ticket.",
        result_contract=IMPLEMENTATION_RESULT_CONTRACT,
        artifact_directory=record.artifact_directory,
        policy=AgentExecutionPolicy(60, NetworkAccess.ALLOWED),
        required_capabilities=required_execution_capabilities(
            RepositoryAccess.WORKSPACE_WRITE
        ),
    )
    inner = InMemoryAgentExecutor(
        {
            AgentTaskKind.IMPLEMENTATION: ImplementationResult(
                ImplementationStatus.COMPLETED, "done", (), (), ()
            )
        },
        capabilities=required_execution_capabilities(RepositoryAccess.WORKSPACE_WRITE),
    )
    provider_artifact = record.artifact_directory / "provider-native.json"
    original_provider_evidence = b'{"provider_owned": true}\n'
    provider_artifact.write_bytes(original_provider_evidence)

    class ProviderNativeEvidenceExecutor:
        @property
        def capabilities(self):
            return inner.capabilities

        def execute(self, execution_request, **kwargs):
            execution = inner.execute(execution_request, **kwargs)
            return replace(
                execution,
                artifacts=(
                    *(
                        item
                        for item in execution.artifacts
                        if item.role is not ArtifactRole.PROVIDER_EXECUTION_DETAILS
                    ),
                    ArtifactReference(
                        ArtifactRole.PROVIDER_EXECUTION_DETAILS,
                        provider_artifact.relative_to(run_dir).as_posix(),
                        "application/json",
                    ),
                ),
            )

    snapshot = WorkspaceSnapshot.capture(repository)
    guarded_request = GuardedWritableRequest(
        phase=AttemptPhase.IMPLEMENTING,
        execution_request=request,
        baseline=WritableBaseline(
            repository.path,
            snapshot.branch,
            snapshot.head_sha or "",
            snapshot.fingerprint,
            True,
        ),
    )
    outcome = GuardedWritableOperation(ProviderNativeEvidenceExecutor()).execute(
        guarded_request
    )

    assert isinstance(outcome, WritableSucceeded)
    assert outcome.audit.execution is not None
    assert outcome.audit.execution.successful
    assert outcome.audit.workspace_guard.artifact_path == (
        record.artifact_directory / "workspace-guard.json"
    )
    assert outcome.audit.workspace_guard.artifact_path.is_file()
    assert provider_artifact.read_bytes() == original_provider_evidence


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
        snapshot.run_record.run_id,
        runs_dir=tmp_path / "runs",
        final_patch_capture=make_final_patch_capture(),
        report_publisher=make_report_publisher(),
        agent_executor_factory=make_resume_agent_executor_factory(
            make_agent_executors(config).implementation
        ),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert "Attempt evidence is invalid" in result.run_record.terminal_reason


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.pop("execution_path"), "missing execution_path"),
        (
            lambda data: data.update({"legacy_result_path": "result.json"}),
            "unexpected legacy_result_path",
        ),
    ],
    ids=["missing-field", "unknown-field"],
)
def test_attempt_codec_requires_the_exact_current_schema(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    record = start_attempt(
        tmp_path,
        phase=AttemptPhase.VERIFYING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )
    data = json.loads(record.path.read_text(encoding="utf-8"))
    mutation(data)
    record.path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(AttemptError, match=message):
        load_attempt_records(tmp_path)


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_attempt_codec_rejects_non_integer_schema_versions_without_rewriting(
    tmp_path: Path,
    schema_version: object,
) -> None:
    record = start_attempt(
        tmp_path,
        phase=AttemptPhase.VERIFYING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )
    data = json.loads(record.path.read_text(encoding="utf-8"))
    data["schema_version"] = schema_version
    record.path.write_text(json.dumps(data), encoding="utf-8")
    rejected_contents = record.path.read_bytes()

    with pytest.raises(AttemptError, match="unsupported schema version"):
        load_attempt_records(tmp_path)

    assert record.path.read_bytes() == rejected_contents


@pytest.mark.parametrize(
    "schema_version",
    [True, float(ATTEMPT_RECORD_SCHEMA_VERSION)],
)
def test_attempt_record_construction_rejects_non_integer_schema_versions(
    tmp_path: Path,
    schema_version: object,
) -> None:
    record = start_attempt(
        tmp_path,
        phase=AttemptPhase.VERIFYING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )

    with pytest.raises(AttemptError, match="unsupported schema version"):
        replace(record, schema_version=schema_version)


def test_resume_rejects_started_writable_attempt_after_persisted_transition(
    tmp_path: Path,
) -> None:
    repository = create_git_repo(tmp_path / "target")
    config = make_config(repository)
    ticket = tmp_path / "TA-LIFE-002.md"
    ticket.write_text("# Ticket\n", encoding="utf-8")
    snapshot = create_trusted_prepared_run(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    fingerprint = WorkspaceSnapshot.capture(GitRepository(repository)).fingerprint
    implementation_attempt = start_attempt(
        snapshot.run_dir,
        phase=AttemptPhase.IMPLEMENTING,
        before_workspace_fingerprint=None,
        clock=fixed_clock,
    )
    complete_stage_attempt(
        snapshot.run_dir,
        implementation_attempt,
        stage_outcome=StageOutcome.COMPLETED,
        after_workspace_fingerprint=fingerprint,
        process_started=True,
        clock=fixed_clock,
    )
    active = snapshot.run_record
    for state in (
        WorkflowState.IMPLEMENTING,
        WorkflowState.VERIFYING,
        WorkflowState.REVIEWING,
        WorkflowState.CORRECTION_PENDING,
        WorkflowState.CORRECTING,
    ):
        active = active.transition_to(
            state,
            updated_timestamp="2026-09-14T10:16:00Z",
        )
    save_run_record(active, snapshot.run_dir / "run.json")
    correction_attempt = start_attempt(
        snapshot.run_dir,
        phase=AttemptPhase.CORRECTING,
        before_workspace_fingerprint=None,
        clock=fixed_clock,
    )
    advanced = active.transition_to(
        WorkflowState.VERIFYING,
        updated_timestamp="2026-09-14T10:17:00Z",
    )
    save_run_record(advanced, snapshot.run_dir / "run.json")

    result = resume_ticket_lifecycle(
        snapshot.run_record.run_id,
        runs_dir=tmp_path / "runs",
        final_patch_capture=make_final_patch_capture(),
        agent_executor_factory=make_resume_agent_executor_factory(
            make_agent_executors(config).implementation
        ),
        report_publisher=make_report_publisher(),
        clock=fixed_clock,
    )

    assert result.run_record.state is WorkflowState.HUMAN_REQUIRED
    assert result.run_record.terminal_reason is not None
    assert (
        f"correction attempt {correction_attempt.sequence} was interrupted"
        in result.run_record.terminal_reason
    )
