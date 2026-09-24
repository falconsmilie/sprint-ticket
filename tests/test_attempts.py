from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.fake_agent_executor import InMemoryAgentExecutor
from tests.helpers import (
    create_git_repo,
    create_trusted_prepared_run,
    make_agent_executors,
    make_config,
    make_final_patch_capture,
    make_report_publisher,
    make_resume_agent_executor_factory,
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
from ticket_automation.attempts import (
    AttemptError,
    StageAttempt,
    attempt_result_path,
    complete_attempt,
    complete_stage_attempt,
    finish_phase_attempt,
    latest_attempt,
    load_attempt_records,
    start_attempt,
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
from ticket_automation.persistence_codecs import write_stage_message_result
from ticket_automation.runs import save_run_record
from ticket_automation.workflow import resume_ticket_lifecycle


def fixed_clock() -> datetime:
    return datetime(2026, 9, 14, 10, 15, tzinfo=UTC)


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
            status="PASS",
            message="must remain in the owning run",
        )
    assert not (second_run / "attempts").exists()


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


def test_finish_phase_attempt_rejects_raw_stage_outcome(tmp_path: Path) -> None:
    start_attempt(
        tmp_path,
        phase=AttemptPhase.VERIFYING,
        before_workspace_fingerprint="before",
        clock=fixed_clock,
    )

    with pytest.raises(AttemptError, match="StageOutcome"):
        finish_phase_attempt(
            tmp_path,
            phase=AttemptPhase.VERIFYING,
            stage_outcome=StageOutcome.COMPLETED.value,  # type: ignore[arg-type]
        )


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
                    *execution.artifacts,
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
