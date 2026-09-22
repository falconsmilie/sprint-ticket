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
)
from ticket_automation.application.agent_execution import (
    IMPLEMENTATION_RESULT_CONTRACT,
    AgentExecutionPolicy,
    AgentExecutionRequest,
    AgentTaskKind,
    ArtifactReference,
    NetworkAccess,
    RepositoryAccess,
    required_execution_capabilities,
)
from ticket_automation.attempts import (
    AttemptError,
    attempt_result_path,
    complete_attempt,
    finish_phase_attempt,
    latest_attempt,
    load_attempt_records,
    start_attempt,
)
from ticket_automation.domain.task_results import (
    ImplementationResult,
    ImplementationStatus,
)
from ticket_automation.git import GitRepository
from ticket_automation.models import (
    AttemptPhase,
    AttemptStatus,
    StageOutcome,
    WorkflowState,
)
from ticket_automation.workflow import resume_ticket_lifecycle
from ticket_automation.writable_worker import run_writable_agent


def fixed_clock() -> datetime:
    return datetime(2026, 9, 14, 10, 15, tzinfo=UTC)


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


def test_writable_worker_rejects_a_non_writable_phase(tmp_path: Path) -> None:
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
        run_writable_agent(
            repository=repository,
            run_dir=run_dir,
            operation="invalid-verification-write",
            phase=AttemptPhase.VERIFYING,
            attempt_record=record,
            executor=executor,
            request=request,
        )


def test_writable_worker_does_not_read_or_modify_provider_native_artifacts(
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
    provider_artifact = tmp_path / "provider-native.json"
    original_provider_evidence = b'{"provider_owned": true}\n'
    provider_artifact.write_bytes(original_provider_evidence)

    class ProviderNativeEvidenceExecutor:
        def execute(self, execution_request, **kwargs):
            execution = inner.execute(execution_request, **kwargs)
            return replace(
                execution,
                artifacts=(
                    *execution.artifacts,
                    ArtifactReference(
                        "provider-native",
                        provider_artifact,
                        "application/json",
                    ),
                ),
            )

    invocation = run_writable_agent(
        repository=repository,
        run_dir=run_dir,
        operation="implementation",
        phase=AttemptPhase.IMPLEMENTING,
        attempt_record=record,
        executor=ProviderNativeEvidenceExecutor(),
        request=request,
    )

    assert invocation.invocation_permitted
    assert invocation.execution is not None
    assert invocation.execution.successful
    assert invocation.workspace_guard.artifact_path == (
        record.artifact_directory / "workspace-guard.json"
    )
    assert invocation.workspace_guard.artifact_path.is_file()
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
        config,
        snapshot.run_record.run_id,
        runs_dir=tmp_path / "runs",
        agent_executor_factory=lambda: make_agent_executors(config),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert "Attempt evidence is invalid" in result.run_record.terminal_reason
