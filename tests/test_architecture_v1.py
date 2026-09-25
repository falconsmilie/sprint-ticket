from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

import ticket_automation.persistence as persistence_module
import ticket_automation.workflow as workflow_module
from tests.fake_agent_executor import InMemoryAgentExecutor
from tests.helpers import (
    GIT,
    create_directory_link,
    create_git_repo,
    create_test_run_snapshot,
    create_trusted_prepared_run,
    make_agent_executor,
    make_agent_executors,
    make_config,
    make_final_patch_capture,
    make_report_publisher,
    make_resume_agent_executor_factory,
    make_run_dependencies,
    remove_directory_link,
    run_test_stage,
)
from ticket_automation.application.agent_execution import (
    AgentTaskKind,
    ProviderId,
    RepositoryAccess,
    required_execution_capabilities,
)
from ticket_automation.attempts import (
    AttemptMetadata,
    attempt_result_path,
    latest_writable_attempt,
    load_attempt_records,
    start_attempt,
    update_attempt,
)
from ticket_automation.domain.task_results import (
    ImplementationResult,
    ImplementationStatus,
    ReviewResult,
    ReviewVerdict,
)
from ticket_automation.git import GitRepository
from ticket_automation.git_safety import WorkspaceSnapshot
from ticket_automation.implementation import run_implementation_stage
from ticket_automation.models import (
    AttemptPhase,
    StageOutcome,
    StopCategory,
    StopReason,
    WorkflowState,
)
from ticket_automation.presentation.reporting import run_report_stage
from ticket_automation.providers.codex_cli import CodexProcessResult
from ticket_automation.review import run_review_stage
from ticket_automation.runs import load_run_record, save_run_record
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


class WorkspaceChangingExecutor:
    def __init__(self, inner: InMemoryAgentExecutor) -> None:
        self.inner = inner

    @property
    def capabilities(self):
        return self.inner.capabilities

    def execute(self, request, *, on_invocation_start=None):
        def start() -> None:
            if on_invocation_start is not None:
                on_invocation_start()
            if request.task_kind is AgentTaskKind.IMPLEMENTATION:
                request.repository_path.joinpath("file.txt").write_text(
                    "implemented\n", encoding="utf-8"
                )
            elif request.task_kind is AgentTaskKind.CORRECTION:
                request.repository_path.joinpath("correction.txt").write_text(
                    "corrected\n", encoding="utf-8"
                )

        return self.inner.execute(request, on_invocation_start=start)


@dataclass
class CompletingCodexRunner:
    calls: int = 0

    def run(self, command, *, stdin, timeout_seconds, on_process_start=None):
        del stdin, timeout_seconds
        if on_process_start is not None:
            on_process_start()
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
    config = make_config(repo)
    runner = PassingVerificationRunner()

    result = run_ticket_lifecycle(
        config,
        _ticket(tmp_path),
        runs_dir=tmp_path / "runs",
        verification_runner=runner,
        **make_run_dependencies(config, process_runner=CompletingCodexRunner()),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.READY_FOR_HUMAN
    assert result.run_record.stop_reason is None
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
    assert "workspace_guard" not in execution_metadata
    guard_metadata = json.loads((implementation / "workspace-guard.json").read_text())
    assert guard_metadata["new_environments"] == []
    assert attempts[1].metadata == AttemptMetadata(
        controller_message="Implementation completed and Git safety checks passed."
    )
    verification = attempts[2].artifact_directory
    verification_result = json.loads((verification / "result.json").read_text())
    assert verification_result["commands"][0]["stdout"] == "verification passed\n"
    assert not list(verification.glob("*.log"))
    assert (result.run_dir / "final.patch").is_file()
    assert (result.run_dir / "report.md").is_file()
    writable = latest_writable_attempt(result.run_dir)
    assert writable is not None
    assert writable.after_workspace_fingerprint is not None
    final_workspace = WorkspaceSnapshot.capture(GitRepository(repo))
    assert final_workspace.matches_fingerprint(writable.after_workspace_fingerprint)
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
        agent_executor=make_agent_executor(
            config, process_runner=CompletingCodexRunner()
        ),
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
    completed_evidence = {
        attempt.artifact_directory.relative_to(snapshot.run_dir): {
            path.relative_to(attempt.artifact_directory): path.read_bytes()
            for path in attempt.artifact_directory.rglob("*")
            if path.is_file()
        }
        for attempt in load_attempt_records(snapshot.run_dir)
    }

    resumed = resume_ticket_lifecycle(
        snapshot.run_record.run_id,
        runs_dir=tmp_path / "runs",
        final_patch_capture=make_final_patch_capture(),
        report_publisher=make_report_publisher(),
        verification_runner=PassingVerificationRunner(),
        agent_executor_factory=make_resume_agent_executor_factory(
            make_agent_executors(
                config, process_runner=CompletingCodexRunner()
            ).implementation
        ),
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
    attempts = load_attempt_records(snapshot.run_dir)
    assert [item.sequence for item in attempts] == list(range(1, len(attempts) + 1))
    for relative_directory, evidence in completed_evidence.items():
        directory = snapshot.run_dir / relative_directory
        current_paths = {
            path.relative_to(directory)
            for path in directory.rglob("*")
            if path.is_file()
        }
        assert current_paths == set(evidence)
        assert all(
            directory.joinpath(relative_path).read_bytes() == contents
            for relative_path, contents in evidence.items()
        )


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_resume_checks_persisted_compatibility_before_constructing_executors(tmp_path):
    repo = create_git_repo(tmp_path / "target")
    config = make_config(repo)
    snapshot = create_test_run_snapshot(
        config,
        _ticket(tmp_path),
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    payload = snapshot.run_record.resolved_policy.provider_policies[0].payload()
    assert isinstance(payload, dict)
    Path(payload["executable"]).unlink()
    repo.joinpath("workspace-drift.txt").write_text("changed\n", encoding="utf-8")

    result = resume_ticket_lifecycle(
        snapshot.run_record.run_id,
        runs_dir=tmp_path / "runs",
        final_patch_capture=make_final_patch_capture(),
        report_publisher=make_report_publisher(),
        agent_executor_factory=make_resume_agent_executor_factory(object()),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert "executable is unavailable" in result.run_record.terminal_reason


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_terminal_resume_does_not_construct_provider_runtime(tmp_path):
    repo = create_git_repo(tmp_path / "target")
    config = make_config(repo)
    snapshot = create_test_run_snapshot(
        config,
        _ticket(tmp_path),
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    terminal = snapshot.run_record.transition_to(
        WorkflowState.HUMAN_REQUIRED,
        updated_timestamp="2026-09-14T10:20:00Z",
        terminal_reason="trusted terminal state",
        stop_reason=StopReason(
            category=StopCategory.CONTROLLER_FAILURE,
            message="trusted terminal state",
            retryable=False,
        ),
    )
    save_run_record(terminal, snapshot.run_dir / "run.json")
    result = resume_ticket_lifecycle(
        terminal.run_id,
        runs_dir=tmp_path / "runs",
        final_patch_capture=make_final_patch_capture(),
        report_publisher=make_report_publisher(),
        agent_executor_factory=make_resume_agent_executor_factory(object()),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED


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
    before = WorkspaceSnapshot.capture(GitRepository(repo))
    attempt = start_attempt(
        snapshot.run_dir,
        phase=AttemptPhase.IMPLEMENTING,
        before_workspace_fingerprint=before.fingerprint,
        clock=fixed_clock,
    )
    attempt = update_attempt(attempt, process_started=True)
    repo.joinpath("partial.txt").write_text("partial work\n", encoding="utf-8")

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
    assert result.run_record.stop_reason is not None
    assert (
        result.run_record.stop_reason.category == StopCategory.HUMAN_JUDGMENT_REQUIRED
    )
    persisted_attempts = load_attempt_records(result.run_dir)
    assert persisted_attempts[-1] == attempt
    assert persisted_attempts[-1].status == "STARTED"
    assert persisted_attempts[-1].process_started is True
    assert repo.joinpath("partial.txt").is_file()
    assert (result.run_dir / "report.md").is_file()


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_reporting_renders_without_changing_controller_state(tmp_path, monkeypatch):
    repo = create_git_repo(tmp_path / "target")
    config = make_config(repo)
    result = run_ticket_lifecycle(
        config,
        _ticket(tmp_path),
        runs_dir=tmp_path / "runs",
        verification_runner=PassingVerificationRunner(),
        **make_run_dependencies(config, process_runner=CompletingCodexRunner()),
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
    config = make_config(repository)
    ticket = _ticket(tmp_path)
    snapshot = create_test_run_snapshot(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    start_attempt(
        snapshot.run_dir,
        phase=AttemptPhase.PREPARING,
        before_workspace_fingerprint=snapshot.run_record.baseline_sha,
        clock=fixed_clock,
    )

    result = resume_ticket_lifecycle(
        snapshot.run_record.run_id,
        runs_dir=tmp_path / "runs",
        final_patch_capture=make_final_patch_capture(),
        report_publisher=make_report_publisher(),
        verification_runner=PassingVerificationRunner(),
        agent_executor_factory=make_resume_agent_executor_factory(
            make_agent_executors(
                config, process_runner=CompletingCodexRunner()
            ).implementation
        ),
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
    implementation = run_test_stage(
        run_implementation_stage,
        AttemptPhase.IMPLEMENTING,
        config,
        snapshot.run_dir,
        agent_executor=make_agent_executor(
            config, process_runner=CompletingCodexRunner()
        ),
        clock=fixed_clock,
    )
    verifying = implementation.run_record.transition_to(
        WorkflowState.VERIFYING,
        updated_timestamp="2026-09-14T10:17:00Z",
    )
    save_run_record(verifying, snapshot.run_dir / "run.json")
    verification = run_test_stage(
        run_verification_stage,
        AttemptPhase.VERIFYING,
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
        phase=AttemptPhase.REVIEWING,
        before_workspace_fingerprint=None,
        clock=fixed_clock,
    )

    result = resume_ticket_lifecycle(
        snapshot.run_record.run_id,
        runs_dir=tmp_path / "runs",
        final_patch_capture=make_final_patch_capture(),
        report_publisher=make_report_publisher(),
        verification_runner=PassingVerificationRunner(),
        agent_executor_factory=make_resume_agent_executor_factory(
            make_agent_executors(
                config, process_runner=CompletingCodexRunner()
            ).implementation
        ),
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
def test_resume_retries_a_completed_review_before_its_transition(tmp_path):
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
    implementation = run_test_stage(
        run_implementation_stage,
        AttemptPhase.IMPLEMENTING,
        config,
        snapshot.run_dir,
        agent_executor=make_agent_executor(
            config, process_runner=CompletingCodexRunner()
        ),
        clock=fixed_clock,
    )
    verifying = implementation.run_record.transition_to(
        WorkflowState.VERIFYING,
        updated_timestamp="2026-09-14T10:17:00Z",
    )
    save_run_record(verifying, snapshot.run_dir / "run.json")
    verification = run_test_stage(
        run_verification_stage,
        AttemptPhase.VERIFYING,
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
    first_review = run_test_stage(
        run_review_stage,
        AttemptPhase.REVIEWING,
        config,
        snapshot.run_dir,
        agent_executor=make_agent_executor(
            config, process_runner=CompletingCodexRunner()
        ),
        clock=fixed_clock,
    )

    assert first_review.outcome is StageOutcome.COMPLETED
    assert load_run_record(snapshot.run_dir / "run.json").state is (
        WorkflowState.REVIEWING
    )

    result = resume_ticket_lifecycle(
        snapshot.run_record.run_id,
        runs_dir=tmp_path / "runs",
        final_patch_capture=make_final_patch_capture(),
        report_publisher=make_report_publisher(),
        verification_runner=PassingVerificationRunner(),
        agent_executor_factory=make_resume_agent_executor_factory(
            make_agent_executors(
                config, process_runner=CompletingCodexRunner()
            ).implementation
        ),
        clock=fixed_clock,
    )

    review_attempts = [
        item
        for item in load_attempt_records(snapshot.run_dir)
        if item.phase == WorkflowState.REVIEWING.value
    ]
    assert result.run_record.state is WorkflowState.READY_FOR_HUMAN
    assert result.run_record.current_review_round == 1
    assert [item.status for item in review_attempts] == ["COMPLETED", "COMPLETED"]


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
    implementation = run_test_stage(
        run_implementation_stage,
        AttemptPhase.IMPLEMENTING,
        config,
        snapshot.run_dir,
        agent_executor=make_agent_executor(
            config, process_runner=CompletingCodexRunner()
        ),
        clock=fixed_clock,
    )
    verifying = implementation.run_record.transition_to(
        WorkflowState.VERIFYING,
        updated_timestamp="2026-09-14T10:17:00Z",
    )
    save_run_record(verifying, snapshot.run_dir / "run.json")
    verification = run_test_stage(
        run_verification_stage,
        AttemptPhase.VERIFYING,
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
    review = run_test_stage(
        run_review_stage,
        AttemptPhase.REVIEWING,
        config,
        snapshot.run_dir,
        agent_executor=make_agent_executor(
            config, process_runner=CompletingCodexRunner()
        ),
        clock=fixed_clock,
    )
    reporting = review.run_record.transition_to(
        WorkflowState.REPORTING,
        updated_timestamp="2026-09-14T10:19:00Z",
        current_review_round=1,
    )
    save_run_record(reporting, snapshot.run_dir / "run.json")
    start_attempt(
        snapshot.run_dir,
        phase=AttemptPhase.REPORTING,
        before_workspace_fingerprint=None,
        clock=fixed_clock,
    )

    result = resume_ticket_lifecycle(
        snapshot.run_record.run_id,
        runs_dir=tmp_path / "runs",
        final_patch_capture=make_final_patch_capture(),
        report_publisher=make_report_publisher(),
        verification_runner=PassingVerificationRunner(),
        agent_executor_factory=make_resume_agent_executor_factory(
            make_agent_executors(
                config, process_runner=CompletingCodexRunner()
            ).implementation
        ),
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
    config = make_config(repository)
    real_replace = persistence_module.os.replace

    def fail_report_replace(source, destination):
        if Path(destination).name == "report.md":
            raise OSError("disk unavailable")
        return real_replace(source, destination)

    monkeypatch.setattr(
        persistence_module.os,
        "replace",
        fail_report_replace,
    )
    result = run_ticket_lifecycle(
        config,
        _ticket(tmp_path),
        runs_dir=tmp_path / "runs",
        verification_runner=PassingVerificationRunner(),
        **make_run_dependencies(config, process_runner=CompletingCodexRunner()),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.READY_FOR_HUMAN
    assert result.report_result is None
    assert load_run_record(result.run_dir / "run.json").state == (
        WorkflowState.READY_FOR_HUMAN
    )
    assert load_attempt_records(result.run_dir)[-1].status == "COMPLETED"
    assert (result.run_dir / "final.patch").is_file()
    assert not tuple(result.run_dir.glob(".report.md.*.tmp"))


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_failed_atomic_report_replacement_preserves_an_existing_report(
    tmp_path,
    monkeypatch,
):
    repository = create_git_repo(tmp_path / "target")
    config = make_config(repository)
    result = run_ticket_lifecycle(
        config,
        _ticket(tmp_path),
        runs_dir=tmp_path / "runs",
        verification_runner=PassingVerificationRunner(),
        **make_run_dependencies(config, process_runner=CompletingCodexRunner()),
        clock=fixed_clock,
    )
    report_path = result.run_dir / "report.md"
    report_path.write_text("existing report\n", encoding="utf-8")
    real_replace = persistence_module.os.replace

    def fail_report_replace(source, destination):
        if Path(destination) == report_path:
            raise OSError("disk unavailable")
        return real_replace(source, destination)

    monkeypatch.setattr(persistence_module.os, "replace", fail_report_replace)

    publication = make_report_publisher().publish(result.run_dir, result.run_record)

    assert publication is None
    assert report_path.read_text(encoding="utf-8") == "existing report\n"
    assert not tuple(result.run_dir.glob(".report.md.*.tmp"))
    assert load_run_record(result.run_dir / "run.json").state is (
        WorkflowState.READY_FOR_HUMAN
    )


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_patch_capture_retarget_fails_without_transition_or_external_writes(
    tmp_path,
):
    probe_target = tmp_path / "probe-target"
    probe_link = tmp_path / "probe-link"
    probe_target.mkdir()
    if create_directory_link(probe_link, probe_target) is None:
        pytest.skip("directory links are unavailable on this platform")
    remove_directory_link(probe_link)

    repository = create_git_repo(tmp_path / "target")
    config = make_config(repository)
    external_run = tmp_path / "external-run"
    external_patch_contents = "external patch evidence\n"

    class RetargetingPatchCapture:
        def __init__(self):
            self.delegate = make_final_patch_capture()
            self.linked_run: Path | None = None

        def capture(self, request):
            linked_run = request.destination.parent
            linked_run.rename(external_run)
            external_run.joinpath("final.patch").write_text(
                external_patch_contents,
                encoding="utf-8",
            )
            assert create_directory_link(linked_run, external_run) is not None
            self.linked_run = linked_run
            return self.delegate.capture(request)

    patch_capture = RetargetingPatchCapture()
    dependencies = make_run_dependencies(
        config,
        process_runner=CompletingCodexRunner(),
    )
    dependencies["final_patch_capture"] = patch_capture
    try:
        result = run_ticket_lifecycle(
            config,
            _ticket(tmp_path),
            runs_dir=tmp_path / "runs",
            verification_runner=PassingVerificationRunner(),
            **dependencies,
            clock=fixed_clock,
        )

        assert result.run_record.state is not WorkflowState.READY_FOR_HUMAN
        assert result.report_result is None
        assert result.controller_error is not None
        assert "ownership was lost" in result.controller_error
        assert load_run_record(external_run / "run.json").state is (
            WorkflowState.REPORTING
        )
        assert external_run.joinpath("final.patch").read_text(encoding="utf-8") == (
            external_patch_contents
        )
        assert not external_run.joinpath("report.md").exists()
    finally:
        if patch_capture.linked_run is not None and patch_capture.linked_run.exists():
            remove_directory_link(patch_capture.linked_run)


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_report_publisher_rejects_a_linked_run_without_external_mutation(tmp_path):
    repository = create_git_repo(tmp_path / "target")
    config = make_config(repository)
    result = run_ticket_lifecycle(
        config,
        _ticket(tmp_path),
        runs_dir=tmp_path / "runs",
        verification_runner=PassingVerificationRunner(),
        **make_run_dependencies(config, process_runner=CompletingCodexRunner()),
        clock=fixed_clock,
    )
    linked_run = tmp_path / "linked-run"
    if create_directory_link(linked_run, result.run_dir) is None:
        pytest.skip("directory links are unavailable on this platform")
    before_files = {
        path.relative_to(result.run_dir).as_posix(): path.read_bytes()
        for path in result.run_dir.rglob("*")
        if path.is_file()
    }
    try:
        publication = make_report_publisher().publish(
            linked_run,
            result.run_record,
        )

        after_files = {
            path.relative_to(result.run_dir).as_posix(): path.read_bytes()
            for path in result.run_dir.rglob("*")
            if path.is_file()
        }
        assert publication is None
        assert after_files == before_files
    finally:
        remove_directory_link(linked_run)


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_report_publisher_replaces_a_hardlink_without_mutating_its_target(
    tmp_path,
):
    repository = create_git_repo(tmp_path / "target")
    config = make_config(repository)
    result = run_ticket_lifecycle(
        config,
        _ticket(tmp_path),
        runs_dir=tmp_path / "runs",
        verification_runner=PassingVerificationRunner(),
        **make_run_dependencies(config, process_runner=CompletingCodexRunner()),
        clock=fixed_clock,
    )
    external_report = tmp_path / "external-report.md"
    external_report.write_text("external report evidence\n", encoding="utf-8")
    report_path = result.run_dir / "report.md"
    report_path.unlink()
    os.link(external_report, report_path)

    publication = make_report_publisher().publish(
        result.run_dir,
        result.run_record,
    )

    assert publication is not None
    assert external_report.read_text(encoding="utf-8") == "external report evidence\n"
    assert report_path.read_text(encoding="utf-8").startswith(
        f"# {result.run_record.ticket_id} report"
    )
    assert not os.path.samefile(external_report, report_path)


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_lifecycle_fails_closed_if_the_selected_run_is_retargeted(
    tmp_path,
    monkeypatch,
):
    probe_target = tmp_path / "probe-target"
    probe_link = tmp_path / "probe-link"
    probe_target.mkdir()
    link_kind = create_directory_link(probe_link, probe_target)
    if link_kind is None:
        pytest.skip("directory links are unavailable on this platform")
    remove_directory_link(probe_link)

    repository = create_git_repo(tmp_path / "target")
    config = make_config(repository)
    moved_run = tmp_path / "moved-run"
    retargeted_path: Path | None = None
    original_dispatch = workflow_module.dispatch_stage

    def dispatch_then_retarget(context, handlers):
        nonlocal retargeted_path
        decision = original_dispatch(context, handlers)
        if context.run_record.state is WorkflowState.PREPARING:
            retargeted_path = context.run_dir
            context.run_dir.rename(moved_run)
            assert create_directory_link(context.run_dir, moved_run) is not None
        return decision

    monkeypatch.setattr(workflow_module, "dispatch_stage", dispatch_then_retarget)
    try:
        result = run_ticket_lifecycle(
            config,
            _ticket(tmp_path),
            runs_dir=tmp_path / "runs",
            verification_runner=PassingVerificationRunner(),
            **make_run_dependencies(config, process_runner=CompletingCodexRunner()),
            clock=fixed_clock,
        )

        assert result.run_record.state is WorkflowState.PREPARING
        assert result.report_result is None
        assert result.controller_error is not None
        assert "ownership was lost" in result.controller_error
        assert [item.name for item in result.safety_violations] == [
            "run-directory-ownership"
        ]
        assert load_run_record(moved_run / "run.json").state is WorkflowState.PREPARING
        attempts = load_attempt_records(moved_run)
        assert len(attempts) == 1
        assert attempts[0].status == "STARTED"
        assert not (moved_run / "final.patch").exists()
        assert not (moved_run / "report.md").exists()
    finally:
        if retargeted_path is not None and retargeted_path.exists():
            remove_directory_link(retargeted_path)


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_agent_cannot_redirect_post_execution_evidence_after_run_retarget(
    tmp_path,
):
    probe_target = tmp_path / "probe-target"
    probe_link = tmp_path / "probe-link"
    probe_target.mkdir()
    if create_directory_link(probe_link, probe_target) is None:
        pytest.skip("directory links are unavailable on this platform")
    remove_directory_link(probe_link)

    repository = create_git_repo(tmp_path / "target")
    config = make_config(repository)
    runs_dir = tmp_path / "runs"
    moved_run = tmp_path / "moved-run"
    external_target = tmp_path / "external-target"
    external_target.mkdir()
    external_target.joinpath("sentinel.txt").write_bytes(b"unchanged\n")
    before = {
        path.relative_to(external_target): path.read_bytes()
        for path in external_target.rglob("*")
        if path.is_file()
    }

    class RetargetingCodexRunner:
        def run(
            self,
            command,
            *,
            stdin,
            timeout_seconds,
            on_process_start=None,
        ):
            del stdin, timeout_seconds
            if on_process_start is not None:
                on_process_start()
            command.cwd.joinpath("file.txt").write_text(
                "implemented\n",
                encoding="utf-8",
            )
            output = Path(command.argv[command.argv.index("--output-last-message") + 1])
            output.write_text(
                json.dumps(
                    {
                        "status": "COMPLETED",
                        "summary": "implementation passed",
                        "tests_run": [],
                        "assumptions": [],
                        "known_issues": [],
                    }
                ),
                encoding="utf-8",
            )
            run_dir = next(path for path in runs_dir.iterdir() if path.is_dir())
            run_dir.rename(moved_run)
            assert create_directory_link(run_dir, external_target) is not None
            return CodexProcessResult(returncode=0, stdout="", stderr="")

    executor = make_agent_executor(
        config,
        process_runner=RetargetingCodexRunner(),
    )

    retargeted_path: Path | None = None
    try:
        result = run_ticket_lifecycle(
            config,
            _ticket(tmp_path),
            runs_dir=runs_dir,
            verification_runner=PassingVerificationRunner(),
            **make_run_dependencies(
                config,
                agent_executor=executor,
            ),
            clock=fixed_clock,
        )
        retargeted_path = result.run_dir

        after = {
            path.relative_to(external_target): path.read_bytes()
            for path in external_target.rglob("*")
            if path.is_file()
        }
        assert after == before
        assert (
            load_run_record(moved_run / "run.json").state is WorkflowState.IMPLEMENTING
        )
        assert result.report_result is None
        assert result.controller_error is not None
        assert "ownership was lost" in result.controller_error
        assert [item.name for item in result.safety_violations] == [
            "run-directory-ownership"
        ]
        implementation_attempt = load_attempt_records(moved_run)[-1]
        assert implementation_attempt.phase is AttemptPhase.IMPLEMENTING
        assert implementation_attempt.status == "STARTED"
        assert not (external_target / "execution.json").exists()
        assert not (external_target / "result.json").exists()
        assert not (external_target / "report.md").exists()
    finally:
        if retargeted_path is not None and retargeted_path.exists():
            remove_directory_link(retargeted_path)


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_verification_runner_cannot_redirect_result_after_run_retarget(tmp_path):
    probe_target = tmp_path / "probe-target"
    probe_link = tmp_path / "probe-link"
    probe_target.mkdir()
    if create_directory_link(probe_link, probe_target) is None:
        pytest.skip("directory links are unavailable on this platform")
    remove_directory_link(probe_link)

    repository = create_git_repo(tmp_path / "target")
    config = make_config(repository)
    runs_dir = tmp_path / "runs"
    moved_run = tmp_path / "moved-run"
    external_target = tmp_path / "external-target"
    external_target.mkdir()
    external_target.joinpath("sentinel.txt").write_bytes(b"unchanged\n")
    before = {
        path.relative_to(external_target): path.read_bytes()
        for path in external_target.rglob("*")
        if path.is_file()
    }

    class RetargetingVerificationRunner:
        calls = 0

        def run(self, command, *, timeout_seconds):
            del command, timeout_seconds
            self.calls += 1
            if self.calls == 2:
                run_dir = next(path for path in runs_dir.iterdir() if path.is_dir())
                run_dir.rename(moved_run)
                assert create_directory_link(run_dir, external_target) is not None
            return VerificationProcessResult(
                returncode=0,
                stdout="verification passed\n",
                stderr="",
            )

    runner = RetargetingVerificationRunner()
    executor = WorkspaceChangingExecutor(
        make_agent_executor(config, process_runner=CompletingCodexRunner())
    )
    retargeted_path: Path | None = None
    try:
        result = run_ticket_lifecycle(
            config,
            _ticket(tmp_path),
            runs_dir=runs_dir,
            verification_runner=runner,
            **make_run_dependencies(config, agent_executor=executor),
            clock=fixed_clock,
        )
        retargeted_path = result.run_dir

        after = {
            path.relative_to(external_target): path.read_bytes()
            for path in external_target.rglob("*")
            if path.is_file()
        }
        assert after == before
        assert load_run_record(moved_run / "run.json").state is WorkflowState.VERIFYING
        assert result.report_result is None
        assert result.controller_error is not None
        assert "ownership was lost" in result.controller_error
        assert [item.name for item in result.safety_violations] == [
            "run-directory-ownership"
        ]
        verification_attempt = load_attempt_records(moved_run)[-1]
        assert verification_attempt.phase is AttemptPhase.VERIFYING
        assert verification_attempt.status == "STARTED"
        assert not (external_target / "result.json").exists()
        assert not (external_target / "report.md").exists()
    finally:
        if retargeted_path is not None and retargeted_path.exists():
            remove_directory_link(retargeted_path)


@pytest.mark.skipif(GIT is None, reason="git executable is required")
@pytest.mark.parametrize("replacement", ["attempts-root", "active-attempt"])
def test_agent_attempt_storage_replacement_stops_without_writing_replacement(
    tmp_path: Path,
    replacement: str,
) -> None:
    repository = create_git_repo(tmp_path / "target")
    config = make_config(repository)
    runs_dir = tmp_path / "runs"
    moved = tmp_path / f"moved-{replacement}"

    class ReplacingAttemptRunner(CompletingCodexRunner):
        replacement_path: Path | None = None
        original_record: Path | None = None

        def run(self, command, *, stdin, timeout_seconds, on_process_start=None):
            if on_process_start is not None:
                on_process_start()
            run_dir = next(path for path in runs_dir.iterdir() if path.is_dir())
            attempts_root = run_dir / "attempts"
            active = next(
                path
                for path in attempts_root.iterdir()
                if path.name.endswith("-implementation")
            )
            if replacement == "attempts-root":
                attempts_root.rename(moved)
                attempts_root.mkdir()
                replacement_path = attempts_root / active.name
                replacement_path.mkdir()
                self.original_record = moved / active.name / "attempt.json"
            else:
                active.rename(moved)
                active.mkdir()
                replacement_path = active
                self.original_record = moved / "attempt.json"
            self.replacement_path = replacement_path
            return super().run(
                command,
                stdin=stdin,
                timeout_seconds=timeout_seconds,
                on_process_start=None,
            )

    runner = ReplacingAttemptRunner()
    result = run_ticket_lifecycle(
        config,
        _ticket(tmp_path),
        runs_dir=runs_dir,
        verification_runner=PassingVerificationRunner(),
        **make_run_dependencies(config, process_runner=runner),
        clock=fixed_clock,
    )

    assert result.run_record.state is not WorkflowState.READY_FOR_HUMAN
    assert result.controller_error is not None
    assert "ownership was lost" in result.controller_error
    assert runner.replacement_path is not None
    assert tuple(runner.replacement_path.iterdir()) == ()
    assert runner.original_record is not None
    assert json.loads(runner.original_record.read_text(encoding="utf-8"))["status"] == (
        "STARTED"
    )


@pytest.mark.skipif(GIT is None, reason="git executable is required")
@pytest.mark.parametrize("replacement", ["attempts-root", "active-attempt"])
def test_verification_attempt_storage_replacement_stops_before_result_write(
    tmp_path: Path,
    replacement: str,
) -> None:
    repository = create_git_repo(tmp_path / "target")
    config = make_config(repository)
    runs_dir = tmp_path / "runs"
    moved = tmp_path / f"moved-verification-{replacement}"

    class ReplacingVerificationRunner(PassingVerificationRunner):
        replacement_path: Path | None = None
        original_record: Path | None = None

        def run(self, command, *, timeout_seconds):
            self.calls += 1
            if self.calls == 2:
                run_dir = next(path for path in runs_dir.iterdir() if path.is_dir())
                attempts_root = run_dir / "attempts"
                active = next(
                    path
                    for path in attempts_root.iterdir()
                    if path.name.endswith("-verification")
                )
                if replacement == "attempts-root":
                    attempts_root.rename(moved)
                    attempts_root.mkdir()
                    replacement_path = attempts_root / active.name
                    replacement_path.mkdir()
                    self.original_record = moved / active.name / "attempt.json"
                else:
                    active.rename(moved)
                    active.mkdir()
                    replacement_path = active
                    self.original_record = moved / "attempt.json"
                self.replacement_path = replacement_path
            del command, timeout_seconds
            return VerificationProcessResult(0, "verification passed\n", "")

    verification_runner = ReplacingVerificationRunner()
    result = run_ticket_lifecycle(
        config,
        _ticket(tmp_path),
        runs_dir=runs_dir,
        verification_runner=verification_runner,
        **make_run_dependencies(config, process_runner=CompletingCodexRunner()),
        clock=fixed_clock,
    )

    assert result.run_record.state is not WorkflowState.READY_FOR_HUMAN
    assert result.controller_error is not None
    assert "ownership was lost" in result.controller_error
    assert verification_runner.replacement_path is not None
    assert tuple(verification_runner.replacement_path.iterdir()) == ()
    assert verification_runner.original_record is not None
    assert (
        json.loads(verification_runner.original_record.read_text(encoding="utf-8"))[
            "status"
        ]
        == "STARTED"
    )


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_new_run_replacement_after_snapshot_is_rejected_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = create_git_repo(tmp_path / "target")
    config = make_config(repository)
    runs_dir = tmp_path / "runs"
    moved_run = tmp_path / "created-run"
    replacement_sentinel = b"replacement must remain untouched\n"
    original_create = workflow_module.create_run_snapshot

    def create_then_replace(*args, **kwargs):
        snapshot = original_create(*args, **kwargs)
        snapshot.run_dir.rename(moved_run)
        snapshot.run_dir.mkdir()
        snapshot.run_dir.joinpath("sentinel").write_bytes(replacement_sentinel)
        return snapshot

    monkeypatch.setattr(workflow_module, "create_run_snapshot", create_then_replace)
    provider_runner = CompletingCodexRunner()
    verification_runner = PassingVerificationRunner()

    result = run_ticket_lifecycle(
        config,
        _ticket(tmp_path),
        runs_dir=runs_dir,
        verification_runner=verification_runner,
        **make_run_dependencies(config, process_runner=provider_runner),
        clock=fixed_clock,
    )

    assert result.run_record.state is WorkflowState.PREPARING
    assert result.controller_error is not None
    assert "ownership was lost" in result.controller_error
    assert [item.name for item in result.safety_violations] == [
        "run-directory-ownership"
    ]
    assert provider_runner.calls == 0
    assert verification_runner.calls == 0
    assert result.report_result is None
    assert snapshot_repository(result.run_record) == repository.resolve()
    assert snapshot_files(result.run_dir) == {"sentinel": replacement_sentinel}
    assert load_run_record(moved_run / "run.json").state is WorkflowState.PREPARING


def snapshot_repository(run_record) -> Path:
    return Path(run_record.target_repository_path).resolve()


def snapshot_files(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_terminal_report_describes_a_failed_baseline_attempt(tmp_path):
    repository = create_git_repo(tmp_path / "target")
    config = make_config(repository)

    class FailingBaselineRunner:
        def run(self, command, *, timeout_seconds):
            del command, timeout_seconds
            return VerificationProcessResult(returncode=1, stdout="failed", stderr="")

    result = run_ticket_lifecycle(
        config,
        _ticket(tmp_path),
        runs_dir=tmp_path / "runs",
        **make_run_dependencies(config),
        verification_runner=FailingBaselineRunner(),
        clock=fixed_clock,
    )

    report = (result.run_dir / "report.md").read_text(encoding="utf-8")
    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert "- Baseline verification: FAIL" in report


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_one_correction_round_preserves_the_ordinary_lifecycle(tmp_path):
    repository = create_git_repo(tmp_path / "target")
    config = make_config(repository)
    verification = FailOnceVerificationRunner()
    capabilities = required_execution_capabilities(
        RepositoryAccess.READ_ONLY
    ) | required_execution_capabilities(RepositoryAccess.WORKSPACE_WRITE)
    executor = WorkspaceChangingExecutor(
        InMemoryAgentExecutor(
            {
                AgentTaskKind.IMPLEMENTATION: ImplementationResult(
                    ImplementationStatus.COMPLETED,
                    "implemented",
                    (),
                    (),
                    (),
                ),
                AgentTaskKind.CORRECTION: ImplementationResult(
                    ImplementationStatus.COMPLETED,
                    "corrected",
                    (),
                    (),
                    (),
                ),
                AgentTaskKind.REVIEW: ReviewResult(
                    ReviewVerdict.PASS,
                    "review passed",
                    (),
                ),
            },
            capabilities=capabilities,
            provider_id=ProviderId("codex-cli"),
        )
    )

    result = run_ticket_lifecycle(
        config,
        _ticket(tmp_path),
        runs_dir=tmp_path / "runs",
        verification_runner=verification,
        **make_run_dependencies(config, agent_executor=executor),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.READY_FOR_HUMAN
    assert result.run_record.current_correction_round == 1
    assert verification.calls == 3
    attempts = load_attempt_records(result.run_dir)
    assert [item.phase for item in attempts] == [
        "PREPARING",
        "IMPLEMENTING",
        "VERIFYING",
        "CORRECTING",
        "VERIFYING",
        "REVIEWING",
        "REPORTING",
    ]
    failed_verification = next(
        item for item in attempts if item.phase == "VERIFYING" and item.sequence == 3
    )
    result_path = attempt_result_path(result.run_dir, failed_verification)
    verification_evidence = json.loads(result_path.read_text(encoding="utf-8"))
    assert [
        reason["kind"] for reason in verification_evidence["correction_reasons"]
    ] == ["VerificationFailure"]
    correction_attempt = next(item for item in attempts if item.phase == "CORRECTING")
    correction_tickets = tuple(
        path
        for path in correction_attempt.artifact_directory.glob("*.md")
        if path.name != "prompt.md"
    )
    assert len(correction_tickets) == 1
    stage_executions = (
        result.implementation_result.agent_execution,
        result.correction_results[0].agent_execution,
        result.review_results[0].agent_execution,
    )
    provider_specific_names = {
        "execution-details",
        "events",
        "standard-error",
        "structured-result",
    }
    assert all(execution is not None for execution in stage_executions)
    assert all(
        provider_specific_names.isdisjoint(
            artifact.name for artifact in execution.artifacts
        )
        for execution in stage_executions
        if execution is not None
    )
