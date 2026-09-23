from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.fake_agent_executor import InMemoryAgentExecutor
from tests.helpers import (
    GIT,
    create_git_repo,
    create_test_run_snapshot,
    create_trusted_prepared_run,
    make_agent_executor,
    make_agent_executors,
    make_config,
    make_resume_agent_executor_factory,
    make_run_dependencies,
)
from ticket_automation.application.agent_execution import (
    AgentTaskKind,
    RepositoryAccess,
    required_execution_capabilities,
)
from ticket_automation.attempts import (
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
    StopCategory,
    StopReason,
    WorkflowState,
)
from ticket_automation.providers.codex_cli import CodexProcessResult
from ticket_automation.reporting import run_report_stage
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

    def _run_with_start_tracking(
        self, command, *, stdin, timeout_seconds, on_process_start
    ):
        return self.run(
            command,
            stdin=stdin,
            timeout_seconds=timeout_seconds,
            on_process_start=on_process_start,
        )


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
    assert "workspace_guard" not in attempts[1].metadata
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
    implementation = run_implementation_stage(
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
        phase=AttemptPhase.REVIEWING,
        before_workspace_fingerprint=None,
        clock=fixed_clock,
    )

    result = resume_ticket_lifecycle(
        snapshot.run_record.run_id,
        runs_dir=tmp_path / "runs",
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
        agent_executor=make_agent_executor(
            config, process_runner=CompletingCodexRunner()
        ),
        clock=fixed_clock,
    )
    reporting = review.run_record.transition_to(
        WorkflowState.REPORTING,
        updated_timestamp="2026-09-14T10:19:00Z",
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
    original_write_text = Path.write_text

    def fail_report_write(path, data, *args, **kwargs):
        if path.name == "report.md":
            raise OSError("disk unavailable")
        return original_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_report_write)
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
