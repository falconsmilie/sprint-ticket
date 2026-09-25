from __future__ import annotations

import json
import shutil
from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path

import pytest

import ticket_automation.corrections as corrections_module
import ticket_automation.persistence as persistence_module
import ticket_automation.persistence_codecs as persistence_codecs_module
from tests.helpers import (
    GIT,
    create_directory_link,
    create_trusted_prepared_run,
    fail_stat_for_path,
    make_agent_executor,
    make_run_dependencies,
    remove_directory_link,
    run_git,
    run_test_stage,
)
from tests.lifecycle_characterization_fixtures import (
    ScriptedVerificationRunner,
    TickingClock,
    assert_attempt_ledger,
    build_lifecycle_workspace,
    configure_fake_codex_actions,
)
from ticket_automation.application.agent_execution import AgentExecutor, AgentTaskKind
from ticket_automation.application.handoff_acceptance import (
    HandoffAcceptanceRequest,
    HandoffAcceptanceService,
    HandoffAccepted,
    HandoffRejected,
)
from ticket_automation.application.ports.handoff import (
    FinalPatchCaptureRequest,
    FinalPatchReference,
)
from ticket_automation.attempts import (
    StageAttempt,
    latest_writable_attempt,
    load_attempt_records,
    start_attempt,
)
from ticket_automation.domain.task_results import ImplementationResult, ReviewResult
from ticket_automation.git import GitRepository
from ticket_automation.git_safety import WorkspaceSnapshot
from ticket_automation.implementation import run_implementation_stage
from ticket_automation.infrastructure.final_patch import FileSystemFinalPatchCapture
from ticket_automation.models import AttemptPhase, StopCategory, WorkflowState
from ticket_automation.presentation.reporting import (
    collect_report_context,
    run_report_stage,
)
from ticket_automation.review import run_review_stage
from ticket_automation.run_ownership import RunOwnershipError
from ticket_automation.runs import load_run_record, save_run_record
from ticket_automation.verification import run_verification_stage
from ticket_automation.workflow import resume_ticket_lifecycle, run_ticket_lifecycle

pytestmark = pytest.mark.skipif(GIT is None, reason="git executable is required")


@dataclass
class HandoffEvidenceTamperingExecutor:
    inner: AgentExecutor
    target_phase: AttemptPhase
    evidence_change: str
    review_calls: int = 0

    @property
    def capabilities(self):
        return self.inner.capabilities

    def execute(self, request, *, on_invocation_start=None):
        execution = self.inner.execute(
            request,
            on_invocation_start=on_invocation_start,
        )
        if request.task_kind is not AgentTaskKind.REVIEW:
            return execution

        self.review_calls += 1
        trigger_review = 2 if self.target_phase is AttemptPhase.CORRECTING else 1
        if self.review_calls != trigger_review:
            return execution

        assert request.artifact_layout is not None
        target = next(
            item
            for item in reversed(load_attempt_records(request.artifact_layout.run_root))
            if item.phase is self.target_phase and item.completed
        )
        evidence_path = target.artifact_directory / "execution.json"
        if self.evidence_change == "missing":
            evidence_path.unlink()
        else:
            data = json.loads(evidence_path.read_text(encoding="utf-8"))
            data["typed_result"]["summary"] = "tampered summary"
            evidence_path.write_text(json.dumps(data), encoding="utf-8")
        return execution


@dataclass(frozen=True)
class WritableStopCase:
    action: str
    state: WorkflowState
    category: StopCategory
    attempt_status: str
    process_started: bool
    workspace_changed: bool
    remove_executable: bool = False


_WRITABLE_STOP_CASES = [
    pytest.param(
        WritableStopCase(
            "blocked",
            WorkflowState.HUMAN_REQUIRED,
            StopCategory.HUMAN_JUDGMENT_REQUIRED,
            "HUMAN_REQUIRED",
            True,
            False,
        ),
        id="blocked",
    ),
    pytest.param(
        WritableStopCase(
            "modify",
            WorkflowState.FAILED,
            StopCategory.EXTERNAL_TOOL_FAILURE,
            "FAILED",
            False,
            False,
            remove_executable=True,
        ),
        id="not-started",
    ),
    pytest.param(
        WritableStopCase(
            "fail",
            WorkflowState.FAILED,
            StopCategory.EXTERNAL_TOOL_FAILURE,
            "FAILED",
            True,
            False,
        ),
        id="started-unchanged",
    ),
    pytest.param(
        WritableStopCase(
            "fail-after-change",
            WorkflowState.HUMAN_REQUIRED,
            StopCategory.REPOSITORY_UNCERTAIN,
            "HUMAN_REQUIRED",
            True,
            True,
        ),
        id="started-changed",
    ),
    pytest.param(
        WritableStopCase(
            "missing-result",
            WorkflowState.HUMAN_REQUIRED,
            StopCategory.REPOSITORY_UNCERTAIN,
            "HUMAN_REQUIRED",
            True,
            False,
        ),
        id="missing-result",
    ),
    pytest.param(
        WritableStopCase(
            "malformed-result",
            WorkflowState.HUMAN_REQUIRED,
            StopCategory.REPOSITORY_UNCERTAIN,
            "HUMAN_REQUIRED",
            True,
            False,
        ),
        id="malformed-result",
    ),
    pytest.param(
        WritableStopCase(
            "timeout",
            WorkflowState.HUMAN_REQUIRED,
            StopCategory.REPOSITORY_UNCERTAIN,
            "HUMAN_REQUIRED",
            True,
            False,
        ),
        id="timeout-after-start",
    ),
    pytest.param(
        WritableStopCase(
            "staging-change",
            WorkflowState.HUMAN_REQUIRED,
            StopCategory.SAFETY_VIOLATION,
            "HUMAN_REQUIRED",
            True,
            True,
        ),
        id="staging-change",
    ),
    pytest.param(
        WritableStopCase(
            "branch-change",
            WorkflowState.HUMAN_REQUIRED,
            StopCategory.SAFETY_VIOLATION,
            "HUMAN_REQUIRED",
            True,
            True,
        ),
        id="branch-change",
    ),
    pytest.param(
        WritableStopCase(
            "head-change",
            WorkflowState.HUMAN_REQUIRED,
            StopCategory.SAFETY_VIOLATION,
            "HUMAN_REQUIRED",
            True,
            True,
        ),
        id="head-change",
    ),
    pytest.param(
        WritableStopCase(
            "environment-change",
            WorkflowState.HUMAN_REQUIRED,
            StopCategory.SAFETY_VIOLATION,
            "HUMAN_REQUIRED",
            True,
            True,
        ),
        id="environment-change",
    ),
]


@pytest.mark.parametrize("case", _WRITABLE_STOP_CASES)
def test_writable_implementation_stop_matrix(
    tmp_path,
    monkeypatch,
    case: WritableStopCase,
):
    workspace = build_lifecycle_workspace(tmp_path)
    baseline = WorkspaceSnapshot.capture(GitRepository(workspace.repository))
    codex_runner = configure_fake_codex_actions(monkeypatch, tmp_path, case.action)
    verification_runner = ScriptedVerificationRunner([0])
    if case.remove_executable:
        original_run = verification_runner.run

        def pass_baseline_then_remove_executable(command, *, timeout_seconds):
            result = original_run(command, timeout_seconds=timeout_seconds)
            workspace.agent_executable.unlink()
            return result

        verification_runner.run = pass_baseline_then_remove_executable

    result = run_ticket_lifecycle(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        **make_run_dependencies(workspace.config, process_runner=codex_runner),
        verification_runner=verification_runner,
        clock=TickingClock(),
    )

    assert result.run_record.state == case.state
    assert result.run_record.stop_reason is not None
    assert result.run_record.stop_reason.category == case.category
    attempts = assert_attempt_ledger(
        result.run_dir,
        [("PREPARING", "COMPLETED"), ("IMPLEMENTING", case.attempt_status)],
    )
    writable = attempts[-1]
    assert writable.process_started is case.process_started
    assert writable.before_workspace_fingerprint is not None
    assert writable.after_workspace_fingerprint is not None
    assert (
        writable.before_workspace_fingerprint != writable.after_workspace_fingerprint
    ) is case.workspace_changed
    current = WorkspaceSnapshot.capture(GitRepository(workspace.repository))
    assert (current.fingerprint != baseline.fingerprint) is case.workspace_changed
    assert (result.run_dir / "report.md").is_file()


@pytest.mark.parametrize("case", _WRITABLE_STOP_CASES)
def test_writable_correction_stop_matrix(
    tmp_path,
    monkeypatch,
    case: WritableStopCase,
):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        case.action,
    )
    verification_runner = ScriptedVerificationRunner([0, 1])
    if case.remove_executable:
        original_run = verification_runner.run

        def fail_implementation_verification_then_remove_executable(
            command, *, timeout_seconds
        ):
            result = original_run(command, timeout_seconds=timeout_seconds)
            if verification_runner.calls == 2:
                workspace.agent_executable.unlink()
            return result

        verification_runner.run = (
            fail_implementation_verification_then_remove_executable
        )

    result = run_ticket_lifecycle(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        **make_run_dependencies(workspace.config, process_runner=codex_runner),
        verification_runner=verification_runner,
        clock=TickingClock(),
    )

    assert result.run_record.state == case.state
    assert result.run_record.stop_reason is not None
    assert result.run_record.stop_reason.category == case.category
    attempts = assert_attempt_ledger(
        result.run_dir,
        [
            ("PREPARING", "COMPLETED"),
            ("IMPLEMENTING", "COMPLETED"),
            ("VERIFYING", "COMPLETED"),
            ("CORRECTING", case.attempt_status),
        ],
    )
    correction = attempts[-1]
    assert correction.process_started is case.process_started
    assert correction.before_workspace_fingerprint is not None
    assert correction.after_workspace_fingerprint is not None
    assert (
        correction.before_workspace_fingerprint
        != correction.after_workspace_fingerprint
    ) is case.workspace_changed
    assert (correction.artifact_directory / "workspace-guard.json").is_file()
    assert (result.run_dir / "report.md").is_file()


def test_correction_preparation_failure_persists_pre_and_post_evidence(
    tmp_path, monkeypatch
):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "modify",
    )
    verification_runner = ScriptedVerificationRunner([0, 1])

    def fail_ticket_write(**_kwargs):
        raise OSError("correction artifact unavailable")

    monkeypatch.setattr(
        corrections_module, "_write_correction_ticket", fail_ticket_write
    )

    result = run_ticket_lifecycle(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        **make_run_dependencies(workspace.config, process_runner=codex_runner),
        verification_runner=verification_runner,
        clock=TickingClock(),
    )

    assert result.run_record.state is WorkflowState.HUMAN_REQUIRED
    attempts = assert_attempt_ledger(
        result.run_dir,
        [
            ("PREPARING", "COMPLETED"),
            ("IMPLEMENTING", "COMPLETED"),
            ("VERIFYING", "COMPLETED"),
            ("CORRECTING", "HUMAN_REQUIRED"),
        ],
    )
    correction = attempts[-1]
    assert correction.process_started is False
    assert correction.before_workspace_fingerprint is not None
    assert correction.after_workspace_fingerprint is not None
    assert (correction.artifact_directory / "workspace-guard.json").is_file()


@pytest.mark.parametrize(
    "relative_runs_dir",
    [False, True],
    ids=["absolute-runs-dir", "relative-runs-dir"],
)
def test_required_review_finding_creates_correction_work(
    tmp_path,
    monkeypatch,
    relative_runs_dir,
):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-corrections",
        "modify-correction",
        "review-pass",
    )
    runs_dir = workspace.runs_dir
    if relative_runs_dir:
        monkeypatch.chdir(tmp_path)
        runs_dir = runs_dir.relative_to(tmp_path)

    result = run_ticket_lifecycle(
        workspace.config,
        workspace.ticket,
        runs_dir=runs_dir,
        **make_run_dependencies(workspace.config, process_runner=codex_runner),
        verification_runner=ScriptedVerificationRunner([0, 0, 0]),
        clock=TickingClock(),
    )

    assert result.run_record.state == WorkflowState.READY_FOR_HUMAN
    assert result.run_record.stop_reason is None
    assert result.run_record.current_correction_round == 1
    assert (result.run_dir / "final.patch").is_file()
    attempts = assert_attempt_ledger(
        result.run_dir.resolve(),
        [
            ("PREPARING", "COMPLETED"),
            ("IMPLEMENTING", "COMPLETED"),
            ("VERIFYING", "COMPLETED"),
            ("REVIEWING", "COMPLETED"),
            ("CORRECTING", "COMPLETED"),
            ("VERIFYING", "COMPLETED"),
            ("REVIEWING", "COMPLETED"),
            ("REPORTING", "COMPLETED"),
        ],
    )
    first_review = next(item for item in attempts if item.phase == "REVIEWING")
    review_evidence = json.loads(
        (first_review.artifact_directory / "result.json").read_text(encoding="utf-8")
    )
    assert review_evidence["verdict"] == "CORRECTIONS_REQUIRED"
    assert {
        (finding["disposition"], finding["scope_relation"])
        for finding in review_evidence["findings"]
    } == {("REQUIRED", "IMPLEMENTATION")}
    correction_attempt = next(item for item in attempts if item.phase == "CORRECTING")
    correction_tickets = tuple(
        path
        for path in correction_attempt.artifact_directory.glob("*.md")
        if path.name != "prompt.md"
    )
    assert len(correction_tickets) == 1

    for attempt in attempts:
        (attempt.artifact_directory / "events.jsonl").unlink(missing_ok=True)
        (attempt.artifact_directory / "stderr.log").unlink(missing_ok=True)
    context = collect_report_context(result.run_dir, result.run_record)
    controller = context.controller
    agent = context.agent
    expected_correction_ticket = (
        Path("attempts")
        / correction_attempt.artifact_directory.name
        / "correction-ticket.md"
    ).as_posix()
    assert controller.correction_ticket_paths == (expected_correction_ticket,)
    assert isinstance(agent.implementation, ImplementationResult)
    assert controller.review_results
    assert all(
        isinstance(review_result, ReviewResult)
        for review_result in controller.review_results
    )
    agent_attempts = tuple(
        attempt
        for attempt in controller.attempts
        if attempt.phase
        in {
            AttemptPhase.IMPLEMENTING,
            AttemptPhase.REVIEWING,
            AttemptPhase.CORRECTING,
        }
    )
    assert all(attempt.execution is not None for attempt in agent_attempts)
    assert {
        attempt.execution.provider_id.value
        for attempt in agent_attempts
        if attempt.execution
    } == {"codex-cli"}


def test_review_finding_outside_correction_scope_remains_human_required(
    tmp_path, monkeypatch
):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-unsafe",
    )

    result = run_ticket_lifecycle(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        **make_run_dependencies(workspace.config, process_runner=codex_runner),
        verification_runner=ScriptedVerificationRunner([0, 0]),
        clock=TickingClock(),
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert result.run_record.terminal_reason == (
        "Review found REQUIRED findings, but none are safely eligible for automatic "
        "correction."
    )
    assert_attempt_ledger(
        result.run_dir,
        [
            ("PREPARING", "COMPLETED"),
            ("IMPLEMENTING", "COMPLETED"),
            ("VERIFYING", "COMPLETED"),
            ("REVIEWING", "HUMAN_REQUIRED"),
        ],
    )


def test_inconsistent_review_result_requires_human_review(tmp_path, monkeypatch):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-inconsistent",
    )

    result = run_ticket_lifecycle(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        **make_run_dependencies(workspace.config, process_runner=codex_runner),
        verification_runner=ScriptedVerificationRunner([0, 0]),
        clock=TickingClock(),
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert result.run_record.stop_reason is not None
    assert (
        result.run_record.stop_reason.category == StopCategory.HUMAN_JUDGMENT_REQUIRED
    )
    assert_attempt_ledger(
        result.run_dir,
        [
            ("PREPARING", "COMPLETED"),
            ("IMPLEMENTING", "COMPLETED"),
            ("VERIFYING", "COMPLETED"),
            ("REVIEWING", "HUMAN_REQUIRED"),
        ],
    )


def test_correction_limit_exhaustion_stops_conservatively(tmp_path, monkeypatch):
    workspace = build_lifecycle_workspace(tmp_path, max_correction_rounds=1)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "modify-correction",
    )

    result = run_ticket_lifecycle(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        **make_run_dependencies(workspace.config, process_runner=codex_runner),
        verification_runner=ScriptedVerificationRunner([0, 1, 1]),
        clock=TickingClock(),
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert result.run_record.stop_reason is not None
    assert (
        result.run_record.stop_reason.category == StopCategory.HUMAN_JUDGMENT_REQUIRED
    )
    assert result.run_record.current_correction_round == 1
    attempts = assert_attempt_ledger(
        result.run_dir,
        [
            ("PREPARING", "COMPLETED"),
            ("IMPLEMENTING", "COMPLETED"),
            ("VERIFYING", "COMPLETED"),
            ("CORRECTING", "COMPLETED"),
            ("VERIFYING", "COMPLETED"),
        ],
    )
    verification_attempts = [item for item in attempts if item.phase == "VERIFYING"]
    assert all(
        json.loads(
            (item.artifact_directory / "result.json").read_text(encoding="utf-8")
        )["correction_reasons"][0]["kind"]
        == "VerificationFailure"
        for item in verification_attempts
    )


@pytest.mark.parametrize(
    "drift",
    [
        "branch",
        "head",
        "staging",
        "fingerprint",
    ],
)
def test_final_handoff_rejects_repository_drift(tmp_path, monkeypatch, drift):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-pass",
    )
    clock = TickingClock()
    reporting_run = _prepare_reporting_run(
        workspace,
        codex_runner=codex_runner,
        clock=clock,
    )
    start_attempt(
        reporting_run.run_dir,
        phase=AttemptPhase.REPORTING,
        before_workspace_fingerprint=None,
        clock=clock,
    )

    if drift == "branch":
        run_git(workspace.repository, "checkout", "-b", "handoff-drift")
    elif drift == "head":
        (workspace.repository / "head.txt").write_text("head drift\n", encoding="utf-8")
        run_git(workspace.repository, "add", "head.txt")
        run_git(workspace.repository, "commit", "-m", "handoff drift")
    elif drift == "staging":
        (workspace.repository / "staged.txt").write_text(
            "staged drift\n", encoding="utf-8"
        )
        run_git(workspace.repository, "add", "staged.txt")
    else:
        (workspace.repository / "untracked.txt").write_text(
            "fingerprint drift\n", encoding="utf-8"
        )

    result = HandoffAcceptanceService(
        patch_capture=FileSystemFinalPatchCapture(),
    ).accept(
        HandoffAcceptanceRequest(
            run_dir=reporting_run.run_dir,
            run_record=reporting_run.run_record,
        )
    )

    assert isinstance(result, HandoffRejected)
    assert result.stop_category is StopCategory.SAFETY_VIOLATION
    assert load_run_record(reporting_run.run_dir / "run.json").state is (
        WorkflowState.REPORTING
    )
    writable = latest_writable_attempt(reporting_run.run_dir)
    assert writable is not None
    current = WorkspaceSnapshot.capture(GitRepository(workspace.repository))
    assert not current.matches_fingerprint(writable.after_workspace_fingerprint)
    if drift == "branch":
        assert current.branch != reporting_run.run_record.starting_branch
    elif drift == "head":
        assert current.head_sha != reporting_run.run_record.baseline_sha
    elif drift == "staging":
        assert current.staged_paths
    else:
        assert current.branch == reporting_run.run_record.starting_branch
        assert current.head_sha == reporting_run.run_record.baseline_sha
        assert not current.staged_paths
    assert not (reporting_run.run_dir / "final.patch").exists()


@pytest.mark.parametrize(
    ("evidence_change", "reason"),
    [
        ("stale-verification", "verification evidence is stale"),
        ("malformed-verification", "lowercase SHA-256 digest"),
        ("missing-verification", "verification evidence is not passing"),
        ("incomplete-verification", "verification evidence is incomplete"),
        (
            "contradictory-verification",
            "Verification command evidence is not passing",
        ),
        ("verification-safety", "contains safety violations"),
        ("verification-missing-safety", "invalid safety evidence"),
        ("verification-corrections", "contains correction reasons"),
        ("verification-missing-corrections", "invalid correction evidence"),
        ("verification-cwd", "unexpected working directory"),
        ("verification-round", "does not match the final correction round"),
        ("stale-review", "review did not pass for the verified final source"),
        ("malformed-review", "lowercase SHA-256 digest"),
        ("missing-review", "review evidence is missing"),
        ("missing-neutral-execution", "Agent execution evidence is invalid"),
        ("drifted-neutral-result", "does not match neutral execution evidence"),
        ("missing-typed-result-artifact", "does not exist"),
        ("wrong-neutral-provider", "provider identity does not match"),
    ],
)
def test_final_handoff_rejects_stale_or_missing_evidence(
    tmp_path,
    monkeypatch,
    evidence_change,
    reason,
):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-pass",
    )
    clock = TickingClock()
    reporting = _prepare_reporting_run(
        workspace,
        codex_runner=codex_runner,
        clock=clock,
    )
    attempts = load_attempt_records(reporting.run_dir)
    if evidence_change == "incomplete-verification":
        start_attempt(
            reporting.run_dir,
            phase=AttemptPhase.VERIFYING,
            before_workspace_fingerprint=WorkspaceSnapshot.capture(
                GitRepository(workspace.repository)
            ).fingerprint,
            clock=clock,
        )
    elif evidence_change.startswith("verification") or evidence_change in {
        "stale-verification",
        "malformed-verification",
        "missing-verification",
        "contradictory-verification",
    }:
        attempt = next(item for item in attempts if item.phase == "VERIFYING")
        if evidence_change == "missing-verification":
            attempt.path.unlink()
        elif evidence_change in {"stale-verification", "malformed-verification"}:
            data = json.loads(attempt.path.read_text(encoding="utf-8"))
            data["after_workspace_fingerprint"] = (
                "invalid" if evidence_change == "malformed-verification" else "0" * 64
            )
            attempt.path.write_text(json.dumps(data), encoding="utf-8")
        else:
            result_path = attempt.artifact_directory / "result.json"
            data = json.loads(result_path.read_text(encoding="utf-8"))
            if evidence_change == "contradictory-verification":
                data["commands"][0]["status"] = "FAIL"
                data["commands"][0]["exit_code"] = 1
            elif evidence_change == "verification-safety":
                data["safety_violations"] = [{"name": "branch"}]
            elif evidence_change == "verification-missing-safety":
                data.pop("safety_violations")
            elif evidence_change == "verification-corrections":
                data["correction_reasons"] = [{"kind": "VerificationFailure"}]
            elif evidence_change == "verification-missing-corrections":
                data.pop("correction_reasons")
            elif evidence_change == "verification-round":
                data["round_index"] = 99
            else:
                data["commands"][0]["cwd"] = str(tmp_path / "other")
            result_path.write_text(json.dumps(data), encoding="utf-8")
    elif evidence_change in {
        "missing-neutral-execution",
        "drifted-neutral-result",
        "missing-typed-result-artifact",
        "wrong-neutral-provider",
    }:
        attempt = next(item for item in attempts if item.phase == "REVIEWING")
        execution_path = attempt.artifact_directory / "execution.json"
        if evidence_change == "missing-neutral-execution":
            execution_path.unlink()
        else:
            data = json.loads(execution_path.read_text(encoding="utf-8"))
            if evidence_change == "drifted-neutral-result":
                data["typed_result"]["summary"] = "tampered summary"
            elif evidence_change == "wrong-neutral-provider":
                data["provider_id"] = "different-provider"
            else:
                typed_reference = data["typed_result_artifact"]
                (reporting.run_dir / typed_reference["path"]).unlink()
            execution_path.write_text(json.dumps(data), encoding="utf-8")
    else:
        attempt = next(item for item in attempts if item.phase == "REVIEWING")
        if evidence_change == "missing-review":
            attempt.path.unlink()
        else:
            data = json.loads(attempt.path.read_text(encoding="utf-8"))
            fingerprint = (
                "invalid" if evidence_change == "malformed-review" else "0" * 64
            )
            data["before_workspace_fingerprint"] = fingerprint
            data["after_workspace_fingerprint"] = fingerprint
            attempt.path.write_text(json.dumps(data), encoding="utf-8")

    dependencies = make_run_dependencies(
        workspace.config,
        process_runner=codex_runner,
    )
    result = resume_ticket_lifecycle(
        reporting.run_record.run_id,
        runs_dir=workspace.runs_dir,
        agent_executor_factory=dependencies["agent_executor_factory"],
        final_patch_capture=dependencies["final_patch_capture"],
        report_publisher=dependencies["report_publisher"],
        verification_runner=ScriptedVerificationRunner([]),
        clock=clock,
    )

    assert result.run_record.state is WorkflowState.HUMAN_REQUIRED
    assert result.run_record.stop_reason is not None
    assert result.run_record.stop_reason.category is StopCategory.SAFETY_VIOLATION
    assert reason in result.run_record.terminal_reason
    assert not (result.run_dir / "final.patch").exists()


@pytest.mark.parametrize(
    ("target_phase", "evidence_change", "actions", "verification_results"),
    [
        pytest.param(
            AttemptPhase.IMPLEMENTING,
            "missing",
            ("modify", "review-pass"),
            [0, 0],
            id="implementation-missing",
        ),
        pytest.param(
            AttemptPhase.IMPLEMENTING,
            "inconsistent",
            ("modify", "review-pass"),
            [0, 0],
            id="implementation-typed-result-inconsistent",
        ),
        pytest.param(
            AttemptPhase.CORRECTING,
            "missing",
            ("modify", "review-corrections", "modify-correction", "review-pass"),
            [0, 0, 0],
            id="correction-missing",
        ),
        pytest.param(
            AttemptPhase.CORRECTING,
            "inconsistent",
            ("modify", "review-corrections", "modify-correction", "review-pass"),
            [0, 0, 0],
            id="correction-typed-result-inconsistent",
        ),
    ],
)
def test_final_handoff_rejects_non_review_agent_evidence_tampering(
    tmp_path,
    monkeypatch,
    target_phase,
    evidence_change,
    actions,
    verification_results,
):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(monkeypatch, tmp_path, *actions)
    executor = HandoffEvidenceTamperingExecutor(
        make_agent_executor(workspace.config, process_runner=codex_runner),
        target_phase=target_phase,
        evidence_change=evidence_change,
    )

    result = run_ticket_lifecycle(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        **make_run_dependencies(workspace.config, agent_executor=executor),
        verification_runner=ScriptedVerificationRunner(verification_results),
        clock=TickingClock(),
    )

    assert result.run_record.state is WorkflowState.HUMAN_REQUIRED
    assert result.run_record.stop_reason is not None
    assert result.run_record.stop_reason.category is StopCategory.SAFETY_VIOLATION
    assert "Agent execution evidence is invalid" in result.run_record.terminal_reason
    if evidence_change == "inconsistent":
        assert "typed stage result does not match" in result.run_record.terminal_reason
    assert not (result.run_dir / "final.patch").exists()


@pytest.mark.parametrize(
    "linked_phase",
    [AttemptPhase.VERIFYING, AttemptPhase.REVIEWING],
    ids=["verification", "review"],
)
def test_final_handoff_rejects_linked_external_evidence(
    tmp_path,
    monkeypatch,
    linked_phase,
):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-pass",
    )
    clock = TickingClock()
    reporting = _prepare_reporting_run(
        workspace,
        codex_runner=codex_runner,
        clock=clock,
    )
    start_attempt(
        reporting.run_dir,
        phase=AttemptPhase.REPORTING,
        before_workspace_fingerprint=None,
        clock=clock,
    )
    linked_attempt = next(
        item
        for item in load_attempt_records(reporting.run_dir)
        if item.phase is linked_phase
    )
    external = tmp_path / f"external-{linked_phase.value.lower()}-attempt"
    linked_attempt.artifact_directory.rename(external)
    link_kind = create_directory_link(linked_attempt.artifact_directory, external)
    if link_kind is None:
        pytest.skip("directory links are unavailable on this platform")
    try:
        result = HandoffAcceptanceService(
            patch_capture=FileSystemFinalPatchCapture(),
        ).accept(
            HandoffAcceptanceRequest(
                run_dir=reporting.run_dir,
                run_record=reporting.run_record,
            )
        )

        assert isinstance(result, HandoffRejected)
        assert result.stop_category is StopCategory.SAFETY_VIOLATION
        assert "confinement" in result.reason or "owning run" in result.reason
        assert not (reporting.run_dir / "final.patch").exists()
    finally:
        remove_directory_link(linked_attempt.artifact_directory)


def test_final_handoff_stops_before_reading_a_replaced_bound_run(
    tmp_path,
    monkeypatch,
):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-pass",
    )
    clock = TickingClock()
    reporting = _prepare_reporting_run(
        workspace,
        codex_runner=codex_runner,
        clock=clock,
    )
    start_attempt(
        reporting.run_dir,
        phase=AttemptPhase.REPORTING,
        before_workspace_fingerprint=None,
        clock=clock,
        run_ownership=reporting.run_ownership,
    )
    original_capture = WorkspaceSnapshot.capture
    moved_run = tmp_path / "original-handoff-run"
    replacement_created = False

    def replace_run_after_workspace_capture(repository):
        nonlocal replacement_created
        snapshot = original_capture(repository)
        if not replacement_created:
            reporting.run_dir.rename(moved_run)
            shutil.copytree(moved_run, reporting.run_dir)
            replacement_created = True
        return snapshot

    patch_capture_called = False

    class RecordingPatchCapture:
        def capture(self, request: FinalPatchCaptureRequest) -> FinalPatchReference:
            nonlocal patch_capture_called
            patch_capture_called = True
            return FileSystemFinalPatchCapture().capture(request)

    monkeypatch.setattr(
        WorkspaceSnapshot,
        "capture",
        staticmethod(replace_run_after_workspace_capture),
    )

    result = HandoffAcceptanceService(
        patch_capture=RecordingPatchCapture(),
    ).accept(
        HandoffAcceptanceRequest(
            run_dir=reporting.run_dir,
            run_record=reporting.run_record,
            run_ownership=reporting.run_ownership,
        )
    )

    assert isinstance(result, HandoffRejected)
    assert "ownership was lost" in result.reason
    assert not patch_capture_called
    assert not (reporting.run_dir / "final.patch").exists()


def test_review_does_not_dispatch_substituted_result_read_from_replaced_run(
    tmp_path,
    monkeypatch,
):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-pass",
    )
    original_read_json = persistence_codecs_module.read_json
    moved_run = tmp_path / "original-review-run"
    replaced = False
    source_fingerprint = None

    def replace_run_after_implementation_result_read(path):
        nonlocal replaced, source_fingerprint
        value = original_read_json(path)
        if not replaced and "implementation" in Path(path).parent.name:
            run_dir = Path(path).parents[2]
            source_fingerprint = WorkspaceSnapshot.capture(
                GitRepository(workspace.repository)
            ).fingerprint
            run_dir.rename(moved_run)
            shutil.copytree(moved_run, run_dir)
            replaced = True
        return value

    monkeypatch.setattr(
        persistence_codecs_module,
        "read_json",
        replace_run_after_implementation_result_read,
    )
    dependencies = make_run_dependencies(
        workspace.config,
        process_runner=codex_runner,
    )

    result = run_ticket_lifecycle(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        **dependencies,
        verification_runner=ScriptedVerificationRunner([0, 0]),
        clock=TickingClock(),
    )

    assert replaced
    assert result.controller_error is not None
    assert "ownership was lost" in result.controller_error
    assert json.loads(codex_runner.action_path.read_text(encoding="utf-8")) == [
        "review-pass"
    ]
    assert load_run_record(moved_run / "run.json").state is WorkflowState.REVIEWING
    assert load_run_record(result.run_dir / "run.json").state is WorkflowState.REVIEWING
    assert source_fingerprint is not None
    assert WorkspaceSnapshot.capture(
        GitRepository(workspace.repository)
    ).fingerprint == (source_fingerprint)


def test_final_handoff_rejects_run_replacement_during_patch_capture(
    tmp_path,
    monkeypatch,
):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-pass",
    )
    clock = TickingClock()
    reporting = _prepare_reporting_run(
        workspace,
        codex_runner=codex_runner,
        clock=clock,
    )
    start_attempt(
        reporting.run_dir,
        phase=AttemptPhase.REPORTING,
        before_workspace_fingerprint=None,
        clock=clock,
        run_ownership=reporting.run_ownership,
    )
    moved_run = tmp_path / "original-patch-capture-run"
    source_fingerprint = WorkspaceSnapshot.capture(
        GitRepository(workspace.repository)
    ).fingerprint

    class ReplacingPatchCapture:
        def capture(self, request: FinalPatchCaptureRequest) -> FinalPatchReference:
            reference = FileSystemFinalPatchCapture().capture(request)
            reporting.run_dir.rename(moved_run)
            shutil.copytree(moved_run, reporting.run_dir)
            return reference

    result = HandoffAcceptanceService(
        patch_capture=ReplacingPatchCapture(),
    ).accept(
        HandoffAcceptanceRequest(
            run_dir=reporting.run_dir,
            run_record=reporting.run_record,
            run_ownership=reporting.run_ownership,
        )
    )

    assert isinstance(result, HandoffRejected)
    assert result.stop_category is StopCategory.SAFETY_VIOLATION
    assert "ownership was lost" in result.reason
    assert load_run_record(moved_run / "run.json").state is WorkflowState.REPORTING
    assert load_run_record(reporting.run_dir / "run.json").state is (
        WorkflowState.REPORTING
    )
    assert WorkspaceSnapshot.capture(
        GitRepository(workspace.repository)
    ).fingerprint == (source_fingerprint)


def test_final_handoff_rejects_run_replacement_during_final_workspace_capture(
    tmp_path,
    monkeypatch,
):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-pass",
    )
    clock = TickingClock()
    reporting = _prepare_reporting_run(
        workspace,
        codex_runner=codex_runner,
        clock=clock,
    )
    start_attempt(
        reporting.run_dir,
        phase=AttemptPhase.REPORTING,
        before_workspace_fingerprint=None,
        clock=clock,
        run_ownership=reporting.run_ownership,
    )
    moved_run = tmp_path / "original-final-workspace-capture-run"
    source_fingerprint = WorkspaceSnapshot.capture(
        GitRepository(workspace.repository)
    ).fingerprint
    original_capture = WorkspaceSnapshot.capture
    patch_was_captured = False
    run_was_replaced = False

    class RecordingPatchCapture:
        def capture(self, request: FinalPatchCaptureRequest) -> FinalPatchReference:
            nonlocal patch_was_captured
            reference = FileSystemFinalPatchCapture().capture(request)
            patch_was_captured = True
            return reference

    def replace_run_after_final_workspace_capture(repository):
        nonlocal run_was_replaced
        snapshot = original_capture(repository)
        if patch_was_captured and not run_was_replaced:
            reporting.run_dir.rename(moved_run)
            shutil.copytree(moved_run, reporting.run_dir)
            run_was_replaced = True
        return snapshot

    monkeypatch.setattr(
        WorkspaceSnapshot,
        "capture",
        staticmethod(replace_run_after_final_workspace_capture),
    )

    result = HandoffAcceptanceService(
        patch_capture=RecordingPatchCapture(),
    ).accept(
        HandoffAcceptanceRequest(
            run_dir=reporting.run_dir,
            run_record=reporting.run_record,
            run_ownership=reporting.run_ownership,
        )
    )

    assert patch_was_captured
    assert run_was_replaced
    assert isinstance(result, HandoffRejected)
    assert result.stop_category is StopCategory.SAFETY_VIOLATION
    assert "ownership was lost" in result.reason
    assert load_run_record(moved_run / "run.json").state is WorkflowState.REPORTING
    assert load_run_record(reporting.run_dir / "run.json").state is (
        WorkflowState.REPORTING
    )
    assert WorkspaceSnapshot.capture(
        GitRepository(workspace.repository)
    ).fingerprint == (source_fingerprint)


def test_terminal_report_rejects_run_replacement_during_patch_read(
    tmp_path,
    monkeypatch,
):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-pass",
    )
    clock = TickingClock()
    reporting = _prepare_reporting_run(
        workspace,
        codex_runner=codex_runner,
        clock=clock,
    )
    start_attempt(
        reporting.run_dir,
        phase=AttemptPhase.REPORTING,
        before_workspace_fingerprint=None,
        clock=clock,
        run_ownership=reporting.run_ownership,
    )
    handoff = HandoffAcceptanceService(
        patch_capture=FileSystemFinalPatchCapture(),
    ).accept(
        HandoffAcceptanceRequest(
            run_dir=reporting.run_dir,
            run_record=reporting.run_record,
            run_ownership=reporting.run_ownership,
        )
    )
    assert isinstance(handoff, HandoffAccepted)
    ready = reporting.run_record.transition_to(
        WorkflowState.READY_FOR_HUMAN,
        updated_timestamp=clock().isoformat(),
    )
    save_run_record(ready, reporting.run_dir / "run.json")

    final_patch = reporting.run_dir / "final.patch"
    moved_run = tmp_path / "original-report-run"
    original_read_text = Path.read_text
    replaced = False

    def replace_run_after_patch_read(path, *args, **kwargs):
        nonlocal replaced
        value = original_read_text(path, *args, **kwargs)
        if not replaced and path == final_patch:
            reporting.run_dir.rename(moved_run)
            shutil.copytree(moved_run, reporting.run_dir)
            replaced = True
        return value

    monkeypatch.setattr(Path, "read_text", replace_run_after_patch_read)

    with pytest.raises(RunOwnershipError, match="ownership was lost|replaced"):
        run_report_stage(
            reporting.run_dir,
            run_ownership=reporting.run_ownership,
        )

    assert replaced
    assert not (moved_run / "final-report.md").exists()
    assert not (reporting.run_dir / "final-report.md").exists()


def test_final_handoff_cannot_fall_back_past_an_unreadable_attempt(
    tmp_path,
    monkeypatch,
):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-pass",
    )
    clock = TickingClock()
    reporting = _prepare_reporting_run(
        workspace,
        codex_runner=codex_runner,
        clock=clock,
    )
    inaccessible = start_attempt(
        reporting.run_dir,
        phase=AttemptPhase.VERIFYING,
        before_workspace_fingerprint=None,
        clock=clock,
    )
    start_attempt(
        reporting.run_dir,
        phase=AttemptPhase.REPORTING,
        before_workspace_fingerprint=None,
        clock=clock,
    )
    fail_stat_for_path(monkeypatch, inaccessible.path)

    result = HandoffAcceptanceService(
        patch_capture=FileSystemFinalPatchCapture(),
    ).accept(
        HandoffAcceptanceRequest(
            run_dir=reporting.run_dir,
            run_record=reporting.run_record,
        )
    )

    assert isinstance(result, HandoffRejected)
    assert result.stop_category is StopCategory.SAFETY_VIOLATION
    assert "simulated filesystem inspection failure" in result.reason
    assert not (reporting.run_dir / "final.patch").exists()


def _prepare_reporting_run(workspace, *, codex_runner, clock):
    snapshot = create_trusted_prepared_run(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        verification_runner=ScriptedVerificationRunner([0]),
        clock=clock,
    )
    implementing = snapshot.run_record.transition_to(
        WorkflowState.IMPLEMENTING,
        updated_timestamp=clock().isoformat(),
    )
    save_run_record(implementing, snapshot.run_dir / "run.json")
    implementation = run_test_stage(
        run_implementation_stage,
        AttemptPhase.IMPLEMENTING,
        workspace.config,
        snapshot.run_dir,
        agent_executor=make_agent_executor(
            workspace.config,
            process_runner=codex_runner,
        ),
        clock=clock,
    )
    verifying = implementation.run_record.transition_to(
        WorkflowState.VERIFYING,
        updated_timestamp=clock().isoformat(),
    )
    save_run_record(verifying, snapshot.run_dir / "run.json")
    verification = run_test_stage(
        run_verification_stage,
        AttemptPhase.VERIFYING,
        workspace.config,
        snapshot.run_dir,
        process_runner=ScriptedVerificationRunner([0]),
        clock=clock,
    )
    reviewing = verification.run_record.transition_to(
        WorkflowState.REVIEWING,
        updated_timestamp=clock().isoformat(),
    )
    save_run_record(reviewing, snapshot.run_dir / "run.json")
    review = run_test_stage(
        run_review_stage,
        AttemptPhase.REVIEWING,
        workspace.config,
        snapshot.run_dir,
        agent_executor=make_agent_executor(
            workspace.config,
            process_runner=codex_runner,
        ),
        clock=clock,
    )
    reporting_record = review.run_record.transition_to(
        WorkflowState.REPORTING,
        updated_timestamp=clock().isoformat(),
        current_review_round=1,
    )
    save_run_record(reporting_record, snapshot.run_dir / "run.json")
    return replace(snapshot, run_record=reporting_record)


def test_handoff_service_uses_persisted_attempt_and_does_not_transition_run(
    tmp_path,
    monkeypatch,
):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-pass",
    )
    clock = TickingClock()
    reporting = _prepare_reporting_run(
        workspace,
        codex_runner=codex_runner,
        clock=clock,
    )
    start_attempt(
        reporting.run_dir,
        phase=AttemptPhase.REPORTING,
        before_workspace_fingerprint=None,
        clock=clock,
    )
    request = HandoffAcceptanceRequest(
        run_dir=reporting.run_dir,
        run_record=reporting.run_record,
    )

    result = HandoffAcceptanceService(
        patch_capture=FileSystemFinalPatchCapture(),
    ).accept(request)

    assert isinstance(result, HandoffAccepted)
    assert load_run_record(reporting.run_dir / "run.json").state is (
        WorkflowState.REPORTING
    )
    persisted_reporting = load_attempt_records(reporting.run_dir)[-1]
    assert persisted_reporting.before_workspace_fingerprint == (
        result.initial_workspace.fingerprint
    )
    with pytest.raises(FrozenInstanceError):
        request.run_dir = tmp_path  # type: ignore[misc]
    with pytest.raises(TypeError, match="unexpected keyword"):
        HandoffAcceptanceRequest(
            run_dir=reporting.run_dir,
            run_record=reporting.run_record,
            reporting_attempt=persisted_reporting,  # type: ignore[call-arg]
        )


def test_handoff_service_rejects_a_reporting_attempt_owned_by_another_run(
    tmp_path,
    monkeypatch,
):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-pass",
    )
    clock = TickingClock()
    reporting = _prepare_reporting_run(
        workspace,
        codex_runner=codex_runner,
        clock=clock,
    )
    fingerprint = WorkspaceSnapshot.capture(
        GitRepository(workspace.repository)
    ).fingerprint
    start_attempt(
        reporting.run_dir,
        phase=AttemptPhase.REPORTING,
        before_workspace_fingerprint=fingerprint,
        clock=clock,
    )
    foreign = start_attempt(
        tmp_path / "foreign-run",
        phase=AttemptPhase.REPORTING,
        before_workspace_fingerprint=fingerprint,
        clock=clock,
    )

    result = HandoffAcceptanceService(
        patch_capture=FileSystemFinalPatchCapture(),
    ).accept(
        HandoffAcceptanceRequest(
            run_dir=reporting.run_dir,
            run_record=reporting.run_record,
            attempt=StageAttempt.from_record(foreign),
        )
    )

    assert isinstance(result, HandoffRejected)
    assert result.stop_category is StopCategory.SAFETY_VIOLATION
    assert "does not belong" in result.reason
    assert not (reporting.run_dir / "final.patch").exists()


@pytest.mark.parametrize("request_problem", ["state", "run-directory"])
def test_handoff_service_rejects_invalid_request_context(
    tmp_path,
    monkeypatch,
    request_problem,
):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-pass",
    )
    reporting = _prepare_reporting_run(
        workspace,
        codex_runner=codex_runner,
        clock=TickingClock(),
    )
    run_dir = reporting.run_dir
    run_record = reporting.run_record
    if request_problem == "state":
        run_record = replace(run_record, state=WorkflowState.REVIEWING)
    else:
        run_dir = reporting.run_dir.with_name("foreign-run")

    result = HandoffAcceptanceService(
        patch_capture=FileSystemFinalPatchCapture(),
    ).accept(HandoffAcceptanceRequest(run_dir=run_dir, run_record=run_record))

    assert isinstance(result, HandoffRejected)
    assert result.stop_category is StopCategory.SAFETY_VIOLATION


def test_final_handoff_rejects_patch_capture_failure(tmp_path, monkeypatch):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-pass",
    )

    class FailingPatchCapture:
        def capture(self, request: FinalPatchCaptureRequest) -> FinalPatchReference:
            del request
            raise OSError("patch storage unavailable")

    dependencies = make_run_dependencies(
        workspace.config,
        process_runner=codex_runner,
    )
    dependencies["final_patch_capture"] = FailingPatchCapture()
    result = run_ticket_lifecycle(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        **dependencies,
        verification_runner=ScriptedVerificationRunner([0, 0]),
        clock=TickingClock(),
    )

    assert result.run_record.state is WorkflowState.HUMAN_REQUIRED
    assert result.run_record.stop_reason is not None
    assert result.run_record.stop_reason.category is StopCategory.SAFETY_VIOLATION
    assert "Could not capture the final handoff patch safely" in (
        result.run_record.terminal_reason
    )
    assert not (result.run_dir / "final.patch").exists()
    reporting = load_attempt_records(result.run_dir)[-1]
    assert reporting.before_workspace_fingerprint is not None


@pytest.mark.parametrize("accepted", [True, False], ids=["accepted", "rejected"])
def test_reporting_attempt_write_failure_cannot_reclassify_handoff_decision(
    tmp_path,
    monkeypatch,
    accepted,
):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-pass",
    )
    original_replace = persistence_module.os.replace

    def fail_reporting_attempt_write(source, destination):
        path = Path(destination)
        if path.name == "result.json" and path.parent.name.endswith("-reporting"):
            raise OSError("reporting attempt storage unavailable")
        return original_replace(source, destination)

    class FailingPatchCapture:
        def capture(self, request: FinalPatchCaptureRequest) -> FinalPatchReference:
            del request
            raise OSError("patch storage unavailable")

    monkeypatch.setattr(
        persistence_module.os,
        "replace",
        fail_reporting_attempt_write,
    )
    dependencies = make_run_dependencies(
        workspace.config,
        process_runner=codex_runner,
    )
    if not accepted:
        dependencies["final_patch_capture"] = FailingPatchCapture()

    result = run_ticket_lifecycle(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        **dependencies,
        verification_runner=ScriptedVerificationRunner([0, 0]),
        clock=TickingClock(),
    )

    assert result.controller_error is not None
    assert "reporting attempt storage unavailable" in result.controller_error
    persisted = load_run_record(result.run_dir / "run.json")
    if accepted:
        assert result.run_record.state is WorkflowState.READY_FOR_HUMAN
        assert persisted.state is WorkflowState.READY_FOR_HUMAN
        assert result.run_record.stop_reason is None
    else:
        assert result.run_record.state is WorkflowState.HUMAN_REQUIRED
        assert persisted.state is WorkflowState.HUMAN_REQUIRED
        assert result.run_record.stop_reason is not None
        assert result.run_record.stop_reason.category is StopCategory.SAFETY_VIOLATION
        assert "Could not capture the final handoff patch safely" in (
            result.run_record.terminal_reason
        )


@pytest.mark.parametrize(
    "reference_problem",
    ["missing", "wrong-type", "wrong-path", "wrong-digest", "wrong-size"],
)
def test_final_handoff_rejects_invalid_patch_reference(
    tmp_path,
    monkeypatch,
    reference_problem,
):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-pass",
    )

    class InvalidPatchCapture:
        def capture(self, request: FinalPatchCaptureRequest) -> FinalPatchReference:
            if reference_problem == "missing":
                return FinalPatchReference(request.destination, "0" * 64, 0)
            if reference_problem == "wrong-type":
                return None  # type: ignore[return-value]
            reference = FileSystemFinalPatchCapture().capture(request)
            if reference_problem == "wrong-path":
                return replace(
                    reference, path=request.destination.with_suffix(".other")
                )
            if reference_problem == "wrong-digest":
                return replace(reference, sha256="0" * 64)
            return replace(reference, size_bytes=reference.size_bytes + 1)

    dependencies = make_run_dependencies(
        workspace.config,
        process_runner=codex_runner,
    )
    dependencies["final_patch_capture"] = InvalidPatchCapture()
    result = run_ticket_lifecycle(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        **dependencies,
        verification_runner=ScriptedVerificationRunner([0, 0]),
        clock=TickingClock(),
    )

    assert result.run_record.state is WorkflowState.HUMAN_REQUIRED
    assert result.run_record.stop_reason is not None
    assert result.run_record.stop_reason.category is StopCategory.SAFETY_VIOLATION
    assert "Could not capture the final handoff patch safely" in (
        result.run_record.terminal_reason
    )
    if reference_problem == "missing":
        assert not (result.run_dir / "final.patch").exists()


def test_final_handoff_rejects_workspace_change_during_patch_capture(
    tmp_path,
    monkeypatch,
):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-pass",
    )

    class MutatingPatchCapture:
        def capture(self, request: FinalPatchCaptureRequest) -> FinalPatchReference:
            reference = FileSystemFinalPatchCapture().capture(request)
            (request.repository_path / "late-change.txt").write_text(
                "changed during capture\n",
                encoding="utf-8",
            )
            return reference

    dependencies = make_run_dependencies(
        workspace.config,
        process_runner=codex_runner,
    )
    dependencies["final_patch_capture"] = MutatingPatchCapture()
    result = run_ticket_lifecycle(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        **dependencies,
        verification_runner=ScriptedVerificationRunner([0, 0]),
        clock=TickingClock(),
    )

    assert result.run_record.state is WorkflowState.HUMAN_REQUIRED
    assert result.run_record.stop_reason is not None
    assert result.run_record.stop_reason.category is StopCategory.SAFETY_VIOLATION
    assert "Workspace changed while" in result.run_record.terminal_reason


def test_final_handoff_rejects_post_capture_inspection_failure(
    tmp_path,
    monkeypatch,
):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-pass",
    )
    patch_was_captured = False
    original_capture = WorkspaceSnapshot.capture

    class CapturingPatch:
        def capture(self, request: FinalPatchCaptureRequest) -> FinalPatchReference:
            nonlocal patch_was_captured
            reference = FileSystemFinalPatchCapture().capture(request)
            patch_was_captured = True
            return reference

    def fail_after_patch(repository):
        if patch_was_captured:
            raise OSError("workspace inspection unavailable")
        return original_capture(repository)

    monkeypatch.setattr(
        WorkspaceSnapshot,
        "capture",
        staticmethod(fail_after_patch),
    )
    dependencies = make_run_dependencies(
        workspace.config,
        process_runner=codex_runner,
    )
    dependencies["final_patch_capture"] = CapturingPatch()
    result = run_ticket_lifecycle(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        **dependencies,
        verification_runner=ScriptedVerificationRunner([0, 0]),
        clock=TickingClock(),
    )

    assert patch_was_captured
    assert result.run_record.state is WorkflowState.HUMAN_REQUIRED
    assert result.run_record.stop_reason is not None
    assert result.run_record.stop_reason.category is StopCategory.SAFETY_VIOLATION
    assert "Could not capture the final handoff patch safely" in (
        result.run_record.terminal_reason
    )


def test_final_handoff_accepts_current_matching_evidence(tmp_path, monkeypatch):
    workspace = build_lifecycle_workspace(tmp_path)
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-pass",
    )

    result = run_ticket_lifecycle(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        **make_run_dependencies(workspace.config, process_runner=codex_runner),
        verification_runner=ScriptedVerificationRunner([0, 0]),
        clock=TickingClock(),
    )

    assert result.run_record.state is WorkflowState.READY_FOR_HUMAN
    assert result.run_record.stop_reason is None
    assert (result.run_dir / "final.patch").is_file()
    reporting = load_attempt_records(result.run_dir)[-1]
    assert reporting.phase == "REPORTING"
    assert reporting.status == "COMPLETED"
    assert reporting.before_workspace_fingerprint is not None
    assert reporting.after_workspace_fingerprint == (
        reporting.before_workspace_fingerprint
    )
