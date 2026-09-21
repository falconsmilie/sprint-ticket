from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from tests.helpers import GIT, run_git
from tests.lifecycle_characterization_fixtures import (
    ScriptedVerificationRunner,
    TickingClock,
    assert_attempt_ledger,
    build_lifecycle_workspace,
    configure_fake_codex_actions,
)
from ticket_automation.attempts import latest_writable_attempt
from ticket_automation.domain.task_results import ImplementationResult, ReviewResult
from ticket_automation.git import GitRepository
from ticket_automation.git_safety import WorkspaceSnapshot
from ticket_automation.models import StopCategory, WorkflowState
from ticket_automation.reporting import collect_report_context
from ticket_automation.workflow import run_ticket_lifecycle

pytestmark = pytest.mark.skipif(GIT is None, reason="git executable is required")


@dataclass(frozen=True)
class WritableStopCase:
    action: str
    state: WorkflowState
    category: StopCategory
    attempt_status: str
    process_started: bool
    workspace_changed: bool
    remove_executable: bool = False


@pytest.mark.parametrize(
    "case",
    [
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
    ],
)
def test_writable_implementation_stop_matrix(
    tmp_path,
    monkeypatch,
    case: WritableStopCase,
):
    workspace = build_lifecycle_workspace(tmp_path)
    baseline = WorkspaceSnapshot.capture(GitRepository(workspace.repository))
    configure_fake_codex_actions(monkeypatch, tmp_path, case.action)
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
        writable.before_workspace_fingerprint
        != writable.after_workspace_fingerprint
    ) is case.workspace_changed
    current = WorkspaceSnapshot.capture(GitRepository(workspace.repository))
    assert (current.fingerprint != baseline.fingerprint) is case.workspace_changed
    assert (result.run_dir / "report.md").is_file()


def test_required_review_finding_creates_correction_work(tmp_path, monkeypatch):
    workspace = build_lifecycle_workspace(tmp_path)
    configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-corrections",
        "modify-correction",
        "review-pass",
    )

    result = run_ticket_lifecycle(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        verification_runner=ScriptedVerificationRunner([0, 0, 0]),
        clock=TickingClock(),
    )

    assert result.run_record.state == WorkflowState.READY_FOR_HUMAN
    assert result.run_record.stop_reason is None
    assert result.run_record.current_correction_round == 1
    attempts = assert_attempt_ledger(
        result.run_dir,
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

    context = collect_report_context(result.run_dir, result.run_record)
    controller = context["controller"]
    agent = context["agent"]
    assert isinstance(agent["implementation"], ImplementationResult)
    assert controller["review_results"]
    assert all(
        isinstance(review_result, ReviewResult)
        for review_result in controller["review_results"]
    )
    assert all("result" not in attempt for attempt in controller["attempts"])


def test_review_finding_outside_correction_scope_remains_human_required(
    tmp_path, monkeypatch
):
    workspace = build_lifecycle_workspace(tmp_path)
    configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-unsafe",
    )

    result = run_ticket_lifecycle(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
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
    configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-inconsistent",
    )

    result = run_ticket_lifecycle(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        verification_runner=ScriptedVerificationRunner([0, 0]),
        clock=TickingClock(),
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert result.run_record.stop_reason is not None
    assert (
        result.run_record.stop_reason.category
        == StopCategory.HUMAN_JUDGMENT_REQUIRED
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
    configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "modify-correction",
    )

    result = run_ticket_lifecycle(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        verification_runner=ScriptedVerificationRunner([0, 1, 1]),
        clock=TickingClock(),
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert result.run_record.stop_reason is not None
    assert (
        result.run_record.stop_reason.category
        == StopCategory.HUMAN_JUDGMENT_REQUIRED
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
    action_path = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "modify",
        "review-pass-arm",
    )
    arm_path = tmp_path / "handoff-drift.arm"
    monkeypatch.setenv("TA_FAKE_CODEX_ARM_FILE", str(arm_path))
    original_capture = WorkspaceSnapshot.capture
    armed_capture_count = 0
    drift_applied = False

    def capture_with_drift(repository):
        nonlocal armed_capture_count, drift_applied
        if arm_path.is_file() and not drift_applied:
            armed_capture_count += 1
        if armed_capture_count == 2 and not drift_applied:
            drift_applied = True
            apply_drift()
        return original_capture(repository)

    def apply_drift():
        if drift == "branch":
            run_git(workspace.repository, "checkout", "-b", "handoff-drift")
        elif drift == "head":
            (workspace.repository / "head.txt").write_text(
                "head drift\n", encoding="utf-8"
            )
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

    monkeypatch.setattr(
        WorkspaceSnapshot,
        "capture",
        staticmethod(capture_with_drift),
    )
    result = run_ticket_lifecycle(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        verification_runner=ScriptedVerificationRunner([0, 0]),
        clock=TickingClock(),
    )

    assert json.loads(action_path.read_text(encoding="utf-8")) == []
    assert drift_applied
    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert result.run_record.stop_reason is not None
    assert result.run_record.stop_reason.category == StopCategory.SAFETY_VIOLATION
    attempts = assert_attempt_ledger(
        result.run_dir,
        [
            ("PREPARING", "COMPLETED"),
            ("IMPLEMENTING", "COMPLETED"),
            ("VERIFYING", "COMPLETED"),
            ("REVIEWING", "COMPLETED"),
            ("REPORTING", "HUMAN_REQUIRED"),
        ],
    )
    reporting = attempts[-1]
    assert reporting.process_started is False
    reporting_evidence = json.loads(
        (reporting.artifact_directory / "result.json").read_text(encoding="utf-8")
    )
    assert reporting_evidence["status"] == "HUMAN_REQUIRED"
    writable = latest_writable_attempt(result.run_dir)
    assert writable is not None
    current = WorkspaceSnapshot.capture(GitRepository(workspace.repository))
    assert not current.matches_fingerprint(writable.after_workspace_fingerprint)
    if drift == "branch":
        assert current.branch != result.run_record.starting_branch
    elif drift == "head":
        assert current.head_sha != result.run_record.baseline_sha
    elif drift == "staging":
        assert current.staged_paths
    else:
        assert current.branch == result.run_record.starting_branch
        assert current.head_sha == result.run_record.baseline_sha
        assert not current.staged_paths
    assert not (result.run_dir / "final.patch").exists()
