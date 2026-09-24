from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import ticket_automation.workflow as workflow_module
from tests.helpers import (
    create_test_run_snapshot,
    create_trusted_prepared_run,
    make_agent_executors,
    make_final_patch_capture,
    make_report_publisher,
    make_run_dependencies,
)
from tests.lifecycle_characterization_fixtures import (
    TickingClock,
    build_lifecycle_workspace,
)
from ticket_automation.application.lifecycle import (
    ReportingEvidence,
    StageContext,
    StageDecision,
    build_active_stage_handlers,
)
from ticket_automation.application.lifecycle.contracts import (
    decision_for_stage_result,
    immediate_transition,
)
from ticket_automation.application.lifecycle.progress import LifecycleProgress
from ticket_automation.attempts import load_attempt_records
from ticket_automation.models import (
    AttemptStatus,
    StageOutcome,
    StopCategory,
    WorkflowState,
)
from ticket_automation.runs import RUN_RECORD_FILE, load_run_record, save_run_record
from ticket_automation.workflow import (
    LifecycleController,
    resume_ticket_lifecycle,
)


@dataclass(frozen=True)
class StubResult:
    source_state: WorkflowState
    outcome: StageOutcome
    controller_message: str = "stub stage result"
    after_workspace_fingerprint: str | None = "stub-fingerprint"
    process_started: bool = False


@dataclass
class StubHandler:
    decisions: dict[WorkflowState, StageDecision]
    calls: list[WorkflowState] = field(default_factory=list)

    def handle(self, context: StageContext) -> StageDecision:
        self.calls.append(context.run_record.state)
        return self.decisions[context.run_record.state]


@dataclass
class RecordingLock:
    states: list[str] = field(default_factory=list)

    def update(self, *, run_id, current_state, target_repository_path) -> None:
        del run_id, target_repository_path
        self.states.append(current_state)


def _completed_decision(record) -> StageDecision:
    result = StubResult(record.state, StageOutcome.COMPLETED)
    return decision_for_stage_result(record, result)


def test_controller_loop_dispatches_stubs_updates_lock_and_persists_transitions(
    tmp_path,
):
    workspace = build_lifecycle_workspace(tmp_path)
    snapshot = create_test_run_snapshot(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        clock=TickingClock(),
    )
    preparing = _completed_decision(snapshot.run_record)
    prepared = immediate_transition(WorkflowState.PREPARED, WorkflowState.IMPLEMENTING)
    implementation_result = StubResult(
        WorkflowState.IMPLEMENTING, StageOutcome.HUMAN_REQUIRED
    )
    implementation = decision_for_stage_result(
        snapshot.run_record.transition_to(
            WorkflowState.PREPARED,
            updated_timestamp="2026-09-24T00:00:00Z",
        ).transition_to(
            WorkflowState.IMPLEMENTING,
            updated_timestamp="2026-09-24T00:00:01Z",
        ),
        implementation_result,
        stop_category=StopCategory.HUMAN_JUDGMENT_REQUIRED,
    )
    handler = StubHandler(
        {
            WorkflowState.PREPARING: preparing,
            WorkflowState.PREPARED: prepared,
            WorkflowState.IMPLEMENTING: implementation,
        }
    )
    lock = RecordingLock()

    result = LifecycleController(
        handlers={
            WorkflowState.PREPARING: handler,
            WorkflowState.PREPARED: handler,
            WorkflowState.IMPLEMENTING: handler,
        },
        progress=LifecycleProgress(),
        repository_lock=lock,
        report_publisher=make_report_publisher(),
        clock=TickingClock(),
    ).drive(
        snapshot.run_dir,
        snapshot.preflight_result,
        snapshot.run_record,
    )

    assert handler.calls == [
        WorkflowState.PREPARING,
        WorkflowState.PREPARED,
        WorkflowState.IMPLEMENTING,
    ]
    assert result.run_record.state is WorkflowState.HUMAN_REQUIRED
    assert load_run_record(snapshot.run_dir / "run.json") == result.run_record
    assert lock.states[-1] == WorkflowState.HUMAN_REQUIRED.value


@dataclass
class InvalidHandler:
    def handle(self, context: StageContext) -> StageDecision:
        return immediate_transition(context.run_record.state, WorkflowState.REPORTING)


@dataclass
class UntrustedHandler:
    def handle(self, context: StageContext):
        del context
        return object()


def test_controller_loop_rejects_illegal_transition_from_stub_handler(tmp_path):
    workspace = build_lifecycle_workspace(tmp_path)
    snapshot = create_test_run_snapshot(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        clock=TickingClock(),
    )

    with pytest.raises(ValueError, match="PREPARING -> REPORTING"):
        LifecycleController(
            handlers={WorkflowState.PREPARING: InvalidHandler()},
            progress=LifecycleProgress(),
            repository_lock=RecordingLock(),
            report_publisher=make_report_publisher(),
            clock=TickingClock(),
        ).drive(
            snapshot.run_dir,
            snapshot.preflight_result,
            snapshot.run_record,
        )

    assert (
        load_run_record(snapshot.run_dir / "run.json").state is WorkflowState.PREPARING
    )


def test_controller_does_not_persist_transition_before_attempt_completion(
    tmp_path,
    monkeypatch,
):
    workspace = build_lifecycle_workspace(tmp_path)
    snapshot = create_test_run_snapshot(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        clock=TickingClock(),
    )
    correcting = snapshot.run_record
    for state in (
        WorkflowState.PREPARED,
        WorkflowState.IMPLEMENTING,
        WorkflowState.VERIFYING,
        WorkflowState.CORRECTION_PENDING,
        WorkflowState.CORRECTING,
    ):
        correcting = correcting.transition_to(
            state,
            updated_timestamp="2026-09-24T00:00:00Z",
        )
    save_run_record(correcting, snapshot.run_dir / RUN_RECORD_FILE)
    decision = decision_for_stage_result(
        correcting,
        StubResult(WorkflowState.CORRECTING, StageOutcome.COMPLETED),
        current_correction_round=1,
    )

    class SimulatedPersistenceInterruption(RuntimeError):
        pass

    def interrupt_attempt_completion(*args, **kwargs):
        del args, kwargs
        raise SimulatedPersistenceInterruption

    monkeypatch.setattr(
        workflow_module,
        "complete_stage_attempt",
        interrupt_attempt_completion,
    )
    controller = LifecycleController(
        handlers={
            WorkflowState.CORRECTING: StubHandler(
                {WorkflowState.CORRECTING: decision}
            )
        },
        progress=LifecycleProgress(),
        repository_lock=RecordingLock(),
        report_publisher=make_report_publisher(),
        clock=TickingClock(),
    )

    with pytest.raises(SimulatedPersistenceInterruption):
        controller.drive(
            snapshot.run_dir,
            snapshot.preflight_result,
            correcting,
        )

    assert (
        load_run_record(snapshot.run_dir / RUN_RECORD_FILE).state
        is WorkflowState.CORRECTING
    )
    assert load_attempt_records(snapshot.run_dir)[-1].status is AttemptStatus.STARTED


def test_controller_rejects_decisions_not_built_by_contract_factories(tmp_path):
    workspace = build_lifecycle_workspace(tmp_path)
    snapshot = create_test_run_snapshot(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        clock=TickingClock(),
    )

    with pytest.raises(RuntimeError, match="decision factories"):
        LifecycleController(
            handlers={WorkflowState.PREPARING: UntrustedHandler()},
            progress=LifecycleProgress(),
            repository_lock=RecordingLock(),
            report_publisher=make_report_publisher(),
            clock=TickingClock(),
        ).drive(
            snapshot.run_dir,
            snapshot.preflight_result,
            snapshot.run_record,
        )


def test_stage_decision_factories_reject_untrusted_result_fields(tmp_path):
    workspace = build_lifecycle_workspace(tmp_path)
    snapshot = create_test_run_snapshot(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        clock=TickingClock(),
    )

    with pytest.raises(RuntimeError, match="does not belong"):
        decision_for_stage_result(
            snapshot.run_record,
            StubResult(WorkflowState.REPORTING, StageOutcome.COMPLETED),
        )

    invalid_fingerprint = StubResult(
        WorkflowState.PREPARING,
        StageOutcome.COMPLETED,
        after_workspace_fingerprint=42,  # type: ignore[arg-type]
    )
    with pytest.raises(TypeError, match="fingerprint"):
        decision_for_stage_result(snapshot.run_record, invalid_fingerprint)

    with pytest.raises(ValueError, match="Only reporting decisions"):
        decision_for_stage_result(
            snapshot.run_record,
            StubResult(WorkflowState.PREPARING, StageOutcome.COMPLETED),
            reporting_evidence=ReportingEvidence("PASS", "stub stage result"),
        )

    reporting_record = snapshot.run_record
    for state in (
        WorkflowState.PREPARED,
        WorkflowState.IMPLEMENTING,
        WorkflowState.VERIFYING,
        WorkflowState.REVIEWING,
        WorkflowState.REPORTING,
    ):
        reporting_record = reporting_record.transition_to(
            state,
            updated_timestamp="2026-09-24T00:00:00Z",
        )
    reporting_result = StubResult(WorkflowState.REPORTING, StageOutcome.COMPLETED)
    with pytest.raises(ValueError, match="message"):
        decision_for_stage_result(
            reporting_record,
            reporting_result,
            reporting_evidence=ReportingEvidence("PASS", "different message"),
        )
    with pytest.raises(ValueError, match="status"):
        decision_for_stage_result(
            reporting_record,
            reporting_result,
            reporting_evidence=ReportingEvidence(
                StageOutcome.HUMAN_REQUIRED.value,
                reporting_result.controller_message,
            ),
        )


def test_active_handler_mapping_covers_every_nonterminal_state(tmp_path):
    workspace = build_lifecycle_workspace(tmp_path)
    dependencies = make_run_dependencies(workspace.config)
    handlers = build_active_stage_handlers(
        config=workspace.config,
        executors=make_agent_executors(workspace.config),
        final_patch_capture=dependencies["final_patch_capture"],
        verification_runner=None,
    )

    assert set(handlers) == set(WorkflowState) - {
        WorkflowState.READY_FOR_HUMAN,
        WorkflowState.HUMAN_REQUIRED,
        WorkflowState.FAILED,
    }


@dataclass
class RaisingHandler:
    def handle(self, context: StageContext):
        del context
        raise RuntimeError("stub handler failed")


def test_handler_exception_uses_conservative_controller_failure_outcome(tmp_path):
    workspace = build_lifecycle_workspace(tmp_path)
    snapshot = create_test_run_snapshot(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        clock=TickingClock(),
    )

    result = LifecycleController(
        handlers={WorkflowState.PREPARING: RaisingHandler()},
        progress=LifecycleProgress(),
        repository_lock=RecordingLock(),
        report_publisher=make_report_publisher(),
        clock=TickingClock(),
    ).drive_safely(
        snapshot.run_dir,
        snapshot.preflight_result,
        snapshot.run_record,
        exception_prefix="Internal TicketAutomation exception",
    )

    assert result.run_record.state is WorkflowState.FAILED
    assert result.run_record.stop_reason is not None
    assert result.run_record.stop_reason.category is StopCategory.CONTROLLER_FAILURE
    assert result.controller_error is not None
    assert "stub handler failed" in result.controller_error


@dataclass
class FailingExecutorFactory:
    def compatibility_problem(self, resolved_policy):
        del resolved_policy

    def create_executors(self, resolved_policy):
        del resolved_policy
        raise RuntimeError("executor construction failed")


def test_resume_default_publisher_generates_report_after_executor_failure(tmp_path):
    workspace = build_lifecycle_workspace(tmp_path)
    snapshot = create_trusted_prepared_run(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        clock=TickingClock(),
    )

    result = resume_ticket_lifecycle(
        snapshot.run_record.run_id,
        runs_dir=workspace.runs_dir,
        agent_executor_factory=FailingExecutorFactory(),
        final_patch_capture=make_final_patch_capture(),
        clock=TickingClock(),
    )

    assert result.run_record.state is WorkflowState.FAILED
    assert (snapshot.run_dir / "report.md").is_file()
    assert "executor construction failed" in (result.controller_error or "")


@dataclass
class RaisingPublisher:
    calls: int = 0

    def publish(self, run_dir, run_record):
        del run_dir, run_record
        self.calls += 1
        raise OSError("report storage unavailable")


def test_terminal_report_failure_is_best_effort_and_attempted_once(tmp_path):
    workspace = build_lifecycle_workspace(tmp_path)
    snapshot = create_test_run_snapshot(
        workspace.config,
        workspace.ticket,
        runs_dir=workspace.runs_dir,
        clock=TickingClock(),
    )
    result_value = StubResult(
        WorkflowState.PREPARING,
        StageOutcome.HUMAN_REQUIRED,
        controller_message="stop after preparation",
        after_workspace_fingerprint=None,
    )
    decision = decision_for_stage_result(
        snapshot.run_record,
        result_value,
        stop_category=StopCategory.BASELINE_FAILURE,
    )
    publisher = RaisingPublisher()

    result = LifecycleController(
        handlers={
            WorkflowState.PREPARING: StubHandler({WorkflowState.PREPARING: decision})
        },
        progress=LifecycleProgress(),
        repository_lock=RecordingLock(),
        report_publisher=publisher,
        clock=TickingClock(),
    ).drive_safely(
        snapshot.run_dir,
        snapshot.preflight_result,
        snapshot.run_record,
        exception_prefix="Internal TicketAutomation exception",
    )

    assert result.run_record.state is WorkflowState.HUMAN_REQUIRED
    assert result.controller_error is None
    assert publisher.calls == 1


def test_lifecycle_handlers_do_not_import_transition_persistence_services():
    lifecycle_dir = (
        Path(__file__).parents[1] / "ticket_automation" / "application" / "lifecycle"
    )
    forbidden = {
        "complete_attempt",
        "complete_stage_attempt",
        "finish_phase_attempt",
        "save_run_record",
        "start_attempt",
    }
    for path in lifecycle_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert imported.isdisjoint(forbidden), path
        persistence_aliases = {
            alias.asname or alias.name.rsplit(".", 1)[-1]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name in {
                "ticket_automation.attempts",
                "ticket_automation.runs",
            }
        }
        qualified_calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in persistence_aliases
        }
        assert qualified_calls.isdisjoint(forbidden), path


def test_established_reporting_import_paths_remain_available():
    from ticket_automation.presentation.lifecycle import (
        format_lifecycle_result as presentation_formatter,
    )
    from ticket_automation.presentation.reporting import (
        run_report_stage as presentation_report_stage,
    )
    from ticket_automation.reporting import run_report_stage
    from ticket_automation.workflow import format_lifecycle_result

    assert run_report_stage is presentation_report_stage
    assert format_lifecycle_result is presentation_formatter
