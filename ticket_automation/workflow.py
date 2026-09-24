"""Synchronous lifecycle orchestration and transition authority."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .application.agent_execution import PersistedAgentExecutorFactory
from .application.lifecycle import (
    StageAttempt,
    StageContext,
    StageDecision,
    StageHandler,
    build_active_stage_handlers,
    dispatch_stage,
)
from .application.lifecycle.progress import LifecycleProgress
from .application.lifecycle.resume import resume_preflight_problem, resume_problem
from .application.ports.handoff import FinalPatchCapture
from .application.ports.preflight import ProviderPreflight
from .application.ports.reporting import ReportPublication, TerminalReportPublisher
from .attempts import (
    AttemptMetadata,
    AttemptRecord,
    complete_stage_attempt,
    start_attempt,
    update_attempt,
)
from .config import AppConfig, ConfigError
from .corrections import CorrectionStageResult
from .failure_classification import (
    TerminalStop,
    classify_unexpected_controller_failure,
)
from .git import GitRepository
from .git_safety import WorkspaceSnapshot
from .implementation import ImplementationStageResult
from .locking import RepositoryRunLock, acquire_repository_run_lock
from .models import (
    StopCategory,
    StopReason,
    WorkflowState,
    phase_for_active_state,
)
from .persistence import timestamp_now
from .persistence_codecs import write_stage_message_result
from .preflight import PreflightResult
from .reporting import FilesystemTerminalReportPublisher, format_lifecycle_result
from .resolved_config import ResolvedRunPolicy, config_from_resolved_run_policy
from .review import ReviewStageResult
from .runs import (
    RUN_RECORD_FILE,
    RunError,
    RunRecord,
    create_run_snapshot,
    load_run_record,
    save_run_record,
)
from .verification import VerificationProcessRunner, VerificationStageResult

TERMINAL_STATES = frozenset(
    {
        WorkflowState.READY_FOR_HUMAN,
        WorkflowState.HUMAN_REQUIRED,
        WorkflowState.FAILED,
    }
)


@dataclass(frozen=True)
class LifecycleSafetyViolation:
    name: str
    expected: str
    actual: str
    message: str


@dataclass(frozen=True)
class LifecycleResult:
    run_dir: Path
    run_record: RunRecord
    preflight_result: PreflightResult
    implementation_result: ImplementationStageResult | None
    verification_results: tuple[VerificationStageResult, ...]
    review_results: tuple[ReviewStageResult, ...]
    correction_results: tuple[CorrectionStageResult, ...]
    report_result: ReportPublication | None = None
    safety_violations: tuple[LifecycleSafetyViolation, ...] = ()
    controller_error: str | None = None

    @property
    def successful(self) -> bool:
        return self.run_record.state is WorkflowState.READY_FOR_HUMAN

    @property
    def terminal_state(self) -> WorkflowState:
        return self.run_record.state


@dataclass
class LifecycleController:
    """Injectable synchronous controller used by run, resume, and unit tests."""

    handlers: Mapping[WorkflowState, StageHandler]
    repository_lock: RepositoryRunLock
    report_publisher: TerminalReportPublisher
    progress: LifecycleProgress
    clock: Callable[[], datetime] | None = None

    def drive(
        self,
        run_dir: Path,
        preflight_result: PreflightResult,
        run_record: RunRecord,
    ) -> LifecycleResult:
        return _drive_lifecycle(
            run_dir,
            preflight_result,
            run_record,
            handlers=self.handlers,
            progress=self.progress,
            repository_lock=self.repository_lock,
            report_publisher=self.report_publisher,
            clock=self.clock,
        )

    def drive_safely(
        self,
        run_dir: Path,
        preflight_result: PreflightResult,
        run_record: RunRecord,
        *,
        exception_prefix: str,
    ) -> LifecycleResult:
        return _drive_lifecycle_safely(
            run_dir,
            preflight_result,
            run_record,
            handlers=self.handlers,
            progress=self.progress,
            repository_lock=self.repository_lock,
            report_publisher=self.report_publisher,
            exception_prefix=exception_prefix,
            clock=self.clock,
        )


def run_ticket_lifecycle(
    config: AppConfig,
    ticket_path: Path | str,
    *,
    runs_dir: Path | str,
    provider_preflight: ProviderPreflight,
    resolved_policy: ResolvedRunPolicy,
    agent_executor_factory: PersistedAgentExecutorFactory,
    final_patch_capture: FinalPatchCapture,
    report_publisher: TerminalReportPublisher | None = None,
    verification_runner: VerificationProcessRunner | None = None,
    clock: Callable[[], datetime] | None = None,
) -> LifecycleResult:
    publisher = _report_publisher_or_default(report_publisher)
    with acquire_repository_run_lock(
        config.project.repo,
        run_id=None,
        current_state=WorkflowState.PREPARING.value,
        clock=clock,
    ) as repository_lock:
        return _run_ticket_lifecycle_locked(
            config,
            ticket_path,
            runs_dir=runs_dir,
            repository_lock=repository_lock,
            provider_preflight=provider_preflight,
            resolved_policy=resolved_policy,
            agent_executor_factory=agent_executor_factory,
            final_patch_capture=final_patch_capture,
            report_publisher=publisher,
            verification_runner=verification_runner,
            clock=clock,
        )


def _run_ticket_lifecycle_locked(
    config: AppConfig,
    ticket_path: Path | str,
    *,
    runs_dir: Path | str,
    repository_lock: RepositoryRunLock,
    provider_preflight: ProviderPreflight,
    resolved_policy: ResolvedRunPolicy,
    agent_executor_factory: PersistedAgentExecutorFactory,
    final_patch_capture: FinalPatchCapture,
    report_publisher: TerminalReportPublisher,
    verification_runner: VerificationProcessRunner | None,
    clock: Callable[[], datetime] | None,
) -> LifecycleResult:
    executors = agent_executor_factory.create_executors(resolved_policy)
    snapshot = create_run_snapshot(
        config,
        ticket_path,
        runs_dir=runs_dir,
        provider_preflight=provider_preflight,
        resolved_policy=resolved_policy,
        clock=clock,
    )
    config = config_from_resolved_run_policy(snapshot.run_record.resolved_policy)
    _update_repository_lock(repository_lock, snapshot.run_record)
    handlers = build_active_stage_handlers(
        config=config,
        executors=executors,
        final_patch_capture=final_patch_capture,
        verification_runner=verification_runner,
    )
    progress = LifecycleProgress()
    return LifecycleController(
        handlers=handlers,
        repository_lock=repository_lock,
        report_publisher=report_publisher,
        progress=progress,
        clock=clock,
    ).drive_safely(
        snapshot.run_dir,
        snapshot.preflight_result,
        snapshot.run_record,
        exception_prefix="Internal TicketAutomation exception",
    )


def resume_ticket_lifecycle(
    run_id: str,
    *,
    runs_dir: Path | str,
    agent_executor_factory: PersistedAgentExecutorFactory,
    final_patch_capture: FinalPatchCapture,
    report_publisher: TerminalReportPublisher | None = None,
    verification_runner: VerificationProcessRunner | None = None,
    clock: Callable[[], datetime] | None = None,
) -> LifecycleResult:
    publisher = _report_publisher_or_default(report_publisher)
    run_dir = Path(runs_dir) / run_id
    if not run_dir.is_dir():
        raise RunError(f"Run directory does not exist: {run_dir}")
    preflight_result = PreflightResult(())
    run_record = load_run_record(run_dir / RUN_RECORD_FILE)
    config = config_from_resolved_run_policy(run_record.resolved_policy)
    with acquire_repository_run_lock(
        run_record.target_repository_path,
        run_id=run_record.run_id,
        current_state=run_record.state.value,
        clock=clock,
    ) as repository_lock:
        return _resume_ticket_lifecycle_locked(
            config,
            run_dir,
            preflight_result,
            run_record,
            repository_lock=repository_lock,
            agent_executor_factory=agent_executor_factory,
            final_patch_capture=final_patch_capture,
            report_publisher=publisher,
            verification_runner=verification_runner,
            clock=clock,
        )


def _resume_ticket_lifecycle_locked(
    config: AppConfig,
    run_dir: Path,
    preflight_result: PreflightResult,
    run_record: RunRecord,
    *,
    repository_lock: RepositoryRunLock,
    agent_executor_factory: PersistedAgentExecutorFactory,
    final_patch_capture: FinalPatchCapture,
    report_publisher: TerminalReportPublisher,
    verification_runner: VerificationProcessRunner | None,
    clock: Callable[[], datetime] | None,
) -> LifecycleResult:
    if run_record.state in TERMINAL_STATES:
        return _empty_result(run_dir, run_record, preflight_result)

    compatibility_problem = agent_executor_factory.compatibility_problem(
        run_record.resolved_policy
    )
    problem = compatibility_problem
    if problem is None:
        problem = resume_preflight_problem(config, run_dir, run_record)
    if problem is None:
        problem = resume_problem(run_dir, run_record)
    if problem is not None:
        run_record = _mark_human_required(
            run_dir,
            terminal_reason=problem,
            clock=clock,
        )
        _update_repository_lock(repository_lock, run_record)
        _publish_terminal_report(report_publisher, run_dir, run_record)
        return _empty_result(run_dir, run_record, preflight_result)

    progress = LifecycleProgress()
    try:
        executors = agent_executor_factory.create_executors(run_record.resolved_policy)
        handlers = build_active_stage_handlers(
            config=config,
            executors=executors,
            final_patch_capture=final_patch_capture,
            verification_runner=verification_runner,
        )
        return LifecycleController(
            handlers=handlers,
            repository_lock=repository_lock,
            report_publisher=report_publisher,
            progress=progress,
            clock=clock,
        ).drive_safely(
            run_dir,
            preflight_result,
            run_record,
            exception_prefix="Internal TicketAutomation exception during resume",
        )
    except ConfigError as error:
        run_record = _mark_human_required(
            run_dir,
            terminal_reason=str(error),
            clock=clock,
        )
        _update_repository_lock(repository_lock, run_record)
        _publish_terminal_report(report_publisher, run_dir, run_record)
        return _empty_result(run_dir, run_record, preflight_result)
    except RunError as error:
        return _controller_failure_result(
            run_dir,
            preflight_result,
            repository_lock,
            progress,
            report_publisher,
            controller_error=str(error),
            clock=clock,
        )
    except Exception as error:  # noqa: BLE001 - terminal evidence must be persisted.
        return _controller_failure_result(
            run_dir,
            preflight_result,
            repository_lock,
            progress,
            report_publisher,
            controller_error=(
                "Internal TicketAutomation exception during resume: "
                f"{type(error).__name__}: {error}"
            ),
            clock=clock,
        )


def _drive_lifecycle(
    run_dir: Path,
    preflight_result: PreflightResult,
    run_record: RunRecord,
    *,
    handlers: Mapping[WorkflowState, StageHandler],
    progress: LifecycleProgress,
    repository_lock: RepositoryRunLock,
    report_publisher: TerminalReportPublisher,
    clock: Callable[[], datetime] | None,
) -> LifecycleResult:
    while run_record.state not in TERMINAL_STATES:
        _update_repository_lock(repository_lock, run_record)
        attempt = _start_stage_attempt(run_dir, run_record, clock=clock)
        context = StageContext(
            run_dir,
            run_record,
            None if attempt is None else StageAttempt.from_record(attempt),
            clock,
            lambda active_attempt=attempt: _mark_attempt_process_started(
                active_attempt
            ),
        )
        decision = dispatch_stage(context, handlers)
        updated = _apply_stage_decision(run_record, decision, clock=clock)
        transition_persisted = False
        completion = decision.completion
        if completion is not None:
            if attempt is None:
                raise RunError(
                    "Executed stage decision has no controller-owned attempt."
                )
            reporting_evidence = completion.reporting_evidence
            if reporting_evidence is not None:
                save_run_record(updated, run_dir / RUN_RECORD_FILE)
                transition_persisted = True
                write_stage_message_result(
                    run_dir,
                    attempt,
                    status=reporting_evidence.status,
                    message=reporting_evidence.message,
                )
            complete_stage_attempt(
                run_dir,
                attempt,
                stage_outcome=completion.outcome,
                after_workspace_fingerprint=completion.after_workspace_fingerprint,
                process_started=completion.process_started,
                metadata=AttemptMetadata(controller_message=completion.message),
                clock=clock,
            )
        if not transition_persisted:
            save_run_record(updated, run_dir / RUN_RECORD_FILE)
        run_record = updated
        _update_repository_lock(repository_lock, run_record)
        progress.record(decision.result, run_record=run_record)

    report_result = _publish_terminal_report(report_publisher, run_dir, run_record)
    _update_repository_lock(repository_lock, run_record)
    return LifecycleResult(
        run_dir=run_dir,
        run_record=run_record,
        preflight_result=preflight_result,
        implementation_result=progress.implementation_result,
        verification_results=tuple(progress.verification_results),
        review_results=tuple(progress.review_results),
        correction_results=tuple(progress.correction_results),
        report_result=report_result,
    )


def _drive_lifecycle_safely(
    run_dir: Path,
    preflight_result: PreflightResult,
    run_record: RunRecord,
    *,
    handlers: Mapping[WorkflowState, StageHandler],
    progress: LifecycleProgress,
    repository_lock: RepositoryRunLock,
    report_publisher: TerminalReportPublisher,
    exception_prefix: str,
    clock: Callable[[], datetime] | None,
) -> LifecycleResult:
    try:
        return _drive_lifecycle(
            run_dir,
            preflight_result,
            run_record,
            handlers=handlers,
            progress=progress,
            repository_lock=repository_lock,
            report_publisher=report_publisher,
            clock=clock,
        )
    except RunError as error:
        controller_error = str(error)
    except Exception as error:  # noqa: BLE001 - terminal evidence must be persisted.
        controller_error = f"{exception_prefix}: {type(error).__name__}: {error}"
    return _controller_failure_result(
        run_dir,
        preflight_result,
        repository_lock,
        progress,
        report_publisher,
        controller_error=controller_error,
        clock=clock,
    )


def _apply_stage_decision(
    run_record: RunRecord,
    decision: StageDecision,
    *,
    clock: Callable[[], datetime] | None = None,
) -> RunRecord:
    """Validate a handler request through the domain transition API."""

    if decision.source_state is not run_record.state:
        raise RunError("Stage decision source state does not match the run record.")

    stop_reason = (
        None if decision.terminal_stop is None else decision.terminal_stop.reason
    )
    return run_record.transition_to(
        decision.requested_state,
        updated_timestamp=_timestamp(clock),
        current_correction_round=decision.current_correction_round,
        current_review_round=decision.current_review_round,
        terminal_reason=None if stop_reason is None else stop_reason.message,
        stop_reason=stop_reason,
    )


def _start_stage_attempt(
    run_dir: Path,
    run_record: RunRecord,
    *,
    clock: Callable[[], datetime] | None,
) -> AttemptRecord | None:
    phase = phase_for_active_state(run_record.state)
    if phase is None:
        return None
    before_fingerprint = None
    if run_record.state in {
        WorkflowState.PREPARING,
        WorkflowState.VERIFYING,
        WorkflowState.REVIEWING,
        WorkflowState.REPORTING,
    }:
        try:
            before_fingerprint = WorkspaceSnapshot.capture(
                GitRepository(Path(run_record.target_repository_path))
            ).fingerprint
        except (OSError, RuntimeError, ValueError):
            pass
    return start_attempt(
        run_dir,
        phase=phase,
        before_workspace_fingerprint=before_fingerprint,
        clock=clock,
    )


def _mark_attempt_process_started(attempt: AttemptRecord | None) -> None:
    if attempt is None:
        raise RunError("A state without an attempt cannot start a process.")
    update_attempt(attempt, process_started=True)


def _controller_failure_result(
    run_dir: Path,
    preflight_result: PreflightResult,
    repository_lock: RepositoryRunLock,
    progress: LifecycleProgress,
    report_publisher: TerminalReportPublisher,
    *,
    controller_error: str,
    clock: Callable[[], datetime] | None,
) -> LifecycleResult:
    run_record = _mark_controller_exception(
        run_dir,
        controller_error=controller_error,
        clock=clock,
    )
    _update_repository_lock(repository_lock, run_record)
    _publish_terminal_report(report_publisher, run_dir, run_record)
    return LifecycleResult(
        run_dir=run_dir,
        run_record=run_record,
        preflight_result=preflight_result,
        implementation_result=progress.implementation_result,
        verification_results=tuple(progress.verification_results),
        review_results=tuple(progress.review_results),
        correction_results=tuple(progress.correction_results),
        report_result=None,
        controller_error=controller_error,
    )


def _empty_result(
    run_dir: Path,
    run_record: RunRecord,
    preflight_result: PreflightResult,
) -> LifecycleResult:
    return LifecycleResult(
        run_dir=run_dir,
        run_record=run_record,
        preflight_result=preflight_result,
        implementation_result=None,
        verification_results=(),
        review_results=(),
        correction_results=(),
    )


def _publish_terminal_report(
    publisher: TerminalReportPublisher,
    run_dir: Path,
    run_record: RunRecord,
) -> ReportPublication | None:
    """Keep terminal report I/O best-effort and exactly once per controller exit."""

    try:
        return publisher.publish(run_dir, run_record)
    except Exception:  # noqa: BLE001 - presentation cannot mask terminal state.
        return None


def _report_publisher_or_default(
    publisher: TerminalReportPublisher | None,
) -> TerminalReportPublisher:
    if publisher is not None:
        return publisher
    return FilesystemTerminalReportPublisher()


def _mark_controller_exception(
    run_dir: Path,
    *,
    controller_error: str,
    clock: Callable[[], datetime] | None,
) -> RunRecord:
    run_record = load_run_record(run_dir / RUN_RECORD_FILE)
    if run_record.state in TERMINAL_STATES:
        return run_record
    stop = classify_unexpected_controller_failure(
        run_dir,
        run_record,
        message=controller_error,
    )
    return _mark_terminal_stop(run_dir, stop=stop, clock=clock)


def _mark_terminal_stop(
    run_dir: Path,
    *,
    stop: TerminalStop,
    clock: Callable[[], datetime] | None,
) -> RunRecord:
    record_path = run_dir / RUN_RECORD_FILE
    run_record = load_run_record(record_path)
    updated = run_record.transition_to(
        stop.state,
        updated_timestamp=_timestamp(clock),
        terminal_reason=stop.reason.message,
        stop_reason=stop.reason,
    )
    save_run_record(updated, record_path)
    return updated


def _mark_human_required(
    run_dir: Path,
    *,
    terminal_reason: str,
    clock: Callable[[], datetime] | None,
    category: StopCategory = StopCategory.HUMAN_JUDGMENT_REQUIRED,
) -> RunRecord:
    return _mark_terminal_stop(
        run_dir,
        stop=TerminalStop(
            state=WorkflowState.HUMAN_REQUIRED,
            reason=StopReason(
                category=category,
                message=terminal_reason,
                retryable=False,
            ),
        ),
        clock=clock,
    )


def _update_repository_lock(
    repository_lock: RepositoryRunLock,
    run_record: RunRecord,
) -> None:
    repository_lock.update(
        run_id=run_record.run_id,
        current_state=run_record.state.value,
        target_repository_path=run_record.target_repository_path,
    )


def _timestamp(clock: Callable[[], datetime] | None) -> str:
    return timestamp_now(clock)


__all__ = [
    "TERMINAL_STATES",
    "LifecycleController",
    "LifecycleResult",
    "LifecycleSafetyViolation",
    "format_lifecycle_result",
    "resume_ticket_lifecycle",
    "run_ticket_lifecycle",
]
