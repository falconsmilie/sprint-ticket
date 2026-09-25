from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from .application.agent_execution import (
    CORRECTION_RESULT_CONTRACT,
    CORRECTION_TICKET_FILE,
    AgentExecution,
    AgentExecutionRequest,
    AgentExecutor,
    AgentTaskKind,
    AttemptArtifactLayout,
    InvocationStart,
    RepositoryAccess,
    required_execution_capabilities,
)
from .application.guarded_writable_operation import (
    GuardedWritableOperation,
    GuardedWritableRejectionRequest,
    GuardedWritableRequest,
    WritableBaseline,
    WritableFailedUncertain,
    WritableFailedUnchanged,
    WritableRejectedBeforeStart,
    WritableSafetyStopped,
    WritableSafetyViolation,
    WritableSucceeded,
    format_writable_failure_audit,
    format_writable_guard_stop,
)
from .attempts import (
    AttemptMetadata,
    StageAttempt,
    complete_stage_attempt,
    require_stage_attempt,
    start_attempt,
)
from .audit import (
    changed_files_including_untracked as _changed_files_including_untracked,
)
from .audit import diff_stats_including_untracked as _diff_stats_including_untracked
from .config import AppConfig, VerificationCommand
from .domain.task_results import (
    ImplementationResult,
    ImplementationStatus,
)
from .git import GitCommandError, GitRepository
from .models import (
    ATTEMPT_RESULT_ARTIFACT_NAME,
    AttemptPhase,
    StageOutcome,
    VerificationStatus,
    WorkflowState,
)
from .persistence_codecs import (
    write_implementation_result,
    write_stage_message,
)
from .resolved_config import config_from_resolved_run_policy
from .runs import (
    BASELINE_RECORD_FILE,
    RUN_RECORD_FILE,
    RUN_TICKET_FILE,
    RunError,
    RunRecord,
    load_baseline_record,
    load_run_record,
)
from .verification_evidence import (
    baseline_verification_evidence_problem,
    read_verification_source_fingerprint,
)
from .workspace_guard import WorkspaceGuardInspection

CORRECTION_TICKET_EXCERPT_CHARS = 1200
_CORRECTION_PROMPT_TEMPLATE = (
    Path(__file__).resolve().parent / "application" / "prompts" / "correct.md"
)


class CorrectionError(RunError):
    """Raised when corrective work cannot be prepared or persisted."""


def _require_correction_text(value: object, *, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CorrectionError(f"{field} must be a non-empty string.")


@dataclass(frozen=True)
class VerificationCorrectionCause:
    gate_name: str
    command: tuple[str, ...]
    failure_summary: str
    stdout_excerpt: str
    stderr_excerpt: str
    exit_code: int | None
    result_path: Path

    def __post_init__(self) -> None:
        _require_correction_text(self.gate_name, field="verification cause gate_name")
        if (
            not isinstance(self.command, tuple)
            or not self.command
            or any(not isinstance(item, str) or not item for item in self.command)
        ):
            raise CorrectionError(
                "verification cause command must be a non-empty string tuple."
            )
        _require_correction_text(
            self.failure_summary,
            field="verification cause failure_summary",
        )
        if not isinstance(self.stdout_excerpt, str) or not isinstance(
            self.stderr_excerpt, str
        ):
            raise CorrectionError("verification cause excerpts must be strings.")
        if self.exit_code is not None and (
            not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool)
        ):
            raise CorrectionError(
                "verification cause exit_code must be an integer or null."
            )
        if not isinstance(self.result_path, Path):
            raise CorrectionError("verification cause result_path must be a Path.")


class ReviewCorrectionScope(StrEnum):
    TICKET = "TICKET"
    IMPLEMENTATION = "IMPLEMENTATION"


@dataclass(frozen=True)
class ReviewCorrectionCause:
    finding_id: str
    summary: str
    details: str
    evidence: str
    required_change: str
    acceptance_criteria: tuple[str, ...]
    scope_relation: ReviewCorrectionScope

    def __post_init__(self) -> None:
        for field, value in (
            ("finding_id", self.finding_id),
            ("summary", self.summary),
            ("details", self.details),
            ("evidence", self.evidence),
            ("required_change", self.required_change),
        ):
            _require_correction_text(value, field=f"review cause {field}")
        if (
            not isinstance(self.acceptance_criteria, tuple)
            or not self.acceptance_criteria
            or any(
                not isinstance(item, str) or not item.strip()
                for item in self.acceptance_criteria
            )
        ):
            raise CorrectionError(
                "review cause acceptance_criteria must be a non-empty string tuple."
            )
        if not isinstance(self.scope_relation, ReviewCorrectionScope):
            raise CorrectionError("review cause scope_relation is not supported.")


CorrectionCause = VerificationCorrectionCause | ReviewCorrectionCause


@dataclass(frozen=True)
class CorrectionCauseSet:
    causes: tuple[CorrectionCause, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.causes, tuple):
            raise CorrectionError("correction causes must be a tuple.")
        if not self.causes:
            raise CorrectionError("Correction requires at least one eligible cause.")
        cause_types = {type(cause) for cause in self.causes}
        supported_types = {VerificationCorrectionCause, ReviewCorrectionCause}
        if not cause_types <= supported_types:
            raise CorrectionError(f"Unsupported correction cause set: {self.causes!r}")
        if len(cause_types) != 1:
            raise CorrectionError(
                "Correction causes must come from one source type per correction round."
            )


@dataclass(frozen=True)
class CorrectionSafetyViolation:
    name: str
    expected: str
    actual: str
    message: str


@dataclass(frozen=True)
class CorrectionStageResult:
    run_dir: Path
    run_record: RunRecord
    outcome: StageOutcome
    advance_correction_round: bool
    correction_round: int
    ticket_path: Path | None
    artifact_directory: Path
    agent_execution: AgentExecution[ImplementationResult] | None
    agent_result: ImplementationResult | None
    safety_violations: tuple[CorrectionSafetyViolation, ...]
    correction_causes: tuple[CorrectionCause, ...]
    workspace_guard: WorkspaceGuardInspection | None
    controller_message: str
    after_workspace_fingerprint: str | None
    process_started: bool

    @property
    def source_state(self) -> WorkflowState:
        return WorkflowState.CORRECTING

    @property
    def successful(self) -> bool:
        return self.outcome == StageOutcome.COMPLETED


def run_correction_stage(
    config: AppConfig,
    run_dir: Path | str,
    *,
    cause_set: CorrectionCauseSet,
    agent_executor: AgentExecutor,
    attempt_record: StageAttempt | None = None,
    clock: Callable[[], datetime] | None = None,
) -> CorrectionStageResult:
    run_path = Path(run_dir)
    if attempt_record is not None:
        return _run_correction_stage(
            config,
            run_path,
            cause_set=cause_set,
            agent_executor=agent_executor,
            attempt_record=attempt_record,
            clock=clock,
        )
    run_record = load_run_record(run_path / RUN_RECORD_FILE)
    if run_record.state is not WorkflowState.CORRECTING:
        raise CorrectionError(
            f"Correction requires run state CORRECTING; found {run_record.state.value}."
        )
    owned_attempt = start_attempt(
        run_path,
        phase=AttemptPhase.CORRECTING,
        before_workspace_fingerprint=None,
        clock=clock,
    )
    result = _run_correction_stage(
        config,
        run_path,
        cause_set=cause_set,
        agent_executor=agent_executor,
        attempt_record=StageAttempt.from_record(owned_attempt),
        clock=clock,
    )
    complete_stage_attempt(
        run_path,
        owned_attempt,
        stage_outcome=result.outcome,
        after_workspace_fingerprint=result.after_workspace_fingerprint,
        process_started=result.process_started,
        metadata=AttemptMetadata(controller_message=result.controller_message),
        clock=clock,
    )
    return result


def _run_correction_stage(
    config: AppConfig,
    run_dir: Path | str,
    *,
    cause_set: CorrectionCauseSet,
    agent_executor: AgentExecutor,
    attempt_record: StageAttempt,
    clock: Callable[[], datetime] | None = None,
) -> CorrectionStageResult:
    run_path = Path(run_dir)
    run_record = load_run_record(run_path / RUN_RECORD_FILE)
    # Corrective writes use the same frozen policy as the initial implementation.
    config = config_from_resolved_run_policy(run_record.resolved_policy)
    if run_record.state != WorkflowState.CORRECTING:
        raise CorrectionError(
            f"Correction requires run state CORRECTING; found {run_record.state.value}."
        )
    attempt = require_stage_attempt(
        run_path,
        attempt_record,
        phase=AttemptPhase.CORRECTING,
    )

    baseline_record = load_baseline_record(run_path / BASELINE_RECORD_FILE)
    if baseline_record.branch != run_record.starting_branch:
        raise CorrectionError("Run record and baseline branch do not match.")
    if baseline_record.head_sha != run_record.baseline_sha:
        raise CorrectionError("Run record and baseline HEAD do not match.")

    correction_round = run_record.current_correction_round + 1
    repository = GitRepository(Path(run_record.target_repository_path))
    artifact_directory = attempt.artifact_directory
    baseline_identity = WritableBaseline(
        repository_path=repository.path,
        branch=run_record.starting_branch,
        head_sha=run_record.baseline_sha,
    )
    writable_operation = GuardedWritableOperation(agent_executor, clock=clock)
    if run_record.current_correction_round >= run_record.max_correction_rounds:
        message = "Maximum corrective rounds exhausted; human intervention is required."
        rejected = writable_operation.reject_before_start(
            GuardedWritableRejectionRequest(
                phase=AttemptPhase.CORRECTING,
                artifact_directory=artifact_directory,
                baseline=baseline_identity,
                failure_message=message,
            )
        )
        audit = rejected.audit
        return _finish(
            run_record=run_record,
            run_dir=run_path,
            correction_round=correction_round,
            ticket_path=None,
            artifact_directory=artifact_directory,
            execution=None,
            agent_result=None,
            safety_violations=_correction_violations(audit.safety_violations),
            correction_causes=(),
            workspace_guard=audit.workspace_guard,
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message=message,
            advance_correction_round=False,
            after_workspace_fingerprint=audit.after_workspace_fingerprint,
            process_started=False,
        )

    evidence_problem = baseline_verification_evidence_problem(
        run_path,
        run_record,
        baseline_record,
        verification_commands=config.verification.commands,
    )
    if evidence_problem is not None:
        rejected = writable_operation.reject_before_start(
            GuardedWritableRejectionRequest(
                phase=AttemptPhase.CORRECTING,
                artifact_directory=artifact_directory,
                baseline=baseline_identity,
                failure_message=evidence_problem,
                safety_violations=(
                    WritableSafetyViolation(
                        name="baseline-verification",
                        expected="persisted passing clean-baseline verification",
                        actual=evidence_problem,
                        message="Writable correction is not authorized.",
                    ),
                ),
            )
        )
        audit = rejected.audit
        return _finish(
            run_record=run_record,
            run_dir=run_path,
            correction_round=correction_round,
            ticket_path=None,
            artifact_directory=artifact_directory,
            execution=None,
            agent_result=None,
            safety_violations=_correction_violations(audit.safety_violations),
            correction_causes=(),
            workspace_guard=audit.workspace_guard,
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message=evidence_problem,
            advance_correction_round=False,
            after_workspace_fingerprint=audit.after_workspace_fingerprint,
            process_started=False,
        )

    expected_source_fingerprint, starting_violations = _correction_source_fingerprint(
        run_path,
        run_record,
        verification_commands=config.verification.commands,
    )
    if starting_violations:
        rejected = writable_operation.reject_before_start(
            GuardedWritableRejectionRequest(
                phase=AttemptPhase.CORRECTING,
                artifact_directory=artifact_directory,
                baseline=baseline_identity,
                failure_message=(
                    "Repository no longer matches the recorded correction baseline."
                ),
                safety_violations=tuple(
                    WritableSafetyViolation(
                        item.name,
                        item.expected,
                        item.actual,
                        item.message,
                    )
                    for item in starting_violations
                ),
            )
        )
        audit = rejected.audit
        return _finish(
            run_record=run_record,
            run_dir=run_path,
            correction_round=correction_round,
            ticket_path=None,
            artifact_directory=artifact_directory,
            execution=None,
            agent_result=None,
            safety_violations=_correction_violations(audit.safety_violations),
            correction_causes=(),
            workspace_guard=audit.workspace_guard,
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message=(
                "Repository no longer matches the recorded correction baseline."
            ),
            advance_correction_round=False,
            after_workspace_fingerprint=audit.after_workspace_fingerprint,
            process_started=False,
        )

    selected_causes = cause_set.causes
    active_record = run_record
    ticket_path: Path | None = None
    try:
        ticket_markdown = render_correction_ticket(
            ticket_id=run_record.ticket_id,
            round_number=correction_round,
            cause_set=cause_set,
            run_dir=run_path,
        )
        ticket_path = _write_correction_ticket(
            artifact_directory=artifact_directory,
            ticket_id=active_record.ticket_id,
            round_number=correction_round,
            markdown=ticket_markdown,
        )
        prompt = render_correction_prompt(
            original_ticket=_read_snapshotted_ticket(run_path / RUN_TICKET_FILE),
            correction_ticket=ticket_markdown,
            repository_context=_render_repository_context(
                repository,
                active_record,
                correction_round=correction_round,
            ),
        )
    except Exception as error:  # noqa: BLE001 - preserve writable safety evidence.
        message = (
            f"Could not prepare correction invocation: {type(error).__name__}: {error}"
        )
        rejected = writable_operation.reject_before_start(
            GuardedWritableRejectionRequest(
                phase=AttemptPhase.CORRECTING,
                artifact_directory=artifact_directory,
                baseline=WritableBaseline(
                    repository_path=repository.path,
                    branch=active_record.starting_branch,
                    head_sha=active_record.baseline_sha,
                    expected_workspace_fingerprint=expected_source_fingerprint,
                ),
                failure_message=message,
            )
        )
        audit = rejected.audit
        return _finish(
            run_record=active_record,
            run_dir=run_path,
            correction_round=correction_round,
            ticket_path=ticket_path,
            artifact_directory=artifact_directory,
            execution=None,
            agent_result=None,
            safety_violations=_correction_violations(audit.safety_violations),
            correction_causes=selected_causes,
            workspace_guard=audit.workspace_guard,
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message=message,
            advance_correction_round=False,
            after_workspace_fingerprint=audit.after_workspace_fingerprint,
            process_started=False,
        )
    request = AgentExecutionRequest(
        task_kind=AgentTaskKind.CORRECTION,
        repository_path=repository.path,
        repository_access=RepositoryAccess.WORKSPACE_WRITE,
        prompt=prompt,
        result_contract=CORRECTION_RESULT_CONTRACT,
        artifact_directory=artifact_directory,
        artifact_layout=AttemptArtifactLayout.for_attempt(run_path, artifact_directory),
        policy=run_record.resolved_policy.task_policy(
            AgentTaskKind.CORRECTION
        ).execution_policy,
        required_capabilities=required_execution_capabilities(
            RepositoryAccess.WORKSPACE_WRITE
        ),
    )
    guarded_request = GuardedWritableRequest(
        phase=AttemptPhase.CORRECTING,
        execution_request=request,
        baseline=WritableBaseline(
            repository_path=repository.path,
            branch=active_record.starting_branch,
            head_sha=active_record.baseline_sha,
            expected_workspace_fingerprint=expected_source_fingerprint,
        ),
    )
    writable_outcome = writable_operation.execute(guarded_request)
    audit = writable_outcome.audit
    workspace_guard = audit.workspace_guard
    safety_violations = _correction_violations(audit.safety_violations)

    if isinstance(writable_outcome, WritableSafetyStopped):
        message = (
            format_writable_guard_stop(
                workspace_guard,
                operation=f"correction round {correction_round}",
                run_dir=run_path,
                git_safety=_format_failure_safety(safety_violations),
            )
            if workspace_guard.requires_human
            else "Repository no longer matches the recorded correction baseline."
            if audit.execution is None
            else "Repository safety invariants were violated."
        )
        return _finish(
            run_record=active_record,
            run_dir=run_path,
            correction_round=correction_round,
            ticket_path=ticket_path,
            artifact_directory=artifact_directory,
            execution=audit.execution,
            agent_result=(
                audit.execution.result
                if audit.execution is not None and audit.execution.successful
                else None
            ),
            safety_violations=safety_violations,
            correction_causes=selected_causes,
            workspace_guard=workspace_guard,
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message=message,
            advance_correction_round=False,
            after_workspace_fingerprint=audit.after_workspace_fingerprint,
            process_started=audit.invocation_start is InvocationStart.STARTED,
        )

    if isinstance(
        writable_outcome,
        (WritableRejectedBeforeStart, WritableFailedUnchanged, WritableFailedUncertain),
    ):
        outcome = (
            StageOutcome.HUMAN_REQUIRED
            if isinstance(writable_outcome, WritableFailedUncertain)
            else StageOutcome.FAILED
        )
        message = (
            (
                "The writable agent invocation did not complete successfully and may "
                "have left partial source changes. Automation has stopped for human "
                "inspection. "
            )
            + format_writable_failure_audit(
                audit,
                operation=f"correction round {correction_round}",
                run_dir=run_path,
            )
            if outcome == StageOutcome.HUMAN_REQUIRED
            else audit.failure_message or "Agent execution failed."
        )
        return _finish(
            run_record=active_record,
            run_dir=run_path,
            correction_round=correction_round,
            ticket_path=ticket_path,
            artifact_directory=artifact_directory,
            execution=audit.execution,
            agent_result=None,
            safety_violations=safety_violations,
            correction_causes=selected_causes,
            workspace_guard=workspace_guard,
            outcome=outcome,
            controller_message=message,
            advance_correction_round=False,
            after_workspace_fingerprint=audit.after_workspace_fingerprint,
            process_started=audit.invocation_start is InvocationStart.STARTED,
        )

    if not isinstance(writable_outcome, WritableSucceeded):
        raise CorrectionError("Unknown guarded writable-operation outcome.")
    execution = audit.execution
    assert execution is not None
    agent_result = _require_agent_result(writable_outcome.result)

    if agent_result.status is ImplementationStatus.BLOCKED:
        return _finish(
            run_record=active_record,
            run_dir=run_path,
            correction_round=correction_round,
            ticket_path=ticket_path,
            artifact_directory=artifact_directory,
            execution=execution,
            agent_result=agent_result,
            safety_violations=(),
            correction_causes=selected_causes,
            workspace_guard=workspace_guard,
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message="Correction agent returned BLOCKED.",
            after_workspace_fingerprint=audit.after_workspace_fingerprint,
            process_started=True,
        )

    return _finish(
        run_record=active_record,
        run_dir=run_path,
        correction_round=correction_round,
        ticket_path=ticket_path,
        artifact_directory=artifact_directory,
        execution=execution,
        agent_result=agent_result,
        safety_violations=(),
        correction_causes=selected_causes,
        workspace_guard=workspace_guard,
        outcome=StageOutcome.COMPLETED,
        controller_message=(
            "Correction completed; deterministic verification must run next."
        ),
        after_workspace_fingerprint=audit.after_workspace_fingerprint,
        process_started=True,
    )


def render_correction_ticket(
    *,
    ticket_id: str,
    round_number: int,
    cause_set: CorrectionCauseSet,
    run_dir: Path | str | None = None,
) -> str:
    selected_causes = cause_set.causes
    first_cause = selected_causes[0]
    if isinstance(first_cause, ReviewCorrectionCause):
        body = _render_review_correction_ticket(
            ticket_id=ticket_id,
            round_number=round_number,
            findings=tuple(
                cause
                for cause in selected_causes
                if isinstance(cause, ReviewCorrectionCause)
            ),
        )
    elif isinstance(first_cause, VerificationCorrectionCause):
        body = _render_verification_correction_ticket(
            ticket_id=ticket_id,
            round_number=round_number,
            failures=tuple(
                cause
                for cause in selected_causes
                if isinstance(cause, VerificationCorrectionCause)
            ),
            run_dir=None if run_dir is None else Path(run_dir),
        )
    else:
        raise CorrectionError(f"Unsupported correction cause: {first_cause!r}")
    return body.rstrip() + "\n"


def render_correction_prompt(
    *,
    original_ticket: str,
    correction_ticket: str,
    repository_context: str,
) -> str:
    replacements = {
        "{{ORIGINAL_TICKET}}": original_ticket,
        "{{CORRECTION_TICKET}}": correction_ticket,
        "{{REPOSITORY_CONTEXT}}": repository_context,
    }
    template = _CORRECTION_PROMPT_TEMPLATE.read_text(encoding="utf-8")
    missing = tuple(key for key in replacements if key not in template)
    if missing:
        raise CorrectionError(
            "Correction prompt template is missing placeholders: "
            + ", ".join(sorted(missing))
        )
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def _finish(
    *,
    run_record: RunRecord,
    run_dir: Path,
    correction_round: int,
    ticket_path: Path | None,
    artifact_directory: Path,
    execution: AgentExecution[ImplementationResult] | None,
    agent_result: ImplementationResult | None,
    safety_violations: tuple[CorrectionSafetyViolation, ...],
    correction_causes: tuple[CorrectionCause, ...],
    workspace_guard: WorkspaceGuardInspection | None = None,
    outcome: StageOutcome,
    controller_message: str,
    advance_correction_round: bool = True,
    after_workspace_fingerprint: str | None = None,
    process_started: bool | None = None,
) -> CorrectionStageResult:
    _write_agent_result(
        artifact_directory,
        agent_result,
        outcome=outcome,
        controller_message=controller_message,
    )
    process_started = (
        process_started
        if process_started is not None
        else False
        if execution is None
        else execution.invocation_start is InvocationStart.STARTED
    )
    return CorrectionStageResult(
        run_dir=run_dir,
        run_record=run_record,
        outcome=outcome,
        advance_correction_round=advance_correction_round,
        correction_round=correction_round,
        ticket_path=ticket_path,
        artifact_directory=artifact_directory,
        agent_execution=execution,
        agent_result=agent_result,
        safety_violations=safety_violations,
        correction_causes=correction_causes,
        workspace_guard=workspace_guard,
        controller_message=controller_message,
        after_workspace_fingerprint=after_workspace_fingerprint,
        process_started=process_started,
    )


def _write_agent_result(
    artifact_directory: Path,
    result: ImplementationResult | None,
    *,
    outcome: StageOutcome,
    controller_message: str,
) -> None:
    path = artifact_directory / ATTEMPT_RESULT_ARTIFACT_NAME
    if result is not None:
        write_implementation_result(path, result)
    else:
        write_stage_message(path, status=outcome, message=controller_message)


def _write_correction_ticket(
    *,
    artifact_directory: Path,
    ticket_id: str,
    round_number: int,
    markdown: str,
) -> Path:
    path = artifact_directory / CORRECTION_TICKET_FILE
    if path.exists():
        raise CorrectionError(f"Correction ticket already exists: {path}")
    path.write_text(markdown, encoding="utf-8", newline="\n")
    return path


def _render_review_correction_ticket(
    *,
    ticket_id: str,
    round_number: int,
    findings: tuple[ReviewCorrectionCause, ...],
) -> str:
    lines = [
        f"# {ticket_id} - Corrective Round {round_number}",
        "",
        "## Context",
        "",
        (
            "This corrective ticket addresses required findings from the "
            f"independent review of {ticket_id}."
        ),
        "",
        (
            "The complete implementation remains governed by the original "
            f"{ticket_id} ticket."
        ),
        "",
        "## Required corrections",
    ]
    for finding in findings:
        lines.extend(
            [
                "",
                f"### {finding.finding_id} - {_text_or_default(finding.summary)}",
                "",
                f"Scope relation: {_display_enum(finding.scope_relation.value)}",
                "",
                "Finding:",
                _text_or_default(finding.details),
                "",
                "Evidence:",
                _text_or_default(finding.evidence),
                "",
                "Required change:",
                _text_or_default(finding.required_change),
                "",
                "Acceptance criteria:",
            ]
        )
        for criterion in finding.acceptance_criteria:
            lines.append(f"- {criterion}")
        if not finding.acceptance_criteria:
            lines.append("- Satisfy the required change above.")
    lines.extend(_constraints_and_validation())
    return "\n".join(lines)


def _render_verification_correction_ticket(
    *,
    ticket_id: str,
    round_number: int,
    failures: tuple[VerificationCorrectionCause, ...],
    run_dir: Path | None,
) -> str:
    lines = [
        f"# {ticket_id} - Corrective Round {round_number}",
        "",
        "## Context",
        "",
        (
            "This corrective ticket addresses deterministic verification "
            f"failures for {ticket_id}."
        ),
        "",
        (
            "The complete implementation remains governed by the original "
            f"{ticket_id} ticket."
        ),
        "",
        "## Required corrections",
    ]
    for failure in failures:
        result_reference = _format_result_reference(failure.result_path, run_dir)
        output_excerpt = _verification_output_excerpt(failure)
        lines.extend(
            [
                "",
                f"### Verification failure - {failure.gate_name}",
                "",
                "Command:",
                "```text",
                _ticket_excerpt(_format_command(failure.command)),
                "```",
                "",
                "Exit code:",
                str(failure.exit_code) if failure.exit_code is not None else "n/a",
                "",
                "Failure:",
                _text_or_default(failure.failure_summary),
                "",
                "Relevant output:",
                "```text",
                output_excerpt,
                "```",
                "",
                "Full typed result:",
                result_reference,
            ]
        )
    lines.extend(_constraints_and_validation())
    return "\n".join(lines)


def _constraints_and_validation() -> list[str]:
    return [
        "",
        "## Constraints",
        "",
        "- Address only the required findings above.",
        "- Preserve valid existing work from the original ticket.",
        "- Do not broaden the ticket.",
        "- Do not perform unrelated refactoring.",
        "- Do not stage or commit changes.",
        "",
        "## Validation",
        "",
        "Run relevant targeted tests after making the correction.",
    ]


def _render_repository_context(
    repository: GitRepository,
    run_record: RunRecord,
    *,
    correction_round: int,
) -> str:
    try:
        changed_files = _worktree_changed_files(repository, run_record.baseline_sha)
        diff_stats = _diff_stats_including_untracked(
            repository,
            run_record.baseline_sha,
        ).strip()
        current_branch = repository.current_branch()
    except (GitCommandError, OSError, ValueError) as error:
        raise CorrectionError(
            f"Could not inspect repository context: {error}"
        ) from error

    lines = [
        f"Run ID: {run_record.run_id}",
        f"Ticket ID: {run_record.ticket_id}",
        f"Correction round: {correction_round}",
        f"Target repository: {run_record.target_repository_path}",
        f"Baseline SHA: {run_record.baseline_sha}",
        f"Starting branch: {run_record.starting_branch}",
        f"Current branch: {_format_optional(current_branch)}",
        "",
        "Current implementation diff range:",
        "baseline SHA -> complete current working tree",
        "",
        "Changed files relative to baseline:",
    ]
    if changed_files:
        lines.extend(f"- {file_path}" for file_path in changed_files)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "Diff statistics:",
            "```text",
            diff_stats if diff_stats else "No diff statistics available.",
            "```",
        ]
    )
    return "\n".join(lines)


def _read_snapshotted_ticket(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except OSError as error:
        raise CorrectionError(
            f"Could not read snapshotted ticket: {path}: {error}"
        ) from error
    except UnicodeDecodeError as error:
        raise CorrectionError(
            f"Snapshotted ticket must be valid UTF-8 Markdown: {path}"
        ) from error


def _require_agent_result(value: Any) -> ImplementationResult:
    if not isinstance(value, ImplementationResult):
        raise CorrectionError("Correction agent result has the wrong domain type.")
    return value


def _correction_violations(
    violations: tuple[WritableSafetyViolation, ...],
) -> tuple[CorrectionSafetyViolation, ...]:
    return tuple(
        CorrectionSafetyViolation(
            name=violation.name,
            expected=violation.expected,
            actual=violation.actual,
            message=violation.message,
        )
        for violation in violations
    )


def _correction_source_fingerprint(
    run_path: Path,
    run_record: RunRecord,
    *,
    verification_commands: tuple[VerificationCommand, ...],
) -> tuple[str | None, tuple[CorrectionSafetyViolation, ...]]:
    try:
        expected = read_verification_source_fingerprint(
            run_path,
            run_record,
            expected_statuses=frozenset(
                {VerificationStatus.FAIL, VerificationStatus.PASS}
            ),
            verification_commands=verification_commands,
        )
    except ValueError as error:
        return (
            None,
            (
                CorrectionSafetyViolation(
                    name="verification-evidence",
                    expected="readable canonical workspace fingerprint",
                    actual=str(error),
                    message="Could not validate the correction source evidence.",
                ),
            ),
        )
    return expected, ()


def _format_failure_safety(
    violations: tuple[CorrectionSafetyViolation, ...],
) -> str:
    if not violations:
        return "branch unchanged; HEAD unchanged; staging empty"
    return "; ".join(
        f"{violation.name} expected {violation.expected}, got {violation.actual}"
        for violation in violations
    )


def _worktree_changed_files(
    repository: GitRepository,
    baseline_sha: str,
) -> tuple[str, ...]:
    return _changed_files_including_untracked(repository, baseline_sha)


def _verification_output_excerpt(failure: VerificationCorrectionCause) -> str:
    output: list[str] = []
    if failure.stdout_excerpt:
        output.extend(["Stdout:", _ticket_excerpt(failure.stdout_excerpt)])
    if failure.stderr_excerpt:
        if output:
            output.append("")
        output.extend(["Stderr:", _ticket_excerpt(failure.stderr_excerpt)])
    if not output:
        return "No stdout or stderr was captured."
    return "\n".join(output).rstrip()


def _ticket_excerpt(value: str) -> str:
    if len(value) <= CORRECTION_TICKET_EXCERPT_CHARS:
        return value.rstrip()
    omission = len(value) - CORRECTION_TICKET_EXCERPT_CHARS
    return (
        f"{value[:CORRECTION_TICKET_EXCERPT_CHARS].rstrip()}\n"
        f"... <truncated {omission} chars; see full typed result>"
    )


def _format_result_reference(result_path: Path, run_dir: Path | None) -> str:
    if run_dir is None:
        return str(result_path)
    try:
        return str(result_path.relative_to(run_dir))
    except ValueError:
        return str(result_path)


def _format_command(command: tuple[str, ...]) -> str:
    if not command:
        return "<unknown command>"
    return " ".join(_quote_arg(argument) for argument in command)


def _quote_arg(argument: str) -> str:
    if not argument or any(character.isspace() for character in argument):
        return repr(argument)
    return argument


def _display_enum(value: str) -> str:
    if not value:
        return "Unspecified"
    return value.replace("_", " ").title()


def _text_or_default(value: str) -> str:
    return value if value else "Not provided."


def _format_optional(value: str | None) -> str:
    return "<detached>" if value is None else value


__all__ = [
    "CorrectionCause",
    "CorrectionCauseSet",
    "CorrectionError",
    "CorrectionSafetyViolation",
    "CorrectionStageResult",
    "ReviewCorrectionCause",
    "ReviewCorrectionScope",
    "VerificationCorrectionCause",
    "render_correction_prompt",
    "render_correction_ticket",
    "run_correction_stage",
]
