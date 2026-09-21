from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from . import writable_worker
from ._verification_artifacts import (
    _baseline_verification_evidence_problem,
    _read_verification_source_fingerprint,
    _VerificationArtifactError,
)
from .application.agent_execution import (
    CORRECTION_RESULT_CONTRACT,
    AgentExecution,
    AgentExecutionPolicy,
    AgentExecutionRequest,
    AgentExecutor,
    AgentFailureCategory,
    AgentTaskKind,
    NetworkAccess,
    RepositoryAccess,
    required_execution_capabilities,
)
from .attempts import (
    finish_phase_attempt,
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
from .failure_classification import classify_writable_failure
from .git import GitCommandError, GitRepository
from .git_safety import (
    WorkspaceChange,
    WorkspaceSnapshot,
    workspace_safety_changes,
)
from .models import (
    ATTEMPT_RESULT_ARTIFACT_NAME,
    AttemptPhase,
    StageOutcome,
    StopCategory,
    WorkflowState,
)
from .resolved_config import config_from_resolved_run_config
from .runs import (
    BASELINE_RECORD_FILE,
    RUN_RECORD_FILE,
    RUN_TICKET_FILE,
    RunError,
    RunRecord,
    load_baseline_record,
    load_run_record,
)
from .task_result_codecs import encode_implementation_result
from .workspace_guard import WorkspaceGuardInspection
from .writable_attempts import WritableAttempt

CORRECTIONS_DIR_NAME = "corrections"
CORRECTION_EXECUTIONS_DIR_NAME = "correction-executions"
CORRECTION_TICKET_SUFFIX = "CORR"
CORRECTION_TICKET_EXCERPT_CHARS = 1200
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CORRECTION_PROMPT_TEMPLATE = _PROJECT_ROOT / "prompts" / "correct.md"
_AGENT_TIMEOUT_SECONDS = 60 * 60


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

    @property
    def successful(self) -> bool:
        return self.outcome == StageOutcome.COMPLETED


@dataclass(frozen=True)
class _FailedWritableAudit:
    safety_violations: tuple[CorrectionSafetyViolation, ...]
    changed_files: tuple[str, ...]
    workspace_guard: WorkspaceGuardInspection | None
    human_required: bool


def run_correction_stage(
    config: AppConfig,
    run_dir: Path | str,
    *,
    cause_set: CorrectionCauseSet,
    agent_executor: AgentExecutor,
    clock: Callable[[], datetime] | None = None,
) -> CorrectionStageResult:
    run_path = Path(run_dir)
    run_record = load_run_record(run_path / RUN_RECORD_FILE)
    # Corrective writes use the same frozen policy as the initial implementation.
    config = config_from_resolved_run_config(run_record.resolved_config)
    if run_record.state != WorkflowState.CORRECTING:
        raise CorrectionError(
            f"Correction requires run state CORRECTING; found {run_record.state.value}."
        )

    baseline_record = load_baseline_record(run_path / BASELINE_RECORD_FILE)
    if baseline_record.branch != run_record.starting_branch:
        raise CorrectionError("Run record and baseline branch do not match.")
    if baseline_record.head_sha != run_record.baseline_sha:
        raise CorrectionError("Run record and baseline HEAD do not match.")

    correction_round = run_record.current_correction_round + 1
    repository = GitRepository(Path(run_record.target_repository_path))
    try:
        before_snapshot = WorkspaceSnapshot.capture(repository)
        before_fingerprint = before_snapshot.fingerprint
    except (OSError, RuntimeError, ValueError):
        before_fingerprint = None
    attempt_record = start_attempt(
        run_path,
        phase=AttemptPhase.CORRECTING,
        before_workspace_fingerprint=before_fingerprint,
        clock=clock,
    )
    artifact_directory = attempt_record.artifact_directory
    if run_record.current_correction_round >= run_record.max_correction_rounds:
        return _finish(
            run_record=run_record,
            run_dir=run_path,
            correction_round=correction_round,
            ticket_path=None,
            artifact_directory=artifact_directory,
            execution=None,
            agent_result=None,
            safety_violations=(),
            correction_causes=(),
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message=(
                "Maximum corrective rounds exhausted; human intervention is required."
            ),
            advance_correction_round=False,
        )

    evidence_problem = _baseline_verification_evidence_problem(
        run_path,
        run_record,
        baseline_record,
        verification_commands=config.verification.commands,
    )
    if evidence_problem is not None:
        return _finish(
            run_record=run_record,
            run_dir=run_path,
            correction_round=correction_round,
            ticket_path=None,
            artifact_directory=artifact_directory,
            execution=None,
            agent_result=None,
            safety_violations=(
                CorrectionSafetyViolation(
                    name="baseline-verification",
                    expected="persisted passing clean-baseline verification",
                    actual=evidence_problem,
                    message="Writable correction is not authorized.",
                ),
            ),
            correction_causes=(),
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message=evidence_problem,
            advance_correction_round=False,
        )

    starting_snapshot = WorkspaceSnapshot.capture(repository)
    starting_violations = _inspect_correction_invariants(
        repository,
        run_record,
        phase="before correction",
        current_snapshot=starting_snapshot,
    )
    starting_violations += _inspect_correction_source_fingerprint(
        run_path,
        run_record,
        starting_snapshot,
        verification_commands=config.verification.commands,
    )
    if starting_violations:
        return _finish(
            run_record=run_record,
            run_dir=run_path,
            correction_round=correction_round,
            ticket_path=None,
            artifact_directory=artifact_directory,
            execution=None,
            agent_result=None,
            safety_violations=starting_violations,
            correction_causes=(),
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message=(
                "Repository no longer matches the recorded correction baseline."
            ),
            advance_correction_round=False,
        )

    selected_causes = cause_set.causes
    ticket_markdown = render_correction_ticket(
        ticket_id=run_record.ticket_id,
        round_number=correction_round,
        cause_set=cause_set,
        run_dir=run_path,
    )
    active_record = run_record
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
    request = AgentExecutionRequest(
        task_kind=AgentTaskKind.CORRECTION,
        repository_path=repository.path,
        repository_access=RepositoryAccess.WORKSPACE_WRITE,
        prompt=prompt,
        result_contract=CORRECTION_RESULT_CONTRACT,
        artifact_directory=artifact_directory,
        policy=AgentExecutionPolicy(
            timeout_seconds=_AGENT_TIMEOUT_SECONDS,
            network_access=NetworkAccess.ALLOWED,
        ),
        required_capabilities=required_execution_capabilities(
            RepositoryAccess.WORKSPACE_WRITE
        ),
    )
    writable_invocation = writable_worker.run_writable_agent(
        repository=repository,
        run_dir=run_path,
        operation=f"correction-round-{correction_round}",
        phase=AttemptPhase.CORRECTING,
        attempt_record=attempt_record,
        executor=agent_executor,
        request=request,
        clock=clock,
    )
    workspace_guard = writable_invocation.workspace_guard
    if not writable_invocation.invocation_permitted:
        return _finish(
            run_record=active_record,
            run_dir=run_path,
            correction_round=correction_round,
            ticket_path=ticket_path,
            artifact_directory=artifact_directory,
            execution=None,
            agent_result=None,
            safety_violations=(),
            correction_causes=selected_causes,
            workspace_guard=workspace_guard,
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message=writable_worker._format_writable_guard_stop(
                workspace_guard,
                operation=f"correction round {correction_round}",
                run_dir=run_path,
                git_safety=_format_failure_safety(()),
            ),
            advance_correction_round=False,
        )

    if (
        writable_invocation.execution is not None
        and not writable_invocation.execution.successful
    ):
        execution = writable_invocation.execution
        failed_audit = _audit_failed_writable_invocation(
            repository,
            run_path,
            active_record,
            round_number=correction_round,
            execution=execution,
            writable_attempt=writable_invocation.attempt,
            after_workspace=writable_invocation.after_workspace,
            workspace_guard=workspace_guard,
        )
        decision = classify_writable_failure(
            repository,
            attempt=writable_invocation.attempt,
            message=execution.failure_message or "Agent execution failed.",
            category_if_safe=StopCategory.EXTERNAL_TOOL_FAILURE,
            retryable_if_safe=True,
            malformed_result=execution.failure_category
            in {
                AgentFailureCategory.MISSING_RESULT,
                AgentFailureCategory.INVALID_RESULT,
            },
            untrusted_completion=execution.failure_category
            in {
                AgentFailureCategory.TIMEOUT,
            },
        )
        outcome = (
            StageOutcome.HUMAN_REQUIRED
            if (
                decision.state == WorkflowState.HUMAN_REQUIRED
                or workspace_guard.requires_human
            )
            else StageOutcome.FAILED
        )
        message = (
            _failed_writable_message(
                operation=f"correction round {correction_round}",
                execution=execution,
                audit=failed_audit,
                run_dir=run_path,
            )
            if outcome == StageOutcome.HUMAN_REQUIRED
            else decision.reason.message
        )
        return _finish(
            run_record=active_record,
            run_dir=run_path,
            correction_round=correction_round,
            ticket_path=ticket_path,
            artifact_directory=artifact_directory,
            execution=execution,
            agent_result=None,
            safety_violations=failed_audit.safety_violations,
            correction_causes=selected_causes,
            workspace_guard=workspace_guard,
            outcome=outcome,
            controller_message=message,
            advance_correction_round=False,
        )

    execution = writable_invocation.execution
    if execution is None:
        raise CorrectionError("Writable agent boundary returned no execution result.")
    agent_result = _require_agent_result(execution.result)
    if writable_invocation.after_workspace is None:
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
            controller_message=writable_worker._format_writable_guard_stop(
                workspace_guard,
                operation=f"correction round {correction_round}",
                run_dir=run_path,
                git_safety=_format_failure_safety(()),
            ),
        )
    safety_violations = _inspect_correction_invariants(
        repository,
        active_record,
        phase="after correction",
        current_snapshot=writable_invocation.after_workspace,
    )
    if workspace_guard.requires_human:
        message = writable_worker._format_writable_guard_stop(
            workspace_guard,
            operation=f"correction round {correction_round}",
            run_dir=run_path,
            git_safety=_format_failure_safety(safety_violations),
        )
        return _finish(
            run_record=active_record,
            run_dir=run_path,
            correction_round=correction_round,
            ticket_path=ticket_path,
            artifact_directory=artifact_directory,
            execution=execution,
            agent_result=agent_result,
            safety_violations=safety_violations,
            correction_causes=selected_causes,
            workspace_guard=workspace_guard,
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message=message,
        )
    if safety_violations:
        return _finish(
            run_record=active_record,
            run_dir=run_path,
            correction_round=correction_round,
            ticket_path=ticket_path,
            artifact_directory=artifact_directory,
            execution=execution,
            agent_result=agent_result,
            safety_violations=safety_violations,
            correction_causes=selected_causes,
            workspace_guard=workspace_guard,
            outcome=StageOutcome.HUMAN_REQUIRED,
            controller_message="Repository safety invariants were violated.",
        )

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


def format_correction_result(result: CorrectionStageResult) -> str:
    rows = [
        f"Correction state: {result.run_record.state.value}",
        f"Correction round: {result.correction_round}",
        f"Artifacts: {result.artifact_directory}",
        result.controller_message,
    ]
    if result.ticket_path is not None:
        rows.append(f"Correction ticket: {result.ticket_path}")
    if result.agent_execution is not None and not result.agent_execution.successful:
        rows.extend(_format_agent_artifacts(result.agent_execution))
    if result.workspace_guard is not None and result.workspace_guard.requires_human:
        if result.workspace_guard.artifact_path is not None:
            rows.append(f"Workspace guard: {result.workspace_guard.artifact_path}")
        if result.workspace_guard.has_violation:
            rows.append("Workspace hygiene violations:")
            rows.extend(
                "  - "
                f"{environment.root_path.relative_to(result.workspace_guard.after.repository_path)}: "
                f"marker {environment.primary_marker_path.relative_to(result.workspace_guard.after.repository_path)}"
                for environment in result.workspace_guard.new_environments
            )
        if result.workspace_guard.has_inspection_failure:
            rows.append("Workspace environment inspection was incomplete.")
    if result.agent_result is not None:
        rows.append(f"Agent status: {result.agent_result.status.value}")
    if result.safety_violations:
        rows.append("Safety violations:")
        rows.extend(
            f"  - {violation.name}: expected {violation.expected}, got {violation.actual}"
            for violation in result.safety_violations
        )
    return "\n".join(rows)


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
) -> CorrectionStageResult:
    _write_agent_result(
        artifact_directory,
        agent_result,
        outcome=outcome,
        controller_message=controller_message,
    )
    after_fingerprint: str | None = None
    try:
        after_fingerprint = WorkspaceSnapshot.capture(
            GitRepository(Path(run_record.target_repository_path))
        ).fingerprint
    except (OSError, RuntimeError, ValueError):
        pass
    finish_phase_attempt(
        run_dir,
        phase=AttemptPhase.CORRECTING,
        stage_outcome=outcome,
        after_workspace_fingerprint=after_fingerprint,
        process_started=(None if execution is None else execution.invocation_started),
        execution_path=None,
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
    )


def _write_agent_result(
    artifact_directory: Path,
    result: ImplementationResult | None,
    *,
    outcome: StageOutcome,
    controller_message: str,
) -> None:
    path = artifact_directory / ATTEMPT_RESULT_ARTIFACT_NAME
    payload = (
        encode_implementation_result(result)
        if result is not None
        else {"status": outcome.value, "message": controller_message}
    )
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_correction_ticket(
    *,
    artifact_directory: Path,
    ticket_id: str,
    round_number: int,
    markdown: str,
) -> Path:
    path = artifact_directory / "correction-ticket.md"
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


def _inspect_correction_invariants(
    repository: GitRepository,
    run_record: RunRecord,
    *,
    phase: str,
    current_snapshot: WorkspaceSnapshot | None = None,
) -> tuple[CorrectionSafetyViolation, ...]:
    snapshot = current_snapshot or WorkspaceSnapshot.capture(repository)
    changes = workspace_safety_changes(
        snapshot,
        expected_repository_path=run_record.target_repository_path,
        expected_branch=run_record.starting_branch,
        expected_head_sha=run_record.baseline_sha,
    )
    return _correction_changes(changes, phase=phase)


def _correction_changes(
    changes: tuple[WorkspaceChange, ...],
    *,
    phase: str,
) -> tuple[CorrectionSafetyViolation, ...]:
    return tuple(
        CorrectionSafetyViolation(
            name=change.name,
            expected=change.expected,
            actual=change.actual,
            message=f"{change.message.rstrip('.')} {phase}.",
        )
        for change in changes
    )


def _inspect_correction_source_fingerprint(
    run_path: Path,
    run_record: RunRecord,
    current: WorkspaceSnapshot,
    *,
    verification_commands: tuple[VerificationCommand, ...],
) -> tuple[CorrectionSafetyViolation, ...]:
    try:
        expected = _read_verification_source_fingerprint(
            run_path,
            run_record,
            expected_statuses=frozenset({"FAIL", "PASS"}),
            verification_commands=verification_commands,
        )
    except _VerificationArtifactError as error:
        return (
            CorrectionSafetyViolation(
                name="verification-evidence",
                expected="readable canonical workspace fingerprint",
                actual=str(error),
                message="Could not validate the correction source evidence.",
            ),
        )
    if not current.inspection_complete:
        return (
            CorrectionSafetyViolation(
                name="inspection-incomplete",
                expected="complete workspace inspection",
                actual="; ".join(current.inspection_errors) or "incomplete",
                message="Could not validate the correction source workspace.",
            ),
        )
    if current.matches_fingerprint(expected):
        return ()
    return (
        CorrectionSafetyViolation(
            name="workspace-fingerprint",
            expected=expected,
            actual=current.fingerprint,
            message="Correction source workspace changed after verification.",
        ),
    )


def _audit_failed_writable_invocation(
    repository: GitRepository,
    run_path: Path,
    run_record: RunRecord,
    *,
    round_number: int,
    execution: AgentExecution[ImplementationResult],
    writable_attempt: WritableAttempt,
    after_workspace: WorkspaceSnapshot | None,
    workspace_guard: WorkspaceGuardInspection | None,
) -> _FailedWritableAudit:
    violations: list[CorrectionSafetyViolation] = []
    if after_workspace is None:
        violations.append(
            CorrectionSafetyViolation(
                name="workspace-inspection",
                expected="complete post-call canonical workspace snapshot",
                actual="unavailable",
                message="Could not inspect the workspace after writable agent execution.",
            )
        )
    else:
        violations.extend(
            _inspect_correction_invariants(
                repository,
                run_record,
                phase="after correction",
                current_snapshot=after_workspace,
            )
        )
    changed_files: tuple[str, ...] = ()
    try:
        changed_files = _worktree_changed_files(repository, run_record.baseline_sha)
    except (GitCommandError, OSError, ValueError) as error:
        violations.append(
            CorrectionSafetyViolation(
                name="worktree-inspection",
                expected="baseline-relative source diff inspection succeeds",
                actual=str(error),
                message=(
                    "Could not inspect source changes after failed writable agent "
                    "invocation."
                ),
            )
        )

    if changed_files and _workspace_changed_since_writable_attempt(
        writable_attempt,
        after_workspace,
    ):
        violations.append(
            CorrectionSafetyViolation(
                name="worktree",
                expected="no baseline-relative source changes after failed writable invocation",
                actual=_format_files(changed_files),
                message=(
                    "Baseline-relative source changes exist after failed writable "
                    "agent invocation."
                ),
            )
        )

    human_required = (
        execution.invocation_started is not False
        or bool(violations)
        or bool(workspace_guard is not None and workspace_guard.requires_human)
    )
    return _FailedWritableAudit(
        safety_violations=tuple(violations),
        changed_files=changed_files,
        workspace_guard=workspace_guard,
        human_required=human_required,
    )


def _workspace_changed_since_writable_attempt(
    writable_attempt: WritableAttempt,
    after_workspace: WorkspaceSnapshot | None,
) -> bool:
    before = writable_attempt.before_snapshot
    if before is None or after_workspace is None:
        return True
    return not before.matches(after_workspace)


def _failed_writable_message(
    *,
    operation: str,
    execution: AgentExecution[ImplementationResult],
    audit: _FailedWritableAudit,
    run_dir: Path,
) -> str:
    rows = [
        (
            "The writable agent invocation did not complete successfully and may "
            "have left partial source changes. Automation has stopped for human "
            "inspection."
        ),
        (
            "Agent failure: "
            f"{_format_failure_category(execution)}: "
            f"{execution.failure_message or 'unknown failure'}"
        ),
        f"Last operation: {operation}",
        f"Process started: {_yes_no(execution.invocation_started is not False)}",
        f"Git safety: {_format_failure_safety(audit.safety_violations)}",
        f"Changed files relative to baseline: {_format_files(audit.changed_files)}",
    ]
    rows.extend(_format_agent_artifacts(execution))
    if audit.workspace_guard is not None and audit.workspace_guard.requires_human:
        rows.append(
            writable_worker._format_writable_guard_stop(
                audit.workspace_guard,
                operation=operation,
                run_dir=run_dir,
                git_safety=_format_failure_safety(audit.safety_violations),
            )
        )
    return " ".join(rows)


def _format_failure_safety(
    violations: tuple[CorrectionSafetyViolation, ...],
) -> str:
    if not violations:
        return "branch unchanged; HEAD unchanged; staging empty"
    return "; ".join(
        f"{violation.name} expected {violation.expected}, got {violation.actual}"
        for violation in violations
    )


def _format_failure_category(execution: AgentExecution[ImplementationResult]) -> str:
    return (
        "UNKNOWN"
        if execution.failure_category is None
        else execution.failure_category.value
    )


def _format_agent_artifacts(
    execution: AgentExecution[ImplementationResult],
) -> list[str]:
    if not execution.artifacts:
        return ["Agent artifacts: none"]
    return [
        "Agent artifacts:",
        *(f"  - {artifact.name}: {artifact.path}" for artifact in execution.artifacts),
    ]


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


def _format_files(files: tuple[str, ...]) -> str:
    if not files:
        return "none"
    shown = ", ".join(files[:5])
    hidden_count = len(files) - 5
    if hidden_count > 0:
        shown = f"{shown}, and {hidden_count} more"
    return shown


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


__all__ = [
    "CORRECTIONS_DIR_NAME",
    "CORRECTION_EXECUTIONS_DIR_NAME",
    "CorrectionCause",
    "CorrectionCauseSet",
    "CorrectionError",
    "CorrectionSafetyViolation",
    "CorrectionStageResult",
    "ReviewCorrectionCause",
    "ReviewCorrectionScope",
    "VerificationCorrectionCause",
    "format_correction_result",
    "render_correction_prompt",
    "render_correction_ticket",
    "run_correction_stage",
]
