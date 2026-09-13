from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from .audit import (
    changed_files_including_untracked as _changed_files_including_untracked,
)
from .audit import (
    diff_including_untracked as _diff_including_untracked,
)
from .audit import (
    diff_stats_including_untracked as _diff_stats_including_untracked,
)
from .codex import (
    CodexExecution,
    CodexExecutionFailure,
    CodexProcessRunner,
    Sandbox,
)
from .codex import (
    execute as execute_codex,
)
from .config import AppConfig
from .git import GitCommandError, GitRepository
from .models import WorkflowState
from .runs import (
    BASELINE_RECORD_FILE,
    RUN_RECORD_FILE,
    RUN_TICKET_FILE,
    RunError,
    RunRecord,
    load_baseline_record,
    load_run_record,
    save_run_record,
)
from .workspace_guard import (
    WorkspaceGuardInspection,
    capture_workspace_environment_snapshot,
    correction_guard_artifact_path,
    format_workspace_hygiene_violation,
    inspect_workspace_environment_change,
)


class CorrectionReasonKind(StrEnum):
    VERIFICATION_FAILURE = "VerificationFailure"
    REVIEW_FINDING = "ReviewFinding"


CORRECTIONS_DIR_NAME = "corrections"
CORRECTION_EXECUTIONS_DIR_NAME = "correction-executions"
VERIFICATION_DIR_NAME = "verification"
REVIEW_DIR_NAME = "reviews"
DIFFS_DIR_NAME = "diffs"
CORRECTION_TICKET_SUFFIX = "CORR"
CORRECTION_TICKET_EXCERPT_CHARS = 1200
AFTER_CORRECTION_PATCH_TEMPLATE = "after-correction-{round_number}.patch"
_FAILED_CORRECTION_PATCH_TEMPLATE = "failed-correction-{round_number}.patch"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CORRECTION_PROMPT_TEMPLATE = _PROJECT_ROOT / "prompts" / "correct.md"
_CORRECTION_RESULT_SCHEMA = (
    _PROJECT_ROOT / "schemas" / "implementation-result.schema.json"
)
_REVIEW_RESULT_FILE = "result.json"
_REVIEW_FINDING_DISPOSITIONS = frozenset({"REQUIRED", "ADVISORY", "FOLLOW_UP"})
_TERMINAL_STATES = frozenset(
    {
        WorkflowState.READY_FOR_HUMAN,
        WorkflowState.HUMAN_REQUIRED,
        WorkflowState.FAILED,
    }
)


class CorrectionError(RunError):
    """Raised when corrective work cannot be prepared or persisted."""


@dataclass(frozen=True)
class VerificationFailure:
    gate_name: str
    command: tuple[str, ...]
    failure_summary: str
    stdout_excerpt: str
    stderr_excerpt: str
    exit_code: int | None
    log_path: Path

    @property
    def kind(self) -> CorrectionReasonKind:
        return CorrectionReasonKind.VERIFICATION_FAILURE

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "gate_name": self.gate_name,
            "command": list(self.command),
            "failure_summary": self.failure_summary,
            "stdout_excerpt": self.stdout_excerpt,
            "stderr_excerpt": self.stderr_excerpt,
            "exit_code": self.exit_code,
            "log_path": str(self.log_path),
        }


@dataclass(frozen=True)
class ReviewFinding:
    finding_id: str
    summary: str
    details: str
    severity: str = ""
    category: str = ""
    evidence: str = ""
    required_change: str = ""
    acceptance_criteria: tuple[str, ...] = ()
    disposition: str = "REQUIRED"

    @property
    def kind(self) -> CorrectionReasonKind:
        return CorrectionReasonKind.REVIEW_FINDING

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "finding_id": self.finding_id,
            "summary": self.summary,
            "details": self.details,
            "severity": self.severity,
            "category": self.category,
            "evidence": self.evidence,
            "required_change": self.required_change,
            "acceptance_criteria": list(self.acceptance_criteria),
            "disposition": self.disposition,
        }


CorrectionReason = VerificationFailure | ReviewFinding


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
    correction_round: int
    ticket_path: Path | None
    artifact_directory: Path
    codex_execution: CodexExecution | None
    agent_result: dict[str, Any] | None
    safety_violations: tuple[CorrectionSafetyViolation, ...]
    correction_reasons: tuple[CorrectionReason, ...]
    patch_path: Path | None
    workspace_guard: WorkspaceGuardInspection | None
    controller_message: str

    @property
    def successful(self) -> bool:
        return self.run_record.state == WorkflowState.VERIFY


@dataclass(frozen=True)
class _FailedWritableAudit:
    safety_violations: tuple[CorrectionSafetyViolation, ...]
    changed_files: tuple[str, ...]
    patch_path: Path | None
    patch_error: str | None
    workspace_guard: WorkspaceGuardInspection | None
    human_required: bool


def run_correction_stage(
    config: AppConfig,
    run_dir: Path | str,
    *,
    codex_runner: CodexProcessRunner | None = None,
    clock: Callable[[], datetime] | None = None,
) -> CorrectionStageResult:
    run_path = Path(run_dir)
    run_record_path = run_path / RUN_RECORD_FILE
    run_record = load_run_record(run_record_path)
    if run_record.state != WorkflowState.CORRECT:
        raise CorrectionError(
            f"Correction requires run state CORRECT; found {run_record.state.value}."
        )

    baseline_record = load_baseline_record(run_path / BASELINE_RECORD_FILE)
    if baseline_record.branch != run_record.starting_branch:
        raise CorrectionError("Run record and baseline branch do not match.")
    if baseline_record.head_sha != run_record.baseline_sha:
        raise CorrectionError("Run record and baseline HEAD do not match.")

    correction_round = run_record.current_correction_round + 1
    repository = GitRepository(Path(run_record.target_repository_path))
    artifact_directory = (
        run_path / CORRECTION_EXECUTIONS_DIR_NAME / f"round-{correction_round}"
    )
    if run_record.current_correction_round >= run_record.max_correction_rounds:
        return _finish(
            run_record=run_record,
            run_record_path=run_record_path,
            run_dir=run_path,
            correction_round=correction_round,
            ticket_path=None,
            artifact_directory=artifact_directory,
            execution=None,
            agent_result=None,
            safety_violations=(),
            correction_reasons=(),
            patch_path=None,
            state=WorkflowState.HUMAN_REQUIRED,
            controller_message=(
                "Maximum corrective rounds exhausted; human intervention is required."
            ),
            clock=clock,
            advance_correction_round=False,
            last_completed_state=run_record.last_completed_state,
        )

    starting_violations = _inspect_correction_invariants(
        repository,
        run_record,
        phase="before correction",
    )
    if starting_violations:
        return _finish(
            run_record=run_record,
            run_record_path=run_record_path,
            run_dir=run_path,
            correction_round=correction_round,
            ticket_path=None,
            artifact_directory=artifact_directory,
            execution=None,
            agent_result=None,
            safety_violations=starting_violations,
            correction_reasons=(),
            patch_path=None,
            state=WorkflowState.HUMAN_REQUIRED,
            controller_message=(
                "Repository no longer matches the recorded correction baseline."
            ),
            clock=clock,
            advance_correction_round=False,
        )

    if artifact_directory.exists() and any(artifact_directory.iterdir()):
        raise CorrectionError(
            f"Correction artifacts already exist for round-{correction_round}."
        )
    reasons = _load_correction_reasons(run_path, run_record)
    ticket_markdown = render_correction_ticket(
        ticket_id=run_record.ticket_id,
        round_number=correction_round,
        reasons=reasons,
        run_dir=run_path,
    )
    active_record = run_record.with_state(
        WorkflowState.CORRECT,
        updated_timestamp=_timestamp(clock),
        last_completed_state=WorkflowState.CORRECT,
    )
    save_run_record(active_record, run_record_path)
    ticket_path = _write_correction_ticket(
        run_path=run_path,
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
    environment_snapshot = capture_workspace_environment_snapshot(repository.path)

    try:
        execution = execute_codex(
            prompt=prompt,
            repo_path=repository.path,
            sandbox=Sandbox.WORKSPACE_WRITE,
            output_schema=_CORRECTION_RESULT_SCHEMA,
            artifact_directory=artifact_directory,
            executable=config.codex.executable,
            execution_config=config.codex.execution,
            runner=codex_runner,
        )
    except CodexExecutionFailure as error:
        workspace_guard = inspect_workspace_environment_change(
            before=environment_snapshot,
            phase=WorkflowState.CORRECT.value,
            artifact_path=correction_guard_artifact_path(
                run_path,
                round_number=correction_round,
            ),
            clock=clock,
        )
        failed_audit = _audit_failed_writable_invocation(
            repository,
            run_path,
            active_record,
            round_number=correction_round,
            execution=error.execution,
            workspace_guard=workspace_guard,
        )
        state = (
            WorkflowState.HUMAN_REQUIRED
            if failed_audit.human_required
            else WorkflowState.FAILED
        )
        message = (
            _failed_writable_message(
                operation=f"correction round {correction_round}",
                execution=error.execution,
                audit=failed_audit,
                run_dir=run_path,
            )
            if failed_audit.human_required
            else error.execution.failure_message or str(error)
        )
        return _finish(
            run_record=active_record,
            run_record_path=run_record_path,
            run_dir=run_path,
            correction_round=correction_round,
            ticket_path=ticket_path,
            artifact_directory=artifact_directory,
            execution=error.execution,
            agent_result=None,
            safety_violations=failed_audit.safety_violations,
            correction_reasons=reasons,
            patch_path=failed_audit.patch_path,
            workspace_guard=workspace_guard,
            state=state,
            controller_message=message,
            clock=clock,
            advance_correction_round=False,
            last_completed_state=run_record.last_completed_state,
        )

    agent_result = _require_agent_result(execution.structured_result)
    workspace_guard = inspect_workspace_environment_change(
        before=environment_snapshot,
        phase=WorkflowState.CORRECT.value,
        artifact_path=correction_guard_artifact_path(
            run_path,
            round_number=correction_round,
        ),
        clock=clock,
    )
    safety_violations = _inspect_correction_invariants(
        repository,
        active_record,
        phase="after correction",
    )
    patch_path, patch_error = _try_capture_correction_diff(
        repository,
        run_path,
        baseline_sha=active_record.baseline_sha,
        round_number=correction_round,
    )
    if workspace_guard.has_violation:
        message = format_workspace_hygiene_violation(
            workspace_guard,
            operation=f"correction round {correction_round}",
            run_dir=run_path,
            git_safety=_format_failure_safety(safety_violations),
        )
        if patch_error is not None:
            message = f"{message} Correction patch capture error: {patch_error}"
        return _finish(
            run_record=active_record,
            run_record_path=run_record_path,
            run_dir=run_path,
            correction_round=correction_round,
            ticket_path=ticket_path,
            artifact_directory=artifact_directory,
            execution=execution,
            agent_result=agent_result,
            safety_violations=safety_violations,
            correction_reasons=reasons,
            patch_path=patch_path,
            workspace_guard=workspace_guard,
            state=WorkflowState.HUMAN_REQUIRED,
            controller_message=message,
            clock=clock,
        )
    if safety_violations:
        return _finish(
            run_record=active_record,
            run_record_path=run_record_path,
            run_dir=run_path,
            correction_round=correction_round,
            ticket_path=ticket_path,
            artifact_directory=artifact_directory,
            execution=execution,
            agent_result=agent_result,
            safety_violations=safety_violations,
            correction_reasons=reasons,
            patch_path=patch_path,
            workspace_guard=workspace_guard,
            state=WorkflowState.HUMAN_REQUIRED,
            controller_message="Repository safety invariants were violated.",
            clock=clock,
        )

    if agent_result["status"] == "BLOCKED":
        return _finish(
            run_record=active_record,
            run_record_path=run_record_path,
            run_dir=run_path,
            correction_round=correction_round,
            ticket_path=ticket_path,
            artifact_directory=artifact_directory,
            execution=execution,
            agent_result=agent_result,
            safety_violations=(),
            correction_reasons=reasons,
            patch_path=patch_path,
            workspace_guard=workspace_guard,
            state=WorkflowState.HUMAN_REQUIRED,
            controller_message="Correction agent returned BLOCKED.",
            clock=clock,
        )

    if patch_error is not None:
        return _finish(
            run_record=active_record,
            run_record_path=run_record_path,
            run_dir=run_path,
            correction_round=correction_round,
            ticket_path=ticket_path,
            artifact_directory=artifact_directory,
            execution=execution,
            agent_result=agent_result,
            safety_violations=(),
            correction_reasons=reasons,
            patch_path=None,
            workspace_guard=workspace_guard,
            state=WorkflowState.HUMAN_REQUIRED,
            controller_message=patch_error,
            clock=clock,
        )

    return _finish(
        run_record=active_record,
        run_record_path=run_record_path,
        run_dir=run_path,
        correction_round=correction_round,
        ticket_path=ticket_path,
        artifact_directory=artifact_directory,
        execution=execution,
        agent_result=agent_result,
        safety_violations=(),
        correction_reasons=reasons,
        patch_path=patch_path,
        workspace_guard=workspace_guard,
        state=WorkflowState.VERIFY,
        controller_message=(
            "Correction completed; deterministic verification must run next."
        ),
        clock=clock,
    )


def render_correction_ticket(
    *,
    ticket_id: str,
    round_number: int,
    reasons: Iterable[CorrectionReason],
    run_dir: Path | str | None = None,
) -> str:
    selected_reasons = _eligible_reasons(tuple(reasons))
    if not selected_reasons:
        raise CorrectionError("Correction requires at least one eligible reason.")
    reason_kind = _single_reason_kind(selected_reasons)
    if reason_kind == CorrectionReasonKind.REVIEW_FINDING:
        body = _render_review_correction_ticket(
            ticket_id=ticket_id,
            round_number=round_number,
            findings=_review_findings(selected_reasons),
        )
    elif reason_kind == CorrectionReasonKind.VERIFICATION_FAILURE:
        body = _render_verification_correction_ticket(
            ticket_id=ticket_id,
            round_number=round_number,
            failures=_verification_failures(selected_reasons),
            run_dir=None if run_dir is None else Path(run_dir),
        )
    else:
        raise CorrectionError(f"Unsupported correction reason kind: {reason_kind}")
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


def review_findings_from_result(result: dict[str, Any]) -> tuple[ReviewFinding, ...]:
    findings = result.get("findings")
    if not isinstance(findings, list):
        raise CorrectionError("Review correction source must contain a findings list.")
    required_findings: list[ReviewFinding] = []
    for index, finding in enumerate(findings, start=1):
        if not isinstance(finding, dict):
            raise CorrectionError(f"Review finding {index} must be an object.")
        disposition = _required_string(
            finding,
            "disposition",
            source=f"review finding {index}",
        )
        if disposition not in _REVIEW_FINDING_DISPOSITIONS:
            raise CorrectionError(
                f"Review finding {index} has unsupported disposition: {disposition!r}."
            )
        if disposition == "REQUIRED":
            required_findings.append(_review_finding_from_dict(finding, index=index))
    return tuple(required_findings)


def correction_reason_from_dict(data: dict[str, Any]) -> CorrectionReason:
    kind = data.get("kind")
    if kind == CorrectionReasonKind.VERIFICATION_FAILURE.value:
        source = "verification correction reason"
        return VerificationFailure(
            gate_name=_required_string(data, "gate_name", source=source),
            command=_required_string_tuple(data, "command", source=source),
            failure_summary=_required_string(
                data,
                "failure_summary",
                source=source,
            ),
            stdout_excerpt=_required_string(
                data,
                "stdout_excerpt",
                source=source,
                allow_empty=True,
            ),
            stderr_excerpt=_required_string(
                data,
                "stderr_excerpt",
                source=source,
                allow_empty=True,
            ),
            exit_code=_required_optional_int(data, "exit_code", source=source),
            log_path=Path(_required_string(data, "log_path", source=source)),
        )
    if kind == CorrectionReasonKind.REVIEW_FINDING.value:
        source = "review correction reason"
        disposition = _required_string(data, "disposition", source=source)
        if disposition not in _REVIEW_FINDING_DISPOSITIONS:
            raise CorrectionError(
                f"{source} has unsupported disposition: {disposition!r}."
            )
        return ReviewFinding(
            finding_id=_required_string(data, "finding_id", source=source),
            summary=_required_string(data, "summary", source=source),
            details=_required_string(data, "details", source=source),
            severity=_required_string(data, "severity", source=source),
            category=_required_string(data, "category", source=source),
            evidence=_required_string(data, "evidence", source=source),
            required_change=_required_string(
                data,
                "required_change",
                source=source,
            ),
            acceptance_criteria=_required_string_tuple(
                data,
                "acceptance_criteria",
                source=source,
            ),
            disposition=disposition,
        )
    raise CorrectionError(f"Unsupported correction reason kind: {kind!r}.")


def format_correction_result(result: CorrectionStageResult) -> str:
    rows = [
        f"Correction state: {result.run_record.state.value}",
        f"Correction round: {result.correction_round}",
        f"Artifacts: {result.artifact_directory}",
        result.controller_message,
    ]
    if result.ticket_path is not None:
        rows.append(f"Correction ticket: {result.ticket_path}")
    if result.patch_path is not None:
        rows.append(f"Patch: {result.patch_path}")
    if result.codex_execution is not None and not result.codex_execution.successful:
        rows.append(f"Codex execution: {result.codex_execution.execution_json_path}")
        rows.append(f"Codex events: {result.codex_execution.events_jsonl_path}")
        rows.append(f"Codex stderr: {result.codex_execution.stderr_log_path}")
    if result.workspace_guard is not None and result.workspace_guard.has_violation:
        rows.append(f"Workspace guard: {result.workspace_guard.artifact_path}")
        rows.append("Workspace hygiene violations:")
        rows.extend(
            "  - "
            f"{environment.root_path.relative_to(result.workspace_guard.after.repository_path)}: "
            f"marker {environment.primary_marker_path.relative_to(result.workspace_guard.after.repository_path)}"
            for environment in result.workspace_guard.new_environments
        )
    if result.agent_result is not None:
        rows.append(f"Agent status: {result.agent_result['status']}")
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
    run_record_path: Path,
    run_dir: Path,
    correction_round: int,
    ticket_path: Path | None,
    artifact_directory: Path,
    execution: CodexExecution | None,
    agent_result: dict[str, Any] | None,
    safety_violations: tuple[CorrectionSafetyViolation, ...],
    correction_reasons: tuple[CorrectionReason, ...],
    patch_path: Path | None,
    workspace_guard: WorkspaceGuardInspection | None = None,
    state: WorkflowState,
    controller_message: str,
    clock: Callable[[], datetime] | None,
    advance_correction_round: bool = True,
    last_completed_state: WorkflowState = WorkflowState.CORRECT,
) -> CorrectionStageResult:
    updated_record = run_record.with_state(
        state,
        updated_timestamp=_timestamp(clock),
        last_completed_state=last_completed_state,
        current_correction_round=(
            correction_round
            if advance_correction_round
            else run_record.current_correction_round
        ),
        terminal_reason=controller_message if state in _TERMINAL_STATES else None,
    )
    save_run_record(updated_record, run_record_path)
    return CorrectionStageResult(
        run_dir=run_dir,
        run_record=updated_record,
        correction_round=correction_round,
        ticket_path=ticket_path,
        artifact_directory=artifact_directory,
        codex_execution=execution,
        agent_result=agent_result,
        safety_violations=safety_violations,
        correction_reasons=correction_reasons,
        patch_path=patch_path,
        workspace_guard=workspace_guard,
        controller_message=controller_message,
    )


def _load_correction_reasons(
    run_path: Path,
    run_record: RunRecord,
) -> tuple[CorrectionReason, ...]:
    verification_reasons = _load_verification_failures(run_path, run_record)
    if verification_reasons:
        return verification_reasons

    review_findings = _load_required_review_findings(run_path, run_record)
    if review_findings:
        return review_findings

    raise CorrectionError(
        "Run is in CORRECT state but no verification failures or REQUIRED review "
        "findings were found."
    )


def _load_verification_failures(
    run_path: Path,
    run_record: RunRecord,
) -> tuple[VerificationFailure, ...]:
    round_index = run_record.current_correction_round
    verification_path = run_path / VERIFICATION_DIR_NAME / f"round-{round_index}.json"
    if not verification_path.is_file():
        return ()
    data = _read_json_object(verification_path)
    if data.get("status") != "FAIL":
        return ()

    if "correction_reasons" in data:
        reasons = data["correction_reasons"]
        if not isinstance(reasons, list):
            raise CorrectionError(
                "Verification correction source must contain a correction_reasons list."
            )
        parsed: list[VerificationFailure] = []
        for index, item in enumerate(reasons, start=1):
            if not isinstance(item, dict):
                raise CorrectionError(
                    f"Verification correction reason {index} must be an object."
                )
            reason = correction_reason_from_dict(item)
            if not isinstance(reason, VerificationFailure):
                raise CorrectionError(
                    "Verification correction source cannot contain review findings."
                )
            parsed.append(reason)
        if parsed:
            return tuple(parsed)

    return _verification_failures_from_commands(
        data,
        log_path=run_path / VERIFICATION_DIR_NAME / f"round-{round_index}.log",
    )


def _verification_failures_from_commands(
    data: dict[str, Any],
    *,
    log_path: Path,
) -> tuple[VerificationFailure, ...]:
    commands = data.get("commands", [])
    if not isinstance(commands, list):
        return ()
    failures: list[VerificationFailure] = []
    for command in commands:
        if not isinstance(command, dict) or command.get("status") != "FAIL":
            continue
        gate_name = _string_value(command.get("name"), "verification")
        exit_code = _optional_int(command.get("exit_code"))
        failures.append(
            VerificationFailure(
                gate_name=gate_name,
                command=_string_tuple(command.get("argv")),
                failure_summary=(
                    f"Verification gate {gate_name!r} exited with code {exit_code}."
                ),
                stdout_excerpt=_ticket_excerpt(
                    _string_value(command.get("stdout"), "")
                ),
                stderr_excerpt=_ticket_excerpt(
                    _string_value(command.get("stderr"), "")
                ),
                exit_code=exit_code,
                log_path=log_path,
            )
        )
    return tuple(failures)


def _load_required_review_findings(
    run_path: Path,
    run_record: RunRecord,
) -> tuple[ReviewFinding, ...]:
    review_round = run_record.current_review_round
    if review_round < 1:
        return ()
    result_path = (
        run_path / REVIEW_DIR_NAME / f"round-{review_round}" / _REVIEW_RESULT_FILE
    )
    if not result_path.is_file():
        return ()
    data = _read_json_object(result_path)
    if data.get("verdict") != "CORRECTIONS_REQUIRED":
        return ()
    findings = review_findings_from_result(data)
    if not findings:
        raise CorrectionError(
            "Review requested corrections but did not contain REQUIRED findings."
        )
    return findings


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CorrectionError(
            f"Could not read correction source data: {path}: {error}"
        ) from error
    if not isinstance(data, dict):
        raise CorrectionError(f"Correction source data must be an object: {path}")
    return data


def _write_correction_ticket(
    *,
    run_path: Path,
    ticket_id: str,
    round_number: int,
    markdown: str,
) -> Path:
    corrections_dir = run_path / CORRECTIONS_DIR_NAME
    corrections_dir.mkdir(parents=True, exist_ok=True)
    path = (
        corrections_dir / f"{ticket_id}-{CORRECTION_TICKET_SUFFIX}-R{round_number}.md"
    )
    if path.exists():
        raise CorrectionError(f"Correction ticket already exists: {path}")
    path.write_text(markdown, encoding="utf-8", newline="\n")
    return path


def _render_review_correction_ticket(
    *,
    ticket_id: str,
    round_number: int,
    findings: tuple[ReviewFinding, ...],
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
                f"Severity: {_display_enum(finding.severity)}",
                f"Category: {_display_enum(finding.category)}",
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
    failures: tuple[VerificationFailure, ...],
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
        log_reference = _format_log_reference(failure.log_path, run_dir)
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
                "Full log:",
                log_reference,
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


def _require_agent_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CorrectionError("Correction agent result must be a JSON object.")
    return value


def _inspect_correction_invariants(
    repository: GitRepository,
    run_record: RunRecord,
    *,
    phase: str,
) -> tuple[CorrectionSafetyViolation, ...]:
    violations: list[CorrectionSafetyViolation] = []
    try:
        current_branch = repository.current_branch()
        current_head = repository.head_sha()
        staged_files = repository.staged_files()
    except GitCommandError as error:
        return (
            CorrectionSafetyViolation(
                name="git-inspection",
                expected=f"{phase} Git inspection succeeds",
                actual=str(error),
                message=f"Could not inspect repository {phase}.",
            ),
        )

    if current_branch != run_record.starting_branch:
        violations.append(
            CorrectionSafetyViolation(
                name="branch",
                expected=run_record.starting_branch,
                actual=_format_optional(current_branch),
                message=f"Current branch changed {phase}.",
            )
        )
    if current_head != run_record.baseline_sha:
        violations.append(
            CorrectionSafetyViolation(
                name="HEAD",
                expected=run_record.baseline_sha,
                actual=current_head,
                message=f"HEAD changed {phase}.",
            )
        )
    if staged_files:
        violations.append(
            CorrectionSafetyViolation(
                name="staging",
                expected="empty",
                actual=_format_files(staged_files),
                message=f"Staging area is not empty {phase}.",
            )
        )
    return tuple(violations)


def _capture_correction_diff(
    repository: GitRepository,
    run_path: Path,
    *,
    baseline_sha: str,
    round_number: int,
    patch_template: str = AFTER_CORRECTION_PATCH_TEMPLATE,
) -> Path:
    diffs_dir = run_path / DIFFS_DIR_NAME
    patch_path = diffs_dir / patch_template.format(
        round_number=round_number,
    )
    try:
        diffs_dir.mkdir(parents=True, exist_ok=True)
        patch = _diff_including_untracked(repository, baseline_sha)
        patch_path.write_text(patch, encoding="utf-8", newline="\n")
    except (GitCommandError, OSError, ValueError) as error:
        raise CorrectionError(f"Could not capture correction diff: {error}") from error
    return patch_path


def _try_capture_correction_diff(
    repository: GitRepository,
    run_path: Path,
    *,
    baseline_sha: str,
    round_number: int,
    patch_template: str = AFTER_CORRECTION_PATCH_TEMPLATE,
) -> tuple[Path | None, str | None]:
    try:
        return (
            _capture_correction_diff(
                repository,
                run_path,
                baseline_sha=baseline_sha,
                round_number=round_number,
                patch_template=patch_template,
            ),
            None,
        )
    except CorrectionError as error:
        return None, str(error)


def _audit_failed_writable_invocation(
    repository: GitRepository,
    run_path: Path,
    run_record: RunRecord,
    *,
    round_number: int,
    execution: CodexExecution,
    workspace_guard: WorkspaceGuardInspection | None,
) -> _FailedWritableAudit:
    violations = list(
        _inspect_correction_invariants(
            repository,
            run_record,
            phase="after correction",
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
                    "Could not inspect source changes after failed writable Codex "
                    "invocation."
                ),
            )
        )

    if changed_files:
        violations.append(
            CorrectionSafetyViolation(
                name="worktree",
                expected="no baseline-relative source changes after failed writable invocation",
                actual=_format_files(changed_files),
                message=(
                    "Baseline-relative source changes exist after failed writable "
                    "Codex invocation."
                ),
            )
        )

    human_required = (
        execution.process_started
        or bool(violations)
        or bool(workspace_guard is not None and workspace_guard.has_violation)
    )
    patch_path: Path | None = None
    patch_error: str | None = None
    if human_required:
        patch_path, patch_error = _try_capture_correction_diff(
            repository,
            run_path,
            baseline_sha=run_record.baseline_sha,
            round_number=round_number,
            patch_template=_FAILED_CORRECTION_PATCH_TEMPLATE,
        )

    return _FailedWritableAudit(
        safety_violations=tuple(violations),
        changed_files=changed_files,
        patch_path=patch_path,
        patch_error=patch_error,
        workspace_guard=workspace_guard,
        human_required=human_required,
    )


def _failed_writable_message(
    *,
    operation: str,
    execution: CodexExecution,
    audit: _FailedWritableAudit,
    run_dir: Path,
) -> str:
    rows = [
        (
            "The writable Codex invocation did not complete successfully and may "
            "have left partial source changes. Automation has stopped for human "
            "inspection."
        ),
        (
            "Codex failure: "
            f"{_format_optional_failure_kind(execution)}: "
            f"{execution.failure_message or 'unknown failure'}"
        ),
        f"Last operation: {operation}",
        f"Process started: {_yes_no(execution.process_started)}",
        f"Execution metadata: {execution.execution_json_path}",
        f"Events: {execution.events_jsonl_path}",
        f"Stderr: {execution.stderr_log_path}",
        f"Git safety: {_format_failure_safety(audit.safety_violations)}",
        f"Changed files relative to baseline: {_format_files(audit.changed_files)}",
    ]
    if audit.patch_path is not None:
        rows.append(f"Baseline-relative failure patch: {audit.patch_path}")
    if audit.patch_error is not None:
        rows.append(f"Failure patch capture error: {audit.patch_error}")
    if audit.workspace_guard is not None and audit.workspace_guard.has_violation:
        rows.append(
            format_workspace_hygiene_violation(
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


def _format_optional_failure_kind(execution: CodexExecution) -> str:
    return "UNKNOWN" if execution.failure_kind is None else execution.failure_kind.value


def _worktree_changed_files(
    repository: GitRepository,
    baseline_sha: str,
) -> tuple[str, ...]:
    return _changed_files_including_untracked(repository, baseline_sha)


def _review_finding_from_dict(data: dict[str, Any], *, index: int) -> ReviewFinding:
    source = f"review finding {index}"
    return ReviewFinding(
        finding_id=_required_string(data, "id", source=source),
        summary=_required_string(data, "title", source=source),
        details=_required_string(data, "description", source=source),
        severity=_required_string(data, "severity", source=source),
        category=_required_string(data, "category", source=source),
        evidence=_required_string(data, "evidence", source=source),
        required_change=_required_string(data, "required_change", source=source),
        acceptance_criteria=_required_string_tuple(
            data,
            "acceptance_criteria",
            source=source,
        ),
        disposition="REQUIRED",
    )


def _eligible_reasons(
    reasons: tuple[CorrectionReason, ...],
) -> tuple[CorrectionReason, ...]:
    return tuple(
        reason
        for reason in reasons
        if not isinstance(reason, ReviewFinding) or reason.disposition == "REQUIRED"
    )


def _single_reason_kind(reasons: tuple[CorrectionReason, ...]) -> CorrectionReasonKind:
    kinds = {reason.kind for reason in reasons}
    if len(kinds) != 1:
        raise CorrectionError(
            "Correction reasons must come from one source type per correction round."
        )
    return next(iter(kinds))


def _review_findings(
    reasons: tuple[CorrectionReason, ...],
) -> tuple[ReviewFinding, ...]:
    return tuple(reason for reason in reasons if isinstance(reason, ReviewFinding))


def _verification_failures(
    reasons: tuple[CorrectionReason, ...],
) -> tuple[VerificationFailure, ...]:
    return tuple(
        reason for reason in reasons if isinstance(reason, VerificationFailure)
    )


def _verification_output_excerpt(failure: VerificationFailure) -> str:
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
        f"... <truncated {omission} chars; see full log>"
    )


def _format_log_reference(log_path: Path, run_dir: Path | None) -> str:
    if run_dir is None:
        return str(log_path)
    try:
        return str(log_path.relative_to(run_dir))
    except ValueError:
        return str(log_path)


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


def _required_string(
    data: dict[str, Any],
    field: str,
    *,
    source: str,
    allow_empty: bool = False,
) -> str:
    value = data.get(field)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise CorrectionError(
            f"{source} must contain a non-empty string field {field!r}."
        )
    return value


def _required_optional_int(
    data: dict[str, Any],
    field: str,
    *,
    source: str,
) -> int | None:
    value = data.get(field)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise CorrectionError(f"{source} field {field!r} must be an integer or null.")


def _required_string_tuple(
    data: dict[str, Any],
    field: str,
    *,
    source: str,
) -> tuple[str, ...]:
    value = data.get(field)
    if not isinstance(value, list | tuple):
        raise CorrectionError(
            f"{source} must contain a non-empty string list field {field!r}."
        )
    strings = tuple(item for item in value if isinstance(item, str) and item.strip())
    if len(strings) != len(value) or not strings:
        raise CorrectionError(
            f"{source} must contain a non-empty string list field {field!r}."
        )
    return strings


def _string_value(value: Any, default: str) -> str:
    return value if isinstance(value, str) else default


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item for item in value if isinstance(item, str))


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


def _timestamp(clock: Callable[[], datetime] | None) -> str:
    now = datetime.now(UTC) if clock is None else clock()
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return now.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "AFTER_CORRECTION_PATCH_TEMPLATE",
    "CORRECTIONS_DIR_NAME",
    "CORRECTION_EXECUTIONS_DIR_NAME",
    "CorrectionError",
    "CorrectionReason",
    "CorrectionReasonKind",
    "CorrectionSafetyViolation",
    "CorrectionStageResult",
    "ReviewFinding",
    "VerificationFailure",
    "correction_reason_from_dict",
    "format_correction_result",
    "render_correction_prompt",
    "render_correction_ticket",
    "review_findings_from_result",
    "run_correction_stage",
]
