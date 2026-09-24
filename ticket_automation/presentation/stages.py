"""Text presentation for individual lifecycle stage results."""

from __future__ import annotations

from typing import Any


def _agent_artifact_rows(execution: Any) -> list[str]:
    if not execution.artifacts:
        return ["Agent artifacts: none"]
    return [
        "Agent artifacts:",
        *(
            f"  - {artifact.name}: {artifact.run_relative_path}"
            for artifact in execution.artifacts
        ),
    ]


def format_implementation_result(result: Any) -> str:
    rows = [
        f"Implementation state: {result.run_record.state.value}",
        f"Artifacts: {result.artifact_directory}",
        result.controller_message,
    ]
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
    if result.agent_execution is not None and not result.agent_execution.successful:
        rows.extend(_agent_artifact_rows(result.agent_execution))
    if result.safety_violations:
        rows.append("Safety violations:")
        rows.extend(
            f"  - {violation.name}: expected {violation.expected}, got {violation.actual}"
            for violation in result.safety_violations
        )
    return "\n".join(rows)


def format_verification_result(result: Any) -> str:
    return "\n".join(
        [
            f"Verification state: {result.run_record.state.value}",
            f"Artifacts: {result.artifact_directory}",
            result.controller_message,
        ]
    )


def format_review_result(result: Any) -> str:
    rows = [
        f"Review state: {result.run_record.state.value}",
        f"Artifacts: {result.artifact_directory}",
        result.controller_message,
    ]
    if result.review_result is not None:
        rows.append(f"Verdict: {result.review_result.verdict.value}")
        rows.append(f"Findings: {len(result.review_result.findings)}")
        rows.append(f"Required findings: {len(result.required_findings)}")
    if result.processing_error:
        rows.append(f"Processing error: {result.processing_error}")
    if result.safety_violations:
        rows.append("Safety violations:")
        rows.extend(
            f"  - {violation.name}: expected {violation.expected}, got {violation.actual}"
            for violation in result.safety_violations
        )
    return "\n".join(rows)


def format_correction_result(result: Any) -> str:
    rows = [
        f"Correction state: {result.run_record.state.value}",
        f"Correction round: {result.correction_round}",
        f"Artifacts: {result.artifact_directory}",
        result.controller_message,
    ]
    if result.ticket_path is not None:
        rows.append(f"Correction ticket: {result.ticket_path}")
    if result.agent_execution is not None and not result.agent_execution.successful:
        rows.extend(_agent_artifact_rows(result.agent_execution))
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


__all__ = [
    "format_correction_result",
    "format_implementation_result",
    "format_review_result",
    "format_verification_result",
]
