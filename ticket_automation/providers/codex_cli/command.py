"""Explicit argument-vector construction for Codex CLI invocations."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from ...application.agent_execution import NetworkAccess, RepositoryAccess
from .process import CodexCommand
from .settings import CodexCliSettings


class CodexSandbox(StrEnum):
    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"


def build_command(
    *,
    executable: str,
    repository_path: Path,
    repository_access: RepositoryAccess,
    network_access: NetworkAccess,
    settings: CodexCliSettings,
    output_schema: Path,
    output_result: Path,
) -> CodexCommand:
    sandbox = sandbox_for(repository_access)
    network_config: tuple[str, ...] = ()
    if repository_access is RepositoryAccess.WORKSPACE_WRITE:
        enabled = "true" if network_access is NetworkAccess.ALLOWED else "false"
        network_config = ("-c", f"sandbox_workspace_write.network_access={enabled}")
    elif network_access is NetworkAccess.ALLOWED:
        raise ValueError("Codex read-only execution cannot guarantee network access.")
    return CodexCommand(
        argv=(
            executable,
            "exec",
            "--ephemeral",
            "--model",
            settings.model,
            "-c",
            f'model_reasoning_effort="{settings.reasoning_effort}"',
            *network_config,
            "--sandbox",
            sandbox.value,
            "--json",
            "--output-schema",
            str(output_schema.resolve()),
            "--output-last-message",
            str(output_result.resolve()),
            "-",
        ),
        cwd=repository_path,
    )


def sandbox_for(access: RepositoryAccess) -> CodexSandbox:
    if access is RepositoryAccess.READ_ONLY:
        return CodexSandbox.READ_ONLY
    if access is RepositoryAccess.WORKSPACE_WRITE:
        return CodexSandbox.WORKSPACE_WRITE
    raise TypeError("repository_access must be a RepositoryAccess value.")


__all__ = ["CodexSandbox", "build_command", "sandbox_for"]
