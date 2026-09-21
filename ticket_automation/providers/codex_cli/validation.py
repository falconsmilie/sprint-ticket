"""Installed Codex CLI validation used before durable run capture."""

from __future__ import annotations

import subprocess
from pathlib import Path


class CodexCliValidationError(ValueError):
    pass


def cli_version(executable: Path | str, *, cwd: Path) -> str:
    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CodexCliValidationError(
            f"Could not determine Codex CLI version: {error}"
        ) from error
    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0 or not output:
        raise CodexCliValidationError(
            "Could not determine Codex CLI version from the resolved executable."
        )
    return " ".join(output.splitlines()[0].split())


def supports_ephemeral(executable: Path | str, *, cwd: Path) -> bool:
    try:
        completed = subprocess.run(
            [str(executable), "exec", "--help"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and "--ephemeral" in (
        (completed.stdout or "") + (completed.stderr or "")
    )


__all__ = ["CodexCliValidationError", "cli_version", "supports_ephemeral"]
