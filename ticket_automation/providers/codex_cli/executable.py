"""Codex CLI executable resolution."""

from __future__ import annotations

from pathlib import Path

from ...executable_resolution import resolve_executable as _resolve_executable


def resolve_executable(
    configured: str,
    *,
    config_dir: Path | str | None = None,
) -> Path | None:
    return _resolve_executable(configured, config_dir=config_dir)


__all__ = ["resolve_executable"]
