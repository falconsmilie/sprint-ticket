"""Provider-neutral executable path resolution."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_PATH_SEPARATORS = tuple(
    separator
    for separator in dict.fromkeys((os.sep, os.altsep, "/", "\\"))
    if separator
)


def resolve_executable(
    configured: str,
    *,
    config_dir: Path | str | None = None,
) -> Path | None:
    if not configured:
        raise ValueError("Executable must be a non-empty string.")
    if _has_path_part(configured):
        candidate = Path(configured)
        if not candidate.is_absolute():
            if config_dir is None:
                return None
            candidate = Path(config_dir) / candidate
        return candidate.resolve() if candidate.is_file() else None
    resolved = shutil.which(configured)
    return None if resolved is None else Path(resolved).resolve()


def _has_path_part(configured: str) -> bool:
    candidate = Path(configured)
    return candidate.is_absolute() or any(
        separator in configured for separator in _PATH_SEPARATORS
    )


__all__ = ["resolve_executable"]
