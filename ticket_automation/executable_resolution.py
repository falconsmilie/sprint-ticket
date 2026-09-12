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
    cwd: Path | str | None = None,
) -> Path | None:
    if not configured:
        raise ValueError("Executable must be a non-empty string.")

    if _has_path_part(configured):
        candidate = Path(configured)
        if not candidate.is_absolute():
            base = Path.cwd() if cwd is None else Path(cwd)
            candidate = base / candidate
        if candidate.is_file():
            return candidate.resolve()
        return None

    resolved = shutil.which(configured)
    if resolved is None:
        return None
    return Path(resolved).resolve()


def _has_path_part(configured: str) -> bool:
    candidate = Path(configured)
    return candidate.is_absolute() or any(
        separator in configured for separator in _PATH_SEPARATORS
    )


__all__ = ["resolve_executable"]
