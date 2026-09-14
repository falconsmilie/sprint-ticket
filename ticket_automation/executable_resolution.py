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
            # A target repository is untrusted input. A relative executable is
            # meaningful only when it has the TicketAutomation configuration
            # directory that supplied it; never fall back to a process or target
            # repository working directory.
            if config_dir is None:
                return None
            candidate = Path(config_dir) / candidate
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
