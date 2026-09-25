"""Infrastructure port for persisting accepted handoff patch evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ...run_ownership import RunOwnership

FINAL_PATCH_FILE = "final.patch"
_GIT_OBJECT_RE = re.compile(r"[0-9a-fA-F]{40,64}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class FinalPatchCaptureRequest:
    repository_path: Path
    baseline_sha: str
    destination: Path
    run_ownership: RunOwnership | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.repository_path, Path):
            raise TypeError("repository_path must be a Path.")
        if not isinstance(self.baseline_sha, str) or not _GIT_OBJECT_RE.fullmatch(
            self.baseline_sha
        ):
            raise ValueError("baseline_sha must be a Git object ID.")
        if not isinstance(self.destination, Path):
            raise TypeError("destination must be a Path.")
        if self.run_ownership is not None and not isinstance(
            self.run_ownership, RunOwnership
        ):
            raise TypeError("run_ownership must be a RunOwnership or None.")


@dataclass(frozen=True)
class FinalPatchReference:
    path: Path
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("patch path must be a Path.")
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("patch sha256 must be a lowercase SHA-256 digest.")
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
        ):
            raise ValueError("patch size_bytes must be a non-negative integer.")


class FinalPatchCapture(Protocol):
    def capture(self, request: FinalPatchCaptureRequest) -> FinalPatchReference: ...


__all__ = [
    "FINAL_PATCH_FILE",
    "FinalPatchCapture",
    "FinalPatchCaptureRequest",
    "FinalPatchReference",
]
