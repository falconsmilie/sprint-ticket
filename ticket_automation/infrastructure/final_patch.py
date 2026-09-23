"""Filesystem adapter for final handoff patch capture."""

from __future__ import annotations

import hashlib

from ..application.ports.handoff import (
    FinalPatchCaptureRequest,
    FinalPatchReference,
)
from ..audit import diff_including_untracked
from ..git import GitRepository


class FileSystemFinalPatchCapture:
    def capture(self, request: FinalPatchCaptureRequest) -> FinalPatchReference:
        patch = diff_including_untracked(
            GitRepository(request.repository_path),
            request.baseline_sha,
        )
        encoded = patch.encode("utf-8")
        request.destination.write_text(
            patch,
            encoding="utf-8",
            newline="\n",
        )
        return FinalPatchReference(
            path=request.destination,
            sha256=hashlib.sha256(encoded).hexdigest(),
            size_bytes=len(encoded),
        )


__all__ = ["FileSystemFinalPatchCapture"]
