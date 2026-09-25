"""Filesystem adapter for final handoff patch capture."""

from __future__ import annotations

import hashlib

from ..application.ports.handoff import (
    FinalPatchCaptureRequest,
    FinalPatchReference,
)
from ..audit import diff_including_untracked
from ..git import GitRepository
from ..persistence import atomic_write_text
from ..run_ownership import RunOwnershipError


class FileSystemFinalPatchCapture:
    def capture(self, request: FinalPatchCaptureRequest) -> FinalPatchReference:
        patch = diff_including_untracked(
            GitRepository(request.repository_path),
            request.baseline_sha,
        )
        encoded = patch.encode("utf-8")
        if request.run_ownership is not None:
            run_dir = request.run_ownership.validate()
            destination = request.run_ownership.validate_descendant(request.destination)
            if destination.parent != run_dir:
                raise RunOwnershipError(
                    "Final patch destination does not match its configured run owner."
                )
        atomic_write_text(request.destination, patch)
        return FinalPatchReference(
            path=request.destination,
            sha256=hashlib.sha256(encoded).hexdigest(),
            size_bytes=len(encoded),
        )


__all__ = ["FileSystemFinalPatchCapture"]
