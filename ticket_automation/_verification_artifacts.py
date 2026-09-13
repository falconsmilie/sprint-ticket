from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from .config import VerificationCommand
from .models import WorkflowState

if TYPE_CHECKING:
    from .runs import RunRecord

VERIFICATION_SCHEMA_VERSION = 1
VERIFICATION_ROUND_FORMAT = "ticket_automation.verification_round"
VERIFICATION_CHECKPOINT_SCHEMA_VERSION = 2
VERIFICATION_CHECKPOINT_FORMAT = "ticket_automation.verification_checkpoint"
VERIFICATION_DIR_NAME = "verification"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class _VerificationArtifactError(ValueError):
    """Raised when controller-owned verification metadata is not trustworthy."""


def _read_verification_source_fingerprint(
    run_path: Path,
    run_record: RunRecord,
    *,
    expected_statuses: frozenset[str],
    verification_commands: tuple[VerificationCommand, ...],
) -> str:
    round_index = run_record.current_correction_round
    artifact_path = run_path / VERIFICATION_DIR_NAME / f"round-{round_index}.json"
    try:
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise _VerificationArtifactError(
            f"Could not read verification artifact: {error}"
        ) from error
    if not isinstance(data, dict):
        raise _VerificationArtifactError("Verification artifact must be an object.")
    if data.get("schema_version") != VERIFICATION_SCHEMA_VERSION:
        raise _VerificationArtifactError(
            "Verification artifact has an unsupported schema version."
        )
    if data.get("format") != VERIFICATION_ROUND_FORMAT:
        raise _VerificationArtifactError(
            "Verification artifact has an unsupported format."
        )
    if data.get("round_index") != round_index:
        raise _VerificationArtifactError("Verification artifact is for another round.")
    if data.get("status") not in expected_statuses:
        raise _VerificationArtifactError(
            "Verification artifact does not have an allowed source status."
        )

    checkpoint = data.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise _VerificationArtifactError(
            "Verification artifact is missing checkpoint metadata."
        )
    expected_metadata = {
        "schema_version": VERIFICATION_CHECKPOINT_SCHEMA_VERSION,
        "format": VERIFICATION_CHECKPOINT_FORMAT,
        "stage": WorkflowState.VERIFYING.value,
        "status": "COMPLETE",
        "run_id": run_record.run_id,
        "round_index": round_index,
        "target_repository_path": str(
            Path(run_record.target_repository_path).resolve()
        ),
        "starting_branch": run_record.starting_branch,
        "baseline_sha": run_record.baseline_sha,
        "verification_commands_fingerprint": _verification_commands_fingerprint(
            verification_commands
        ),
    }
    for key, expected in expected_metadata.items():
        if checkpoint.get(key) != expected:
            raise _VerificationArtifactError(
                f"Verification checkpoint metadata does not match: {key}."
            )

    fingerprint = checkpoint.get("source_fingerprint")
    if not isinstance(fingerprint, str) or not _SHA256_RE.fullmatch(fingerprint):
        raise _VerificationArtifactError(
            "Verification checkpoint source fingerprint is invalid."
        )
    return fingerprint


def _verification_commands_fingerprint(
    commands: tuple[VerificationCommand, ...],
) -> str:
    payload = [
        {
            "name": command.name,
            "argv": list(command.argv),
            "timeout_seconds": command.timeout_seconds,
        }
        for command in commands
    ]
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


__all__ = [
    "VERIFICATION_CHECKPOINT_FORMAT",
    "VERIFICATION_CHECKPOINT_SCHEMA_VERSION",
    "VERIFICATION_DIR_NAME",
    "VERIFICATION_ROUND_FORMAT",
    "VERIFICATION_SCHEMA_VERSION",
]
