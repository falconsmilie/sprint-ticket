from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from .config import VerificationCommand
from .models import WorkflowState

if TYPE_CHECKING:
    from .runs import BaselineRecord, RunRecord

VERIFICATION_SCHEMA_VERSION = 1
VERIFICATION_ROUND_FORMAT = "ticket_automation.verification_round"
VERIFICATION_CHECKPOINT_SCHEMA_VERSION = 2
VERIFICATION_CHECKPOINT_FORMAT = "ticket_automation.verification_checkpoint"
VERIFICATION_DIR_NAME = "verification"
BASELINE_VERIFICATION_DIR_NAME = "baseline-verification"
BASELINE_VERIFICATION_JSON_FILE = "verification.json"
BASELINE_VERIFICATION_LOG_FILE = "verification.log"

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
    return _read_verification_artifact_source_fingerprint(
        artifact_path,
        run_record,
        round_index=round_index,
        stage=WorkflowState.VERIFYING,
        expected_statuses=expected_statuses,
        verification_commands=verification_commands,
        require_empty_correction_reasons=False,
    )


def _baseline_verification_evidence_problem(
    run_path: Path,
    run_record: RunRecord,
    baseline_record: BaselineRecord,
    *,
    verification_commands: tuple[VerificationCommand, ...],
) -> str | None:
    commands_fingerprint = _verification_commands_fingerprint(verification_commands)
    if baseline_record.verification_commands_fingerprint != commands_fingerprint:
        return (
            "Configured verification_commands_fingerprint no longer matches the "
            "recorded baseline."
        )

    ticket_path = run_path / "ticket.md"
    try:
        ticket_sha256 = hashlib.sha256(ticket_path.read_bytes()).hexdigest()
    except OSError as error:
        return f"Could not inspect the snapshotted baseline ticket: {error}"
    if ticket_sha256 != baseline_record.ticket_sha256:
        return "Snapshotted ticket no longer matches the recorded baseline."

    artifact_path = (
        run_path / BASELINE_VERIFICATION_DIR_NAME / BASELINE_VERIFICATION_JSON_FILE
    )
    log_path = (
        run_path / BASELINE_VERIFICATION_DIR_NAME / BASELINE_VERIFICATION_LOG_FILE
    )
    try:
        source_fingerprint = _read_verification_artifact_source_fingerprint(
            artifact_path,
            run_record,
            round_index=0,
            stage=WorkflowState.PREPARING,
            expected_statuses=frozenset({"PASS"}),
            verification_commands=verification_commands,
            require_empty_correction_reasons=True,
        )
        if source_fingerprint != baseline_record.workspace_fingerprint:
            raise _VerificationArtifactError(
                "Baseline verification source does not match the clean workspace baseline."
            )
        if not log_path.is_file() or log_path.stat().st_size <= 0:
            raise _VerificationArtifactError("Baseline verification log is empty.")
    except (OSError, _VerificationArtifactError) as error:
        return f"Clean baseline verification evidence is invalid: {error}"
    return None


def _read_verification_artifact_source_fingerprint(
    artifact_path: Path,
    run_record: RunRecord,
    *,
    round_index: int,
    stage: WorkflowState,
    expected_statuses: frozenset[str],
    verification_commands: tuple[VerificationCommand, ...],
    require_empty_correction_reasons: bool,
) -> str:
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
    if require_empty_correction_reasons and data.get("correction_reasons") != []:
        raise _VerificationArtifactError(
            "Baseline verification artifact contains correction reasons."
        )
    if require_empty_correction_reasons:
        if data.get("safety_violations") != []:
            raise _VerificationArtifactError(
                "Baseline verification artifact contains safety violations."
            )
        command_results = data.get("commands")
        if not isinstance(command_results, list) or len(command_results) != len(
            verification_commands
        ):
            raise _VerificationArtifactError(
                "Baseline verification artifact has an unexpected command set."
            )
        for result, command in zip(command_results, verification_commands, strict=True):
            if not isinstance(result, dict):
                raise _VerificationArtifactError(
                    "Baseline verification command evidence must be an object."
                )
            if (
                result.get("name") != command.name
                or result.get("argv") != list(command.argv)
                or result.get("status") != "PASS"
                or result.get("exit_code") != 0
                or result.get("error_kind") is not None
                or result.get("error_message") is not None
                or not isinstance(result.get("stdout"), str)
                or not isinstance(result.get("stderr"), str)
            ):
                raise _VerificationArtifactError(
                    "Baseline verification command evidence is not a passing result."
                )

    checkpoint = data.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise _VerificationArtifactError(
            "Verification artifact is missing checkpoint metadata."
        )
    expected_metadata = {
        "schema_version": VERIFICATION_CHECKPOINT_SCHEMA_VERSION,
        "format": VERIFICATION_CHECKPOINT_FORMAT,
        "stage": stage.value,
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
    "BASELINE_VERIFICATION_DIR_NAME",
    "BASELINE_VERIFICATION_JSON_FILE",
    "BASELINE_VERIFICATION_LOG_FILE",
    "VERIFICATION_CHECKPOINT_FORMAT",
    "VERIFICATION_CHECKPOINT_SCHEMA_VERSION",
    "VERIFICATION_DIR_NAME",
    "VERIFICATION_ROUND_FORMAT",
    "VERIFICATION_SCHEMA_VERSION",
]
