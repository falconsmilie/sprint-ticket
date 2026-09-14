"""Readers for typed verification results stored in attempt directories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .attempts import AttemptError, attempt_result_path, latest_attempt
from .config import VerificationCommand

if TYPE_CHECKING:
    from .runs import BaselineRecord, RunRecord


VERIFICATION_SCHEMA_VERSION = 1
VERIFICATION_ROUND_FORMAT = "ticket_automation.verification_round"


class _VerificationArtifactError(ValueError):
    pass


def _read_verification_source_fingerprint(
    run_path: Path,
    run_record: RunRecord,
    *,
    expected_statuses: frozenset[str],
    verification_commands: tuple[VerificationCommand, ...],
) -> str:
    try:
        record = latest_attempt(
            run_path,
            phases=("VERIFYING",),
            statuses=("COMPLETED",),
        )
        if record is None:
            raise _VerificationArtifactError(
                "No completed verification attempt exists."
            )
        data = _read_result(run_path, record)
        _validate_verification_result(
            data,
            expected_statuses=expected_statuses,
            verification_commands=verification_commands,
        )
        fingerprint = record.after_workspace_fingerprint
        if not isinstance(fingerprint, str):
            raise _VerificationArtifactError(
                "Verification attempt has no post-verification workspace fingerprint."
            )
        return fingerprint
    except AttemptError as error:
        raise _VerificationArtifactError(
            f"Attempt evidence is invalid: {error}"
        ) from error


def _baseline_verification_evidence_problem(
    run_path: Path,
    run_record: RunRecord,
    baseline_record: BaselineRecord,
    *,
    verification_commands: tuple[VerificationCommand, ...],
) -> str | None:
    if (
        baseline_record.verification_commands_fingerprint
        != _verification_commands_fingerprint(verification_commands)
    ):
        return "Configured verification commands no longer match the recorded baseline."
    try:
        ticket_sha256 = hashlib.sha256(
            (run_path / "ticket.md").read_bytes()
        ).hexdigest()
    except OSError as error:
        return f"Could not inspect the snapshotted baseline ticket: {error}"
    if ticket_sha256 != baseline_record.ticket_sha256:
        return "Snapshotted ticket no longer matches the recorded baseline."

    try:
        record = latest_attempt(
            run_path,
            phases=("PREPARING",),
            statuses=("COMPLETED",),
        )
        if record is None:
            return "No completed clean-baseline verification attempt exists."
        if (
            record.before_workspace_fingerprint != baseline_record.workspace_fingerprint
            or record.after_workspace_fingerprint
            != baseline_record.workspace_fingerprint
        ):
            return "Clean-baseline verification does not match the recorded baseline workspace."
        data = _read_result(run_path, record)
        _validate_verification_result(
            data,
            expected_statuses=frozenset({"PASS"}),
            verification_commands=verification_commands,
            require_all_commands_pass=True,
        )
    except (_VerificationArtifactError, AttemptError) as error:
        return f"Clean baseline verification evidence is invalid: {error}"
    return None


def _read_result(run_path: Path, record: Any) -> dict[str, Any]:
    path = attempt_result_path(run_path, record)
    if path is None:
        raise _VerificationArtifactError("Verification attempt has no result path.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise _VerificationArtifactError(
            f"Could not read verification result: {error}"
        ) from error
    if not isinstance(data, dict):
        raise _VerificationArtifactError("Verification result must be an object.")
    return data


def _validate_verification_result(
    data: dict[str, Any],
    *,
    expected_statuses: frozenset[str],
    verification_commands: tuple[VerificationCommand, ...],
    require_all_commands_pass: bool = False,
) -> None:
    if data.get("schema_version") != VERIFICATION_SCHEMA_VERSION:
        raise _VerificationArtifactError(
            "Verification result has an unsupported schema version."
        )
    if data.get("format") != VERIFICATION_ROUND_FORMAT:
        raise _VerificationArtifactError(
            "Verification result has an unsupported format."
        )
    if data.get("status") not in expected_statuses:
        raise _VerificationArtifactError(
            "Verification result does not have an allowed status."
        )
    commands = data.get("commands")
    if not isinstance(commands, list) or len(commands) != len(verification_commands):
        raise _VerificationArtifactError(
            "Verification result has an unexpected command set."
        )
    for result, command in zip(commands, verification_commands, strict=True):
        if not isinstance(result, dict):
            raise _VerificationArtifactError(
                "Verification command evidence must be an object."
            )
        if result.get("name") != command.name or result.get("argv") != list(
            command.argv
        ):
            raise _VerificationArtifactError(
                "Verification command evidence does not match configuration."
            )
        if require_all_commands_pass and (
            result.get("status") != "PASS"
            or result.get("exit_code") != 0
            or result.get("error_kind") is not None
            or result.get("error_message") is not None
        ):
            raise _VerificationArtifactError(
                "Baseline verification command evidence is not passing."
            )


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
    "VERIFICATION_ROUND_FORMAT",
    "VERIFICATION_SCHEMA_VERSION",
]
