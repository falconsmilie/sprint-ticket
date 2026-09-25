"""Public codecs for typed verification evidence stored by attempts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .attempts import AttemptError, AttemptRecord, attempt_result_path, latest_attempt
from .config import VerificationCommand
from .models import AttemptPhase, AttemptStatus, VerificationStatus
from .persistence import CodecError, read_json_object

if TYPE_CHECKING:
    from .runs import BaselineRecord, RunRecord


VERIFICATION_SCHEMA_VERSION = 1
VERIFICATION_ROUND_FORMAT = "ticket_automation.verification_round"


class _VerificationArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class VerificationEvidence:
    round_index: int
    status: VerificationStatus
    started_at: str
    ended_at: str
    duration_seconds: float
    command_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.status, VerificationStatus):
            raise TypeError(
                "Verification evidence status must be a VerificationStatus."
            )


def read_verification_evidence(
    run_path: Path, record: AttemptRecord
) -> VerificationEvidence:
    data = _read_result(run_path, record)
    _validate_verification_result_shape(data)
    commands = data["commands"]
    assert isinstance(commands, list)
    return VerificationEvidence(
        round_index=data["round_index"],
        status=_verification_status(data["status"], source="result"),
        started_at=data["started_at"],
        ended_at=data["ended_at"],
        duration_seconds=float(data["duration_seconds"]),
        command_count=len(commands),
    )


def read_verification_result_text(
    run_path: Path,
    record: AttemptRecord,
    *,
    expected_statuses: frozenset[VerificationStatus],
) -> str:
    """Render validated verification evidence without exposing its JSON object."""

    data = _read_result(run_path, record)
    _validate_verification_result_shape(data)
    if _verification_status(data["status"], source="result") not in expected_statuses:
        raise _VerificationArtifactError(
            "Verification result does not have an allowed status."
        )
    return json.dumps(data, indent=2, sort_keys=True)


def _validate_verification_result_shape(data: dict[str, Any]) -> None:
    if data.get("schema_version") != VERIFICATION_SCHEMA_VERSION:
        raise _VerificationArtifactError(
            "Verification result has an unsupported schema version."
        )
    if data.get("format") != VERIFICATION_ROUND_FORMAT:
        raise _VerificationArtifactError(
            "Verification result has an unsupported format."
        )
    _verification_status(data.get("status"), source="result")
    if not isinstance(data.get("round_index"), int) or isinstance(
        data.get("round_index"), bool
    ):
        raise _VerificationArtifactError(
            "Verification result has an invalid round index."
        )
    for field in ("started_at", "ended_at"):
        if not isinstance(data.get(field), str) or not data[field]:
            raise _VerificationArtifactError(
                f"Verification result has invalid {field}."
            )
    duration = data.get("duration_seconds")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, int | float)
        or duration < 0
    ):
        raise _VerificationArtifactError(
            "Verification result has invalid duration_seconds."
        )
    if not isinstance(data.get("commands"), list):
        raise _VerificationArtifactError(
            "Verification result commands must be an array."
        )


def read_verification_source_fingerprint(
    run_path: Path,
    run_record: RunRecord,
    *,
    expected_statuses: frozenset[VerificationStatus],
    verification_commands: tuple[VerificationCommand, ...],
    require_all_commands_pass: bool = False,
    require_authoritative_pass: bool = False,
    expected_command_cwd: Path | None = None,
    expected_attempt_sequence: int | None = None,
    expected_round_index: int | None = None,
) -> str:
    try:
        record = latest_attempt(
            run_path,
            phases=(AttemptPhase.VERIFYING,),
            statuses=(AttemptStatus.COMPLETED,),
        )
        if record is None:
            raise _VerificationArtifactError(
                "No completed verification attempt exists."
            )
        if (
            expected_attempt_sequence is not None
            and record.sequence != expected_attempt_sequence
        ):
            raise _VerificationArtifactError(
                "Completed verification evidence is not the current verification "
                "attempt."
            )
        data = _read_result(run_path, record)
        _validate_verification_result(
            data,
            expected_statuses=expected_statuses,
            verification_commands=verification_commands,
            require_all_commands_pass=require_all_commands_pass,
            require_authoritative_pass=require_authoritative_pass,
            expected_command_cwd=expected_command_cwd,
            expected_round_index=expected_round_index,
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


def baseline_verification_evidence_problem(
    run_path: Path,
    run_record: RunRecord,
    baseline_record: BaselineRecord,
    *,
    verification_commands: tuple[VerificationCommand, ...],
) -> str | None:
    if (
        baseline_record.verification_commands_fingerprint
        != verification_commands_fingerprint(verification_commands)
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
            phases=(AttemptPhase.PREPARING,),
            statuses=(AttemptStatus.COMPLETED,),
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
            expected_statuses=frozenset({VerificationStatus.PASS}),
            verification_commands=verification_commands,
            require_all_commands_pass=True,
        )
    except (_VerificationArtifactError, AttemptError) as error:
        return f"Clean baseline verification evidence is invalid: {error}"
    return None


def _read_result(run_path: Path, record: AttemptRecord) -> dict[str, Any]:
    path = attempt_result_path(run_path, record)
    try:
        data = read_json_object(path)
    except CodecError as error:
        raise _VerificationArtifactError(
            f"Could not read verification result: {error}"
        ) from error
    return data


def _validate_verification_result(
    data: dict[str, Any],
    *,
    expected_statuses: frozenset[VerificationStatus],
    verification_commands: tuple[VerificationCommand, ...],
    require_all_commands_pass: bool = False,
    require_authoritative_pass: bool = False,
    expected_command_cwd: Path | None = None,
    expected_round_index: int | None = None,
) -> None:
    if data.get("schema_version") != VERIFICATION_SCHEMA_VERSION:
        raise _VerificationArtifactError(
            "Verification result has an unsupported schema version."
        )
    if data.get("format") != VERIFICATION_ROUND_FORMAT:
        raise _VerificationArtifactError(
            "Verification result has an unsupported format."
        )
    if (
        _verification_status(data.get("status"), source="result")
        not in expected_statuses
    ):
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
            _verification_status(result.get("status"), source="command")
            is not VerificationStatus.PASS
            or result.get("exit_code") != 0
            or result.get("error_kind") is not None
            or result.get("error_message") is not None
        ):
            message = (
                "Verification command evidence is not passing."
                if require_authoritative_pass
                else "Baseline verification command evidence is not passing."
            )
            raise _VerificationArtifactError(message)
        if require_authoritative_pass:
            _validate_authoritative_command_evidence(
                result,
                expected_command_cwd=expected_command_cwd,
            )
    if require_authoritative_pass:
        _validate_authoritative_round_evidence(
            data,
            expected_round_index=expected_round_index,
        )


def _verification_status(value: object, *, source: str) -> VerificationStatus:
    if not isinstance(value, str):
        raise _VerificationArtifactError(
            f"Verification {source} has an unsupported status."
        )
    try:
        return VerificationStatus(value)
    except ValueError as error:
        raise _VerificationArtifactError(
            f"Verification {source} has an unsupported status."
        ) from error


def _validate_authoritative_command_evidence(
    data: dict[str, Any],
    *,
    expected_command_cwd: Path | None,
) -> None:
    cwd = data.get("cwd")
    if not isinstance(cwd, str) or not cwd:
        raise _VerificationArtifactError(
            "Verification command evidence has no working directory."
        )
    if expected_command_cwd is None:
        raise _VerificationArtifactError(
            "Authoritative verification requires an expected working directory."
        )
    if Path(cwd).resolve(strict=False) != expected_command_cwd.resolve(strict=False):
        raise _VerificationArtifactError(
            "Verification command evidence has an unexpected working directory."
        )
    for field in ("started_at", "ended_at"):
        if not isinstance(data.get(field), str) or not data[field]:
            raise _VerificationArtifactError(
                f"Verification command evidence has invalid {field}."
            )
    duration = data.get("duration_seconds")
    if (
        not isinstance(duration, int | float)
        or isinstance(duration, bool)
        or duration < 0
    ):
        raise _VerificationArtifactError(
            "Verification command evidence has invalid duration_seconds."
        )
    if not isinstance(data.get("stdout"), str) or not isinstance(
        data.get("stderr"), str
    ):
        raise _VerificationArtifactError(
            "Verification command evidence has invalid process output."
        )


def _validate_authoritative_round_evidence(
    data: dict[str, Any],
    *,
    expected_round_index: int | None,
) -> None:
    round_index = data.get("round_index")
    if (
        not isinstance(round_index, int)
        or isinstance(round_index, bool)
        or round_index < 0
    ):
        raise _VerificationArtifactError(
            "Verification result has an invalid round index."
        )
    if expected_round_index is None or round_index != expected_round_index:
        raise _VerificationArtifactError(
            "Verification result round does not match the final correction round."
        )
    for field in ("started_at", "ended_at"):
        if not isinstance(data.get(field), str) or not data[field]:
            raise _VerificationArtifactError(
                f"Verification result has invalid {field}."
            )
    duration = data.get("duration_seconds")
    if (
        not isinstance(duration, int | float)
        or isinstance(duration, bool)
        or duration < 0
    ):
        raise _VerificationArtifactError(
            "Verification result has invalid duration_seconds."
        )
    if data.get("safety_violations") != []:
        raise _VerificationArtifactError(
            "Verification result contains safety violations or invalid safety evidence."
        )
    if data.get("correction_reasons") != []:
        raise _VerificationArtifactError(
            "Verification result contains correction reasons or invalid correction "
            "evidence."
        )


def verification_commands_fingerprint(
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
    "VerificationEvidence",
    "baseline_verification_evidence_problem",
    "read_verification_evidence",
    "read_verification_result_text",
    "read_verification_source_fingerprint",
    "verification_commands_fingerprint",
]
