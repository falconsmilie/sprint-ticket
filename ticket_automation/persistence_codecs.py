"""Public JSON codecs for lifecycle-owned persistence artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .attempts import (
    AttemptRecord,
    attempt_artifact_file_exists,
    attempt_result_path,
)
from .domain.task_results import (
    ImplementationResult,
    ResultValidationError,
    ReviewResult,
)
from .models import StageOutcome, VerificationStatus
from .persistence import CodecError, atomic_write_json, read_json
from .run_ownership import RunOwnership
from .task_result_codecs import (
    decode_implementation_result,
    decode_review_result,
    encode_implementation_result,
    encode_review_result,
)
from .verification import VerificationError, VerificationFailure


class PersistenceCodecError(ValueError):
    """Raised when a lifecycle artifact cannot be decoded as declared."""


@dataclass(frozen=True)
class StageMessageResult:
    status: StageOutcome
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, StageOutcome):
            raise PersistenceCodecError("stage message status must be a StageOutcome.")
        if not isinstance(self.message, str) or not self.message.strip():
            raise PersistenceCodecError("stage message must be non-empty.")


def _read_attempt_result_value(
    run_dir: Path | str,
    attempt: AttemptRecord,
    *,
    run_ownership: RunOwnership | None = None,
) -> object | None:
    """Decode an optional attempt result as JSON without interpreting its schema."""

    path = attempt_result_path(
        run_dir,
        attempt,
        run_ownership=run_ownership,
    )
    if not attempt_artifact_file_exists(
        run_dir,
        attempt,
        attempt.result_path,
        run_ownership=run_ownership,
    ):
        return None
    try:
        return (
            read_json(path)
            if run_ownership is None
            else run_ownership.read_descendant(path, read_json)
        )
    except CodecError as error:
        raise PersistenceCodecError(
            f"Attempt result artifact could not be read as JSON: {path}: {error}"
        ) from error


def read_implementation_result(
    run_dir: Path | str,
    attempt: AttemptRecord,
    *,
    run_ownership: RunOwnership | None = None,
) -> ImplementationResult | None:
    value = _read_attempt_result_value(
        run_dir,
        attempt,
        run_ownership=run_ownership,
    )
    if value is None:
        return None
    try:
        return decode_implementation_result(value)
    except ResultValidationError as error:
        raise PersistenceCodecError(str(error)) from error


def read_review_result(
    run_dir: Path | str,
    attempt: AttemptRecord,
    *,
    run_ownership: RunOwnership | None = None,
) -> ReviewResult | None:
    value = _read_attempt_result_value(
        run_dir,
        attempt,
        run_ownership=run_ownership,
    )
    if value is None:
        return None
    try:
        return decode_review_result(value)
    except ResultValidationError as error:
        raise PersistenceCodecError(str(error)) from error


def read_stage_message_result(
    run_dir: Path | str,
    attempt: AttemptRecord,
    *,
    run_ownership: RunOwnership | None = None,
) -> StageMessageResult | None:
    value = _read_attempt_result_value(
        run_dir,
        attempt,
        run_ownership=run_ownership,
    )
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"status", "message"}:
        raise PersistenceCodecError("stage message result has unsupported fields.")
    try:
        return StageMessageResult(
            status=_decode_stage_message_status(value["status"]),
            message=value["message"],
        )
    except (TypeError, ValueError) as error:
        raise PersistenceCodecError(str(error)) from error


def read_verification_failures(
    run_dir: Path | str,
    attempt: AttemptRecord,
    *,
    run_ownership: RunOwnership | None = None,
) -> tuple[VerificationFailure, ...]:
    value = _read_attempt_result_value(
        run_dir,
        attempt,
        run_ownership=run_ownership,
    )
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise PersistenceCodecError("verification result must be an object.")
    try:
        status = VerificationStatus(value.get("status"))
    except (TypeError, ValueError) as error:
        raise PersistenceCodecError(
            "verification result status is unsupported."
        ) from error
    if status is not VerificationStatus.FAIL:
        return ()
    encoded = value.get("correction_reasons")
    if not isinstance(encoded, list) or not encoded:
        raise PersistenceCodecError(
            "failed verification evidence must contain correction reasons."
        )
    failures: list[VerificationFailure] = []
    for index, item in enumerate(encoded, start=1):
        try:
            failures.append(VerificationFailure.from_dict(item))
        except VerificationError as error:
            raise PersistenceCodecError(
                f"verification correction reason {index} is invalid: {error}"
            ) from error
    return tuple(failures)


def write_stage_message_result(
    run_dir: Path | str,
    attempt: AttemptRecord,
    *,
    status: StageOutcome,
    message: str,
) -> Path:
    """Persist the small status/message result used by coordination stages."""

    path = attempt_result_path(run_dir, attempt)
    result = StageMessageResult(status, message)
    wire_status = (
        VerificationStatus.PASS.value
        if result.status is StageOutcome.COMPLETED
        else result.status.value
    )
    _write_stage_message(path, result, wire_status=wire_status)
    return path


def write_stage_message(path: Path, *, status: StageOutcome, message: str) -> None:
    result = StageMessageResult(status, message)
    _write_stage_message(path, result, wire_status=result.status.value)


def _write_stage_message(
    path: Path,
    result: StageMessageResult,
    *,
    wire_status: str,
) -> None:
    atomic_write_json(path, {"status": wire_status, "message": result.message})


def _decode_stage_message_status(value: object) -> StageOutcome:
    if value == VerificationStatus.PASS.value:
        return StageOutcome.COMPLETED
    if not isinstance(value, str):
        raise PersistenceCodecError("stage message status is unsupported.")
    try:
        return StageOutcome(value)
    except ValueError as error:
        raise PersistenceCodecError("stage message status is unsupported.") from error


def write_implementation_result(path: Path, result: ImplementationResult) -> None:
    atomic_write_json(path, encode_implementation_result(result))


def write_review_result(path: Path, result: ReviewResult) -> None:
    atomic_write_json(path, encode_review_result(result))


__all__ = [
    "PersistenceCodecError",
    "StageMessageResult",
    "read_implementation_result",
    "read_review_result",
    "read_stage_message_result",
    "read_verification_failures",
    "write_implementation_result",
    "write_review_result",
    "write_stage_message",
    "write_stage_message_result",
]
