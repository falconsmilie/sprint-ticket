"""Public JSON codecs for lifecycle-owned persistence artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .attempts import AttemptRecord, attempt_result_path
from .domain.task_results import (
    ImplementationResult,
    ResultValidationError,
    ReviewResult,
)
from .persistence import CodecError, atomic_write_json, read_json
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
    status: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, str) or not self.status.strip():
            raise PersistenceCodecError("stage message status must be non-empty.")
        if not isinstance(self.message, str) or not self.message.strip():
            raise PersistenceCodecError("stage message must be non-empty.")


def _read_attempt_result_value(
    run_dir: Path | str,
    attempt: AttemptRecord,
) -> object | None:
    """Decode an optional attempt result as JSON without interpreting its schema."""

    path = attempt_result_path(run_dir, attempt)
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except CodecError as error:
        raise PersistenceCodecError(
            f"Attempt result artifact could not be read as JSON: {path}: {error}"
        ) from error


def read_implementation_result(
    run_dir: Path | str,
    attempt: AttemptRecord,
) -> ImplementationResult | None:
    value = _read_attempt_result_value(run_dir, attempt)
    if value is None:
        return None
    try:
        return decode_implementation_result(value)
    except ResultValidationError as error:
        raise PersistenceCodecError(str(error)) from error


def read_review_result(
    run_dir: Path | str,
    attempt: AttemptRecord,
) -> ReviewResult | None:
    value = _read_attempt_result_value(run_dir, attempt)
    if value is None:
        return None
    try:
        return decode_review_result(value)
    except ResultValidationError as error:
        raise PersistenceCodecError(str(error)) from error


def read_stage_message_result(
    run_dir: Path | str,
    attempt: AttemptRecord,
) -> StageMessageResult | None:
    value = _read_attempt_result_value(run_dir, attempt)
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"status", "message"}:
        raise PersistenceCodecError("stage message result has unsupported fields.")
    try:
        return StageMessageResult(status=value["status"], message=value["message"])
    except (TypeError, ValueError) as error:
        raise PersistenceCodecError(str(error)) from error


def read_verification_failures(
    run_dir: Path | str,
    attempt: AttemptRecord,
) -> tuple[VerificationFailure, ...]:
    value = _read_attempt_result_value(run_dir, attempt)
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise PersistenceCodecError("verification result must be an object.")
    if value.get("status") != "FAIL":
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
    status: str,
    message: str,
) -> Path:
    """Persist the small status/message result used by coordination stages."""

    path = attempt_result_path(run_dir, attempt)
    write_stage_message(path, status=status, message=message)
    return path


def write_stage_message(path: Path, *, status: str, message: str) -> None:
    result = StageMessageResult(status, message)
    atomic_write_json(path, {"status": result.status, "message": result.message})


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
