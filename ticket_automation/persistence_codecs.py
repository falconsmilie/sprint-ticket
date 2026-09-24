"""Public JSON codecs for lifecycle-owned persistence artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .attempts import AttemptRecord, attempt_result_path
from .domain.task_results import ImplementationResult, ReviewResult
from .persistence import CodecError, atomic_write_json, read_json
from .task_result_codecs import encode_implementation_result, encode_review_result


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


def read_attempt_result_value(
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
    "read_attempt_result_value",
    "write_implementation_result",
    "write_review_result",
    "write_stage_message",
    "write_stage_message_result",
]
