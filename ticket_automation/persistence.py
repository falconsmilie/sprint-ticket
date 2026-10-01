"""Durable JSON and timestamp primitives for persisted records."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, TextIO, TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
JsonMapping: TypeAlias = Mapping[str, JsonValue]


class PersistenceError(RuntimeError):
    """Raised when a durable persistence operation fails."""


class CodecError(ValueError):
    """Raised when persisted data does not satisfy its declared codec."""


AtomicCommit: TypeAlias = Callable[[Callable[[], None]], None]


def atomic_write_json(
    path: Path | str,
    value: JsonMapping,
    *,
    commit: AtomicCommit | None = None,
) -> None:
    """Write one JSON object using the repository-wide durability policy."""

    def write_payload(output: TextIO) -> None:
        json.dump(value, output, indent=2, sort_keys=True, ensure_ascii=False)
        output.write("\n")

    _atomic_write_text_payload(
        Path(path),
        write_payload,
        description="JSON",
        commit=commit,
    )


def atomic_write_text(path: Path | str, value: str) -> None:
    """Atomically replace one UTF-8 text artifact without following file links."""

    if not isinstance(value, str):
        raise TypeError("text artifact value must be a string.")

    def write_payload(output: TextIO) -> None:
        output.write(value)

    _atomic_write_text_payload(Path(path), write_payload, description="text")


def atomic_write_bytes(path: Path | str, value: bytes) -> None:
    """Atomically replace one byte artifact without following file links."""

    if not isinstance(value, bytes):
        raise TypeError("byte artifact value must be bytes.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    descriptor = -1
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        _sync_directory(destination.parent)
    except BaseException as error:
        if descriptor != -1:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if not isinstance(error, Exception):
            raise
        if isinstance(error, PersistenceError):
            raise
        raise PersistenceError(
            f"Could not atomically write bytes {destination}: {error}"
        ) from error


def atomic_copy_file(source: Path | str, destination: Path | str) -> None:
    """Atomically publish a file without loading it all or following the target."""

    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    descriptor = -1
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination_path.parent,
            prefix=f".{destination_path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        with (
            source_path.open("rb") as input_file,
            os.fdopen(descriptor, "wb") as output_file,
        ):
            descriptor = -1
            shutil.copyfileobj(input_file, output_file, length=64 * 1024)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, destination_path)
        _sync_directory(destination_path.parent)
    except BaseException as error:
        if descriptor != -1:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if not isinstance(error, Exception):
            raise
        if isinstance(error, PersistenceError):
            raise
        raise PersistenceError(
            f"Could not atomically publish file {destination_path}: {error}"
        ) from error


def exclusive_write_text(path: Path | str, value: str) -> None:
    """Create one immutable UTF-8 artifact without following an existing entry."""

    if not isinstance(value, str):
        raise TypeError("text artifact value must be a string.")
    destination = Path(path)

    def write_payload(output: IO[Any]) -> None:
        output.write(value)

    _exclusive_write_payload(
        destination,
        write_payload,
        binary=False,
        description="text",
    )


def exclusive_write_bytes(path: Path | str, value: bytes) -> None:
    """Create one immutable byte artifact without following an existing entry."""

    if not isinstance(value, bytes):
        raise TypeError("byte artifact value must be bytes.")
    destination = Path(path)

    def write_payload(output: IO[Any]) -> None:
        output.write(value)

    _exclusive_write_payload(
        destination,
        write_payload,
        binary=True,
        description="byte",
    )


def _atomic_write_text_payload(
    destination: Path,
    write_payload: Callable[[TextIO], None],
    *,
    description: str,
    commit: AtomicCommit | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    descriptor = -1
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            text=True,
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            descriptor = -1
            write_payload(output)
            output.flush()
            os.fsync(output.fileno())
        def replace_destination() -> None:
            assert temporary is not None
            os.replace(temporary, destination)

        if commit is None:
            replace_destination()
        else:
            # Payload creation and fsync intentionally happen before entering a
            # caller-owned commit fence. The fence therefore covers only the
            # final validity check and atomic replacement, not potentially slow
            # JSON serialization or storage flushes.
            commit(replace_destination)
        _sync_directory(destination.parent)
    except BaseException as error:
        if descriptor != -1:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if not isinstance(error, Exception):
            raise
        if isinstance(error, PersistenceError):
            raise
        raise PersistenceError(
            f"Could not atomically write {description} {destination}: {error}"
        ) from error


def _exclusive_write_payload(
    destination: Path,
    write_payload: Callable[[IO[Any]], None],
    *,
    binary: bool,
    description: str,
) -> None:
    """Create a new file entry exclusively and durably, never opening a target."""

    descriptor = -1
    created = False
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if binary and hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(destination, flags, 0o600)
        created = True
        mode = "wb" if binary else "w"
        kwargs = {} if binary else {"encoding": "utf-8", "newline": "\n"}
        with os.fdopen(descriptor, mode, **kwargs) as output:
            descriptor = -1
            write_payload(output)
            output.flush()
            os.fsync(output.fileno())
        _sync_directory(destination.parent)
    except BaseException as error:
        if descriptor != -1:
            os.close(descriptor)
        if created:
            destination.unlink(missing_ok=True)
        if not isinstance(error, Exception):
            raise
        if isinstance(error, PersistenceError):
            raise
        raise PersistenceError(
            f"Could not exclusively create {description} {destination}: {error}"
        ) from error


def read_json(path: Path | str) -> JsonValue:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CodecError(f"Could not read JSON {source}: {error}") from error
    return _require_json_value(value, source=str(source))


def read_json_object(path: Path | str) -> JsonObject:
    value = read_json(path)
    if not isinstance(value, dict):
        raise CodecError(f"JSON record must be an object: {Path(path)}")
    return value


def format_timestamp(value: datetime | None = None) -> str:
    timestamp = datetime.now(UTC) if value is None else value
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def timestamp_now(clock: Callable[[], datetime] | None = None) -> str:
    return format_timestamp(None if clock is None else clock())


def parse_timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise CodecError(f"{field} must be a non-empty timestamp string.")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as error:
        raise CodecError(f"{field} must be an ISO-8601 timestamp.") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CodecError(f"{field} must include a timezone.")
    return parsed.astimezone(UTC)


def _require_json_value(value: object, *, source: str) -> JsonValue:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not (-float("inf") < value < float("inf")):
            raise CodecError(f"{source} contains a non-finite JSON number.")
        return value
    if isinstance(value, list):
        return [_require_json_value(item, source=source) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise CodecError(f"{source} contains a non-string object key.")
        return {
            key: _require_json_value(item, source=source) for key, item in value.items()
        }
    if isinstance(value, Sequence):
        return [_require_json_value(item, source=source) for item in value]
    raise CodecError(f"{source} contains a non-JSON value: {type(value).__name__}.")


def _sync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "CodecError",
    "JsonMapping",
    "JsonObject",
    "JsonScalar",
    "JsonValue",
    "PersistenceError",
    "atomic_copy_file",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "exclusive_write_bytes",
    "exclusive_write_text",
    "format_timestamp",
    "parse_timestamp",
    "read_json",
    "read_json_object",
    "timestamp_now",
]
