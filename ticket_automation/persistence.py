"""Durable JSON and timestamp primitives for persisted records."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
JsonMapping: TypeAlias = Mapping[str, JsonValue]


class PersistenceError(RuntimeError):
    """Raised when a durable persistence operation fails."""


class CodecError(ValueError):
    """Raised when persisted data does not satisfy its declared codec."""


def atomic_write_json(path: Path | str, value: JsonMapping) -> None:
    """Write one JSON object using the repository-wide durability policy."""

    destination = Path(path)
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
            json.dump(value, output, indent=2, sort_keys=True, ensure_ascii=False)
            output.write("\n")
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
            f"Could not atomically write JSON {destination}: {error}"
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
    "atomic_write_json",
    "format_timestamp",
    "parse_timestamp",
    "read_json",
    "read_json_object",
    "timestamp_now",
]
