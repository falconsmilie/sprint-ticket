from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

UTF8 = "utf-8"


class ProcessOutputDecodeError(UnicodeError):
    """Raised when protocol output cannot be decoded with its required codec."""

    def __init__(self, stream_name: str, encoding: str, error: UnicodeDecodeError):
        super().__init__(f"{stream_name} was not valid {encoding}: byte {error.start}")
        self.stream_name = stream_name
        self.encoding = encoding
        self.__cause__ = error


@dataclass(frozen=True)
class TextProcessResult:
    args: Any
    returncode: int
    stdout: str
    stderr: str


def decode_human_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _normalize_newlines(value)
    return _normalize_newlines(value.decode(UTF8, errors="replace"))


def decode_protocol_output(
    value: str | bytes | None,
    *,
    stream_name: str,
    encoding: str = UTF8,
) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return value.decode(encoding)
    except UnicodeDecodeError as error:
        raise ProcessOutputDecodeError(stream_name, encoding, error) from error


def run_human_text_command(
    args: Sequence[str],
    *,
    cwd: Path,
    timeout: float | None = None,
) -> TextProcessResult:
    completed = subprocess.run(
        tuple(args),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=False,
        timeout=timeout,
        shell=False,
    )
    return TextProcessResult(
        args=completed.args,
        returncode=completed.returncode,
        stdout=decode_human_output(completed.stdout),
        stderr=decode_human_output(completed.stderr),
    )


def _normalize_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


__all__ = [
    "ProcessOutputDecodeError",
    "TextProcessResult",
    "decode_human_output",
    "decode_protocol_output",
    "run_human_text_command",
]
