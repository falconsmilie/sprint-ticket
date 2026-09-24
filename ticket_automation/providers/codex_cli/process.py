"""Codex CLI subprocess boundary."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ...process_output import decode_human_output


@dataclass(frozen=True)
class CodexCommand:
    argv: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str] | None = None


@dataclass(frozen=True)
class CodexProcessResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class CodexProcessTimeout:
    stdout: str
    stderr: str
    timeout_seconds: float


class CodexProcessTimedOut(TimeoutError):
    def __init__(self, result: CodexProcessTimeout):
        super().__init__(
            f"Codex process timed out after {result.timeout_seconds:g} seconds."
        )
        self.result = result


class CodexScratchDirectoryError(RuntimeError):
    pass


class CodexProcessRunner(Protocol):
    def run(
        self,
        command: CodexCommand,
        *,
        stdin: str,
        timeout_seconds: float | None,
        on_process_start: Callable[[], None] | None = None,
    ) -> CodexProcessResult: ...


class SubprocessCodexRunner:
    def run(
        self,
        command: CodexCommand,
        *,
        stdin: str,
        timeout_seconds: float | None,
        on_process_start: Callable[[], None] | None = None,
    ) -> CodexProcessResult:
        process = subprocess.Popen(
            command.argv,
            cwd=command.cwd,
            env=_process_environment(command.environment),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
        try:
            if on_process_start is not None:
                on_process_start()
            stdout, stderr = process.communicate(
                input=stdin.encode("utf-8"), timeout=timeout_seconds
            )
        except subprocess.TimeoutExpired as error:
            process.kill()
            final_stdout, final_stderr = process.communicate()
            raise CodexProcessTimedOut(
                CodexProcessTimeout(
                    stdout=decode_human_output(final_stdout),
                    stderr=decode_human_output(final_stderr),
                    timeout_seconds=float(timeout_seconds or 0),
                )
            ) from error
        except BaseException:
            process.kill()
            process.wait()
            raise
        return CodexProcessResult(
            returncode=process.returncode,
            stdout=decode_human_output(stdout),
            stderr=decode_human_output(stderr),
        )


def with_environment(
    command: CodexCommand, environment: Mapping[str, str]
) -> CodexCommand:
    return CodexCommand(command.argv, command.cwd, environment)


@contextmanager
def external_scratch_environment(repo_path: Path) -> Iterator[dict[str, str]]:
    try:
        scratch_path = Path(
            tempfile.mkdtemp(prefix="ticket-automation-codex-")
        ).resolve()
    except OSError as error:
        raise CodexScratchDirectoryError(
            f"Could not create an external Codex scratch directory: {error}"
        ) from error
    try:
        repository_path = Path(repo_path).resolve()
        try:
            scratch_path.relative_to(repository_path)
        except ValueError:
            pass
        else:
            raise CodexScratchDirectoryError(
                "Codex scratch directory must be outside the target repository: "
                f"{scratch_path}"
            )
        scratch = str(scratch_path)
        yield {"TEMP": scratch, "TMP": scratch, "TMPDIR": scratch}
    finally:
        shutil.rmtree(scratch_path, ignore_errors=True)


def _process_environment(overrides: Mapping[str, str] | None) -> dict[str, str]:
    environment = os.environ.copy()
    if overrides is not None:
        environment.update(overrides)
    return environment


__all__ = [
    "CodexCommand",
    "CodexProcessResult",
    "CodexProcessRunner",
    "CodexProcessTimedOut",
    "CodexProcessTimeout",
    "CodexScratchDirectoryError",
    "SubprocessCodexRunner",
    "external_scratch_environment",
    "with_environment",
]
