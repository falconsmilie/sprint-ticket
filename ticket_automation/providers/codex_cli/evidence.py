"""Provider-native diagnostic artifacts for Codex CLI executions."""

from __future__ import annotations

import os
import stat
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ...application.agent_execution import (
    ArtifactReference,
    ArtifactRole,
    AttemptArtifactLayout,
)
from ...persistence import (
    JsonValue,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    exclusive_write_text,
)

PROMPT_ARTIFACT = "prompt.md"
EVENTS_ARTIFACT = "events.jsonl"
STDERR_ARTIFACT = "stderr.log"
EXECUTION_ARTIFACT = "codex-execution.json"
RESULT_ARTIFACT = "codex-result.json"
DIAGNOSTIC_RESULT_ARTIFACT = "codex-diagnostic-result.json"
DIAGNOSTIC_RESULT_BINARY_ARTIFACT = "codex-diagnostic-result.bin"


@dataclass(frozen=True)
class CodexArtifactPaths:
    layout: AttemptArtifactLayout
    directory: Path
    prompt: Path
    events: Path
    stderr: Path
    execution: Path
    result: Path
    diagnostic_result: Path
    diagnostic_result_binary: Path

    @classmethod
    def create(cls, layout: AttemptArtifactLayout) -> CodexArtifactPaths:
        if not isinstance(layout, AttemptArtifactLayout):
            raise TypeError("layout must be an AttemptArtifactLayout.")
        directory = layout.attempt_root
        return cls(
            layout=layout,
            directory=directory,
            prompt=directory / PROMPT_ARTIFACT,
            events=directory / EVENTS_ARTIFACT,
            stderr=directory / STDERR_ARTIFACT,
            execution=directory / EXECUTION_ARTIFACT,
            result=directory / RESULT_ARTIFACT,
            diagnostic_result=directory / DIAGNOSTIC_RESULT_ARTIFACT,
            diagnostic_result_binary=directory / DIAGNOSTIC_RESULT_BINARY_ARTIFACT,
        )

    def references(self) -> tuple[ArtifactReference, ...]:
        self.revalidate()
        references = [
            self.layout.reference(ArtifactRole.PROMPT, self.prompt, "text/markdown"),
            self.layout.reference(
                ArtifactRole.PROVIDER_EVENTS,
                self.events,
                "application/x-ndjson",
            ),
            self.layout.reference(
                ArtifactRole.STANDARD_ERROR, self.stderr, "text/plain"
            ),
            self.layout.reference(
                ArtifactRole.PROVIDER_EXECUTION_DETAILS,
                self.execution,
                "application/json",
            ),
        ]
        if self.result.is_file():
            references.append(
                self.layout.reference(
                    ArtifactRole.TYPED_RESULT, self.result, "application/json"
                )
            )
        if self.diagnostic_result.is_file():
            references.append(
                self.layout.reference(
                    ArtifactRole.PROVIDER_DIAGNOSTIC_RESULT,
                    self.diagnostic_result,
                    "application/json",
                )
            )
        if self.diagnostic_result_binary.is_file():
            references.append(
                self.layout.reference(
                    ArtifactRole.PROVIDER_DIAGNOSTIC_RESULT,
                    self.diagnostic_result_binary,
                    "application/octet-stream",
                )
            )
        return tuple(references)

    def revalidate(self) -> None:
        """Reject cached provider paths after their owning run is retargeted."""

        self.layout.revalidate()
        expected = {
            "prompt": self.layout.path(PROMPT_ARTIFACT),
            "events": self.layout.path(EVENTS_ARTIFACT),
            "stderr": self.layout.path(STDERR_ARTIFACT),
            "execution": self.layout.path(EXECUTION_ARTIFACT),
            "result": self.layout.path(RESULT_ARTIFACT),
            "diagnostic_result": self.layout.path(DIAGNOSTIC_RESULT_ARTIFACT),
            "diagnostic_result_binary": self.layout.path(
                DIAGNOSTIC_RESULT_BINARY_ARTIFACT
            ),
        }
        for name, path in expected.items():
            if getattr(self, name) != path:
                raise RuntimeError(f"Cached Codex artifact path changed: {name}.")


@dataclass(frozen=True)
class ProcessOutputPublication:
    stdout_truncated: bool
    stderr_truncated: bool

    @property
    def truncated(self) -> bool:
        return self.stdout_truncated or self.stderr_truncated


def create_process_capture_paths(paths: CodexArtifactPaths) -> tuple[Path, Path]:
    """Choose exclusive attempt-local staging paths for O(1) publication."""

    paths.revalidate()
    nonce = uuid.uuid4().hex
    stdout = paths.layout.path(f".codex-events-{nonce}.capture")
    stderr = paths.layout.path(f".codex-stderr-{nonce}.capture")
    paths.revalidate()
    return stdout, stderr


@contextmanager
def process_capture_environment(
    paths: CodexArtifactPaths,
) -> Iterator[tuple[Path, Path]]:
    captures = create_process_capture_paths(paths)
    try:
        yield captures
    finally:
        paths.revalidate()
        for capture in captures:
            capture.unlink(missing_ok=True)
        paths.revalidate()


def prepare(paths: CodexArtifactPaths, prompt: str) -> None:
    paths.revalidate()
    paths.directory.mkdir(parents=True, exist_ok=True)
    paths.revalidate()
    paths.result.unlink(missing_ok=True)
    paths.diagnostic_result.unlink(missing_ok=True)
    paths.diagnostic_result_binary.unlink(missing_ok=True)
    exclusive_write_text(paths.prompt, prompt)
    atomic_write_text(paths.events, "")
    atomic_write_text(paths.stderr, "")


def write_process_output(
    paths: CodexArtifactPaths,
    *,
    stdout: str,
    stderr: str,
) -> None:
    paths.revalidate()
    atomic_write_text(paths.events, stdout)
    atomic_write_text(paths.stderr, stderr)


def write_execution(paths: CodexArtifactPaths, record: Mapping[str, JsonValue]) -> None:
    paths.revalidate()
    atomic_write_json(paths.execution, record)


def write_result(paths: CodexArtifactPaths, result: str) -> None:
    paths.revalidate()
    atomic_write_text(paths.result, result)


def write_diagnostic_result(paths: CodexArtifactPaths, result: str) -> None:
    paths.revalidate()
    paths.diagnostic_result_binary.unlink(missing_ok=True)
    atomic_write_text(paths.diagnostic_result, result)


def write_diagnostic_result_bytes(paths: CodexArtifactPaths, result: bytes) -> None:
    paths.revalidate()
    paths.diagnostic_result.unlink(missing_ok=True)
    atomic_write_bytes(paths.diagnostic_result_binary, result)


def publish_process_output(
    paths: CodexArtifactPaths,
    *,
    stdout_source: Path | None,
    stderr_source: Path | None,
    stdout_fallback: str,
    stderr_fallback: str,
    stdout_capture_complete: bool = True,
    stderr_capture_complete: bool = True,
) -> ProcessOutputPublication:
    paths.revalidate()
    stdout_promoted = False
    if stdout_capture_complete and stdout_source is not None:
        stdout_promoted = _promote_capture(paths, stdout_source, paths.events)
    if not stdout_promoted:
        atomic_write_text(paths.events, stdout_fallback)
    paths.revalidate()
    stderr_promoted = False
    if stderr_capture_complete and stderr_source is not None:
        stderr_promoted = _promote_capture(paths, stderr_source, paths.stderr)
    if not stderr_promoted:
        atomic_write_text(paths.stderr, stderr_fallback)
    paths.revalidate()
    return ProcessOutputPublication(
        stdout_truncated=not stdout_capture_complete,
        stderr_truncated=not stderr_capture_complete,
    )


def _promote_capture(
    paths: CodexArtifactPaths,
    source: Path,
    destination: Path,
) -> bool:
    """Validate and atomically promote an attempt-local regular capture file."""

    paths.revalidate()
    source_path = Path(os.path.abspath(source))
    if source_path.parent != paths.directory or not source_path.name.endswith(
        ".capture"
    ):
        raise RuntimeError("Codex capture path is not attempt-owned staging storage.")
    try:
        details = source_path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise RuntimeError(
            "Codex capture staging path is linked or not a regular file."
        )
    paths.revalidate()
    os.replace(source_path, destination)
    paths.revalidate()
    return True


__all__ = [
    "DIAGNOSTIC_RESULT_ARTIFACT",
    "DIAGNOSTIC_RESULT_BINARY_ARTIFACT",
    "EVENTS_ARTIFACT",
    "EXECUTION_ARTIFACT",
    "PROMPT_ARTIFACT",
    "RESULT_ARTIFACT",
    "STDERR_ARTIFACT",
    "CodexArtifactPaths",
    "ProcessOutputPublication",
    "create_process_capture_paths",
    "prepare",
    "process_capture_environment",
    "publish_process_output",
    "write_diagnostic_result",
    "write_diagnostic_result_bytes",
    "write_execution",
    "write_process_output",
    "write_result",
]
