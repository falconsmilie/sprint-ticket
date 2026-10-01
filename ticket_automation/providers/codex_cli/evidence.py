"""Provider-native diagnostic artifacts for Codex CLI executions."""

from __future__ import annotations

import os
import stat
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
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


class _PartialCaptureUnavailable(RuntimeError):
    pass


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
    deadline_exhausted: bool = False

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


def recovery_capture_path(capture: Path) -> Path:
    """Return the private recovery spool paired with a process capture."""

    return capture.with_name(f"{capture.name}.recovery.capture")


@contextmanager
def process_capture_environment(
    paths: CodexArtifactPaths,
) -> Iterator[tuple[Path, Path]]:
    captures = create_process_capture_paths(paths)
    try:
        yield captures
    finally:
        paths.revalidate()
        for capture in (*captures, *(recovery_capture_path(path) for path in captures)):
            try:
                capture.unlink(missing_ok=True)
            except PermissionError:
                # A bounded return takes precedence over waiting for a broken
                # capture sink. Such a writer is never promoted to a published
                # artifact, and Windows may keep its staging handle undeletable.
                pass
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
    stdout_capture_stable: bool = False,
    stderr_capture_stable: bool = False,
    stdout_capture_prefix_bytes: int = 0,
    stderr_capture_prefix_bytes: int = 0,
    stdout_observed_bytes: int = 0,
    stderr_observed_bytes: int = 0,
    stdout_observed_tail: bytes = b"",
    stderr_observed_tail: bytes = b"",
    deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ProcessOutputPublication:
    paths.revalidate()
    deadline_exhausted = deadline is not None and clock() >= deadline
    stdout_promoted = False
    if stdout_capture_complete and stdout_source is not None:
        stdout_promoted = _promote_capture(paths, stdout_source, paths.events)
    elif stdout_source is not None:
        stdout_promoted, exhausted = _publish_partial_capture(
            paths,
            stdout_source,
            paths.events,
            capture_stable=stdout_capture_stable,
            prefix_bytes=stdout_capture_prefix_bytes,
            observed_bytes=stdout_observed_bytes,
            observed_tail=stdout_observed_tail,
            deadline=deadline,
            clock=clock,
        )
        deadline_exhausted = deadline_exhausted or exhausted
    if not stdout_promoted:
        atomic_write_text(paths.events, stdout_fallback)
    paths.revalidate()
    stderr_promoted = False
    if stderr_capture_complete and stderr_source is not None:
        stderr_promoted = _promote_capture(paths, stderr_source, paths.stderr)
    elif stderr_source is not None:
        stderr_promoted, exhausted = _publish_partial_capture(
            paths,
            stderr_source,
            paths.stderr,
            capture_stable=stderr_capture_stable,
            prefix_bytes=stderr_capture_prefix_bytes,
            observed_bytes=stderr_observed_bytes,
            observed_tail=stderr_observed_tail,
            deadline=deadline,
            clock=clock,
        )
        deadline_exhausted = deadline_exhausted or exhausted
    if not stderr_promoted:
        atomic_write_text(paths.stderr, stderr_fallback)
    paths.revalidate()
    return ProcessOutputPublication(
        stdout_truncated=not stdout_capture_complete,
        stderr_truncated=not stderr_capture_complete,
        deadline_exhausted=deadline_exhausted,
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


def _publish_partial_capture(
    paths: CodexArtifactPaths,
    source: Path,
    destination: Path,
    *,
    capture_stable: bool,
    prefix_bytes: int,
    observed_bytes: int,
    observed_tail: bytes,
    deadline: float | None,
    clock: Callable[[], float],
) -> tuple[bool, bool]:
    """Publish a partial stream without recopying a stable durable prefix."""

    paths.revalidate()
    source_path = Path(os.path.abspath(source))
    if source_path.parent != paths.directory or not source_path.name.endswith(
        ".capture"
    ):
        raise RuntimeError("Codex capture path is not attempt-owned staging storage.")
    try:
        details = source_path.lstat()
    except FileNotFoundError:
        return False, _deadline_exhausted(deadline, clock)
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise RuntimeError(
            "Codex capture staging path is linked or not a regular file."
        )
    if capture_stable:
        try:
            published = _complete_stable_capture(
                paths,
                source_path,
                destination,
                captured_bytes=details.st_size,
                observed_bytes=observed_bytes,
                observed_tail=observed_tail,
                clock=clock,
            )
        except TimeoutError:
            return False, True
        return published, _deadline_exhausted(deadline, clock)

    # A worker that did not stop cannot donate its staging inode to published
    # evidence. Copy only while the invocation's one overall allowance remains;
    # the bounded tail is used solely when it covers the complete missing suffix.
    temporary = paths.layout.path(f".codex-publish-{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    copied = 0
    try:
        with (
            os.fdopen(descriptor, "wb", buffering=0) as output,
            source_path.open("rb", buffering=0) as captured,
        ):
            remaining = max(0, prefix_bytes)
            while remaining:
                if _deadline_exhausted(deadline, clock):
                    raise TimeoutError("Codex output publication deadline expired.")
                chunk = captured.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                _write_all(output, chunk, deadline=deadline, clock=clock)
                copied += len(chunk)
                remaining -= len(chunk)
            tail_start = max(0, observed_bytes - len(observed_tail))
            if copied > observed_bytes or copied < tail_start:
                raise _PartialCaptureUnavailable(
                    "Observed stream suffix is not completely retained."
                )
            suffix = observed_tail[copied - tail_start :]
            _write_all(output, suffix, deadline=deadline, clock=clock)
        paths.revalidate()
        os.replace(temporary, destination)
        paths.revalidate()
    except (TimeoutError, _PartialCaptureUnavailable) as error:
        temporary.unlink(missing_ok=True)
        return False, isinstance(error, TimeoutError)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return True, _deadline_exhausted(deadline, clock)


def _complete_stable_capture(
    paths: CodexArtifactPaths,
    source: Path,
    destination: Path,
    *,
    captured_bytes: int,
    observed_bytes: int,
    observed_tail: bytes,
    clock: Callable[[], float],
) -> bool:
    """Append at most one retained read chunk, then promote in constant time."""

    if captured_bytes < 0 or observed_bytes < captured_bytes:
        return False
    if captured_bytes < observed_bytes:
        tail_start = max(0, observed_bytes - len(observed_tail))
        if captured_bytes < tail_start:
            return False
        suffix = observed_tail[captured_bytes - tail_start :]
        if len(suffix) != observed_bytes - captured_bytes:
            return False
        _append_capture_suffix(
            paths,
            source,
            suffix,
            expected_size=captured_bytes,
            clock=clock,
        )
    return _promote_capture(paths, source, destination)


def _append_capture_suffix(
    paths: CodexArtifactPaths,
    source: Path,
    suffix: bytes,
    *,
    expected_size: int,
    clock: Callable[[], float],
) -> None:
    """Append one bounded chunk without following a replaced staging path."""

    paths.revalidate()
    flags = os.O_WRONLY | os.O_APPEND
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source, flags)
    try:
        opened = os.fstat(descriptor)
        current = source.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
            or opened.st_size != expected_size
        ):
            raise RuntimeError(
                "Codex capture staging path changed before suffix publication."
            )
        with os.fdopen(descriptor, "wb", buffering=0) as output:
            descriptor = -1
            # The worker contract limits this to its single retained read chunk.
            # Finish that bounded suffix even if the deadline is crossed during
            # the write; the caller reports exhaustion, and no work scales with
            # the already captured log prefix.
            _write_all(output, suffix, deadline=None, clock=clock)
        paths.revalidate()
    finally:
        if descriptor != -1:
            os.close(descriptor)


def _deadline_exhausted(
    deadline: float | None,
    clock: Callable[[], float],
) -> bool:
    return deadline is not None and clock() >= deadline


def _write_all(
    output,
    value: bytes,
    *,
    deadline: float | None,
    clock: Callable[[], float],
) -> None:
    view = memoryview(value)
    while view:
        if _deadline_exhausted(deadline, clock):
            raise TimeoutError("Codex output publication deadline expired.")
        written = output.write(view)
        if written is None:
            written = len(view)
        if written <= 0:
            raise OSError("Codex artifact publication made no progress.")
        view = view[written:]


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
    "recovery_capture_path",
    "write_diagnostic_result",
    "write_diagnostic_result_bytes",
    "write_execution",
    "write_process_output",
    "write_result",
]
