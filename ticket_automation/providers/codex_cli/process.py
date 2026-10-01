"""Codex CLI subprocess boundary with live deadline observation."""

from __future__ import annotations

import codecs
import ctypes
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Protocol

from ...process_output import decode_human_output

FINALIZATION_SECONDS = 5.0
CLEANUP_SECONDS = 10.0
_READ_SIZE = 64 * 1024
_DIAGNOSTIC_TAIL_BYTES = _READ_SIZE
_MAX_OBSERVED_RECORD_CHARS = 8 * 1024 * 1024
_CAPTURE_POPEN = subprocess.Popen


def _capture_helper_source() -> str:
    """Return the isolated staging writer used for one output stream."""

    return r"""
import os
import sys

path = sys.argv[1]
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
if hasattr(os, "O_BINARY"):
    flags |= os.O_BINARY
if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
descriptor = os.open(path, flags, 0o600)

def read_exact(size):
    value = bytearray()
    while len(value) < size:
        chunk = os.read(sys.stdin.fileno(), size - len(value))
        if not chunk:
            raise EOFError
        value.extend(chunk)
    return bytes(value)

def write_all(target, value):
    view = memoryview(value)
    while view:
        written = os.write(target, view)
        if written <= 0:
            raise OSError("capture write made no progress")
        view = view[written:]

try:
    write_all(sys.stdout.fileno(), b"R")
    while True:
        header = os.read(sys.stdin.fileno(), 8)
        if not header:
            break
        while len(header) < 8:
            header += read_exact(8 - len(header))
        size = int.from_bytes(header, "big")
        payload = read_exact(size)
        write_all(descriptor, payload)
        write_all(sys.stdout.fileno(), b"A")
    os.close(descriptor)
    descriptor = -1
except BaseException:
    try:
        write_all(sys.stdout.fileno(), b"E")
    except BaseException:
        pass
    raise
finally:
    if descriptor >= 0:
        try:
            os.close(descriptor)
        except OSError:
            pass
"""


@dataclass(frozen=True)
class CodexCommand:
    argv: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str] | None = None
    stdout_capture: Path | None = None
    stderr_capture: Path | None = None


@dataclass(frozen=True)
class CodexProcessEvidence:
    work_timeout_seconds: float | None = None
    work_elapsed_seconds: float = 0.0
    total_elapsed_seconds: float = 0.0
    terminal_event_type: str | None = None
    terminal_event_elapsed_seconds: float | None = None
    completion_before_deadline: bool = False
    structured_message: str | None = None
    structured_message_elapsed_seconds: float | None = None
    structured_message_before_deadline: bool = False
    deadline_outcome: str = "process-exited"
    finalization_outcome: str = "not-started"
    cleanup_outcome: str = "not-required"
    termination_method: str | None = None
    tree_termination_confirmed: bool | None = None
    cleanup_duration_seconds: float = 0.0
    output_draining_truncated: bool = False
    event_stream_problem: str | None = None


@dataclass(frozen=True)
class CodexProcessResult:
    """Process outcome with bounded tails; capture paths retain complete streams."""

    returncode: int
    stdout: str
    stderr: str
    evidence: CodexProcessEvidence = field(default_factory=CodexProcessEvidence)
    stdout_capture: Path | None = None
    stderr_capture: Path | None = None
    stdout_capture_complete: bool = True
    stderr_capture_complete: bool = True
    stdout_capture_stable: bool = False
    stderr_capture_stable: bool = False
    transport_failure: str | None = None
    stdout_capture_prefix_bytes: int = 0
    stderr_capture_prefix_bytes: int = 0
    stdout_observed_bytes: int = 0
    stderr_observed_bytes: int = 0
    stdout_observed_tail: bytes = b""
    stderr_observed_tail: bytes = b""


@dataclass(frozen=True)
class CodexProcessTimeout:
    """Timeout outcome with bounded tails; capture paths retain complete streams."""

    stdout: str
    stderr: str
    timeout_seconds: float
    evidence: CodexProcessEvidence = field(default_factory=CodexProcessEvidence)
    stdout_capture: Path | None = None
    stderr_capture: Path | None = None
    stdout_capture_complete: bool = True
    stderr_capture_complete: bool = True
    stdout_capture_stable: bool = False
    stderr_capture_stable: bool = False
    stdout_capture_prefix_bytes: int = 0
    stderr_capture_prefix_bytes: int = 0
    stdout_observed_bytes: int = 0
    stderr_observed_bytes: int = 0
    stdout_observed_tail: bytes = b""
    stderr_observed_tail: bytes = b""


class CodexProcessTimedOut(TimeoutError):
    def __init__(self, result: CodexProcessTimeout):
        super().__init__(
            f"Codex process timed out after {result.timeout_seconds:g} seconds."
        )
        self.result = result


class _ProcessStartObserverRaised(BaseException):
    """Carry a start-observer exception across its bounded helper thread."""

    def __init__(self, error: BaseException) -> None:
        super().__init__(str(error))
        self.error = error


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


class _CodexEventObserver:
    def __init__(
        self,
        *,
        started: float,
        deadline: float | None,
        clock: Callable[[], float],
    ) -> None:
        self.started = started
        self.deadline = deadline
        self.clock = clock
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.pending = ""
        self.discarding_oversized_record = False
        self.lock = threading.Lock()
        self.terminal_event_type: str | None = None
        self.terminal_event_elapsed_seconds: float | None = None
        self.structured_message: str | None = None
        self.structured_message_elapsed_seconds: float | None = None
        self.problem: str | None = None
        self.problem_elapsed_seconds: float | None = None

    def feed(self, chunk: bytes, *, final: bool = False) -> None:
        text = self.decoder.decode(chunk, final=final)
        with self.lock:
            if self.discarding_oversized_record:
                newline = text.find("\n")
                if newline < 0:
                    return
                text = text[newline + 1 :]
                self.discarding_oversized_record = False
            self.pending += text
            while "\n" in self.pending:
                line, self.pending = self.pending.split("\n", 1)
                self._observe_line(line.rstrip("\r"))
            if len(self.pending) > _MAX_OBSERVED_RECORD_CHARS:
                # The complete raw record remains in the capture. Avoid retaining
                # an unbounded malformed/incomplete JSONL record for observation.
                self.pending = ""
                self.discarding_oversized_record = True
            if final and self.pending:
                # A partial record is diagnostic only. It is deliberately not parsed.
                self.pending = ""

    def _observe_line(self, line: str) -> None:
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(record, dict) or not isinstance(record.get("type"), str):
            return
        observed = max(0.0, self.clock() - self.started)
        event_type = record["type"]
        if event_type == "item.completed":
            item = record.get("item")
            if (
                isinstance(item, dict)
                and item.get("type") == "agent_message"
                and isinstance(item.get("text"), str)
            ):
                if self.terminal_event_type is not None:
                    self.problem = "agent message arrived after a terminal event"
                    self.problem_elapsed_seconds = observed
                self.structured_message = item["text"]
                self.structured_message_elapsed_seconds = observed
            return
        if event_type not in {"turn.completed", "turn.failed", "turn.cancelled"}:
            return
        if self.terminal_event_type is not None:
            if self.terminal_event_type != event_type:
                self.problem = (
                    "contradictory terminal events: "
                    f"{self.terminal_event_type} and {event_type}"
                )
                self.problem_elapsed_seconds = observed
            return
        self.terminal_event_type = event_type
        self.terminal_event_elapsed_seconds = observed

    def snapshot(
        self,
    ) -> tuple[str | None, float | None, str | None, float | None, str | None]:
        with self.lock:
            return (
                self.terminal_event_type,
                self.terminal_event_elapsed_seconds,
                self.structured_message,
                self.structured_message_elapsed_seconds,
                self.problem,
            )

    def detailed_snapshot(
        self,
    ) -> tuple[
        str | None,
        float | None,
        str | None,
        float | None,
        str | None,
        float | None,
    ]:
        """Return all decision fields from one observer-lock acquisition."""

        with self.lock:
            return (
                self.terminal_event_type,
                self.terminal_event_elapsed_seconds,
                self.structured_message,
                self.structured_message_elapsed_seconds,
                self.problem,
                self.problem_elapsed_seconds,
            )


def _worker_descriptor(stream: BinaryIO) -> tuple[int | None, bool]:
    """Give workers cancellable native descriptors without buffered-I/O locks."""

    try:
        descriptor = stream.fileno()
    except (AttributeError, OSError, TypeError, ValueError):
        return None, False
    if os.name == "nt":
        try:
            return os.dup(descriptor), True
        except OSError:
            return None, False
    try:
        os.set_blocking(descriptor, False)
    except OSError:
        return None, False
    return descriptor, False


def _write_descriptor(descriptor: int, value: bytes | memoryview) -> None:
    view = memoryview(value)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("Codex capture transport made no progress")
        view = view[written:]


def _read_descriptor(descriptor: int, size: int) -> bytes:
    value = bytearray()
    while len(value) < size:
        chunk = os.read(descriptor, size - len(value))
        if not chunk:
            break
        value.extend(chunk)
    return bytes(value)


class _CaptureSink:
    """Write staging bytes in a process that cleanup can always terminate."""

    _codex_close_is_nonblocking = True

    def __init__(
        self,
        destination: Path,
        *,
        deadline: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.destination = destination
        self._clock = clock
        self._lock = threading.Lock()
        self._closed = False
        self._aborted = False
        creationflags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        )
        self.process = _CAPTURE_POPEN(
            (sys.executable, "-u", "-c", _capture_helper_source(), str(destination)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            bufsize=0,
            creationflags=creationflags,
        )
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self._input_descriptor: int | None = os.dup(self.process.stdin.fileno())
        self._ack_descriptor: int | None = os.dup(self.process.stdout.fileno())
        self.process.stdin.close()
        self.process.stdout.close()
        if deadline is None:
            deadline = clock() + CLEANUP_SECONDS
        try:
            remaining = max(0.0, deadline - clock())
            reap_reserve = min(0.02, remaining / 4.0)
            ready_deadline = max(clock(), deadline - reap_reserve)
            if self._read_ack_until(ready_deadline, clock) != b"R":
                raise OSError("Codex capture helper did not initialize")
        except BaseException:
            self.abort(deadline=deadline, clock=clock)
            raise

    def write(self, value: bytes | memoryview) -> int:
        payload = bytes(value)
        with self._lock:
            if self._closed or self._aborted or self._input_descriptor is None:
                raise OSError("Codex capture helper is closed")
            input_descriptor = self._input_descriptor
        _write_descriptor(input_descriptor, len(payload).to_bytes(8, "big"))
        _write_descriptor(input_descriptor, payload)
        if self._read_ack() != b"A":
            raise OSError("Codex capture helper failed while writing output")
        return len(payload)

    def flush(self) -> None:
        # A per-chunk acknowledgement is emitted only after the helper's
        # unbuffered staging write completes, so there is no buffered flush.
        return None

    def close(
        self,
        *,
        deadline: float | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        clock = self._clock if clock is None else clock
        if deadline is None:
            deadline = clock() + CLEANUP_SECONDS
        with self._lock:
            if self._closed:
                return
            self._closed = True
            descriptor = self._input_descriptor
            self._input_descriptor = None
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        remaining = max(0.0, deadline - clock())
        reap_reserve = min(0.1, remaining / 2.0)
        graceful_timeout = max(0.0, remaining - reap_reserve)
        try:
            self.process.wait(timeout=graceful_timeout)
        except (OSError, subprocess.TimeoutExpired) as error:
            confirmed = self.abort(deadline=deadline, clock=clock)
            if not confirmed:
                raise OSError(
                    "Codex capture helper shutdown could not be confirmed"
                ) from error
            raise OSError("Codex capture helper did not close promptly") from error
        finally:
            self._close_ack_descriptor()
        if self.process.returncode != 0 and not self._aborted:
            raise OSError("Codex capture helper exited unsuccessfully")

    def abort(
        self,
        *,
        deadline: float | None = None,
        clock: Callable[[], float] | None = None,
    ) -> bool:
        """Force-stop the writer under the caller's remaining deadline."""

        clock = self._clock if clock is None else clock
        if deadline is None:
            deadline = clock() + CLEANUP_SECONDS
        self.request_abort()
        try:
            self.process.wait(timeout=max(0.0, deadline - clock()))
        except (OSError, subprocess.TimeoutExpired):
            pass
        self._close_ack_descriptor()
        return self.process.poll() is not None

    def request_abort(self) -> None:
        """Revoke descriptors and request forceful stop without waiting."""

        with self._lock:
            self._aborted = True
            input_descriptor = self._input_descriptor
            self._input_descriptor = None
        if input_descriptor is not None:
            try:
                os.close(input_descriptor)
            except OSError:
                pass
        try:
            if self.process.poll() is None:
                self.process.kill()
        except OSError:
            pass

    @property
    def stopped(self) -> bool:
        return self.process.poll() is not None

    def _read_ack(self) -> bytes:
        with self._lock:
            descriptor = self._ack_descriptor
        if descriptor is None:
            return b""
        return _read_descriptor(descriptor, 1)

    def _read_ack_until(
        self,
        deadline: float,
        clock: Callable[[], float],
    ) -> bytes:
        """Read one helper acknowledgement without owning an unbounded wait."""

        result: list[bytes] = []

        def read() -> None:
            result.append(self._read_ack())

        reader = threading.Thread(
            target=read,
            name="codex-capture-readiness",
            daemon=True,
        )
        reader.start()
        reader.join(max(0.0, deadline - clock()))
        if reader.is_alive():
            return b""
        return result[0] if result else b""

    def _close_ack_descriptor(self) -> None:
        with self._lock:
            descriptor = self._ack_descriptor
            self._ack_descriptor = None
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


class _StreamWorker:
    def __init__(
        self,
        stream: BinaryIO,
        destination: Path | None,
        *,
        stream_name: str,
        observer: _CodexEventObserver | None = None,
        output: object | None = None,
    ) -> None:
        self.stream = stream
        self.destination = destination
        self.observer = observer
        self.output = output
        self.done = threading.Event()
        self.cancel_requested = threading.Event()
        self.error: BaseException | None = None
        self.capture_failed = False
        self.buffer = bytearray()
        self.state_lock = threading.Lock()
        self.capture_revoked = threading.Event()
        self.capture_prefix_bytes = 0
        self.observed_bytes = 0
        self.capture_operation_active = False
        self.capture_sink = output
        self.file_descriptor, self.owns_descriptor = _worker_descriptor(stream)
        self.uses_descriptor = self.file_descriptor is not None
        self.descriptor_lock = threading.Lock()
        self.thread = threading.Thread(
            target=self._run,
            name=f"codex-{stream_name}-reader",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def cancel(
        self,
        *,
        deadline: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> bool:
        self.cancel_requested.set()
        self.revoke_capture(deadline=deadline, clock=clock)
        if os.name == "nt" and self.thread.is_alive():
            return _cancel_blocked_windows_thread(self.thread)
        return True

    def force_close_descriptor(self) -> bool:
        if not self.owns_descriptor:
            return True
        with self.descriptor_lock:
            if self.file_descriptor is None:
                return True
            descriptor = self.file_descriptor
            self.file_descriptor = None
        try:
            os.close(descriptor)
        except OSError:
            return False
        return True

    def _read(self) -> bytes:
        if not self.uses_descriptor:
            read = getattr(self.stream, "read1", self.stream.read)
            return read(_READ_SIZE)
        while not self.cancel_requested.is_set():
            with self.descriptor_lock:
                descriptor = self.file_descriptor
            if descriptor is None:
                return b""
            try:
                return os.read(descriptor, _READ_SIZE)
            except BlockingIOError:
                self.cancel_requested.wait(0.005)
        return b""

    def _run(self) -> None:
        try:
            if self.destination is not None and self.output is None:
                opened = _open_capture(self.destination)
                if self.capture_revoked.is_set():
                    if opened is not None:
                        opened.close()
                else:
                    self.output = opened
                    self.capture_sink = opened
            while not self.cancel_requested.is_set():
                chunk = self._read()
                if not chunk:
                    break
                # Retain and observe bytes immediately after reading them. Capture
                # storage is fallible and must not erase the chunk that triggered
                # a streaming failure.
                with self.state_lock:
                    self.observed_bytes += len(chunk)
                    self.buffer.extend(chunk)
                    if len(self.buffer) > _DIAGNOSTIC_TAIL_BYTES:
                        del self.buffer[:-_DIAGNOSTIC_TAIL_BYTES]
                if self.observer is not None:
                    self.observer.feed(chunk)
                try:
                    with self.state_lock:
                        output = self.output
                        if output is None or self.capture_revoked.is_set():
                            output = None
                        else:
                            self.capture_operation_active = True
                    if output is None:
                        if self.capture_revoked.is_set():
                            # Revocation is used only after the bounded healthy-
                            # drain interval. Preserve this one already-observed
                            # chunk in the diagnostic suffix, then stop before
                            # reading and silently discarding more pipe data.
                            break
                        # A caller may intentionally omit provider capture while
                        # still requiring the stream to be drained.
                        continue
                    try:
                        view = memoryview(chunk)
                        while view and not self.capture_revoked.is_set():
                            written = output.write(view)
                            if written is None:
                                written = len(view)
                            if written <= 0:
                                raise OSError("Codex capture write made no progress")
                            with self.state_lock:
                                self.capture_prefix_bytes += written
                            view = view[written:]
                    finally:
                        with self.state_lock:
                            self.capture_operation_active = False
                except (BufferError, OSError, RuntimeError, TypeError, ValueError):
                    self.capture_failed = True
                    raise
            if self.observer is not None:
                self.observer.feed(b"", final=True)
        except (BufferError, OSError, RuntimeError, TypeError, ValueError) as error:
            self.error = error
        finally:
            self.capture_revoked.set()
            with self.state_lock:
                self.capture_operation_active = True
                output = self.output
                self.output = None
            try:
                if output is not None:
                    output.close()
            except (OSError, RuntimeError, ValueError) as error:
                self.capture_failed = True
                if self.error is None:
                    self.error = error
            self.force_close_descriptor()
            self.done.set()

    def revoke_capture(
        self,
        *,
        deadline: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Prevent writes and terminate the isolated staging writer."""

        already_revoked = self.capture_revoked.is_set()
        self.capture_revoked.set()
        with self.state_lock:
            output = self.capture_sink
        if already_revoked and bool(output is None or getattr(output, "stopped", False)):
            return
        abort = getattr(output, "abort", None)
        if abort is not None:
            try:
                abort(deadline=deadline, clock=clock)
            except TypeError:
                # Non-production test sinks retain the historical no-argument
                # seam; they never own a helper process.
                try:
                    abort()
                except (OSError, RuntimeError, TypeError, ValueError):
                    pass
            except (OSError, RuntimeError, ValueError):
                pass

    def request_capture_abort(self) -> bool:
        """Revoke this worker and signal its helper before shared waiting."""

        self.capture_revoked.set()
        with self.state_lock:
            output = self.capture_sink
        request_abort = getattr(output, "request_abort", None)
        if request_abort is None:
            return output is None
        try:
            request_abort()
        except (OSError, RuntimeError, TypeError, ValueError):
            return False
        return True

    def close_capture_if_idle(
        self,
        *,
        deadline: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> bool:
        """Close an idle native capture without contending with sink I/O."""

        self.capture_revoked.set()
        with self.state_lock:
            output = self.output
            if output is None:
                sink = self.capture_sink
                return bool(sink is None or getattr(sink, "stopped", False))
            if self.capture_operation_active:
                abort = getattr(self.capture_sink, "abort", None)
                if abort is not None:
                    try:
                        try:
                            return bool(abort(deadline=deadline, clock=clock))
                        except TypeError:
                            return bool(abort())
                    except (OSError, RuntimeError, TypeError, ValueError):
                        return False
                return False
            if type(output).__module__ != "_io" and not getattr(
                output, "_codex_close_is_nonblocking", False
            ):
                return False
            self.output = None
        try:
            try:
                output.close(deadline=deadline, clock=clock)
            except TypeError:
                output.close()
        except (OSError, RuntimeError, ValueError) as error:
            self.capture_failed = True
            if self.error is None:
                self.error = error
            return False
        return True

    def capture_snapshot(self) -> tuple[int, int, bytes]:
        """Return the durable-prefix claim and bounded observed tail."""

        with self.state_lock:
            return (
                self.capture_prefix_bytes,
                self.observed_bytes,
                bytes(self.buffer),
            )

    def recovery_snapshot(self) -> tuple[Path | None, int, bool]:
        """Compatibility hook: isolated primary capture needs no recovery spool."""

        return None, 0, False

    def capture_stable(self) -> bool:
        with self.state_lock:
            sink = self.capture_sink
            operation_active = self.capture_operation_active
        if self.done.is_set() and not self.thread.is_alive():
            return True
        return bool(
            self.capture_revoked.is_set()
            and not operation_active
            and (sink is None or getattr(sink, "stopped", False))
        )

    def durable_prefix_bytes(self) -> int:
        prefix_bytes, _, _ = self.capture_snapshot()
        if not self.capture_stable() or self.destination is None:
            return prefix_bytes
        try:
            return self.destination.stat().st_size
        except OSError:
            return prefix_bytes

    def text(self) -> str:
        return decode_human_output(self.capture_snapshot()[2])


class _InputWorker:
    def __init__(
        self,
        stream: BinaryIO,
        value: str,
        *,
        clock: Callable[[], float],
    ) -> None:
        self.stream = stream
        self.value = value.encode("utf-8")
        self.clock = clock
        self.completed_at: float | None = None
        self.done = threading.Event()
        self.cancel_requested = threading.Event()
        self.error: BaseException | None = None
        self.file_descriptor, self.owns_descriptor = _worker_descriptor(stream)
        self.uses_descriptor = self.file_descriptor is not None
        self.descriptor_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="codex-stdin-writer")
        self.thread.daemon = True

    def start(self) -> None:
        self.thread.start()

    def cancel(self) -> bool:
        self.cancel_requested.set()
        if os.name == "nt" and self.thread.is_alive():
            return _cancel_blocked_windows_thread(self.thread)
        return True

    def force_close_descriptor(self) -> bool:
        if not self.owns_descriptor:
            return True
        with self.descriptor_lock:
            if self.file_descriptor is None:
                return True
            descriptor = self.file_descriptor
            self.file_descriptor = None
        try:
            os.close(descriptor)
        except OSError:
            return False
        return True

    def _write(self, view: memoryview) -> int | None:
        if not self.uses_descriptor:
            return self.stream.write(view)
        while not self.cancel_requested.is_set():
            with self.descriptor_lock:
                descriptor = self.file_descriptor
            if descriptor is None:
                raise BrokenPipeError("Codex stdin delivery was cancelled")
            try:
                return os.write(descriptor, view)
            except BlockingIOError:
                self.cancel_requested.wait(0.005)
        raise BrokenPipeError("Codex stdin delivery was cancelled")

    def _run(self) -> None:
        try:
            view = memoryview(self.value)
            while view and not self.cancel_requested.is_set():
                written = self._write(view[:_READ_SIZE])
                if written is None:
                    written = min(len(view), _READ_SIZE)
                if written <= 0:
                    raise OSError("Codex stdin write made no progress")
                view = view[written:]
            self.stream.flush()
        except (BufferError, OSError, RuntimeError, TypeError, ValueError) as error:
            self.error = error
        finally:
            try:
                self.stream.close()
            except (OSError, RuntimeError, ValueError) as error:
                if self.error is None:
                    self.error = error
            finally:
                self.force_close_descriptor()
                self.completed_at = self.clock()
                self.done.set()


class _StartObserverWorker:
    """Run post-launch recording once without letting it own the deadline."""

    def __init__(self, observer: Callable[[], None]) -> None:
        self.observer = observer
        self.done = threading.Event()
        self.cancel_requested = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run,
            name="codex-invocation-start-observer",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def cancel(self) -> bool:
        self.cancel_requested.set()
        cancel = getattr(self.observer, "cancel", None)
        if cancel is None:
            return True
        try:
            cancel()
        except (OSError, RuntimeError, TypeError, ValueError):
            return False
        return True

    def _run(self) -> None:
        try:
            if not self.cancel_requested.is_set():
                self.observer()
        except BaseException as error:  # noqa: BLE001 - preserve interruptions
            self.error = error
        finally:
            self.done.set()


class _InvocationContainment:
    def terminate(self) -> str:
        raise NotImplementedError

    def active(self) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _PosixProcessGroup(_InvocationContainment):
    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self.process = process
        self.pgid = process.pid
        self.killed = False

    def terminate(self) -> str:
        try:
            os.killpg(self.pgid, signal.SIGTERM if not self.killed else signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.killed = True
        return "posix-process-group"

    def active(self) -> bool:
        try:
            os.killpg(self.pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


if os.name == "nt":

    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_ulonglong)
            for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )
        ]

    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class _BasicAccountingInformation(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_longlong),
            ("TotalKernelTime", ctypes.c_longlong),
            ("ThisPeriodTotalUserTime", ctypes.c_longlong),
            ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
            ("TotalPageFaultCount", ctypes.c_uint32),
            ("TotalProcesses", ctypes.c_uint32),
            ("ActiveProcesses", ctypes.c_uint32),
            ("TotalTerminatedProcesses", ctypes.c_uint32),
        ]


class _WindowsJob(_InvocationContainment):
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        kernel32 = _windows_kernel32()
        self.kernel32 = kernel32
        self.handle = kernel32.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            info = _ExtendedLimitInformation()
            info.BasicLimitInformation.LimitFlags = (
                self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            )
            if not kernel32.SetInformationJobObject(
                self.handle,
                self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(info),
                ctypes.sizeof(info),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if not kernel32.AssignProcessToJobObject(
                self.handle,
                ctypes.c_void_p(int(process._handle)),  # type: ignore[attr-defined]
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            _resume_windows_process(process.pid)
        except BaseException:
            # If assignment succeeded, kill-on-close contains a later setup
            # failure. Otherwise the caller kills the still-suspended process.
            self.close()
            raise

    def terminate(self) -> str:
        if self.handle and not self.kernel32.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())
        return "windows-job-object"

    def active(self) -> bool:
        if not self.handle:
            return False
        info = _BasicAccountingInformation()
        if not self.kernel32.QueryInformationJobObject(
            self.handle,
            1,
            ctypes.byref(info),
            ctypes.sizeof(info),
            None,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return bool(info.ActiveProcesses)

    def close(self) -> None:
        if self.handle:
            self.kernel32.CloseHandle(self.handle)
            self.handle = None


class _SingleProcessContainment(_InvocationContainment):
    """Compatibility containment for test doubles that are not real Popen objects."""

    def __init__(self, process: object) -> None:
        self.process = process

    def terminate(self) -> str:
        self.process.kill()  # type: ignore[attr-defined]
        return "single-process-fallback"

    def active(self) -> bool:
        return getattr(self.process, "returncode", None) is None


class SubprocessCodexRunner:
    def __init__(self, *, monotonic: Callable[[], float] | None = None) -> None:
        self._monotonic = monotonic or time.monotonic

    def run(
        self,
        command: CodexCommand,
        *,
        stdin: str,
        timeout_seconds: float | None,
        on_process_start: Callable[[], None] | None = None,
    ) -> CodexProcessResult:
        # Capture-helper startup precedes the Codex launch and therefore uses
        # its own bounded native-monotonic setup window. Do not sample the
        # injectable work clock until immediately before the Codex Popen call.
        capture_setup_deadline = time.monotonic() + CLEANUP_SECONDS
        stdout_capture = _open_capture(
            command.stdout_capture,
            deadline=capture_setup_deadline,
            clock=time.monotonic,
        )
        try:
            capture_setup_deadline = time.monotonic() + CLEANUP_SECONDS
            stderr_capture = _open_capture(
                command.stderr_capture,
                deadline=capture_setup_deadline,
                clock=time.monotonic,
            )
        except BaseException:
            if stdout_capture is not None:
                _close_captures(
                    (stdout_capture, None),
                    deadline=capture_setup_deadline,
                    clock=time.monotonic,
                    force=True,
                )
            raise
        capture_streams = (stdout_capture, stderr_capture)
        popen_options: dict[str, object] = {
            "cwd": command.cwd,
            "env": _process_environment(command.environment),
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "shell": False,
        }
        if os.name == "nt":
            popen_options["creationflags"] = getattr(
                subprocess, "CREATE_SUSPENDED", 0x4
            )
        else:
            popen_options["start_new_session"] = True
        launched_at = self._monotonic()
        deadline = None if timeout_seconds is None else launched_at + timeout_seconds
        try:
            process = subprocess.Popen(command.argv, **popen_options)
        except BaseException:
            launch_cleanup_deadline = min(
                self._monotonic() + CLEANUP_SECONDS,
                (deadline or self._monotonic()) + CLEANUP_SECONDS,
            )
            _close_captures(
                capture_streams,
                deadline=launch_cleanup_deadline,
                clock=self._monotonic,
                force=True,
            )
            raise
        containment: _InvocationContainment
        try:
            containment = _contain_process(process)
        except BaseException as error:
            cleanup_started = self._monotonic()
            cleanup_deadline = self._monotonic() + CLEANUP_SECONDS
            if deadline is not None:
                cleanup_deadline = min(cleanup_deadline, deadline + CLEANUP_SECONDS)
            _request_capture_aborts(capture_streams)
            cleanup = _emergency_cleanup(
                process,
                _SingleProcessContainment(process),
                cleanup_deadline,
                self._monotonic,
            )
            capture_confirmed = _close_captures(
                capture_streams,
                deadline=cleanup_deadline,
                clock=self._monotonic,
                force=True,
            )
            if not isinstance(error, Exception):
                raise
            uncertain_cleanup = _CleanupResult(
                "incomplete",
                "containment-setup-failed",
                False,
                max(cleanup.duration, self._monotonic() - cleanup_started),
                True,
            )
            observer = _CodexEventObserver(
                started=launched_at,
                deadline=deadline,
                clock=self._monotonic,
            )
            return CodexProcessResult(
                returncode=(
                    process.returncode if process.returncode is not None else -1
                ),
                stdout="",
                stderr="",
                evidence=_evidence(
                    observer,
                    launched_at=launched_at,
                    clock=self._monotonic,
                    timeout_seconds=timeout_seconds,
                    outcome_elapsed_seconds=max(0.0, self._monotonic() - launched_at),
                    deadline_outcome="containment-failure",
                    finalization_outcome="failed",
                    cleanup=uncertain_cleanup,
                ),
                stdout_capture=command.stdout_capture,
                stderr_capture=command.stderr_capture,
                stdout_capture_complete=False,
                stderr_capture_complete=False,
                transport_failure=(
                    f"Codex invocation containment failed after launch: {error}"
                    + (
                        ""
                        if capture_confirmed
                        else "; capture-helper shutdown could not be confirmed"
                    )
                ),
            )
        workers_for_cleanup: (
            tuple[_StreamWorker, _StreamWorker, _InputWorker] | None
        ) = None
        start_observer_worker: _StartObserverWorker | None = None
        observer: _CodexEventObserver | None = None
        capture_close_deadline: float | None = None
        capture_helpers_confirmed = True
        try:
            if not all(
                hasattr(process, name) for name in ("stdin", "stdout", "stderr", "poll")
            ):
                notified, start_observer_worker = self._notify_process_start(
                    on_process_start, deadline
                )
                if not notified:
                    cleanup = _emergency_cleanup(
                        process,
                        containment,
                        (deadline or self._monotonic()) + CLEANUP_SECONDS,
                        self._monotonic,
                    )
                    cleanup = _include_start_observer_cleanup(
                        cleanup,
                        start_observer_worker,
                        (deadline or self._monotonic()) + CLEANUP_SECONDS,
                        self._monotonic,
                    )
                    evidence = CodexProcessEvidence(
                        work_timeout_seconds=timeout_seconds,
                        work_elapsed_seconds=float(timeout_seconds or 0.0),
                        total_elapsed_seconds=max(0.0, self._monotonic() - launched_at),
                        deadline_outcome="timed-out",
                        finalization_outcome="not-eligible",
                        cleanup_outcome=cleanup.outcome,
                        termination_method=cleanup.method,
                        tree_termination_confirmed=cleanup.confirmed,
                        cleanup_duration_seconds=cleanup.duration,
                        output_draining_truncated=cleanup.drain_truncated,
                    )
                    raise CodexProcessTimedOut(
                        CodexProcessTimeout(
                            "",
                            "",
                            float(timeout_seconds or 0.0),
                            evidence,
                            command.stdout_capture,
                            command.stderr_capture,
                            False,
                            False,
                        )
                    )
                return self._run_compatibility_process(
                    process,
                    containment,
                    stdin=stdin,
                    timeout_seconds=timeout_seconds,
                    launched_at=launched_at,
                    deadline=deadline,
                )
            assert process.stdin is not None
            assert process.stdout is not None
            assert process.stderr is not None
            observer = _CodexEventObserver(
                started=launched_at,
                deadline=deadline,
                clock=self._monotonic,
            )
            stdout_worker = _StreamWorker(
                process.stdout,
                command.stdout_capture,
                stream_name="stdout",
                observer=observer,
                output=stdout_capture,
            )
            stderr_worker = _StreamWorker(
                process.stderr,
                command.stderr_capture,
                stream_name="stderr",
                output=stderr_capture,
            )
            input_worker = _InputWorker(
                process.stdin,
                stdin,
                clock=self._monotonic,
            )
            workers = (stdout_worker, stderr_worker, input_worker)
            workers_for_cleanup = workers
            for worker in workers[:2]:
                worker.start()
            notified, start_observer_worker = self._notify_process_start(
                on_process_start, deadline
            )
            if not notified:
                assert timeout_seconds is not None
                return self._timeout(
                    process,
                    containment,
                    command,
                    observer,
                    stdout_worker,
                    stderr_worker,
                    input_worker,
                    launched_at=launched_at,
                    deadline=deadline,
                    timeout_seconds=float(timeout_seconds),
                    start_observer_worker=start_observer_worker,
                )
            input_worker.start()
            return self._monitor(
                process,
                containment,
                command,
                observer,
                stdout_worker,
                stderr_worker,
                input_worker,
                launched_at=launched_at,
                deadline=deadline,
                timeout_seconds=timeout_seconds,
            )
        except BaseException as original:
            if isinstance(original, CodexProcessTimedOut):
                raise
            cleanup_deadline = self._monotonic() + CLEANUP_SECONDS
            if deadline is not None:
                cleanup_deadline = min(cleanup_deadline, deadline + CLEANUP_SECONDS)
            capture_close_deadline = cleanup_deadline
            if workers_for_cleanup is None:
                _request_capture_aborts(capture_streams)
                cleanup = _emergency_cleanup(
                    process, containment, cleanup_deadline, self._monotonic
                )
                cleanup = _include_start_observer_cleanup(
                    cleanup,
                    start_observer_worker,
                    cleanup_deadline,
                    self._monotonic,
                )
            else:
                cleanup = _cleanup(
                    process,
                    containment,
                    workers_for_cleanup,
                    cleanup_deadline,
                    self._monotonic,
                    start_observer_worker=start_observer_worker,
                )
            if workers_for_cleanup is None:
                capture_helpers_confirmed = _close_captures(
                    capture_streams,
                    deadline=cleanup_deadline,
                    clock=self._monotonic,
                    force=True,
                )
                if not capture_helpers_confirmed:
                    cleanup = _CleanupResult(
                        "incomplete",
                        cleanup.method,
                        cleanup.confirmed,
                        max(
                            cleanup.duration,
                            max(0.0, self._monotonic() - launched_at),
                        ),
                        True,
                    )
            if isinstance(original, _ProcessStartObserverRaised):
                raise original.error
            if not isinstance(original, Exception):
                raise
            if observer is None:
                observer = _CodexEventObserver(
                    started=launched_at,
                    deadline=deadline,
                    clock=self._monotonic,
                )
            stdout_worker = (
                None if workers_for_cleanup is None else workers_for_cleanup[0]
            )
            stderr_worker = (
                None if workers_for_cleanup is None else workers_for_cleanup[1]
            )
            stdout_capture = _capture_result(
                stdout_worker, command.stdout_capture, cleanup
            )
            stderr_capture = _capture_result(
                stderr_worker, command.stderr_capture, cleanup
            )
            return CodexProcessResult(
                returncode=(
                    process.returncode if process.returncode is not None else -1
                ),
                stdout="" if stdout_worker is None else stdout_worker.text(),
                stderr="" if stderr_worker is None else stderr_worker.text(),
                evidence=_evidence(
                    observer,
                    launched_at=launched_at,
                    clock=self._monotonic,
                    timeout_seconds=timeout_seconds,
                    outcome_elapsed_seconds=max(0.0, self._monotonic() - launched_at),
                    deadline_outcome="setup-failure",
                    finalization_outcome="failed",
                    cleanup=cleanup,
                ),
                stdout_capture=stdout_capture.source,
                stderr_capture=stderr_capture.source,
                stdout_capture_complete=stdout_capture.complete,
                stderr_capture_complete=stderr_capture.complete,
                stdout_capture_stable=stdout_capture.stable,
                stderr_capture_stable=stderr_capture.stable,
                transport_failure=(
                    f"Codex transport setup failed after launch: {original}"
                    + (
                        ""
                        if capture_helpers_confirmed
                        else "; capture-helper shutdown could not be confirmed"
                    )
                ),
                stdout_capture_prefix_bytes=stdout_capture.prefix_bytes,
                stderr_capture_prefix_bytes=stderr_capture.prefix_bytes,
                stdout_observed_bytes=stdout_capture.observed_bytes,
                stderr_observed_bytes=stderr_capture.observed_bytes,
                stdout_observed_tail=stdout_capture.observed_tail,
                stderr_observed_tail=stderr_capture.observed_tail,
            )
        finally:
            containment.close()
            if workers_for_cleanup is None:
                if capture_close_deadline is None:
                    capture_close_deadline = min(
                        self._monotonic() + CLEANUP_SECONDS,
                        (deadline or self._monotonic()) + CLEANUP_SECONDS,
                    )
                _close_captures(
                    capture_streams,
                    deadline=capture_close_deadline,
                    clock=self._monotonic,
                    force=True,
                )

    def _notify_process_start(
        self,
        observer: Callable[[], None] | None,
        deadline: float | None,
    ) -> tuple[bool, _StartObserverWorker | None]:
        if observer is None:
            return True, None
        worker = _StartObserverWorker(observer)
        worker.start()
        while not worker.done.is_set():
            now = self._monotonic()
            if deadline is not None and now >= deadline:
                worker.cancel()
                return False, worker
            wait = 0.005 if deadline is None else min(0.005, deadline - now)
            worker.done.wait(max(0.0, wait))
        if worker.error is not None:
            raise _ProcessStartObserverRaised(worker.error)
        return True, worker

    def _monitor(
        self,
        process: subprocess.Popen[bytes],
        containment: _InvocationContainment,
        command: CodexCommand,
        observer: _CodexEventObserver,
        stdout_worker: _StreamWorker,
        stderr_worker: _StreamWorker,
        input_worker: _InputWorker,
        *,
        launched_at: float,
        deadline: float | None,
        timeout_seconds: float | None,
    ) -> CodexProcessResult:
        finalization_deadline: float | None = None
        deadline_outcome = "process-exited"
        finalization_outcome = "not-started"
        transport_failure: str | None = None
        workers = (stdout_worker, stderr_worker, input_worker)
        while True:
            observed = observer.detailed_snapshot()
            process_exited = process.poll() is not None
            workers_done = all(worker.done.is_set() for worker in workers)
            if workers_done:
                # Once every worker has published done, no reader can mutate the
                # observer or expose a new worker error. Re-read both before any
                # terminal decision so EOF cannot race the earlier snapshot.
                observed = observer.detailed_snapshot()
            invocation_finished = (
                process_exited and not containment.active() and workers_done
            )
            # Sample after observer/worker/exit state. This timestamp, rather
            # than one captured at the top of the polling iteration, governs
            # both deadline application and acceptance of a completed exit.
            now = self._monotonic()
            if (
                invocation_finished
                or (deadline is not None and now >= deadline)
                or (finalization_deadline is not None and now >= finalization_deadline)
            ):
                # A worker may publish an event/error while poll() observes EOF
                # or while the boundary sample is taken. Refresh all terminal
                # state and then sample time again before committing the choice.
                observed = observer.detailed_snapshot()
                process_exited = process.poll() is not None
                workers_done = all(worker.done.is_set() for worker in workers)
                if workers_done:
                    observed = observer.detailed_snapshot()
                invocation_finished = (
                    process_exited and not containment.active() and workers_done
                )
                now = self._monotonic()
            (
                terminal_type,
                terminal_elapsed,
                message,
                message_elapsed,
                problem,
                problem_elapsed,
            ) = observed
            timely_terminal = (
                terminal_type == "turn.completed"
                and terminal_elapsed is not None
                and (timeout_seconds is None or terminal_elapsed < timeout_seconds)
            )
            timely_message = (
                message is not None
                and message_elapsed is not None
                and (timeout_seconds is None or message_elapsed < timeout_seconds)
            )
            timely_input = (
                input_worker.done.is_set()
                and input_worker.completed_at is not None
                and (deadline is None or input_worker.completed_at < deadline)
            )
            for label, worker in (
                ("stdout", stdout_worker),
                ("stderr", stderr_worker),
                ("stdin", input_worker),
            ):
                if worker.error is not None:
                    transport_failure = (
                        f"Codex {label} streaming failed: {worker.error}"
                    )
                    deadline_outcome = "stream-failure"
                    finalization_outcome = "failed"
                    break
            if transport_failure is not None:
                break
            timely_terminal_failure = (
                terminal_type in {"turn.failed", "turn.cancelled"}
                and terminal_elapsed is not None
                and (timeout_seconds is None or terminal_elapsed < timeout_seconds)
            )
            timely_problem = (
                problem is not None
                and problem_elapsed is not None
                and (timeout_seconds is None or problem_elapsed < timeout_seconds)
            )
            completion_eligible = timely_terminal and timely_message and timely_input
            if problem is not None and (
                finalization_deadline is not None
                or completion_eligible
                or timely_terminal_failure
                or timely_problem
            ):
                deadline_outcome = "event-conflict"
                finalization_outcome = "failed"
                transport_failure = problem
                break
            if timely_terminal_failure:
                deadline_outcome = "terminal-failure"
                finalization_outcome = "failed"
                transport_failure = f"Codex emitted {terminal_type}."
                break
            if completion_eligible:
                deadline_outcome = "completed-before-deadline"
                if finalization_deadline is None:
                    assert terminal_elapsed is not None
                    finalization_deadline = (
                        launched_at + terminal_elapsed + FINALIZATION_SECONDS
                    )
                    finalization_outcome = "pending"
                if invocation_finished and now < finalization_deadline:
                    finalization_outcome = "completed"
                    break
                if now >= finalization_deadline:
                    finalization_outcome = "expired"
                    transport_failure = (
                        "Codex did not exit and publish its final output within "
                        f"{FINALIZATION_SECONDS:g} seconds of timely completion."
                    )
                    break
            elif deadline is not None and now >= deadline:
                return self._timeout(
                    process,
                    containment,
                    command,
                    observer,
                    stdout_worker,
                    stderr_worker,
                    input_worker,
                    launched_at=launched_at,
                    deadline=deadline,
                    timeout_seconds=float(timeout_seconds),
                )
            elif invocation_finished:
                deadline_outcome = "process-exited-without-timely-completion"
                finalization_outcome = "not-eligible"
                break
            threading.Event().wait(0.005)

        # Freeze the phase duration before termination, draining, reaping, or
        # stream closure. Cleanup time is reported separately and must never be
        # folded into the work/finalisation decision.
        if (
            deadline_outcome == "completed-before-deadline"
            and finalization_outcome == "completed"
            and terminal_elapsed is not None
        ):
            outcome_elapsed_seconds = terminal_elapsed
        else:
            outcome_elapsed_seconds = max(0.0, now - launched_at)
        cleanup_required = (
            transport_failure is not None
            or containment.active()
            or not all(
                worker.done.is_set()
                for worker in (stdout_worker, stderr_worker, input_worker)
            )
        )
        cleanup = _CleanupResult.not_required()
        if cleanup_required:
            overall_deadline = (
                launched_at + timeout_seconds + CLEANUP_SECONDS
                if timeout_seconds is not None
                else self._monotonic() + CLEANUP_SECONDS
            )
            cleanup = _cleanup(
                process,
                containment,
                (stdout_worker, stderr_worker, input_worker),
                min(overall_deadline, self._monotonic() + CLEANUP_SECONDS),
                self._monotonic,
            )
            if not cleanup.confirmed:
                detail = "invocation-tree cleanup could not be confirmed"
                transport_failure = (
                    detail
                    if transport_failure is None
                    else f"{transport_failure}; {detail}"
                )
        else:
            close_started = self._monotonic()
            if not _close_process_streams(process, ("stdin", "stdout", "stderr")):
                cleanup = _CleanupResult(
                    "incomplete",
                    "stream-close",
                    True,
                    max(0.0, self._monotonic() - close_started),
                    False,
                )
                transport_failure = "Codex process streams could not be closed."
        process.poll()
        returncode = process.returncode if process.returncode is not None else -1
        evidence = _evidence(
            observer,
            launched_at=launched_at,
            clock=self._monotonic,
            timeout_seconds=timeout_seconds,
            outcome_elapsed_seconds=outcome_elapsed_seconds,
            deadline_outcome=deadline_outcome,
            finalization_outcome=finalization_outcome,
            cleanup=cleanup,
        )
        stdout_capture = _capture_result(stdout_worker, command.stdout_capture, cleanup)
        stderr_capture = _capture_result(stderr_worker, command.stderr_capture, cleanup)
        return CodexProcessResult(
            returncode=returncode,
            stdout=stdout_worker.text(),
            stderr=stderr_worker.text(),
            evidence=evidence,
            stdout_capture=stdout_capture.source,
            stderr_capture=stderr_capture.source,
            stdout_capture_complete=stdout_capture.complete,
            stderr_capture_complete=stderr_capture.complete,
            stdout_capture_stable=stdout_capture.stable,
            stderr_capture_stable=stderr_capture.stable,
            transport_failure=transport_failure,
            stdout_capture_prefix_bytes=stdout_capture.prefix_bytes,
            stderr_capture_prefix_bytes=stderr_capture.prefix_bytes,
            stdout_observed_bytes=stdout_capture.observed_bytes,
            stderr_observed_bytes=stderr_capture.observed_bytes,
            stdout_observed_tail=stdout_capture.observed_tail,
            stderr_observed_tail=stderr_capture.observed_tail,
        )

    def _timeout(
        self,
        process: subprocess.Popen[bytes],
        containment: _InvocationContainment,
        command: CodexCommand,
        observer: _CodexEventObserver,
        stdout_worker: _StreamWorker,
        stderr_worker: _StreamWorker,
        input_worker: _InputWorker,
        *,
        launched_at: float,
        deadline: float,
        timeout_seconds: float,
        start_observer_worker: _StartObserverWorker | None = None,
    ) -> CodexProcessResult:
        cleanup = _cleanup(
            process,
            containment,
            (stdout_worker, stderr_worker, input_worker),
            deadline + CLEANUP_SECONDS,
            self._monotonic,
            start_observer_worker=start_observer_worker,
        )
        evidence = _evidence(
            observer,
            launched_at=launched_at,
            clock=self._monotonic,
            timeout_seconds=timeout_seconds,
            outcome_elapsed_seconds=timeout_seconds,
            deadline_outcome="timed-out",
            finalization_outcome="not-eligible",
            cleanup=cleanup,
        )
        stdout_capture = _capture_result(stdout_worker, command.stdout_capture, cleanup)
        stderr_capture = _capture_result(stderr_worker, command.stderr_capture, cleanup)
        raise CodexProcessTimedOut(
            CodexProcessTimeout(
                stdout=stdout_worker.text(),
                stderr=stderr_worker.text(),
                timeout_seconds=timeout_seconds,
                evidence=evidence,
                stdout_capture=stdout_capture.source,
                stderr_capture=stderr_capture.source,
                stdout_capture_complete=stdout_capture.complete,
                stderr_capture_complete=stderr_capture.complete,
                stdout_capture_stable=stdout_capture.stable,
                stderr_capture_stable=stderr_capture.stable,
                stdout_capture_prefix_bytes=stdout_capture.prefix_bytes,
                stderr_capture_prefix_bytes=stderr_capture.prefix_bytes,
                stdout_observed_bytes=stdout_capture.observed_bytes,
                stderr_observed_bytes=stderr_capture.observed_bytes,
                stdout_observed_tail=stdout_capture.observed_tail,
                stderr_observed_tail=stderr_capture.observed_tail,
            )
        )

    def _run_compatibility_process(
        self,
        process: object,
        containment: _InvocationContainment,
        *,
        stdin: str,
        timeout_seconds: float | None,
        launched_at: float,
        deadline: float | None,
    ) -> CodexProcessResult:
        """Keep lightweight historical Popen test doubles usable."""

        try:
            stdout, stderr = process.communicate(  # type: ignore[attr-defined]
                input=stdin.encode("utf-8"), timeout=timeout_seconds
            )
        except subprocess.TimeoutExpired as error:
            containment.terminate()
            cleanup_deadline = (deadline or self._monotonic()) + CLEANUP_SECONDS
            try:
                stdout, stderr = process.communicate(  # type: ignore[attr-defined]
                    timeout=max(0.0, cleanup_deadline - self._monotonic())
                )
            except subprocess.TimeoutExpired:
                stdout = error.output or b""
                stderr = error.stderr or b""
            evidence = CodexProcessEvidence(
                work_timeout_seconds=timeout_seconds,
                work_elapsed_seconds=float(timeout_seconds or 0),
                total_elapsed_seconds=max(0.0, self._monotonic() - launched_at),
                deadline_outcome="timed-out",
                finalization_outcome="not-eligible",
                cleanup_outcome="completed",
                termination_method="single-process-fallback",
                tree_termination_confirmed=True,
            )
            raise CodexProcessTimedOut(
                CodexProcessTimeout(
                    decode_human_output(stdout),
                    decode_human_output(stderr),
                    float(timeout_seconds or 0),
                    evidence,
                )
            ) from error
        return CodexProcessResult(
            returncode=process.returncode,  # type: ignore[attr-defined]
            stdout=decode_human_output(stdout),
            stderr=decode_human_output(stderr),
            evidence=CodexProcessEvidence(
                work_timeout_seconds=timeout_seconds,
                work_elapsed_seconds=max(0.0, self._monotonic() - launched_at),
                total_elapsed_seconds=max(0.0, self._monotonic() - launched_at),
                terminal_event_type="turn.completed",
                terminal_event_elapsed_seconds=0.0,
                completion_before_deadline=True,
                structured_message_before_deadline=True,
                deadline_outcome="completed-before-deadline",
                finalization_outcome="completed",
                tree_termination_confirmed=True,
            ),
        )


@dataclass(frozen=True)
class _CleanupResult:
    outcome: str
    method: str | None
    confirmed: bool | None
    duration: float
    drain_truncated: bool

    @classmethod
    def not_required(cls) -> _CleanupResult:
        return cls("not-required", None, True, 0.0, False)


@dataclass(frozen=True)
class _CaptureResult:
    source: Path | None
    complete: bool
    stable: bool
    prefix_bytes: int
    observed_bytes: int
    observed_tail: bytes


def _capture_result(
    worker: _StreamWorker | None,
    source: Path | None,
    cleanup: _CleanupResult,
) -> _CaptureResult:
    if worker is None:
        return _CaptureResult(source, False, False, 0, 0, b"")
    _, observed_bytes, observed_tail = worker.capture_snapshot()
    prefix_bytes = worker.durable_prefix_bytes()
    complete = _capture_complete(worker, cleanup)
    if complete:
        return _CaptureResult(
            source,
            True,
            True,
            prefix_bytes,
            observed_bytes,
            observed_tail,
        )
    recovery_source, recovery_prefix, recovery_stable = worker.recovery_snapshot()
    if recovery_stable and recovery_prefix == observed_bytes:
        # The recovery spool was populated before each primary-sink operation,
        # then fenced and closed during cleanup. It can be atomically promoted
        # even when the primary writer remains blocked on another inode.
        return _CaptureResult(
            recovery_source,
            False,
            True,
            recovery_prefix,
            observed_bytes,
            observed_tail,
        )
    return _CaptureResult(
        source,
        False,
        _capture_stable(worker),
        prefix_bytes,
        observed_bytes,
        observed_tail,
    )


def _capture_complete(
    worker: _StreamWorker | None,
    cleanup: _CleanupResult,
) -> bool:
    """A capture is promotable only after its sole writer stopped cleanly."""

    if worker is None:
        return False
    _, observed_bytes, _ = worker.capture_snapshot()
    prefix_bytes = worker.durable_prefix_bytes()
    return bool(
        _capture_stable(worker)
        and worker.error is None
        and not worker.capture_failed
        and not cleanup.drain_truncated
        and prefix_bytes == observed_bytes
    )


def _capture_stable(worker: _StreamWorker | None) -> bool:
    """Return whether no worker can append to its staging capture."""

    return bool(worker is not None and worker.capture_stable())


def _cleanup(
    process: subprocess.Popen[bytes],
    containment: _InvocationContainment,
    workers: tuple[_StreamWorker, _StreamWorker, _InputWorker],
    deadline: float,
    clock: Callable[[], float],
    *,
    start_observer_worker: _StartObserverWorker | None = None,
) -> _CleanupResult:
    started = clock()
    method: str | None = None
    confirmed = False
    termination_failed = False
    try:
        method = containment.terminate()
    except (OSError, RuntimeError):
        method = "termination-failed"
        termination_failed = True
        try:
            process.kill()
        except (OSError, RuntimeError):
            pass
    graceful_until = min(deadline, clock() + 1.0)
    while clock() < graceful_until:
        try:
            process.poll()
        except (OSError, RuntimeError):
            break
        try:
            if not containment.active():
                confirmed = True
                break
        except (OSError, RuntimeError):
            break
        threading.Event().wait(min(0.01, max(0.0, graceful_until - clock())))
    if not confirmed and clock() < deadline:
        try:
            method = containment.terminate()
        except (OSError, RuntimeError):
            method = "termination-failed"
            termination_failed = True
            try:
                process.kill()
            except (OSError, RuntimeError):
                pass

    # The readers are already draining concurrently. Give buffered bytes made
    # available by process termination a bounded opportunity to reach capture
    # storage before forcing the pipe handles closed.
    remaining = max(0.0, deadline - clock())
    drain_until = min(deadline, clock() + remaining / 2.0)
    for worker in workers[:2]:
        if worker.thread.ident is not None:
            worker.thread.join(max(0.0, drain_until - clock()))

    forced_output_close = any(worker.thread.is_alive() for worker in workers[:2])
    # Healthy readers retain every byte made available by containment
    # termination during the bounded drain interval. Revoke only workers that
    # failed to finish; cancellation below prevents another read after the
    # single already-observed chunk has been accounted for.
    for worker in workers[:2]:
        if (
            worker.thread.is_alive()
            and hasattr(worker, "revoke_capture")
            and hasattr(worker, "request_capture_abort")
        ):
            worker.request_capture_abort()
    for worker in workers[:2]:
        if worker.thread.is_alive() and hasattr(worker, "revoke_capture"):
            worker.revoke_capture(deadline=deadline, clock=clock)
    all_workers = (
        workers if start_observer_worker is None else (*workers, start_observer_worker)
    )
    cancellation_results: list[bool] = []
    for worker in all_workers:
        if not hasattr(worker, "cancel"):
            cancellation_results.append(True)
        elif isinstance(worker, _StreamWorker):
            cancellation_results.append(
                worker.cancel(deadline=deadline, clock=clock)
            )
        else:
            cancellation_results.append(worker.cancel())
    initial_join_deadline = min(deadline, clock() + 0.05)
    for worker in all_workers:
        if worker.thread.ident is not None:
            worker.thread.join(max(0.0, initial_join_deadline - clock()))
    if os.name == "nt":
        for worker in all_workers:
            if not worker.thread.is_alive():
                continue
            cancellation_results.append(
                _cancel_windows_descriptor_io(getattr(worker, "file_descriptor", None))
            )
            cancellation_results.append(_cancel_blocked_windows_thread(worker.thread))
    secondary_join_deadline = min(deadline, clock() + 0.05)
    for worker in all_workers:
        if worker.thread.ident is not None and worker.thread.is_alive():
            worker.thread.join(max(0.0, secondary_join_deadline - clock()))
    for worker in all_workers:
        if worker.thread.is_alive() and hasattr(worker, "force_close_descriptor"):
            cancellation_results.append(worker.force_close_descriptor())
    for worker in workers[:2]:
        if hasattr(worker, "close_recovery_if_idle"):
            cancellation_results.append(worker.close_recovery_if_idle())
    for worker in workers[:2]:
        if hasattr(worker, "close_capture_if_idle"):
            cancellation_results.append(
                worker.close_capture_if_idle(deadline=deadline, clock=clock)
            )
    for worker in workers:
        if worker.thread.ident is not None and worker.thread.is_alive():
            worker.thread.join(max(0.0, deadline - clock()))
    for worker in all_workers:
        if not worker.thread.is_alive() and hasattr(worker, "force_close_descriptor"):
            cancellation_results.append(worker.force_close_descriptor())
    streams_closed = _close_process_streams(
        process,
        tuple(
            name
            for name, worker in zip(("stdout", "stderr", "stdin"), workers, strict=True)
            if not worker.thread.is_alive()
        ),
    )
    reaped = False
    try:
        process.wait(timeout=max(0.0, deadline - clock()))
        reaped = True
    except (subprocess.TimeoutExpired, OSError, RuntimeError, TypeError):
        pass
    try:
        confirmed = not containment.active()
    except (OSError, RuntimeError):
        confirmed = False
    if termination_failed:
        # A failed containment operation leaves invocation ownership uncertain
        # even if a later status probe happens to report no active process.
        method = "termination-failed"
        confirmed = False
    truncated = (
        forced_output_close
        or any(getattr(worker, "error", None) is not None for worker in workers[:2])
        or any(worker.thread.is_alive() for worker in workers[:2])
    )
    workers_stopped = not any(worker.thread.is_alive() for worker in all_workers)
    cancellation_confirmed = workers_stopped or all(cancellation_results)
    return _CleanupResult(
        (
            "completed"
            if confirmed
            and reaped
            and streams_closed
            and workers_stopped
            and cancellation_confirmed
            else "incomplete"
        ),
        method,
        confirmed,
        max(0.0, clock() - started),
        truncated,
    )


def _emergency_cleanup(
    process: object,
    containment: _InvocationContainment,
    deadline: float,
    clock: Callable[[], float],
) -> _CleanupResult:
    """Stop a contained invocation before stream workers have all started."""

    started = clock()
    method: str | None = None
    termination_failed = False
    try:
        method = containment.terminate()
    except (OSError, RuntimeError):
        method = "termination-failed"
        termination_failed = True
        try:
            process.kill()  # type: ignore[attr-defined]
        except (OSError, RuntimeError):
            pass
    if isinstance(containment, _SingleProcessContainment):
        try:
            process.wait(  # type: ignore[attr-defined]
                timeout=max(0.0, deadline - clock())
            )
        except (OSError, RuntimeError, TypeError, subprocess.TimeoutExpired):
            pass
        streams_closed = _close_process_streams(process, ("stdin", "stdout", "stderr"))
        reaped = getattr(process, "returncode", None) is not None
        confirmed = reaped and not termination_failed
        return _CleanupResult(
            "completed" if confirmed and streams_closed else "incomplete",
            method,
            confirmed,
            max(0.0, clock() - started),
            True,
        )
    graceful_until = min(deadline, clock() + 1.0)
    while clock() < graceful_until:
        try:
            if not containment.active():
                break
        except (OSError, RuntimeError):
            break
        threading.Event().wait(min(0.01, max(0.0, graceful_until - clock())))
    if clock() < deadline:
        try:
            if containment.active():
                containment.terminate()
        except (OSError, RuntimeError):
            pass
    reaped = False
    try:
        process.wait(timeout=max(0.0, deadline - clock()))  # type: ignore[attr-defined]
        reaped = True
    except (OSError, RuntimeError, TypeError, subprocess.TimeoutExpired):
        pass
    streams_closed = _close_process_streams(process, ("stdin", "stdout", "stderr"))
    try:
        confirmed = not containment.active()
    except (OSError, RuntimeError):
        confirmed = False
    if termination_failed:
        confirmed = False
    return _CleanupResult(
        "completed" if confirmed and reaped and streams_closed else "incomplete",
        method,
        confirmed,
        max(0.0, clock() - started),
        True,
    )


def _include_start_observer_cleanup(
    cleanup: _CleanupResult,
    worker: _StartObserverWorker | None,
    deadline: float,
    clock: Callable[[], float],
) -> _CleanupResult:
    """Cancel and bound a start recorder under the existing cleanup deadline."""

    if worker is None or not worker.thread.is_alive():
        return cleanup
    started = clock()
    cancelled = worker.cancel()
    observer_deadline = min(deadline, clock() + 0.05)
    worker.thread.join(max(0.0, observer_deadline - clock()))
    stopped = not worker.thread.is_alive()
    return _CleanupResult(
        cleanup.outcome if cancelled and stopped else "incomplete",
        cleanup.method,
        cleanup.confirmed,
        cleanup.duration + max(0.0, clock() - started),
        cleanup.drain_truncated,
    )


def _close_process_streams(
    process: object,
    names: tuple[str, ...],
) -> bool:
    """Close native process streams after their I/O workers have stopped."""

    succeeded = True
    for name in names:
        stream = getattr(process, name, None)
        if stream is None:
            continue
        if type(stream).__module__ != "_io" and not getattr(
            stream, "_codex_close_is_nonblocking", False
        ):
            succeeded = False
            continue
        try:
            stream.close()
        except (OSError, RuntimeError, ValueError):
            succeeded = False
    return succeeded


def _evidence(
    observer: _CodexEventObserver,
    *,
    launched_at: float,
    clock: Callable[[], float],
    timeout_seconds: float | None,
    outcome_elapsed_seconds: float,
    deadline_outcome: str,
    finalization_outcome: str,
    cleanup: _CleanupResult,
) -> CodexProcessEvidence:
    terminal, terminal_elapsed, message, message_elapsed, problem = observer.snapshot()
    total = max(0.0, clock() - launched_at)
    work_elapsed = max(0.0, outcome_elapsed_seconds)
    if timeout_seconds is not None:
        work_elapsed = min(work_elapsed, timeout_seconds + FINALIZATION_SECONDS)
    return CodexProcessEvidence(
        work_timeout_seconds=timeout_seconds,
        work_elapsed_seconds=work_elapsed,
        total_elapsed_seconds=total,
        terminal_event_type=terminal,
        terminal_event_elapsed_seconds=terminal_elapsed,
        completion_before_deadline=(
            terminal == "turn.completed"
            and terminal_elapsed is not None
            and (timeout_seconds is None or terminal_elapsed < timeout_seconds)
        ),
        structured_message=message,
        structured_message_elapsed_seconds=message_elapsed,
        structured_message_before_deadline=(
            message is not None
            and message_elapsed is not None
            and (timeout_seconds is None or message_elapsed < timeout_seconds)
        ),
        deadline_outcome=deadline_outcome,
        finalization_outcome=finalization_outcome,
        cleanup_outcome=cleanup.outcome,
        termination_method=cleanup.method,
        tree_termination_confirmed=cleanup.confirmed,
        cleanup_duration_seconds=cleanup.duration,
        output_draining_truncated=cleanup.drain_truncated,
        event_stream_problem=problem,
    )


def _contain_process(process: subprocess.Popen[bytes]) -> _InvocationContainment:
    if not hasattr(process, "pid"):
        return _SingleProcessContainment(process)
    if os.name == "nt":
        if not hasattr(process, "_handle"):
            return _SingleProcessContainment(process)
        return _WindowsJob(process)
    return _PosixProcessGroup(process)


def _open_native_capture(path: Path | None) -> BinaryIO | None:
    if path is None:
        return None
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        # Unbuffered staging makes capture_prefix_bytes a durable-file prefix
        # claim and prevents a late flush from changing an O(1)-promoted file.
        return os.fdopen(descriptor, "wb", buffering=0)
    except BaseException:
        os.close(descriptor)
        raise


def _open_capture(
    path: Path | None,
    *,
    deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> object | None:
    """Open a killable isolated primary sink through the transport-test seam."""

    return (
        None
        if path is None
        else _CaptureSink(path, deadline=deadline, clock=clock)
    )


def _close_captures(
    captures: tuple[object | None, object | None],
    *,
    deadline: float,
    clock: Callable[[], float],
    force: bool = False,
) -> bool:
    """Stop capture helpers concurrently under one shared absolute deadline."""

    active = tuple(capture for capture in captures if capture is not None)
    if not active:
        return True
    results: list[bool] = [False] * len(active)

    def stop(index: int, capture: object) -> None:
        try:
            operation = getattr(capture, "abort" if force else "close")
            try:
                value = operation(deadline=deadline, clock=clock)
            except TypeError:
                value = operation()
            results[index] = (
                bool(value)
                if force
                else bool(getattr(capture, "stopped", True))
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            results[index] = False

    threads = tuple(
        threading.Thread(
            target=stop,
            args=(index, capture),
            name=f"codex-capture-{index}-closer",
            daemon=True,
        )
        for index, capture in enumerate(active)
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(max(0.0, deadline - clock()))
    for capture, thread in zip(active, threads, strict=True):
        if thread.is_alive():
            request_abort = getattr(capture, "request_abort", None)
            if request_abort is not None:
                try:
                    request_abort()
                except (OSError, RuntimeError, TypeError, ValueError):
                    pass
    for thread in threads:
        if thread.is_alive():
            thread.join(max(0.0, deadline - clock()))
    return bool(
        all(results)
        and not any(thread.is_alive() for thread in threads)
        and all(getattr(capture, "stopped", True) for capture in active)
    )


def _request_capture_aborts(
    captures: tuple[object | None, object | None],
) -> bool:
    """Signal every helper before any shared-deadline reaping begins."""

    succeeded = True
    for capture in captures:
        if capture is None:
            continue
        request_abort = getattr(capture, "request_abort", None)
        if request_abort is None:
            succeeded = False
            continue
        try:
            request_abort()
        except (OSError, RuntimeError, TypeError, ValueError):
            succeeded = False
    return succeeded


def _resume_windows_process(pid: int) -> None:
    kernel32 = _windows_kernel32()

    class ThreadEntry(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.c_uint32),
            ("cntUsage", ctypes.c_uint32),
            ("th32ThreadID", ctypes.c_uint32),
            ("th32OwnerProcessID", ctypes.c_uint32),
            ("tpBasePri", ctypes.c_long),
            ("tpDeltaPri", ctypes.c_long),
            ("dwFlags", ctypes.c_uint32),
        ]

    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000004, 0)
    invalid = ctypes.c_void_p(-1).value
    if snapshot == invalid:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        entry = ThreadEntry()
        entry.dwSize = ctypes.sizeof(entry)
        found = kernel32.Thread32First(snapshot, ctypes.byref(entry))
        while found:
            if entry.th32OwnerProcessID == pid:
                thread = kernel32.OpenThread(0x0002, False, entry.th32ThreadID)
                if not thread:
                    raise ctypes.WinError(ctypes.get_last_error())
                try:
                    if kernel32.ResumeThread(thread) == 0xFFFFFFFF:
                        raise ctypes.WinError(ctypes.get_last_error())
                finally:
                    kernel32.CloseHandle(thread)
                return
            found = kernel32.Thread32Next(snapshot, ctypes.byref(entry))
        raise RuntimeError("Could not find the suspended Codex process thread.")
    finally:
        kernel32.CloseHandle(snapshot)


def _cancel_blocked_windows_thread(thread: threading.Thread) -> bool:
    if os.name != "nt" or thread.native_id is None:
        return os.name != "nt"
    kernel32 = _windows_kernel32()
    # CancelSynchronousIo requires THREAD_TERMINATE, not merely SYNCHRONIZE.
    handle = kernel32.OpenThread(0x0001, False, thread.native_id)
    if not handle:
        return False
    try:
        cancelled = bool(kernel32.CancelSynchronousIo(handle))
        error = ctypes.get_last_error()
        # ERROR_NOT_FOUND means the thread had no cancellable synchronous I/O
        # at this instant. The cooperative cancellation flag still governs its
        # next operation, so this is not an API failure.
        return cancelled or error == 1168
    finally:
        kernel32.CloseHandle(handle)


def _cancel_windows_descriptor_io(descriptor: int | None) -> bool:
    if os.name != "nt":
        return True
    if descriptor is None:
        return False
    try:
        import msvcrt

        native_handle = msvcrt.get_osfhandle(descriptor)
    except (OSError, TypeError, ValueError):
        return False
    kernel32 = _windows_kernel32()
    cancelled = bool(kernel32.CancelIoEx(ctypes.c_void_p(native_handle), None))
    error = ctypes.get_last_error()
    return cancelled or error == 1168


def _windows_kernel32():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    void_p = ctypes.c_void_p
    kernel32.CreateJobObjectW.argtypes = (void_p, ctypes.c_wchar_p)
    kernel32.CreateJobObjectW.restype = void_p
    kernel32.SetInformationJobObject.argtypes = (
        void_p,
        ctypes.c_int,
        void_p,
        ctypes.c_uint32,
    )
    kernel32.SetInformationJobObject.restype = ctypes.c_int
    kernel32.AssignProcessToJobObject.argtypes = (void_p, void_p)
    kernel32.AssignProcessToJobObject.restype = ctypes.c_int
    kernel32.TerminateJobObject.argtypes = (void_p, ctypes.c_uint32)
    kernel32.TerminateJobObject.restype = ctypes.c_int
    kernel32.QueryInformationJobObject.argtypes = (
        void_p,
        ctypes.c_int,
        void_p,
        ctypes.c_uint32,
        void_p,
    )
    kernel32.QueryInformationJobObject.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = (void_p,)
    kernel32.CloseHandle.restype = ctypes.c_int
    kernel32.CreateToolhelp32Snapshot.argtypes = (ctypes.c_uint32, ctypes.c_uint32)
    kernel32.CreateToolhelp32Snapshot.restype = void_p
    kernel32.Thread32First.argtypes = (void_p, void_p)
    kernel32.Thread32First.restype = ctypes.c_int
    kernel32.Thread32Next.argtypes = (void_p, void_p)
    kernel32.Thread32Next.restype = ctypes.c_int
    kernel32.OpenThread.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    kernel32.OpenThread.restype = void_p
    kernel32.ResumeThread.argtypes = (void_p,)
    kernel32.ResumeThread.restype = ctypes.c_uint32
    kernel32.CancelSynchronousIo.argtypes = (void_p,)
    kernel32.CancelSynchronousIo.restype = ctypes.c_int
    kernel32.CancelIoEx.argtypes = (void_p, void_p)
    kernel32.CancelIoEx.restype = ctypes.c_int
    return kernel32


def with_environment(
    command: CodexCommand, environment: Mapping[str, str]
) -> CodexCommand:
    return CodexCommand(
        command.argv,
        command.cwd,
        environment,
        command.stdout_capture,
        command.stderr_capture,
    )


def with_capture_paths(
    command: CodexCommand,
    *,
    stdout: Path,
    stderr: Path,
) -> CodexCommand:
    return CodexCommand(command.argv, command.cwd, command.environment, stdout, stderr)


@contextmanager
def external_scratch_environment(
    repo_path: Path,
    *,
    cleanup_deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Iterator[dict[str, str]]:
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
        _remove_external_scratch(
            scratch_path,
            cleanup_deadline=cleanup_deadline,
            clock=clock,
        )


def _remove_external_scratch(
    scratch_path: Path,
    *,
    cleanup_deadline: float | None,
    clock: Callable[[], float],
) -> None:
    """Remove scratch without letting a large tree extend invocation return."""

    try:
        scratch_path.rmdir()
        return
    except FileNotFoundError:
        return
    except OSError:
        pass
    if cleanup_deadline is None:
        shutil.rmtree(scratch_path, ignore_errors=True)
        return
    remaining = max(0.0, cleanup_deadline - clock())
    if remaining <= 0:
        return
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        subprocess.run(
            (
                sys.executable,
                "-c",
                "import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)",
                str(scratch_path),
            ),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=creationflags,
            timeout=remaining,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def _process_environment(overrides: Mapping[str, str] | None) -> dict[str, str]:
    environment = os.environ.copy()
    if overrides is not None:
        environment.update(overrides)
    return environment


__all__ = [
    "CLEANUP_SECONDS",
    "FINALIZATION_SECONDS",
    "CodexCommand",
    "CodexProcessEvidence",
    "CodexProcessResult",
    "CodexProcessRunner",
    "CodexProcessTimedOut",
    "CodexProcessTimeout",
    "CodexScratchDirectoryError",
    "SubprocessCodexRunner",
    "external_scratch_environment",
    "with_capture_paths",
    "with_environment",
]
