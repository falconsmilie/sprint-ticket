"""Codex CLI subprocess boundary with live deadline observation."""

from __future__ import annotations

import codecs
import ctypes
import json
import os
import shutil
import signal
import subprocess
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
    returncode: int
    stdout: str
    stderr: str
    evidence: CodexProcessEvidence = field(default_factory=CodexProcessEvidence)
    stdout_capture: Path | None = None
    stderr_capture: Path | None = None
    transport_failure: str | None = None


@dataclass(frozen=True)
class CodexProcessTimeout:
    stdout: str
    stderr: str
    timeout_seconds: float
    evidence: CodexProcessEvidence = field(default_factory=CodexProcessEvidence)
    stdout_capture: Path | None = None
    stderr_capture: Path | None = None


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
        self.lock = threading.Lock()
        self.terminal_event_type: str | None = None
        self.terminal_event_elapsed_seconds: float | None = None
        self.structured_message: str | None = None
        self.structured_message_elapsed_seconds: float | None = None
        self.problem: str | None = None

    def feed(self, chunk: bytes, *, final: bool = False) -> None:
        text = self.decoder.decode(chunk, final=final)
        with self.lock:
            self.pending += text
            while "\n" in self.pending:
                line, self.pending = self.pending.split("\n", 1)
                self._observe_line(line.rstrip("\r"))
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


class _StreamWorker:
    def __init__(
        self,
        stream: BinaryIO,
        destination: Path | None,
        *,
        observer: _CodexEventObserver | None = None,
        output: BinaryIO | None = None,
    ) -> None:
        self.stream = stream
        self.destination = destination
        self.observer = observer
        self.output = output
        self.done = threading.Event()
        self.error: BaseException | None = None
        self.buffer = bytearray()
        self.thread = threading.Thread(target=self._run)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        output = self.output
        try:
            if self.destination is not None and output is None:
                output = self.destination.open("wb")
            while True:
                read = getattr(self.stream, "read1", self.stream.read)
                chunk = read(_READ_SIZE)
                if not chunk:
                    break
                if output is not None:
                    output.write(chunk)
                    output.flush()
                self.buffer.extend(chunk)
                if self.destination is not None and len(self.buffer) > _READ_SIZE:
                    del self.buffer[:-_READ_SIZE]
                if self.observer is not None:
                    self.observer.feed(chunk)
            if self.observer is not None:
                self.observer.feed(b"", final=True)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self.error = error
        finally:
            if output is not None:
                try:
                    output.close()
                except (OSError, ValueError):
                    pass
            self.done.set()

    def text(self) -> str:
        return decode_human_output(bytes(self.buffer))


class _InputWorker:
    def __init__(self, stream: BinaryIO, value: str) -> None:
        self.stream = stream
        self.value = value.encode("utf-8")
        self.done = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            view = memoryview(self.value)
            while view:
                written = self.stream.write(view[:_READ_SIZE])
                if written is None:
                    written = min(len(view), _READ_SIZE)
                view = view[written:]
            self.stream.flush()
        except (BrokenPipeError, OSError, ValueError) as error:
            self.error = error
        finally:
            try:
                self.stream.close()
            except OSError:
                pass
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
        info = _ExtendedLimitInformation()
        info.BasicLimitInformation.LimitFlags = self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            self.handle,
            self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel32.AssignProcessToJobObject(
            self.handle,
            ctypes.c_void_p(int(process._handle)),  # type: ignore[attr-defined]
        ):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())
        _resume_windows_process(process.pid)

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
        stdout_capture = _open_capture(command.stdout_capture)
        try:
            stderr_capture = _open_capture(command.stderr_capture)
        except BaseException:
            if stdout_capture is not None:
                stdout_capture.close()
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
            popen_options["creationflags"] = getattr(subprocess, "CREATE_SUSPENDED", 0x4)
        else:
            popen_options["start_new_session"] = True
        launched_at = self._monotonic()
        deadline = None if timeout_seconds is None else launched_at + timeout_seconds
        try:
            process = subprocess.Popen(command.argv, **popen_options)
        except BaseException:
            _close_captures(capture_streams)
            raise
        containment: _InvocationContainment
        try:
            containment = _contain_process(process)
        except BaseException as error:
            process.kill()
            try:
                process.wait(timeout=CLEANUP_SECONDS)
            except (OSError, subprocess.TimeoutExpired, TypeError) as cleanup_error:
                error.add_note(f"Suspended process cleanup also failed: {cleanup_error}")
            _close_captures(capture_streams)
            raise
        workers_for_cleanup: tuple[
            _StreamWorker, _StreamWorker, _InputWorker
        ] | None = None
        try:
            if on_process_start is not None:
                on_process_start()
            if not all(
                hasattr(process, name) for name in ("stdin", "stdout", "stderr", "poll")
            ):
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
                observer=observer,
                output=stdout_capture,
            )
            stderr_worker = _StreamWorker(
                process.stderr,
                command.stderr_capture,
                output=stderr_capture,
            )
            input_worker = _InputWorker(process.stdin, stdin)
            workers = (stdout_worker, stderr_worker, input_worker)
            workers_for_cleanup = workers
            for worker in workers:
                worker.start()
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
            cleanup_deadline = min(
                launched_at + (timeout_seconds or CLEANUP_SECONDS) + CLEANUP_SECONDS,
                self._monotonic() + CLEANUP_SECONDS,
            )
            if workers_for_cleanup is None:
                _emergency_cleanup(
                    process, containment, cleanup_deadline, self._monotonic
                )
            else:
                _cleanup(
                    process,
                    containment,
                    workers_for_cleanup,
                    cleanup_deadline,
                    self._monotonic,
                )
            raise
        finally:
            containment.close()
            _close_captures(capture_streams)

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
        while True:
            now = self._monotonic()
            terminal_type, terminal_elapsed, message, message_elapsed, problem = (
                observer.snapshot()
            )
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
            if problem is not None:
                deadline_outcome = "event-conflict"
                finalization_outcome = "failed"
                transport_failure = problem
                break
            if terminal_type in {"turn.failed", "turn.cancelled"}:
                deadline_outcome = "terminal-failure"
                finalization_outcome = "failed"
                transport_failure = f"Codex emitted {terminal_type}."
                break
            if timely_terminal and timely_message:
                deadline_outcome = "completed-before-deadline"
                if finalization_deadline is None:
                    assert terminal_elapsed is not None
                    finalization_deadline = (
                        launched_at + terminal_elapsed + FINALIZATION_SECONDS
                    )
                    finalization_outcome = "pending"
                if (
                    process.poll() is not None
                    and not containment.active()
                    and stdout_worker.done.is_set()
                    and stderr_worker.done.is_set()
                    and input_worker.done.is_set()
                ):
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
            elif (
                process.poll() is not None
                and not containment.active()
                and stdout_worker.done.is_set()
                and stderr_worker.done.is_set()
            ):
                deadline_outcome = "process-exited-without-timely-completion"
                finalization_outcome = "not-eligible"
                break
            for worker in (stdout_worker, stderr_worker):
                if worker.error is not None:
                    transport_failure = f"Codex output streaming failed: {worker.error}"
                    deadline_outcome = "stream-failure"
                    finalization_outcome = "failed"
                    break
            if transport_failure is not None:
                break
            threading.Event().wait(0.005)

        cleanup_required = transport_failure is not None or containment.active()
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
        process.poll()
        returncode = process.returncode if process.returncode is not None else -1
        evidence = _evidence(
            observer,
            launched_at=launched_at,
            clock=self._monotonic,
            timeout_seconds=timeout_seconds,
            deadline_outcome=deadline_outcome,
            finalization_outcome=finalization_outcome,
            cleanup=cleanup,
        )
        return CodexProcessResult(
            returncode=returncode,
            stdout=stdout_worker.text(),
            stderr=stderr_worker.text(),
            evidence=evidence,
            stdout_capture=command.stdout_capture,
            stderr_capture=command.stderr_capture,
            transport_failure=transport_failure,
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
    ) -> CodexProcessResult:
        cleanup = _cleanup(
            process,
            containment,
            (stdout_worker, stderr_worker, input_worker),
            deadline + CLEANUP_SECONDS,
            self._monotonic,
        )
        evidence = _evidence(
            observer,
            launched_at=launched_at,
            clock=self._monotonic,
            timeout_seconds=timeout_seconds,
            deadline_outcome="timed-out",
            finalization_outcome="not-eligible",
            cleanup=cleanup,
        )
        raise CodexProcessTimedOut(
            CodexProcessTimeout(
                stdout=stdout_worker.text(),
                stderr=stderr_worker.text(),
                timeout_seconds=timeout_seconds,
                evidence=evidence,
                stdout_capture=command.stdout_capture,
                stderr_capture=command.stderr_capture,
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


def _cleanup(
    process: subprocess.Popen[bytes],
    containment: _InvocationContainment,
    workers: tuple[_StreamWorker, _StreamWorker, _InputWorker],
    deadline: float,
    clock: Callable[[], float],
) -> _CleanupResult:
    started = clock()
    method: str | None = None
    confirmed = False
    try:
        method = containment.terminate()
    except (OSError, RuntimeError):
        method = "termination-failed"
    graceful_until = min(deadline, clock() + 1.0)
    while clock() < graceful_until:
        process.poll()
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
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
    for worker in workers:
        remaining = max(0.0, deadline - clock())
        worker.thread.join(remaining)
        if worker.thread.is_alive():
            _cancel_blocked_windows_thread(worker.thread)
            worker.thread.join(max(0.0, deadline - clock()))
    try:
        process.wait(timeout=max(0.0, deadline - clock()))
    except (subprocess.TimeoutExpired, OSError):
        pass
    try:
        confirmed = not containment.active()
    except (OSError, RuntimeError):
        confirmed = False
    truncated = any(worker.thread.is_alive() for worker in workers[:2])
    return _CleanupResult(
        (
            "completed"
            if confirmed and not any(w.thread.is_alive() for w in workers)
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
) -> None:
    try:
        containment.terminate()
    except (OSError, RuntimeError):
        try:
            process.kill()  # type: ignore[attr-defined]
        except (OSError, RuntimeError):
            return
    try:
        process.wait(timeout=max(0.0, deadline - clock()))  # type: ignore[attr-defined]
    except (OSError, RuntimeError, TypeError, subprocess.TimeoutExpired):
        return


def _evidence(
    observer: _CodexEventObserver,
    *,
    launched_at: float,
    clock: Callable[[], float],
    timeout_seconds: float | None,
    deadline_outcome: str,
    finalization_outcome: str,
    cleanup: _CleanupResult,
) -> CodexProcessEvidence:
    terminal, terminal_elapsed, message, message_elapsed, problem = observer.snapshot()
    total = max(0.0, clock() - launched_at)
    if deadline_outcome == "completed-before-deadline" and terminal_elapsed is not None:
        work_elapsed = terminal_elapsed
    elif deadline_outcome == "timed-out" and timeout_seconds is not None:
        work_elapsed = timeout_seconds
    else:
        work_elapsed = (
            min(total, timeout_seconds) if timeout_seconds is not None else total
        )
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


def _open_capture(path: Path | None) -> BinaryIO | None:
    if path is None:
        return None
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        return os.fdopen(descriptor, "wb")
    except BaseException:
        os.close(descriptor)
        raise


def _close_captures(captures: tuple[BinaryIO | None, BinaryIO | None]) -> None:
    for capture in captures:
        if capture is not None:
            try:
                capture.close()
            except (OSError, ValueError):
                pass


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


def _cancel_blocked_windows_thread(thread: threading.Thread) -> None:
    if os.name != "nt" or thread.native_id is None:
        return
    kernel32 = _windows_kernel32()
    handle = kernel32.OpenThread(0x00100000, False, thread.native_id)
    if handle:
        try:
            kernel32.CancelSynchronousIo(handle)
        finally:
            kernel32.CloseHandle(handle)


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
