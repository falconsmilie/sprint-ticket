from __future__ import annotations

import errno
import hashlib
import json
import os
import socket
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

if os.name == "nt":
    import msvcrt
else:
    import fcntl


LOCKS_DIR_NAME = ".locks"
LOCK_METADATA_FORMAT = "ticket_automation.repository_lock"
LOCK_METADATA_SCHEMA_VERSION = 1
PENDING_RUN_ID = "<pending>"
_LOCK_ROOT_ENV = "_TICKET_AUTOMATION_LOCK_ROOT"


class RepositoryLockError(RuntimeError):
    """Raised when another active process owns a target repository lock."""


@dataclass(frozen=True)
class RepositoryLockMetadata:
    run_id: str
    target_repository_path: str
    canonical_repository_identity: str
    owner_pid: int
    owner_hostname: str
    acquired_timestamp: str
    updated_timestamp: str
    current_state: str | None
    lock_file: Path
    metadata_file: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LOCK_METADATA_SCHEMA_VERSION,
            "format": LOCK_METADATA_FORMAT,
            "run_id": self.run_id,
            "target_repository_path": self.target_repository_path,
            "canonical_repository_identity": self.canonical_repository_identity,
            "owner_pid": self.owner_pid,
            "owner_hostname": self.owner_hostname,
            "acquired_timestamp": self.acquired_timestamp,
            "updated_timestamp": self.updated_timestamp,
            "current_state": self.current_state,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        lock_file: Path,
        metadata_file: Path,
    ) -> RepositoryLockMetadata | None:
        if data.get("format") != LOCK_METADATA_FORMAT:
            return None
        if data.get("schema_version") != LOCK_METADATA_SCHEMA_VERSION:
            return None
        run_id = data.get("run_id")
        target_repository_path = data.get("target_repository_path")
        canonical_repository_identity = data.get("canonical_repository_identity")
        owner_pid = data.get("owner_pid")
        owner_hostname = data.get("owner_hostname")
        acquired_timestamp = data.get("acquired_timestamp")
        updated_timestamp = data.get("updated_timestamp")
        current_state = data.get("current_state")
        if not isinstance(run_id, str) or not run_id:
            return None
        if not isinstance(target_repository_path, str) or not target_repository_path:
            return None
        if (
            not isinstance(canonical_repository_identity, str)
            or not canonical_repository_identity
        ):
            return None
        if not isinstance(owner_pid, int) or isinstance(owner_pid, bool):
            return None
        if not isinstance(owner_hostname, str) or not owner_hostname:
            return None
        if not isinstance(acquired_timestamp, str) or not acquired_timestamp:
            return None
        if not isinstance(updated_timestamp, str) or not updated_timestamp:
            return None
        if current_state is not None and not isinstance(current_state, str):
            return None
        return cls(
            run_id=run_id,
            target_repository_path=target_repository_path,
            canonical_repository_identity=canonical_repository_identity,
            owner_pid=owner_pid,
            owner_hostname=owner_hostname,
            acquired_timestamp=acquired_timestamp,
            updated_timestamp=updated_timestamp,
            current_state=current_state,
            lock_file=lock_file,
            metadata_file=metadata_file,
        )


class RepositoryRunLock:
    def __init__(
        self,
        *,
        file_handle: Any,
        lock_file: Path,
        metadata_file: Path,
        canonical_repository_identity: str,
        target_repository_path: str,
        run_id: str | None,
        current_state: str | None,
        acquired_timestamp: str,
        clock: Callable[[], datetime] | None,
    ) -> None:
        self._file_handle = file_handle
        self._metadata_file = metadata_file
        self._lock_file = lock_file
        self._canonical_repository_identity = canonical_repository_identity
        self._target_repository_path = target_repository_path
        self._run_id = run_id or PENDING_RUN_ID
        self._current_state = current_state
        self._acquired_timestamp = acquired_timestamp
        self._clock = clock
        self._released = False
        self.update(run_id=run_id, current_state=current_state)

    @property
    def metadata(self) -> RepositoryLockMetadata:
        return RepositoryLockMetadata(
            run_id=self._run_id,
            target_repository_path=self._target_repository_path,
            canonical_repository_identity=self._canonical_repository_identity,
            owner_pid=os.getpid(),
            owner_hostname=socket.gethostname(),
            acquired_timestamp=self._acquired_timestamp,
            updated_timestamp=_timestamp(self._clock),
            current_state=self._current_state,
            lock_file=self._lock_file,
            metadata_file=self._metadata_file,
        )

    def update(
        self,
        *,
        run_id: str | None = None,
        current_state: str | None = None,
        target_repository_path: str | Path | None = None,
    ) -> None:
        if self._released:
            raise RuntimeError("Cannot update a released repository lock.")
        if run_id:
            self._run_id = run_id
        if current_state is not None:
            self._current_state = current_state
        if target_repository_path is not None:
            self._target_repository_path = str(
                Path(target_repository_path).resolve(strict=False)
            )
        _write_metadata(self._metadata_file, self.metadata.to_dict())

    def release(self) -> None:
        if self._released:
            return
        try:
            self._metadata_file.unlink(missing_ok=True)
        finally:
            try:
                _unlock_file(self._file_handle)
            finally:
                self._file_handle.close()
                self._released = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


def acquire_repository_run_lock(
    target_repository: str | Path,
    *,
    run_id: str | None,
    current_state: str | None,
    clock: Callable[[], datetime] | None = None,
) -> RepositoryRunLock:
    canonical_identity = canonical_repository_identity(target_repository)
    target_repository_path = str(Path(target_repository).resolve(strict=False))
    locks_dir = _lock_root()
    locks_dir.mkdir(parents=True, exist_ok=True)
    lock_file, metadata_file = _lock_paths(locks_dir, canonical_identity)
    file_handle = _open_lock_file(lock_file)
    try:
        if not _try_lock_file(file_handle):
            metadata = _read_metadata(metadata_file, lock_file=lock_file)
            raise RepositoryLockError(
                _format_lock_error(
                    target_repository=target_repository_path,
                    canonical_identity=canonical_identity,
                    metadata=metadata,
                )
            )
        return RepositoryRunLock(
            file_handle=file_handle,
            lock_file=lock_file,
            metadata_file=metadata_file,
            canonical_repository_identity=canonical_identity,
            target_repository_path=target_repository_path,
            run_id=run_id,
            current_state=current_state,
            acquired_timestamp=_timestamp(clock),
            clock=clock,
        )
    except Exception:
        file_handle.close()
        raise


def canonical_repository_identity(target_repository: str | Path) -> str:
    path = Path(target_repository).resolve(strict=False)
    git_root = _git_working_tree_root(path)
    identity_path = path if git_root is None else git_root
    return os.path.normcase(os.path.realpath(str(identity_path)))


def active_repository_locks() -> tuple[RepositoryLockMetadata, ...]:
    locks_dir = _lock_root()
    if not locks_dir.is_dir():
        return ()

    active: list[RepositoryLockMetadata] = []
    for metadata_file in sorted(locks_dir.glob("*.json")):
        lock_file = metadata_file.with_suffix(".lock")
        metadata = _read_metadata(metadata_file, lock_file=lock_file)
        if metadata is None or not lock_file.is_file():
            continue
        if _lock_is_currently_held(lock_file):
            active.append(metadata)
    return tuple(active)


def _open_lock_file(lock_file: Path) -> Any:
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    file_handle = lock_file.open("a+b")
    file_handle.seek(0, os.SEEK_END)
    if file_handle.tell() == 0:
        file_handle.write(b"\0")
        file_handle.flush()
        os.fsync(file_handle.fileno())
    file_handle.seek(0)
    return file_handle


def _try_lock_file(file_handle: Any) -> bool:
    file_handle.seek(0)
    try:
        if os.name == "nt":
            msvcrt.locking(file_handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno in (
            errno.EACCES,
            errno.EAGAIN,
            errno.EDEADLK,
            errno.EPERM,
        ):
            return False
        raise
    return True


def _unlock_file(file_handle: Any) -> None:
    file_handle.seek(0)
    if os.name == "nt":
        msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)


def _lock_is_currently_held(lock_file: Path) -> bool:
    file_handle = _open_lock_file(lock_file)
    try:
        if _try_lock_file(file_handle):
            _unlock_file(file_handle)
            return False
        return True
    finally:
        file_handle.close()


def _git_working_tree_root(path: Path) -> Path | None:
    if not path.exists():
        return None
    cwd = path if path.is_dir() else path.parent
    try:
        result = subprocess.run(
            ("git", "rev-parse", "--show-toplevel"),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    root = result.stdout.strip()
    if not root:
        return None
    return Path(root).resolve(strict=False)


def _lock_paths(locks_dir: Path, canonical_identity: str) -> tuple[Path, Path]:
    digest = hashlib.sha256(canonical_identity.encode("utf-8")).hexdigest()
    stem = f"repository-{digest}"
    return locks_dir / f"{stem}.lock", locks_dir / f"{stem}.json"


def _lock_root() -> Path:
    override = os.environ.get(_LOCK_ROOT_ENV)
    if override:
        return Path(override)
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir())
        return base / "TicketAutomation" / "locks"
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        return Path(runtime_dir) / "ticket-automation" / "locks"
    owner = os.getuid() if hasattr(os, "getuid") else "user"
    return Path(tempfile.gettempdir()) / f"ticket-automation-{owner}" / "locks"


def _write_metadata(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(data), indent=2, sort_keys=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(f"{payload}\n", encoding="utf-8", newline="\n")
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _read_metadata(
    metadata_file: Path,
    *,
    lock_file: Path,
) -> RepositoryLockMetadata | None:
    try:
        data = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return RepositoryLockMetadata.from_dict(
        data,
        lock_file=lock_file,
        metadata_file=metadata_file,
    )


def _format_lock_error(
    *,
    target_repository: str,
    canonical_identity: str,
    metadata: RepositoryLockMetadata | None,
) -> str:
    lines = [
        "Target repository is already owned by an active TicketAutomation process.",
        f"Target repository: {target_repository}",
        f"Canonical repository identity: {canonical_identity}",
    ]
    if metadata is None:
        lines.append("Owning run ID: unknown")
        lines.append("Owning state: unknown")
        return "\n".join(lines)

    lines.extend(
        [
            f"Owning run ID: {metadata.run_id}",
            f"Owning state: {metadata.current_state or 'unknown'}",
            (
                "Owner process: "
                f"{metadata.owner_hostname} pid {metadata.owner_pid}"
            ),
            f"Acquired: {metadata.acquired_timestamp}",
        ]
    )
    return "\n".join(lines)


def _timestamp(clock: Callable[[], datetime] | None) -> str:
    now = datetime.now(UTC) if clock is None else clock()
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return now.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "RepositoryLockError",
    "acquire_repository_run_lock",
    "active_repository_locks",
]
