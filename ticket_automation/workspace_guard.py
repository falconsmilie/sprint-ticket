from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

WORKSPACE_GUARD_DIR_NAME = "workspace-guard"
WORKSPACE_GUARD_FORMAT = "ticket_automation.workspace_environment_guard"
WORKSPACE_GUARD_SCHEMA_VERSION = 1

_SKIPPED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)


@dataclass(frozen=True)
class LocalEnvironment:
    kind: str
    root_path: Path
    marker_paths: tuple[Path, ...]

    @property
    def primary_marker_path(self) -> Path:
        return self.marker_paths[0]

    def to_dict(
        self, repository_path: Path, *, existed_before: bool | None = None
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind,
            "root": _relative_path(self.root_path, repository_path),
            "root_absolute_path": str(self.root_path),
            "markers": tuple(
                _relative_path(marker_path, repository_path)
                for marker_path in self.marker_paths
            ),
        }
        if existed_before is not None:
            data["existed_before"] = existed_before
        return data


@dataclass(frozen=True)
class WorkspaceEnvironmentSnapshot:
    repository_path: Path
    environments: tuple[LocalEnvironment, ...]
    inspection_errors: tuple[str, ...] = ()

    @property
    def inspection_complete(self) -> bool:
        """Whether Python/Conda root detection inspected the whole repository."""

        return not self.inspection_errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_repository_path": str(self.repository_path),
            "environments": tuple(
                environment.to_dict(self.repository_path)
                for environment in self.environments
            ),
            "inspection_errors": self.inspection_errors,
        }


@dataclass(frozen=True)
class WorkspaceGuardInspection:
    phase: str
    timestamp: str
    artifact_path: Path | None
    before: WorkspaceEnvironmentSnapshot
    after: WorkspaceEnvironmentSnapshot
    new_environments: tuple[LocalEnvironment, ...]

    @property
    def has_violation(self) -> bool:
        return bool(self.new_environments)

    @property
    def has_inspection_failure(self) -> bool:
        return not (self.before.inspection_complete and self.after.inspection_complete)

    @property
    def requires_human(self) -> bool:
        """The deterministic V1 policy cannot safely permit this invocation."""

        return self.has_violation or self.has_inspection_failure

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WORKSPACE_GUARD_SCHEMA_VERSION,
            "format": WORKSPACE_GUARD_FORMAT,
            "phase": self.phase,
            "timestamp": self.timestamp,
            "target_repository_path": str(self.after.repository_path),
            "environments_before": tuple(
                environment.to_dict(self.before.repository_path)
                for environment in self.before.environments
            ),
            "environments_after": tuple(
                environment.to_dict(self.after.repository_path)
                for environment in self.after.environments
            ),
            "new_environments": tuple(
                environment.to_dict(self.after.repository_path, existed_before=False)
                for environment in self.new_environments
            ),
            "inspection_errors_before": self.before.inspection_errors,
            "inspection_errors_after": self.after.inspection_errors,
        }


def capture_workspace_environment_snapshot(
    repository_path: Path | str,
) -> WorkspaceEnvironmentSnapshot:
    """Detect credible Python virtual-environment and Conda roots in a repository.

    This is intentionally a narrow deterministic check.  It does not claim to
    identify package caches, dependency trees, or environments outside the
    target repository.
    """

    path = Path(repository_path)
    environments: list[LocalEnvironment] = []
    errors: list[str] = []
    try:
        root = path.resolve()
    except OSError as error:
        root = path.absolute()
        errors.append(f"repository-path: {error}")
    _walk_repository(root, root, environments=environments, errors=errors)
    return WorkspaceEnvironmentSnapshot(
        repository_path=root,
        environments=tuple(sorted(environments, key=_environment_sort_key)),
        inspection_errors=tuple(errors),
    )


def _compare_workspace_environment_change(
    *,
    before: WorkspaceEnvironmentSnapshot,
    after: WorkspaceEnvironmentSnapshot,
    phase: str,
    clock: Callable[[], datetime] | None = None,
) -> WorkspaceGuardInspection:
    """Build environment evidence from the shared boundary's two snapshots."""
    return WorkspaceGuardInspection(
        phase=phase,
        timestamp=_timestamp(clock),
        artifact_path=None,
        before=before,
        after=after,
        new_environments=new_environments(before, after),
    )


def new_environments(
    before: WorkspaceEnvironmentSnapshot,
    after: WorkspaceEnvironmentSnapshot,
) -> tuple[LocalEnvironment, ...]:
    before_roots = {
        _path_identity(environment.root_path) for environment in before.environments
    }
    return tuple(
        environment
        for environment in after.environments
        if _path_identity(environment.root_path) not in before_roots
    )


def write_workspace_guard_inspection(
    inspection: WorkspaceGuardInspection,
) -> None:
    if inspection.artifact_path is None:
        raise ValueError("A workspace-guard artifact path is required for persistence.")
    if not inspection.requires_human:
        raise ValueError(
            "A standalone workspace-guard artifact is only permitted for an "
            "inspection failure or a prohibited environment."
        )
    _atomic_write_json(inspection.artifact_path, inspection.to_dict())


def format_workspace_hygiene_violation(
    inspection: WorkspaceGuardInspection,
    *,
    operation: str,
    run_dir: Path | str,
    git_safety: str,
) -> str:
    environments = "; ".join(
        _format_environment(environment, inspection.after.repository_path)
        for environment in inspection.new_environments
    )
    artifact = (
        "writable execution record"
        if inspection.artifact_path is None
        else _relative_path(inspection.artifact_path, Path(run_dir))
    )
    return (
        "Workspace hygiene violation: "
        f"The writable Codex operation for {operation} created a new local "
        f"Python/Conda environment inside the target repository: {environments}. "
        "The environment did not exist before that writable operation. "
        f"Git safety: {git_safety}. "
        "TicketAutomation has stopped for human inspection. "
        "No files were deleted automatically. "
        f"Workspace-guard artifact: {artifact}."
    )


def _format_workspace_environment_inspection_failure(
    inspection: WorkspaceGuardInspection,
    *,
    operation: str,
    run_dir: Path | str,
) -> str:
    """Describe a fail-closed scanner error without overstating its coverage."""

    errors = tuple(
        f"before: {error}" for error in inspection.before.inspection_errors
    ) + tuple(f"after: {error}" for error in inspection.after.inspection_errors)
    artifact = (
        "writable execution record"
        if inspection.artifact_path is None
        else _relative_path(inspection.artifact_path, Path(run_dir))
    )
    details = "; ".join(errors) or "unknown scanner failure"
    return (
        "Workspace environment inspection could not complete for the writable "
        f"Codex operation {operation}. TicketAutomation has stopped for human "
        "inspection. The deterministic check covers Python and Conda "
        f"environment roots inside the target repository. Details: {details}. "
        "No files were deleted automatically. "
        f"Workspace-guard artifact: {artifact}."
    )


def _walk_repository(
    path: Path,
    repository_path: Path,
    *,
    environments: list[LocalEnvironment],
    errors: list[str],
) -> None:
    environment = _detect_environment_root(path)
    if environment is not None:
        environments.append(environment)
        return

    try:
        with _scandir(path) as iterator:
            entries = tuple(iterator)
    except OSError as error:
        errors.append(f"{_relative_path(path, repository_path)}: {error}")
        return

    for entry in entries:
        if entry.name in _SKIPPED_DIRECTORY_NAMES:
            continue
        try:
            if not entry.is_dir(follow_symlinks=False):
                continue
        except OSError as error:
            errors.append(
                f"{_relative_path(Path(entry.path), repository_path)}: {error}"
            )
            continue
        entry_path = Path(entry.path)
        if _is_junction(entry_path):
            continue
        _walk_repository(
            entry_path,
            repository_path,
            environments=environments,
            errors=errors,
        )


def _detect_environment_root(path: Path) -> LocalEnvironment | None:
    conda_history = path / "conda-meta" / "history"
    if conda_history.is_file():
        return LocalEnvironment(
            kind="conda",
            root_path=path,
            marker_paths=(conda_history,),
        )

    pyvenv = path / "pyvenv.cfg"
    if pyvenv.is_file():
        return LocalEnvironment(
            kind="python_venv",
            root_path=path,
            marker_paths=(pyvenv,),
        )

    markers = _credible_python_environment_markers(path)
    if markers:
        return LocalEnvironment(
            kind="python_venv",
            root_path=path,
            marker_paths=markers,
        )
    return None


def _credible_python_environment_markers(path: Path) -> tuple[Path, ...]:
    windows_python = _first_existing_file(
        path / "Scripts" / "python.exe",
        path / "Scripts" / "python",
    )
    windows_site_packages = path / "Lib" / "site-packages"
    if windows_python is not None and windows_site_packages.is_dir():
        return (windows_python, windows_site_packages)

    posix_python = path / "bin" / "python"
    posix_site_packages = _posix_site_packages(path / "lib")
    if posix_python.is_file() and posix_site_packages is not None:
        return (posix_python, posix_site_packages)

    return ()


def _first_existing_file(*paths: Path) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def _posix_site_packages(lib_path: Path) -> Path | None:
    try:
        with _scandir(lib_path) as iterator:
            entries = tuple(iterator)
    except OSError:
        return None
    for entry in entries:
        if not entry.name.startswith("python"):
            continue
        try:
            if not entry.is_dir(follow_symlinks=False):
                continue
        except OSError:
            continue
        site_packages = Path(entry.path) / "site-packages"
        if site_packages.is_dir():
            return site_packages
    return None


def _format_environment(environment: LocalEnvironment, repository_path: Path) -> str:
    root = _relative_path(environment.root_path, repository_path)
    marker = _relative_path(environment.primary_marker_path, repository_path)
    return f"{_format_root(root)} ({_kind_label(environment.kind)}; marker: {marker})"


def _kind_label(kind: str) -> str:
    if kind == "conda":
        return "Conda environment"
    return "Python virtual environment"


def _format_root(value: str) -> str:
    if value == ".":
        return value
    return value.rstrip("/") + "/"


def _environment_sort_key(environment: LocalEnvironment) -> str:
    return _path_identity(environment.root_path)


def _path_identity(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path.resolve(strict=False))))


def _relative_path(path: Path, base: Path) -> str:
    resolved_path = path.resolve(strict=False)
    resolved_base = base.resolve(strict=False)
    try:
        relative = resolved_path.relative_to(resolved_base)
    except ValueError:
        relative = Path(os.path.relpath(resolved_path, resolved_base))
    value = relative.as_posix()
    return "." if value == "." else value


def _scandir(path: Path) -> os.ScandirIterator[os.DirEntry[str]]:
    return os.scandir(path)


def _is_junction(path: Path) -> bool:
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction is not None and isjunction(path))


def _timestamp(clock: Callable[[], datetime] | None) -> str:
    now = datetime.now(UTC) if clock is None else clock()
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return now.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True)
    temp_path: Path | None = None
    file_descriptor = -1
    try:
        file_descriptor, temp_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            text=True,
        )
        temp_path = Path(temp_name)
        with os.fdopen(
            file_descriptor, "w", encoding="utf-8", newline="\n"
        ) as temp_file:
            file_descriptor = -1
            temp_file.write(payload)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    except Exception:
        if file_descriptor != -1:
            os.close(file_descriptor)
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


__all__ = [
    "WORKSPACE_GUARD_DIR_NAME",
    "LocalEnvironment",
    "WorkspaceEnvironmentSnapshot",
    "WorkspaceGuardInspection",
    "capture_workspace_environment_snapshot",
    "format_workspace_hygiene_violation",
    "new_environments",
    "write_workspace_guard_inspection",
]
