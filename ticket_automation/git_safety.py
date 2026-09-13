from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .git import GitRepository
from .workspace_guard import capture_workspace_environment_snapshot

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT_RE = re.compile(r"[0-9a-fA-F]{40,64}")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_WORKSPACE_SNAPSHOT_FORMAT = "ticket_automation.workspace_snapshot"
_WORKSPACE_SNAPSHOT_SCHEMA_VERSION = 1


class _WorkspaceInspectionError(RuntimeError):
    """Raised when a complete workspace identity could not be captured."""


@dataclass(frozen=True)
class WorkspaceChange:
    name: str
    expected: str
    actual: str
    message: str


@dataclass(frozen=True)
class WorkspaceSnapshot:
    """Canonical, read-only identity of a repository and its local workspace."""

    repository_path: Path
    branch: str | None
    head_sha: str | None
    staged_paths: tuple[str, ...]
    staged_diff_sha256: str | None
    tracked_diff_sha256: str | None
    untracked_paths: tuple[str, ...]
    untracked_file_hashes: tuple[tuple[str, str], ...]
    environment_roots: tuple[str, ...]
    inspection_complete: bool
    inspection_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        errors = set(self.inspection_errors)
        repository_value = Path(self.repository_path)
        try:
            repository_path = repository_value.resolve(strict=False)
        except OSError as error:
            repository_path = repository_value.absolute()
            errors.add(f"repository-path: {error}")
        object.__setattr__(
            self,
            "repository_path",
            repository_path,
        )
        object.__setattr__(
            self,
            "staged_paths",
            tuple(sorted({_canonical_path(path) for path in self.staged_paths})),
        )
        object.__setattr__(
            self,
            "untracked_paths",
            tuple(sorted({_canonical_path(path) for path in self.untracked_paths})),
        )
        hashes: dict[str, str] = {}
        for path, content_hash in self.untracked_file_hashes:
            canonical_path = _canonical_path(path)
            previous = hashes.get(canonical_path)
            if previous is not None and previous != content_hash:
                errors.add(
                    "untracked-content: conflicting hashes for " + canonical_path
                )
            hashes[canonical_path] = content_hash
            if not _is_sha256(content_hash):
                errors.add(
                    f"untracked-content:{canonical_path}: invalid SHA-256 digest"
                )
        object.__setattr__(self, "untracked_file_hashes", tuple(sorted(hashes.items())))
        object.__setattr__(
            self,
            "environment_roots",
            tuple(sorted({_canonical_path(path) for path in self.environment_roots})),
        )
        if not isinstance(self.head_sha, str) or not _GIT_OBJECT_RE.fullmatch(
            self.head_sha
        ):
            errors.add("HEAD: unavailable or invalid Git object ID")
        if not _is_sha256(self.staged_diff_sha256):
            errors.add("staged-diff: unavailable or invalid SHA-256 digest")
        if not _is_sha256(self.tracked_diff_sha256):
            errors.add("tracked-worktree: unavailable or invalid SHA-256 digest")
        if set(self.untracked_paths) != set(hashes):
            errors.add(
                "untracked-content: paths and content hashes do not describe "
                "the same files"
            )
        sorted_errors = tuple(sorted(errors))
        object.__setattr__(self, "inspection_errors", sorted_errors)
        if sorted_errors:
            object.__setattr__(self, "inspection_complete", False)

    @classmethod
    def capture(
        cls,
        repository: GitRepository | Path | str,
    ) -> WorkspaceSnapshot:
        return capture_workspace_snapshot(repository)

    @property
    def fingerprint(self) -> str:
        return _sha256(self.canonical_json().encode("utf-8"))

    @property
    def tracked_worktree_clean(self) -> bool:
        return self.inspection_complete and self.tracked_diff_sha256 == _EMPTY_SHA256

    @property
    def worktree_clean(self) -> bool:
        return self.tracked_worktree_clean and not self.untracked_paths

    @property
    def unstaged_worktree_clean(self) -> bool:
        return (
            self.inspection_complete
            and self.staged_diff_sha256 == self.tracked_diff_sha256
        )

    def canonical_data(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "environment_roots": list(self.environment_roots),
            "format": _WORKSPACE_SNAPSHOT_FORMAT,
            "head_sha": self.head_sha,
            "inspection_complete": self.inspection_complete,
            "inspection_errors": list(self.inspection_errors),
            "repository_path": self.repository_path.as_posix(),
            "schema_version": _WORKSPACE_SNAPSHOT_SCHEMA_VERSION,
            "staged_diff_sha256": self.staged_diff_sha256,
            "staged_paths": list(self.staged_paths),
            "tracked_diff_sha256": self.tracked_diff_sha256,
            "untracked_file_hashes": [
                [path, content_hash]
                for path, content_hash in self.untracked_file_hashes
            ],
            "untracked_paths": list(self.untracked_paths),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_data(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def compare(self, current: WorkspaceSnapshot) -> tuple[WorkspaceChange, ...]:
        return compare_workspace_snapshots(self, current)

    def matches(self, current: WorkspaceSnapshot) -> bool:
        return not compare_workspace_snapshots(self, current)

    def matches_fingerprint(self, expected_fingerprint: str) -> bool:
        return (
            self.inspection_complete
            and _is_sha256(expected_fingerprint)
            and self.fingerprint == expected_fingerprint
        )


def capture_workspace_snapshot(
    repository: GitRepository | Path | str,
) -> WorkspaceSnapshot:
    git_repository = (
        repository
        if isinstance(repository, GitRepository)
        else GitRepository(Path(repository))
    )
    errors: list[str] = []
    try:
        repository_path = git_repository.path.resolve(strict=False)
    except OSError as error:
        repository_path = git_repository.path.absolute()
        errors.append(f"repository-path: {error}")

    branch: str | None = None
    try:
        branch = git_repository.current_branch()
    except (OSError, RuntimeError, ValueError) as error:
        errors.append(f"branch: {error}")

    head_sha: str | None = None
    try:
        head_sha = git_repository.head_sha()
    except (OSError, RuntimeError, ValueError) as error:
        errors.append(f"HEAD: {error}")

    staged_paths: tuple[str, ...] = ()
    try:
        staged_paths = git_repository.staged_files()
    except (OSError, RuntimeError, ValueError) as error:
        errors.append(f"staging: {error}")

    staged_diff_sha256: str | None = None
    try:
        staged_diff_sha256 = _sha256(git_repository.staged_diff().encode("utf-8"))
    except (OSError, RuntimeError, UnicodeError, ValueError) as error:
        errors.append(f"staged-diff: {error}")

    tracked_diff_sha256: str | None = None
    try:
        tracked_diff_sha256 = _sha256(git_repository.tracked_diff().encode("utf-8"))
    except (OSError, RuntimeError, UnicodeError, ValueError) as error:
        errors.append(f"tracked-worktree: {error}")

    untracked_paths: tuple[str, ...] = ()
    untracked_file_hashes: tuple[tuple[str, str], ...] = ()
    try:
        untracked_paths = git_repository.untracked_files()
    except (OSError, RuntimeError, ValueError) as error:
        errors.append(f"untracked-paths: {error}")
    else:
        hashes: list[tuple[str, str]] = []
        for relative_path in sorted(untracked_paths):
            try:
                hashes.append(
                    (
                        relative_path,
                        _file_sha256(repository_path / relative_path),
                    )
                )
            except (OSError, ValueError) as error:
                errors.append(f"untracked-content:{relative_path}: {error}")
        untracked_file_hashes = tuple(hashes)

    environment_roots: tuple[str, ...] = ()
    try:
        environment_scan = capture_workspace_environment_snapshot(repository_path)
        environment_roots = tuple(
            _relative_path(environment.root_path, repository_path)
            for environment in environment_scan.environments
        )
    except (OSError, RuntimeError, ValueError) as error:
        errors.append(f"environment-roots: {error}")
    else:
        errors.extend(
            f"environment-roots: {error}"
            for error in environment_scan.inspection_errors
        )

    return WorkspaceSnapshot(
        repository_path=repository_path,
        branch=branch,
        head_sha=head_sha,
        staged_paths=staged_paths,
        staged_diff_sha256=staged_diff_sha256,
        tracked_diff_sha256=tracked_diff_sha256,
        untracked_paths=untracked_paths,
        untracked_file_hashes=untracked_file_hashes,
        environment_roots=environment_roots,
        inspection_complete=not errors,
        inspection_errors=tuple(errors),
    )


def compare_workspace_snapshots(
    expected: WorkspaceSnapshot,
    current: WorkspaceSnapshot,
) -> tuple[WorkspaceChange, ...]:
    changes: list[WorkspaceChange] = []
    if not expected.inspection_complete or not current.inspection_complete:
        changes.append(
            WorkspaceChange(
                name="inspection-incomplete",
                expected=_inspection_label(expected),
                actual=_inspection_label(current),
                message=(
                    "Workspace inspection was incomplete; unchanged state cannot "
                    "be established."
                ),
            )
        )
    if expected.repository_path != current.repository_path:
        changes.append(
            _change(
                "repository-path",
                str(expected.repository_path),
                str(current.repository_path),
                "Repository path changed.",
            )
        )
    if expected.branch != current.branch:
        changes.append(
            _change(
                "branch",
                _optional(expected.branch),
                _optional(current.branch),
                "Repository branch changed.",
            )
        )
    if expected.head_sha != current.head_sha:
        changes.append(
            _change(
                "HEAD",
                _optional(expected.head_sha),
                _optional(current.head_sha),
                "Repository HEAD changed.",
            )
        )
    if (
        expected.staged_paths != current.staged_paths
        or expected.staged_diff_sha256 != current.staged_diff_sha256
    ):
        changes.append(
            _change(
                "staging",
                _paths(expected.staged_paths),
                _paths(current.staged_paths),
                "Repository staging area changed.",
            )
        )
    if expected.tracked_diff_sha256 != current.tracked_diff_sha256:
        changes.append(
            _change(
                "tracked-diff",
                _optional(expected.tracked_diff_sha256),
                _optional(current.tracked_diff_sha256),
                "Tracked worktree diff changed.",
            )
        )
    if expected.untracked_paths != current.untracked_paths:
        changes.append(
            _change(
                "untracked-files",
                _paths(expected.untracked_paths),
                _paths(current.untracked_paths),
                "Untracked path set changed.",
            )
        )
    expected_hashes = dict(expected.untracked_file_hashes)
    current_hashes = dict(current.untracked_file_hashes)
    shared_untracked_paths = set(expected_hashes) & set(current_hashes)
    if any(
        expected_hashes[path] != current_hashes[path] for path in shared_untracked_paths
    ):
        changes.append(
            _change(
                "untracked-content",
                "unchanged",
                "changed",
                "Untracked file content changed.",
            )
        )
    if expected.environment_roots != current.environment_roots:
        changes.append(
            _change(
                "environment-roots",
                _paths(expected.environment_roots),
                _paths(current.environment_roots),
                "Detected local Python/Conda environment roots changed.",
            )
        )
    return tuple(changes)


def workspace_safety_changes(
    snapshot: WorkspaceSnapshot,
    *,
    expected_repository_path: Path | str,
    expected_branch: str | None,
    expected_head_sha: str,
    require_empty_staging: bool = True,
    require_clean_worktree: bool = False,
) -> tuple[WorkspaceChange, ...]:
    changes: list[WorkspaceChange] = []
    if not snapshot.inspection_complete:
        changes.append(
            WorkspaceChange(
                name="inspection-incomplete",
                expected="complete workspace inspection",
                actual=_inspection_label(snapshot),
                message="Workspace safety could not be established.",
            )
        )
    expected_path = Path(expected_repository_path).resolve(strict=False)
    if snapshot.repository_path != expected_path:
        changes.append(
            _change(
                "repository-path",
                str(expected_path),
                str(snapshot.repository_path),
                "Repository path no longer matches the recorded workspace.",
            )
        )
    if snapshot.branch != expected_branch:
        changes.append(
            _change(
                "branch",
                _optional(expected_branch),
                _optional(snapshot.branch),
                "Repository branch no longer matches the recorded workspace.",
            )
        )
    if snapshot.head_sha != expected_head_sha:
        changes.append(
            _change(
                "HEAD",
                expected_head_sha,
                _optional(snapshot.head_sha),
                "Repository HEAD no longer matches the recorded workspace.",
            )
        )
    if require_empty_staging and snapshot.staged_paths:
        changes.append(
            _change(
                "staging",
                "empty",
                _paths(snapshot.staged_paths),
                "Repository staging area is not empty.",
            )
        )
    if require_clean_worktree:
        if (
            snapshot.tracked_diff_sha256 is not None
            and snapshot.tracked_diff_sha256 != _EMPTY_SHA256
        ):
            changes.append(
                _change(
                    "tracked-diff",
                    _EMPTY_SHA256,
                    snapshot.tracked_diff_sha256,
                    "Tracked worktree is not clean.",
                )
            )
        if snapshot.untracked_paths:
            changes.append(
                _change(
                    "untracked-files",
                    "empty",
                    _paths(snapshot.untracked_paths),
                    "Untracked paths are present.",
                )
            )
    return tuple(changes)


def _workspace_fingerprint_path(patch_path: Path | str) -> Path:
    path = Path(patch_path)
    return path.with_suffix(".workspace.sha256")


def _write_workspace_fingerprint(
    path: Path | str,
    snapshot: WorkspaceSnapshot,
) -> None:
    if not snapshot.inspection_complete:
        raise _WorkspaceInspectionError(
            "Cannot persist an incomplete workspace fingerprint: "
            + "; ".join(snapshot.inspection_errors)
        )
    Path(path).write_text(snapshot.fingerprint + "\n", encoding="ascii", newline="\n")


def _read_workspace_fingerprint(path: Path | str) -> str:
    value = Path(path).read_text(encoding="ascii").strip()
    if not _SHA256_RE.fullmatch(value):
        raise ValueError("Workspace fingerprint must be a lowercase SHA-256 digest.")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _relative_path(path: Path, repository_path: Path) -> str:
    resolved_path = path.resolve(strict=False)
    try:
        return resolved_path.relative_to(repository_path).as_posix() or "."
    except ValueError:
        return _canonical_path(os.path.relpath(resolved_path, repository_path))


def _canonical_path(value: str) -> str:
    canonical = value.replace("\\", "/")
    while canonical.startswith("./"):
        canonical = canonical[2:]
    return canonical or "."


def _inspection_label(snapshot: WorkspaceSnapshot) -> str:
    if snapshot.inspection_complete:
        return "complete"
    if not snapshot.inspection_errors:
        return "incomplete"
    return "incomplete: " + "; ".join(snapshot.inspection_errors)


def _optional(value: str | None) -> str:
    return "<unknown>" if value is None else value


def _paths(values: tuple[str, ...]) -> str:
    return "empty" if not values else ", ".join(values)


def _change(
    name: str,
    expected: str,
    actual: str,
    message: str,
) -> WorkspaceChange:
    return WorkspaceChange(
        name=name,
        expected=expected,
        actual=actual,
        message=message,
    )


__all__ = [
    "WorkspaceChange",
    "WorkspaceSnapshot",
    "capture_workspace_snapshot",
    "compare_workspace_snapshots",
    "workspace_safety_changes",
]
