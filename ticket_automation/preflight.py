from __future__ import annotations

import os
from pathlib import Path

from . import executable_resolution
from .application.ports.preflight import (
    PreflightCheck,
    PreflightResult,
    PreflightStatus,
)
from .config import AppConfig, VerificationCommand
from .git import GitCommandError, GitRepository
from .git_safety import WorkspaceSnapshot
from .locking import active_repository_locks, canonical_repository_identity


def _run_common_preflight(config: AppConfig) -> PreflightResult:
    """Run repository and verification checks shared by every provider."""
    checks: list[PreflightCheck] = []
    repo_path = config.project.repo

    if not repo_path.exists():
        return PreflightResult(
            (
                _fail(
                    "Repository",
                    f"Configured repository path does not exist: {repo_path}",
                ),
            )
        )
    checks.append(_pass("Repository"))

    repository = GitRepository(repo_path)
    try:
        is_repository = repository.is_repository()
    except OSError as error:
        return PreflightResult(
            (
                *checks,
                _fail("Git repository", f"Could not inspect repository path: {error}"),
            )
        )
    if not is_repository:
        return PreflightResult(
            (
                *checks,
                _fail("Git repository", f"Path is not a Git working tree: {repo_path}"),
            )
        )
    checks.append(_pass("Git repository"))
    _check_repository_lock(repo_path, checks)

    snapshot = WorkspaceSnapshot.capture(repository)
    if not snapshot.inspection_complete:
        checks.append(_fail("Git inspection", "; ".join(snapshot.inspection_errors)))
    else:
        _check_working_tree(repository, snapshot, checks)
        _check_staging_area(snapshot, checks)
        branch = _check_branch(snapshot, checks)
        if branch is not None:
            _check_protected_branch(branch, config.project.protected_branches, checks)

    for command in config.verification.commands:
        _check_verification_command(command, checks, cwd=repo_path)

    return PreflightResult(tuple(checks))


def run_preflight(
    config: AppConfig,
    *,
    provider_result: PreflightResult,
) -> PreflightResult:
    """Combine common checks with already-dispatched provider checks."""

    common = _run_common_preflight(config)
    return PreflightResult((*common.checks, *provider_result.checks))


def format_preflight_result(result: PreflightResult) -> str:
    rows = [_format_check(check) for check in result.checks]
    rows.append("")
    rows.append("PREFLIGHT PASSED" if result.passed else "PREFLIGHT FAILED")
    return "\n".join(rows)


def _check_working_tree(
    repository: GitRepository,
    snapshot: WorkspaceSnapshot,
    checks: list[PreflightCheck],
) -> None:
    if snapshot.unstaged_worktree_clean and not snapshot.untracked_paths:
        checks.append(_pass("Working tree"))
        return

    try:
        tracked_paths = repository.unstaged_files()
    except GitCommandError as error:
        checks.append(
            _fail(
                "Working tree",
                "Canonical workspace state is dirty; changed paths could not be "
                f"listed: {error}",
            )
        )
        return
    changed_paths = tuple(sorted((*tracked_paths, *snapshot.untracked_paths)))
    reason = (
        _format_file_reason(
            "Unstaged or untracked changes are present",
            changed_paths,
        )
        if changed_paths
        else "Canonical workspace state contains unstaged changes."
    )
    checks.append(_fail("Working tree", reason))


def _check_staging_area(
    snapshot: WorkspaceSnapshot, checks: list[PreflightCheck]
) -> None:
    if snapshot.staged_paths:
        checks.append(
            _fail(
                "Staging area",
                _format_file_reason(
                    "Staged files are present",
                    snapshot.staged_paths,
                ),
            )
        )
        return
    checks.append(_pass("Staging area"))


def _check_branch(
    snapshot: WorkspaceSnapshot, checks: list[PreflightCheck]
) -> str | None:
    branch = snapshot.branch
    if branch is None:
        checks.append(_fail("Branch", "Repository is in detached HEAD state."))
        return None
    checks.append(_pass("Branch", branch))
    return branch


def _check_protected_branch(
    branch: str,
    protected_branches: tuple[str, ...],
    checks: list[PreflightCheck],
) -> None:
    if branch in protected_branches:
        checks.append(
            _fail(
                "Protected branch",
                f"Current branch is protected: {branch}",
            )
        )
        return
    checks.append(_pass("Protected branch"))


def _check_verification_command(
    command: VerificationCommand,
    checks: list[PreflightCheck],
    *,
    cwd: Path,
) -> None:
    executable = command.argv[0]
    _check_executable(
        f"Verification {command.name}",
        executable,
        checks,
        config_dir=cwd,
    )


def _check_executable(
    name: str,
    executable: str,
    checks: list[PreflightCheck],
    *,
    config_dir: Path,
) -> None:
    if (
        executable_resolution.resolve_executable(
            executable,
            config_dir=config_dir,
        )
        is not None
    ):
        checks.append(_pass(name))
        return
    checks.append(_fail(name, f"Executable not found: {executable}"))


def _check_repository_lock(
    repo_path: Path,
    checks: list[PreflightCheck],
) -> None:
    identity = canonical_repository_identity(repo_path)
    conflicts = tuple(
        lock
        for lock in active_repository_locks()
        if lock.canonical_repository_identity == identity
        and lock.owner_pid != os.getpid()
    )
    if not conflicts:
        checks.append(_pass("Repository lock"))
        return
    assignments = ", ".join(
        f"{lock.run_id} ({lock.current_state or 'unknown state'})" for lock in conflicts
    )
    checks.append(
        _fail(
            "Repository lock",
            f"Repository is owned by active run(s): {assignments}",
        )
    )


def _pass(name: str, message: str = "") -> PreflightCheck:
    return PreflightCheck(name=name, status=PreflightStatus.PASS, message=message)


def _fail(name: str, message: str) -> PreflightCheck:
    return PreflightCheck(name=name, status=PreflightStatus.FAIL, message=message)


def _format_check(check: PreflightCheck) -> str:
    if check.passed and check.name == "Branch" and check.message:
        value = check.message
    elif check.passed:
        value = check.status.value
    else:
        value = f"{check.status.value} - {check.message}"
    return f"{check.name:<18} {value}"


def _format_file_reason(prefix: str, files: tuple[str, ...]) -> str:
    shown = ", ".join(files[:5])
    hidden_count = len(files) - 5
    if hidden_count > 0:
        shown = f"{shown}, and {hidden_count} more"
    return f"{prefix}: {shown}"


__all__ = [
    "format_preflight_result",
    "run_preflight",
]
