from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .config import AppConfig, VerificationCommand
from .git import GitCommandError, GitRepository


class PreflightStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: PreflightStatus
    message: str = ""

    @property
    def passed(self) -> bool:
        return self.status == PreflightStatus.PASS


@dataclass(frozen=True)
class PreflightResult:
    checks: tuple[PreflightCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> tuple[PreflightCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


def run_preflight(config: AppConfig) -> PreflightResult:
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

    try:
        _check_working_tree(repository, checks)
        _check_staging_area(repository, checks)
        branch = _check_branch(repository, checks)
        if branch is not None:
            _check_protected_branch(branch, config.project.protected_branches, checks)
    except GitCommandError as error:
        checks.append(_fail("Git inspection", str(error)))

    _check_executable("Codex CLI", config.codex.executable, checks, cwd=repo_path)
    for command in config.verification.commands:
        _check_verification_command(command, checks, cwd=repo_path)

    return PreflightResult(tuple(checks))


def format_preflight_result(result: PreflightResult) -> str:
    rows = [_format_check(check) for check in result.checks]
    rows.append("")
    rows.append("PREFLIGHT PASSED" if result.passed else "PREFLIGHT FAILED")
    return "\n".join(rows)


def _check_working_tree(
    repository: GitRepository, checks: list[PreflightCheck]
) -> None:
    changed_files = repository.unstaged_files()
    untracked_files = repository.untracked_files()
    if changed_files or untracked_files:
        checks.append(
            _fail(
                "Working tree",
                _format_file_reason(
                    "Unstaged or untracked changes are present",
                    (*changed_files, *untracked_files),
                ),
            )
        )
        return
    checks.append(_pass("Working tree"))


def _check_staging_area(
    repository: GitRepository, checks: list[PreflightCheck]
) -> None:
    staged_files = repository.staged_files()
    if staged_files:
        checks.append(
            _fail(
                "Staging area",
                _format_file_reason("Staged files are present", staged_files),
            )
        )
        return
    checks.append(_pass("Staging area"))


def _check_branch(
    repository: GitRepository, checks: list[PreflightCheck]
) -> str | None:
    branch = repository.current_branch()
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
    _check_executable(f"Verification {command.name}", executable, checks, cwd=cwd)


def _check_executable(
    name: str,
    executable: str,
    checks: list[PreflightCheck],
    *,
    cwd: Path,
) -> None:
    if _executable_exists(executable, cwd=cwd):
        checks.append(_pass(name))
        return
    checks.append(_fail(name, f"Executable not found: {executable}"))


def _executable_exists(executable: str, *, cwd: Path) -> bool:
    executable_path = Path(executable)
    has_path_part = executable_path.is_absolute() or any(
        separator in executable for separator in (os.sep, os.altsep) if separator
    )
    if has_path_part:
        candidate = (
            executable_path if executable_path.is_absolute() else cwd / executable_path
        )
        return candidate.is_file()
    return shutil.which(executable) is not None


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
    "PreflightCheck",
    "PreflightResult",
    "PreflightStatus",
    "format_preflight_result",
    "run_preflight",
]
