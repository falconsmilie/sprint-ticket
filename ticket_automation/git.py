from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_MUTATING_SUBCOMMANDS = frozenset(
    {
        "add",
        "commit",
        "reset",
        "checkout",
        "switch",
        "stash",
        "clean",
        "push",
        "pull",
        "merge",
        "rebase",
        "revert",
    }
)
_SHA_RE = re.compile(r"[0-9a-fA-F]{4,64}")


class GitCommandError(RuntimeError):
    """Raised when a read-only Git inspection command fails."""


class GitStateChangedError(RuntimeError):
    """Raised when a repository safety snapshot no longer matches."""


@dataclass(frozen=True)
class GitCommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class GitSafetyViolation:
    name: str
    expected: str
    actual: str
    message: str


@dataclass(frozen=True)
class GitSafetySnapshot:
    branch: str | None
    head_sha: str
    has_staged_files: bool

    @classmethod
    def capture(cls, repository: GitRepository) -> GitSafetySnapshot:
        return cls(
            branch=repository.current_branch(),
            head_sha=repository.head_sha(),
            has_staged_files=repository.has_staged_files(),
        )

    def compare(self, repository: GitRepository) -> tuple[GitSafetyViolation, ...]:
        current = GitSafetySnapshot.capture(repository)
        violations: list[GitSafetyViolation] = []

        if self.branch != current.branch:
            violations.append(
                GitSafetyViolation(
                    name="branch",
                    expected=_format_optional(self.branch),
                    actual=_format_optional(current.branch),
                    message="Repository branch changed after the safety snapshot.",
                )
            )
        if self.head_sha != current.head_sha:
            violations.append(
                GitSafetyViolation(
                    name="HEAD",
                    expected=self.head_sha,
                    actual=current.head_sha,
                    message="Repository HEAD changed after the safety snapshot.",
                )
            )
        if self.has_staged_files != current.has_staged_files:
            violations.append(
                GitSafetyViolation(
                    name="staging",
                    expected=str(self.has_staged_files),
                    actual=str(current.has_staged_files),
                    message="Repository staging state changed after the safety snapshot.",
                )
            )

        return tuple(violations)

    def assert_matches(self, repository: GitRepository) -> None:
        violations = self.compare(repository)
        if violations:
            details = "; ".join(violation.message for violation in violations)
            raise GitStateChangedError(details)


class GitRepository:
    def __init__(self, path: Path | str):
        self.path = Path(path)

    def is_repository(self) -> bool:
        if not self.path.is_dir():
            return False
        result = _git(self.path, ("rev-parse", "--is-inside-work-tree"), check=False)
        return result.returncode == 0 and result.stdout.strip() == "true"

    def current_branch(self) -> str | None:
        result = _git(
            self.path, ("symbolic-ref", "--quiet", "--short", "HEAD"), check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
        if result.returncode == 1:
            return None
        raise _command_error(result)

    def head_sha(self) -> str:
        return _git(self.path, ("rev-parse", "--verify", "HEAD")).stdout.strip()

    def is_detached_head(self) -> bool:
        return self.current_branch() is None

    def is_working_tree_clean(self) -> bool:
        return (
            not self.has_unstaged_changes()
            and not self.has_staged_files()
            and not self.untracked_files()
        )

    def has_unstaged_changes(self) -> bool:
        result = _git(self.path, ("diff", "--quiet", "--"), check=False)
        if result.returncode == 0:
            return False
        if result.returncode == 1:
            return True
        raise _command_error(result)

    def unstaged_files(self) -> tuple[str, ...]:
        output = _git(self.path, ("diff", "--name-only", "--")).stdout
        return _split_lines(output)

    def untracked_files(self) -> tuple[str, ...]:
        output = _git(self.path, ("ls-files", "--others", "--exclude-standard")).stdout
        return _split_lines(output)

    def has_staged_files(self) -> bool:
        result = _git(self.path, ("diff", "--cached", "--quiet", "--"), check=False)
        if result.returncode == 0:
            return False
        if result.returncode == 1:
            return True
        raise _command_error(result)

    def staged_files(self) -> tuple[str, ...]:
        output = _git(self.path, ("diff", "--cached", "--name-only", "--")).stdout
        return _split_lines(output)

    def changed_files(self, baseline_sha: str) -> tuple[str, ...]:
        baseline = _validate_baseline_sha(baseline_sha)
        output = _git(self.path, ("diff", "--name-only", baseline, "--")).stdout
        return _split_lines(output)

    def diff(self, baseline_sha: str) -> str:
        baseline = _validate_baseline_sha(baseline_sha)
        return _git(self.path, ("diff", "--no-ext-diff", baseline, "--")).stdout

    def diff_stats(self, baseline_sha: str) -> str:
        baseline = _validate_baseline_sha(baseline_sha)
        return _git(self.path, ("diff", "--stat", baseline, "--")).stdout


def _git(
    repo_path: Path, args: Iterable[str], *, check: bool = True
) -> GitCommandResult:
    argv = tuple(args)
    if not argv:
        raise ValueError("Git inspection command cannot be empty.")
    if argv[0] in _MUTATING_SUBCOMMANDS:
        raise ValueError(
            f"Git subcommand is not available in read-only V1 support: {argv[0]}"
        )

    command = ("git", *argv)
    completed = subprocess.run(
        command,
        cwd=repo_path,
        check=False,
        capture_output=True,
        text=True,
    )
    result = GitCommandResult(
        argv=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if check and result.returncode != 0:
        raise _command_error(result)
    return result


def _command_error(result: GitCommandResult) -> GitCommandError:
    message = result.stderr.strip() or result.stdout.strip() or "no output"
    command = " ".join(result.argv)
    return GitCommandError(
        f"{command} failed with exit code {result.returncode}: {message}"
    )


def _split_lines(output: str) -> tuple[str, ...]:
    return tuple(line for line in output.splitlines() if line)


def _validate_baseline_sha(baseline_sha: str) -> str:
    if not _SHA_RE.fullmatch(baseline_sha):
        raise ValueError("baseline_sha must be a Git object SHA.")
    return baseline_sha


def _format_optional(value: str | None) -> str:
    return "<none>" if value is None else value


__all__ = [
    "GitCommandError",
    "GitRepository",
    "GitSafetySnapshot",
    "GitSafetyViolation",
    "GitStateChangedError",
]
