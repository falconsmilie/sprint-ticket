from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .process_output import run_human_text_command

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


@dataclass(frozen=True)
class GitCommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


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

    def status_short(self) -> str:
        return _git(self.path, ("status", "--short")).stdout

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

    def staged_diff(self) -> str:
        return _git(
            self.path,
            (
                "diff",
                "--cached",
                "--no-ext-diff",
                "--no-textconv",
                "--no-renames",
                "--binary",
                "--full-index",
                "HEAD",
                "--",
            ),
        ).stdout

    def tracked_diff(self) -> str:
        return _git(
            self.path,
            (
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--no-renames",
                "--binary",
                "--full-index",
                "HEAD",
                "--",
            ),
        ).stdout

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
    completed = run_human_text_command(command, cwd=repo_path)
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


__all__ = [
    "GitCommandError",
    "GitRepository",
]
