from __future__ import annotations

from collections.abc import Iterable
from os import devnull
from pathlib import Path

from .git import GitCommandError, GitRepository
from .process_output import TextProcessResult, run_human_text_command


def diff_including_untracked(repository: GitRepository, baseline_sha: str) -> str:
    parts = [repository.diff(baseline_sha).rstrip()]
    for file_path in repository.untracked_files():
        parts.append(
            git_no_index_diff(repository.path, file_path, stats=False).rstrip()
        )
    return _join_git_sections(parts)


def diff_stats_including_untracked(
    repository: GitRepository,
    baseline_sha: str,
) -> str:
    parts = [repository.diff_stats(baseline_sha).rstrip()]
    for file_path in repository.untracked_files():
        parts.append(git_no_index_diff(repository.path, file_path, stats=True).rstrip())
    return _join_git_sections(parts)


def changed_files_including_untracked(
    repository: GitRepository,
    baseline_sha: str,
) -> tuple[str, ...]:
    return _unique(
        (*repository.changed_files(baseline_sha), *repository.untracked_files())
    )


def git_no_index_diff(repo_path: Path, file_path: str, *, stats: bool) -> str:
    null_candidates = (
        ("/dev/null",) if devnull == "/dev/null" else ("/dev/null", devnull)
    )
    last_result: TextProcessResult | None = None
    for null_path in null_candidates:
        command = ["git", "diff", "--no-ext-diff", "--no-index"]
        if stats:
            command.append("--stat")
        command.extend(("--", null_path, file_path))
        result = run_human_text_command(command, cwd=repo_path)
        if not isinstance(result.stdout, str) or not isinstance(result.stderr, str):
            raise GitCommandError(
                "git diff --no-index did not return textual stdout/stderr."
            )
        if result.returncode in (0, 1):
            return result.stdout
        last_result = result
    assert last_result is not None
    raise _git_no_index_error(last_result)


def _git_no_index_error(result: TextProcessResult) -> GitCommandError:
    message = result.stderr.strip() or result.stdout.strip() or "no output"
    command = " ".join(str(argument) for argument in result.args)
    return GitCommandError(
        f"{command} failed with exit code {result.returncode}: {message}"
    )


def _join_git_sections(parts: Iterable[str]) -> str:
    content = "\n".join(part for part in parts if part)
    if not content:
        return ""
    return f"{content}\n"


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


__all__ = [
    "changed_files_including_untracked",
    "diff_including_untracked",
    "diff_stats_including_untracked",
    "git_no_index_diff",
]
