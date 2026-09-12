from __future__ import annotations

import pytest

from tests.helpers import GIT, create_git_repo
from ticket_automation import audit as audit_module
from ticket_automation.audit import (
    diff_including_untracked,
    diff_stats_including_untracked,
    git_no_index_diff,
)
from ticket_automation.git import GitCommandError, GitRepository
from ticket_automation.process_output import TextProcessResult


@pytest.mark.skipif(GIT is None, reason="git executable is required for audit tests")
def test_unicode_untracked_file_is_represented_in_baseline_diff(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    repository = GitRepository(repo)
    baseline = repository.head_sha()
    contents = "kinase α/β signal č - en dash – em dash — José 😀\n"
    repo.joinpath("unicode.txt").write_text(contents, encoding="utf-8")

    patch = diff_including_untracked(repository, baseline)
    stats = diff_stats_including_untracked(repository, baseline)

    assert isinstance(patch, str)
    assert "diff --git a/unicode.txt b/unicode.txt" in patch
    assert f"+{contents.rstrip()}" in patch
    assert "unicode.txt" in stats


@pytest.mark.skipif(GIT is None, reason="git executable is required for audit tests")
def test_binary_untracked_file_has_safe_diff_representation(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    repository = GitRepository(repo)
    baseline = repository.head_sha()
    repo.joinpath("payload.bin").write_bytes(b"\x00\x8d\xffbinary\x00payload")

    patch = diff_including_untracked(repository, baseline)
    stats = diff_stats_including_untracked(repository, baseline)

    assert isinstance(patch, str)
    assert "payload.bin" in patch
    assert "payload.bin" in stats


def test_no_index_diff_raises_controlled_error_instead_of_returning_none(
    monkeypatch, tmp_path
):
    def fake_run(command, *, cwd):
        del command, cwd
        return TextProcessResult(
            args=("git", "diff", "--no-index"),
            returncode=1,
            stdout=None,  # type: ignore[arg-type]
            stderr="",
        )

    monkeypatch.setattr(audit_module, "run_human_text_command", fake_run)

    with pytest.raises(GitCommandError, match="textual stdout/stderr"):
        git_no_index_diff(tmp_path, "file.txt", stats=False)
