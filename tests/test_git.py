from __future__ import annotations

import pytest

import ticket_automation.git as git_module
from tests.helpers import GIT, create_git_repo, run_git
from ticket_automation.git import GitRepository
from ticket_automation.git_safety import WorkspaceSnapshot


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for Git inspection tests"
)
def test_detects_git_repository_branch_and_sha(tmp_path):
    repo = create_git_repo(tmp_path / "repo", branch="feature/example")
    repository = GitRepository(repo)
    nested = repo / "nested"
    nested.mkdir()

    assert repository.is_repository()
    assert GitRepository(nested).is_repository()
    assert repository.current_branch() == "feature/example"
    assert not repository.is_detached_head()
    assert repository.head_sha() == run_git(repository.path, "rev-parse", "HEAD")


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for Git inspection tests"
)
def test_changed_files_relative_to_baseline(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    repository = GitRepository(repo)
    baseline = repository.head_sha()
    (repository.path / "file.txt").write_text("changed\n", encoding="utf-8")
    (repository.path / "added.txt").write_text("added\n", encoding="utf-8")
    run_git(repository.path, "add", "file.txt", "added.txt")

    assert set(repository.changed_files(baseline)) == {"added.txt", "file.txt"}


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for Git inspection tests"
)
def test_diff_and_stats_relative_to_baseline(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    repository = GitRepository(repo)
    baseline = repository.head_sha()
    (repository.path / "file.txt").write_text("changed\n", encoding="utf-8")

    diff = repository.diff(baseline)
    stats = repository.diff_stats(baseline)

    assert "diff --git a/file.txt b/file.txt" in diff
    assert "+changed" in diff
    assert "file.txt" in stats


def test_api_does_not_expose_mutation_operations():
    forbidden_names = {
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
        "run_git",
    }

    assert not forbidden_names.intersection(dir(GitRepository))
    assert not forbidden_names.intersection(git_module.__all__)


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for Git inspection tests"
)
def test_safety_snapshot_detects_branch_head_and_staging_changes(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    repository = GitRepository(repo)
    snapshot = WorkspaceSnapshot.capture(repository)

    assert snapshot.compare(WorkspaceSnapshot.capture(repository)) == ()

    (repository.path / "file.txt").write_text("staged\n", encoding="utf-8")
    run_git(repository.path, "add", "file.txt")
    assert "staging" in {
        violation.name
        for violation in snapshot.compare(WorkspaceSnapshot.capture(repository))
    }

    run_git(repository.path, "commit", "-m", "change")
    assert "HEAD" in {
        violation.name
        for violation in snapshot.compare(WorkspaceSnapshot.capture(repository))
    }

    run_git(repository.path, "checkout", "-b", "another-branch")
    assert "branch" in {
        violation.name
        for violation in snapshot.compare(WorkspaceSnapshot.capture(repository))
    }
