from __future__ import annotations

import pytest

from tests.helpers import (
    GIT,
    create_git_repo,
    make_config,
    prepend_executable_path,
    run_git,
    write_path_executable,
)
from ticket_automation.preflight import PreflightStatus, run_preflight


def check(result, name: str):
    return next(item for item in result.checks if item.name == name)


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for preflight tests"
)
def test_clean_feature_branch_passes(tmp_path):
    repo = create_git_repo(tmp_path / "repo")

    result = run_preflight(make_config(repo))

    assert result.passed
    assert check(result, "Branch").message == "feature/example"


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for preflight tests"
)
def test_dirty_working_tree_fails_without_modifying_repository(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    (repo / "file.txt").write_text("dirty\n", encoding="utf-8")
    status_before = run_git(repo, "status", "--short")

    result = run_preflight(make_config(repo))

    assert not result.passed
    assert check(result, "Working tree").status == PreflightStatus.FAIL
    assert "file.txt" in check(result, "Working tree").message
    assert run_git(repo, "status", "--short") == status_before


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for preflight tests"
)
def test_staged_files_fail(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    (repo / "file.txt").write_text("staged\n", encoding="utf-8")
    run_git(repo, "add", "file.txt")

    result = run_preflight(make_config(repo))

    assert not result.passed
    assert check(result, "Staging area").status == PreflightStatus.FAIL
    assert "file.txt" in check(result, "Staging area").message


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for preflight tests"
)
def test_protected_branch_fails(tmp_path):
    repo = create_git_repo(tmp_path / "repo", branch="main")

    result = run_preflight(make_config(repo, protected_branches=("main",)))

    assert not result.passed
    assert check(result, "Protected branch").status == PreflightStatus.FAIL
    assert "main" in check(result, "Protected branch").message


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for preflight tests"
)
def test_detached_head_fails(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    run_git(repo, "checkout", "--detach", "HEAD")

    result = run_preflight(make_config(repo))

    assert not result.passed
    assert check(result, "Branch").status == PreflightStatus.FAIL
    assert "detached HEAD" in check(result, "Branch").message


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for preflight tests"
)
def test_non_git_directory_fails(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    result = run_preflight(make_config(repo))

    assert not result.passed
    assert check(result, "Git repository").status == PreflightStatus.FAIL
    assert "not a Git working tree" in check(result, "Git repository").message


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for preflight tests"
)
def test_missing_codex_executable_fails_with_useful_reason(tmp_path):
    repo = create_git_repo(tmp_path / "repo")

    result = run_preflight(
        make_config(repo, codex_executable="ticket-automation-missing-codex")
    )

    assert not result.passed
    assert check(result, "Codex CLI").status == PreflightStatus.FAIL
    assert "ticket-automation-missing-codex" in check(result, "Codex CLI").message


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for preflight tests"
)
def test_codex_cli_check_accepts_path_resolved_bare_command(monkeypatch, tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    executable = write_path_executable(tmp_path / "tool dir")
    prepend_executable_path(monkeypatch, executable.parent)

    result = run_preflight(make_config(repo, codex_executable="codex"))

    assert result.passed
    assert check(result, "Codex CLI").status == PreflightStatus.PASS
