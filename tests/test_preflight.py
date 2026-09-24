from __future__ import annotations

from types import SimpleNamespace

import pytest

import ticket_automation.preflight as preflight_module
from tests.helpers import (
    GIT,
    create_git_repo,
    make_config,
    prepend_executable_path,
    run_git,
    write_path_executable,
)
from ticket_automation.application.ports.preflight import PreflightStatus
from ticket_automation.composition import prepare_production_agents
from ticket_automation.git import GitRepository
from ticket_automation.locking import canonical_repository_identity
from ticket_automation.preflight import run_preflight as combine_preflight


def configured_preflight(config):
    providers = prepare_production_agents(config)
    return combine_preflight(
        config,
        provider_result=providers.run_preflight(repository_path=config.project.repo),
    )


def check(result, name: str):
    return next(item for item in result.checks if item.name == name)


def provider_check(result, suffix: str):
    return next(item for item in result.checks if item.name.endswith(f": {suffix}"))


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for preflight tests"
)
def test_clean_feature_branch_passes(tmp_path):
    repo = create_git_repo(tmp_path / "repo")

    result = configured_preflight(make_config(repo))

    assert result.passed
    assert check(result, "Branch").message == "feature/example"


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for preflight tests"
)
def test_dirty_working_tree_fails_without_modifying_repository(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    (repo / "file.txt").write_text("dirty\n", encoding="utf-8")
    status_before = run_git(repo, "status", "--short")

    result = configured_preflight(make_config(repo))

    assert not result.passed
    assert check(result, "Working tree").status == PreflightStatus.FAIL
    assert "file.txt" in check(result, "Working tree").message
    assert run_git(repo, "status", "--short") == status_before


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for preflight tests"
)
def test_dirty_decision_comes_from_canonical_snapshot(monkeypatch, tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    (repo / "file.txt").write_text("dirty\n", encoding="utf-8")
    monkeypatch.setattr(GitRepository, "unstaged_files", lambda self: ())

    result = configured_preflight(make_config(repo))

    assert not result.passed
    assert check(result, "Working tree").status == PreflightStatus.FAIL


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for preflight tests"
)
def test_staged_files_fail(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    (repo / "file.txt").write_text("staged\n", encoding="utf-8")
    run_git(repo, "add", "file.txt")

    result = configured_preflight(make_config(repo))

    assert not result.passed
    assert check(result, "Working tree").status == PreflightStatus.PASS
    assert check(result, "Staging area").status == PreflightStatus.FAIL
    assert "file.txt" in check(result, "Staging area").message


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for preflight tests"
)
def test_active_repository_lock_fails_common_preflight(tmp_path, monkeypatch):
    repo = create_git_repo(tmp_path / "repo")
    ownership = SimpleNamespace(
        canonical_repository_identity=canonical_repository_identity(repo),
        owner_pid=-1,
        run_id="active-run",
        current_state="IMPLEMENTING",
    )
    monkeypatch.setattr(
        preflight_module, "active_repository_locks", lambda: (ownership,)
    )

    result = configured_preflight(make_config(repo))

    assert not result.passed
    assert check(result, "Repository lock").status == PreflightStatus.FAIL
    assert "active-run" in check(result, "Repository lock").message


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for preflight tests"
)
def test_protected_branch_fails(tmp_path):
    repo = create_git_repo(tmp_path / "repo", branch="main")

    result = configured_preflight(make_config(repo, protected_branches=("main",)))

    assert not result.passed
    assert check(result, "Protected branch").status == PreflightStatus.FAIL
    assert "main" in check(result, "Protected branch").message


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for preflight tests"
)
def test_detached_head_fails(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    run_git(repo, "checkout", "--detach", "HEAD")

    result = configured_preflight(make_config(repo))

    assert not result.passed
    assert check(result, "Branch").status == PreflightStatus.FAIL
    assert "detached HEAD" in check(result, "Branch").message


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for preflight tests"
)
def test_non_git_directory_fails(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))

    result = configured_preflight(make_config(repo))

    assert not result.passed
    assert check(result, "Git repository").status == PreflightStatus.FAIL
    assert "not a Git working tree" in check(result, "Git repository").message


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for preflight tests"
)
def test_missing_codex_executable_fails_with_useful_reason(tmp_path):
    repo = create_git_repo(tmp_path / "repo")

    result = configured_preflight(
        make_config(repo, codex_executable="ticket-automation-missing-codex")
    )

    assert not result.passed
    assert provider_check(result, "executable").status == PreflightStatus.FAIL
    assert (
        "ticket-automation-missing-codex"
        in provider_check(result, "executable").message
    )


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for preflight tests"
)
def test_codex_cli_check_accepts_path_resolved_bare_command(monkeypatch, tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    executable = write_path_executable(tmp_path / "tool dir")
    prepend_executable_path(monkeypatch, executable.parent)

    result = configured_preflight(make_config(repo, codex_executable="codex"))

    assert result.passed
    assert provider_check(result, "executable").status == PreflightStatus.PASS


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for preflight tests"
)
def test_target_codex_project_configuration_is_rejected(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    project_config = repo / ".codex" / "config.toml"
    project_config.parent.mkdir()
    project_config.write_text("model = 'target-controlled'\n", encoding="utf-8")

    result = configured_preflight(make_config(repo))

    assert not result.passed
    assert (
        provider_check(result, "project configuration").status == PreflightStatus.FAIL
    )
    assert (
        ".codex/config.toml" in provider_check(result, "project configuration").message
    )
