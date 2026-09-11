from __future__ import annotations

import pytest

from tests.helpers import (
    GIT,
    copy_example_config,
    create_git_repo,
    run_cli,
    write_preflight_config,
)


def test_cli_help_succeeds(tmp_path):
    result = run_cli("--help", cwd=tmp_path)

    assert result.returncode == 0
    assert "config" in result.stdout
    assert "preflight" in result.stdout


def test_cli_config_output(tmp_path):
    copy_example_config(tmp_path)

    result = run_cli("config", cwd=tmp_path)

    assert result.returncode == 0
    assert "TicketAutomation configuration" in result.stdout
    assert "PhosPy" in result.stdout
    assert "C:\\Projects\\phospy" in result.stdout
    assert "tests: python -m pytest" in result.stdout


@pytest.mark.skipif(GIT is None, reason="git executable is required for preflight CLI tests")
def test_cli_preflight_output_and_exit_code(tmp_path):
    config_dir = tmp_path / "config"
    repo = create_git_repo(tmp_path / "repo")
    config_dir.mkdir()
    write_preflight_config(config_dir, repo)

    result = run_cli("--config-dir", str(config_dir), "preflight", cwd=config_dir)

    assert result.returncode == 0
    assert "Repository" in result.stdout
    assert "Branch" in result.stdout
    assert "feature/example" in result.stdout
    assert "PREFLIGHT PASSED" in result.stdout


@pytest.mark.skipif(GIT is None, reason="git executable is required for preflight CLI tests")
def test_cli_preflight_failure_returns_nonzero(tmp_path):
    config_dir = tmp_path / "config"
    repo = create_git_repo(tmp_path / "repo")
    config_dir.mkdir()
    write_preflight_config(config_dir, repo, codex="ticket-automation-missing-codex")

    result = run_cli("--config-dir", str(config_dir), "preflight", cwd=config_dir)

    assert result.returncode == 1
    assert "Codex CLI" in result.stdout
    assert "ticket-automation-missing-codex" in result.stdout
    assert "PREFLIGHT FAILED" in result.stdout
