from __future__ import annotations

import json
import sys

import pytest

from tests.helpers import (
    GIT,
    copy_example_config,
    create_git_repo,
    run_cli,
    run_git,
    write_fake_codex_executable,
    write_preflight_config,
)
from ticket_automation.config import VerificationCommand


def test_cli_help_succeeds(tmp_path):
    result = run_cli("--help", cwd=tmp_path)

    assert result.returncode == 0
    assert "config" in result.stdout
    assert "preflight" in result.stdout
    assert "run" in result.stdout
    assert "status" in result.stdout


def test_cli_config_output(tmp_path):
    copy_example_config(tmp_path)

    result = run_cli("config", cwd=tmp_path)

    assert result.returncode == 0
    assert "TicketAutomation configuration" in result.stdout
    assert "PhosPy" in result.stdout
    assert "C:\\Projects\\phospy" in result.stdout
    assert "tests (1800s): python -m pytest" in result.stdout


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


@pytest.mark.skipif(GIT is None, reason="git executable is required for run CLI tests")
def test_cli_run_invokes_fake_codex_and_runs_verification(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "QDEB-003.md"
    ticket.write_text("# Ticket\n\nDo the thing.\n", encoding="utf-8")
    fake_codex = write_fake_codex_executable(tmp_path / "fake-bin")
    record_path = tmp_path / "fake-codex-record.json"
    config_dir.mkdir()
    write_preflight_config(config_dir, repo, codex=str(fake_codex))
    monkeypatch.setenv("TA_FAKE_CODEX_ACTION", "modify")
    monkeypatch.setenv("TA_FAKE_CODEX_RECORD", str(record_path))

    result = run_cli("--config-dir", str(config_dir), "run", str(ticket), cwd=config_dir)

    assert result.returncode == 0
    assert "Snapshot created for run" in result.stdout
    assert "Implementation state: IMPLEMENT" in result.stdout
    assert "Verification state: VERIFY" in result.stdout
    assert "No review has been attempted." in result.stdout
    assert run_git(repo, "diff", "--name-only") == "file.txt"
    fake_record = json.loads(record_path.read_text(encoding="utf-8"))
    assert sandbox_value(tuple(fake_record["argv"])) == "workspace-write"


@pytest.mark.skipif(GIT is None, reason="git executable is required for run CLI tests")
def test_cli_verification_failure_overrides_agent_claimed_tests(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "QDEB-003.md"
    ticket.write_text("# Ticket\n\nDo the thing.\n", encoding="utf-8")
    fake_codex = write_fake_codex_executable(tmp_path / "fake-bin")
    config_dir.mkdir()
    write_preflight_config(
        config_dir,
        repo,
        codex=str(fake_codex),
        verification_commands=(
            VerificationCommand(
                name="runner-gate",
                argv=(
                    sys.executable,
                    "-c",
                    "import sys; print('runner failed'); raise SystemExit(7)",
                ),
                timeout_seconds=1800,
            ),
        ),
    )
    monkeypatch.setenv("TA_FAKE_CODEX_ACTION", "modify")

    result = run_cli("--config-dir", str(config_dir), "run", str(ticket), cwd=config_dir)

    assert result.returncode == 1
    assert "Implementation state: IMPLEMENT" in result.stdout
    assert "Verification state: CORRECT" in result.stdout
    assert "runner-gate: FAIL" in result.stdout


@pytest.mark.skipif(GIT is None, reason="git executable is required for status CLI tests")
def test_cli_status_reads_multiple_run_records(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    repo = create_git_repo(tmp_path / "repo")
    first_ticket = tmp_path / "QDEB-003.md"
    second_ticket = tmp_path / "QDEB-004.md"
    first_ticket.write_text("# First\n", encoding="utf-8")
    second_ticket.write_text("# Second\n", encoding="utf-8")
    fake_codex = write_fake_codex_executable(tmp_path / "fake-bin")
    config_dir.mkdir()
    write_preflight_config(config_dir, repo, codex=str(fake_codex))
    monkeypatch.setenv("TA_FAKE_CODEX_ACTION", "no-change")

    first_result = run_cli(
        "--config-dir",
        str(config_dir),
        "run",
        str(first_ticket),
        cwd=config_dir,
    )
    second_result = run_cli(
        "--config-dir",
        str(config_dir),
        "run",
        str(second_ticket),
        cwd=config_dir,
    )
    status_result = run_cli("--config-dir", str(config_dir), "status", cwd=config_dir)

    assert first_result.returncode == 1
    assert second_result.returncode == 1
    assert status_result.returncode == 0
    assert "QDEB-003" in status_result.stdout
    assert "QDEB-004" in status_result.stdout
    assert "HUMAN_REQUIRED" in status_result.stdout
    assert "feature/example" in status_result.stdout


def sandbox_value(argv: tuple[str, ...]) -> str:
    sandbox_index = argv.index("--sandbox")
    return argv[sandbox_index + 1]
