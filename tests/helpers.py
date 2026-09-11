from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ticket_automation.config import (
    AppConfig,
    CodexSettings,
    ProjectSettings,
    RunnerSettings,
    VerificationCommand,
    VerificationSettings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GIT = shutil.which("git")


def run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "ticket_automation", *args],
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def run_git(repo: Path, *args: str) -> str:
    assert GIT is not None
    result = subprocess.run(
        [GIT, *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def create_git_repo(repo: Path, *, branch: str = "feature/example") -> Path:
    repo.mkdir()
    run_git(repo, "init")
    run_git(repo, "config", "user.email", "ticket-automation@example.test")
    run_git(repo, "config", "user.name", "Ticket Automation")
    (repo / "file.txt").write_text("initial\n", encoding="utf-8")
    run_git(repo, "add", "file.txt")
    run_git(repo, "commit", "-m", "initial")
    if run_git(repo, "rev-parse", "--abbrev-ref", "HEAD") != branch:
        run_git(repo, "checkout", "-b", branch)
    return repo


def copy_example_config(config_dir: Path) -> Path:
    source = PROJECT_ROOT / "config.example.toml"
    destination = config_dir / source.name
    shutil.copyfile(source, destination)
    return destination


def make_config(
    repo: Path,
    *,
    protected_branches: tuple[str, ...] = ("main", "master"),
    codex_executable: str = sys.executable,
) -> AppConfig:
    executable = (
        Path(codex_executable).as_posix()
        if os.path.isabs(codex_executable)
        else codex_executable
    )
    return AppConfig(
        project=ProjectSettings(
            name="Example",
            repo=repo,
            protected_branches=protected_branches,
        ),
        runner=RunnerSettings(max_correction_rounds=1),
        codex=CodexSettings(
            executable=executable,
            implementation_sandbox="workspace-write",
            review_sandbox="read-only",
        ),
        verification=VerificationSettings(
            commands=(
                VerificationCommand(
                    name="python",
                    argv=(Path(sys.executable).as_posix(),),
                ),
            )
        ),
        source_files=(),
    )


def write_preflight_config(
    config_dir: Path,
    repo: Path,
    *,
    codex: str | None = None,
) -> None:
    executable = codex or Path(sys.executable).as_posix()
    config_dir.joinpath("config.example.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "Example"',
                f"repo = {json.dumps(repo.as_posix())}",
                'protected_branches = ["main", "master"]',
                "",
                "[runner]",
                "max_correction_rounds = 1",
                "",
                "[codex]",
                f"executable = {json.dumps(executable)}",
                'implementation_sandbox = "workspace-write"',
                'review_sandbox = "read-only"',
                "",
                "[[verification.commands]]",
                'name = "python"',
                f"argv = [{json.dumps(Path(sys.executable).as_posix())}]",
            ]
        ),
        encoding="utf-8",
    )
