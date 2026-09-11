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
FAKE_CODEX_SCRIPT = r'''
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys


def implementation_result(status: str) -> dict[str, object]:
    return {
        "status": status,
        "summary": "fake implementation result",
        "tests_run": [{"command": "fake validation", "result": "PASS"}],
        "assumptions": [],
        "known_issues": [] if status == "COMPLETED" else ["blocked by fake codex"],
    }


def emit_result(result: dict[str, object]) -> None:
    print(json.dumps({"type": "thread.started", "thread_id": "fake-thread"}))
    print(json.dumps({"type": "turn.started"}))
    print(
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "agent_message",
                    "text": json.dumps(result),
                },
            }
        )
    )
    print(json.dumps({"type": "turn.completed"}))


prompt = sys.stdin.read()
record_path = os.environ.get("TA_FAKE_CODEX_RECORD")
if record_path:
    pathlib.Path(record_path).write_text(
        json.dumps({"argv": sys.argv[1:], "prompt": prompt}, indent=2),
        encoding="utf-8",
    )

action = os.environ.get("TA_FAKE_CODEX_ACTION", "modify")
if action == "fail":
    sys.stderr.write("fake codex failed\n")
    raise SystemExit(2)

if action == "modify":
    pathlib.Path("file.txt").write_text("implemented by fake codex\n", encoding="utf-8")
    emit_result(implementation_result("COMPLETED"))
elif action == "untracked":
    pathlib.Path("added.txt").write_text("new file from fake codex\n", encoding="utf-8")
    emit_result(implementation_result("COMPLETED"))
elif action == "blocked":
    emit_result(implementation_result("BLOCKED"))
elif action == "no-change":
    emit_result(implementation_result("COMPLETED"))
elif action == "stage":
    pathlib.Path("file.txt").write_text("staged by fake codex\n", encoding="utf-8")
    subprocess.run(["git", "add", "file.txt"], check=True)
    emit_result(implementation_result("COMPLETED"))
elif action == "commit":
    pathlib.Path("file.txt").write_text("committed by fake codex\n", encoding="utf-8")
    subprocess.run(["git", "add", "file.txt"], check=True)
    subprocess.run(["git", "commit", "-m", "fake codex commit"], check=True)
    emit_result(implementation_result("COMPLETED"))
elif action == "branch":
    subprocess.run(["git", "checkout", "-b", "fake-codex-branch"], check=True)
    emit_result(implementation_result("COMPLETED"))
else:
    sys.stderr.write(f"unknown fake codex action: {action}\n")
    raise SystemExit(2)
'''


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


def write_fake_codex_executable(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "fake_codex.py"
    script.write_text(FAKE_CODEX_SCRIPT, encoding="utf-8")
    if os.name == "nt":
        launcher = directory / "fake-codex.cmd"
        launcher.write_text(
            f'@echo off\n"{sys.executable}" "{script}" %*\n',
            encoding="utf-8",
        )
        return launcher

    launcher = directory / "fake-codex"
    launcher.write_text(f"#!{sys.executable}\n{FAKE_CODEX_SCRIPT}", encoding="utf-8")
    launcher.chmod(0o755)
    return launcher


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
    max_correction_rounds: int = 1,
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
        runner=RunnerSettings(max_correction_rounds=max_correction_rounds),
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
    max_correction_rounds: int = 1,
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
                f"max_correction_rounds = {max_correction_rounds}",
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
