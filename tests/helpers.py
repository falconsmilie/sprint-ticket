from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ticket_automation.config import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING_EFFORT,
    AppConfig,
    CodexSettings,
    ProjectSettings,
    RunnerSettings,
    VerificationCommand,
    VerificationSettings,
)

if TYPE_CHECKING:
    from ticket_automation.runs import RunCreationResult
    from ticket_automation.verification import VerificationProcessRunner

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GIT = shutil.which("git")
FAKE_CODEX_SCRIPT = r"""
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


def review_result() -> dict[str, object]:
    return {
        "verdict": "PASS",
        "summary": "fake review accepted the implementation",
        "findings": [],
    }


def emit_result(result: dict[str, object]) -> None:
    if "--output-last-message" in sys.argv:
        output_path = pathlib.Path(
            sys.argv[sys.argv.index("--output-last-message") + 1]
        )
        output_path.write_text(json.dumps(result), encoding="utf-8")
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


if "--version" in sys.argv:
    print("fake-codex 1.0")
    raise SystemExit(0)

if sys.argv[1:] == ["exec", "--help"]:
    print("--ephemeral  Run without persisting session files to disk")
    raise SystemExit(0)

prompt = sys.stdin.read()
record_path = os.environ.get("TA_FAKE_CODEX_RECORD")
if record_path:
    path = pathlib.Path(record_path)
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
    else:
        record = {"calls": []}
    record["calls"].append({"argv": sys.argv[1:], "prompt": prompt})
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")

action = os.environ.get("TA_FAKE_CODEX_ACTION", "modify")
if action == "fail":
    sys.stderr.write("fake codex failed\n")
    raise SystemExit(2)

if "--sandbox" in sys.argv and sys.argv[sys.argv.index("--sandbox") + 1] == "read-only":
    emit_result(review_result())
elif action == "modify":
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
"""


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


def write_path_executable(directory: Path, *, name: str = "codex") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        executable = directory / f"{name}.CMD"
        executable.write_text("@echo off\nexit /b 0\n", encoding="utf-8")
        return executable

    executable = directory / name
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    return executable


def prepend_executable_path(monkeypatch, directory: Path) -> None:
    existing_path = os.environ.get("PATH")
    if existing_path:
        monkeypatch.setenv("PATH", f"{directory}{os.pathsep}{existing_path}")
    else:
        monkeypatch.setenv("PATH", str(directory))
    if os.name == "nt":
        monkeypatch.setenv("PATHEXT", ".CMD;.EXE;.BAT")


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
    codex_executable: str | None = None,
    codex_model: str = DEFAULT_CODEX_MODEL,
    codex_reasoning_effort: str = DEFAULT_CODEX_REASONING_EFFORT,
    max_correction_rounds: int = 1,
    verification_commands: tuple[VerificationCommand, ...] | None = None,
) -> AppConfig:
    if codex_executable is None:
        codex_executable = str(
            write_fake_codex_executable(repo.parent / "fake-codex-bin")
        )
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
            model=codex_model,
            reasoning_effort=codex_reasoning_effort,
        ),
        verification=VerificationSettings(
            commands=verification_commands
            or (
                VerificationCommand(
                    name="python",
                    argv=(Path(sys.executable).as_posix(), "-c", "raise SystemExit(0)"),
                    timeout_seconds=1800,
                ),
            ),
        ),
        source_files=(),
        configuration_directory=repo.parent,
    )


def create_trusted_prepared_run(
    config: AppConfig,
    ticket_path: Path | str,
    *,
    runs_dir: Path | str,
    verification_runner: VerificationProcessRunner | None = None,
    clock: Callable[[], datetime] | None = None,
) -> RunCreationResult:
    """Build a PREPARED run through the real baseline-verification boundary."""
    from ticket_automation.models import WorkflowState
    from ticket_automation.runs import create_run_snapshot, save_run_record
    from ticket_automation.verification import (
        VerificationProcessResult,
        _run_baseline_verification_stage,
    )

    class PassingBaselineRunner:
        def run(self, command, *, timeout_seconds):
            del command, timeout_seconds
            return VerificationProcessResult(returncode=0, stdout="", stderr="")

    snapshot = create_run_snapshot(
        config,
        ticket_path,
        runs_dir=runs_dir,
        clock=clock,
    )
    verification = _run_baseline_verification_stage(
        config,
        snapshot.run_dir,
        process_runner=verification_runner or PassingBaselineRunner(),
        clock=clock,
    )
    if not verification.successful:
        raise AssertionError(verification.controller_message)
    prepared = verification.run_record.transition_to(
        WorkflowState.PREPARED,
        updated_timestamp=verification.round_result.ended_at,
    )
    save_run_record(prepared, snapshot.run_dir / "run.json")
    return replace(snapshot, run_record=prepared)


def write_preflight_config(
    config_dir: Path,
    repo: Path,
    *,
    codex: str | None = None,
    max_correction_rounds: int = 1,
    verification_commands: tuple[VerificationCommand, ...] | None = None,
) -> None:
    executable = codex or Path(sys.executable).as_posix()
    commands = verification_commands or (
        VerificationCommand(
            name="python",
            argv=(Path(sys.executable).as_posix(), "-c", "raise SystemExit(0)"),
            timeout_seconds=1800,
        ),
    )
    lines = [
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
        f"model = {json.dumps(DEFAULT_CODEX_MODEL)}",
        f"reasoning_effort = {json.dumps(DEFAULT_CODEX_REASONING_EFFORT)}",
    ]
    for command in commands:
        lines.extend(
            [
                "",
                "[[verification.commands]]",
                f"name = {json.dumps(command.name)}",
                f"argv = {json.dumps(list(command.argv))}",
                f"timeout_seconds = {command.timeout_seconds}",
            ]
        )
    config_dir.joinpath("config.local.toml").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
