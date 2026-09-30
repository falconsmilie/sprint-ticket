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

from ticket_automation.application.agent_execution import (
    AgentExecutorAssignments,
    AgentTaskKind,
)
from ticket_automation.composition import (
    prepare_production_agents,
    production_final_patch_capture,
)
from ticket_automation.composition.providers import RegisteredProviderExecutorFactory
from ticket_automation.config import (
    AgentSettings,
    AppConfig,
    ProjectSettings,
    RunnerSettings,
    VerificationCommand,
    VerificationSettings,
)
from ticket_automation.presentation.reporting import FilesystemTerminalReportPublisher
from ticket_automation.providers.codex_cli import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING_EFFORT,
    CodexCliAgentExecutor,
    CodexCliSettings,
    CodexProcessEvidence,
    CodexProcessResult,
)
from ticket_automation.providers.codex_cli.composition import (
    CodexCliProviderRegistration,
)
from ticket_automation.providers.codex_cli.identity import PROVIDER_ID


def completed_codex_process_result(
    payload: object,
    *,
    stdout: str = "",
    stderr: str = "",
    timeout_seconds: float = 60,
) -> CodexProcessResult:
    message = payload if isinstance(payload, str) else json.dumps(payload)
    return CodexProcessResult(
        0,
        stdout,
        stderr,
        evidence=CodexProcessEvidence(
            work_timeout_seconds=timeout_seconds,
            work_elapsed_seconds=1,
            total_elapsed_seconds=1,
            terminal_event_type="turn.completed",
            terminal_event_elapsed_seconds=1,
            completion_before_deadline=True,
            structured_message=message,
            structured_message_elapsed_seconds=0.9,
            structured_message_before_deadline=True,
            deadline_outcome="completed-before-deadline",
            finalization_outcome="completed",
            tree_termination_confirmed=True,
        ),
    )


def create_directory_link(link: Path, target: Path) -> str | None:
    """Create the platform directory-link type used by confinement tests."""

    if os.name == "nt":
        result = subprocess.run(
            ("cmd", "/c", "mklink", "/J", str(link), str(target)),
            check=False,
            capture_output=True,
            text=True,
        )
        return "junction" if result.returncode == 0 else None
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        return None
    return "symlink"


def remove_directory_link(link: Path) -> None:
    """Remove a link without traversing or deleting its target."""

    if os.name == "nt":
        link.rmdir()
    else:
        link.unlink()


if TYPE_CHECKING:
    import pytest

    from ticket_automation.runs import RunCreationResult
    from ticket_automation.verification import VerificationProcessRunner


def fail_stat_for_path(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    *,
    message: str = "simulated filesystem inspection failure",
) -> None:
    """Make the OS stat call for exactly one path fail without hiding the error."""

    original_stat = os.stat
    denied = os.path.normcase(os.path.abspath(path))

    def failing_stat(candidate, *args, **kwargs):
        try:
            inspected = os.path.normcase(os.path.abspath(os.fspath(candidate)))
        except TypeError:
            return original_stat(candidate, *args, **kwargs)
        if inspected == denied:
            raise PermissionError(message)
        return original_stat(candidate, *args, **kwargs)

    monkeypatch.setattr(os, "stat", failing_stat)


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


def review_result(*, corrections_required: bool = False) -> dict[str, object]:
    if corrections_required:
        return {
            "verdict": "CORRECTIONS_REQUIRED",
            "summary": "fake review requires one correction",
            "findings": [
                {
                    "id": "R1",
                    "disposition": "REQUIRED",
                    "scope_relation": "IMPLEMENTATION",
                    "title": "Correct the implementation",
                    "description": "The implementation needs a correction.",
                    "evidence": "The deterministic fake review found the defect.",
                    "required_change": "Apply the correction.",
                    "acceptance_criteria": ["The corrected verification passes."],
                }
            ],
        }
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

action_sequence_path = os.environ.get("TA_FAKE_CODEX_ACTION_SEQUENCE")
if action_sequence_path:
    path = pathlib.Path(action_sequence_path)
    actions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(actions, list) or not actions:
        sys.stderr.write("fake codex action sequence is exhausted\n")
        raise SystemExit(2)
    action = actions.pop(0)
    path.write_text(json.dumps(actions), encoding="utf-8")
else:
    action = os.environ.get("TA_FAKE_CODEX_ACTION", "modify")

if action == "fail":
    sys.stderr.write("fake codex failed\n")
    raise SystemExit(2)
if action == "fail-after-change":
    pathlib.Path("partial.txt").write_text("partial change\n", encoding="utf-8")
    sys.stderr.write("fake codex failed after changing the workspace\n")
    raise SystemExit(2)
if action == "missing-result":
    raise SystemExit(0)
if action == "malformed-result":
    output_path = pathlib.Path(
        sys.argv[sys.argv.index("--output-last-message") + 1]
    )
    output_path.write_text("{malformed", encoding="utf-8")
    raise SystemExit(0)

if "--sandbox" in sys.argv and sys.argv[sys.argv.index("--sandbox") + 1] == "read-only":
    result = review_result(corrections_required=action == "review-corrections")
    if action == "review-unsafe":
        result = review_result(corrections_required=True)
        result["findings"][0]["scope_relation"] = "REPOSITORY_AUTHORITY"
    if action == "review-inconsistent":
        result = review_result(corrections_required=True)
        result["verdict"] = "PASS"
    emit_result(result)
    if action == "review-pass-arm":
        arm_path = os.environ.get("TA_FAKE_CODEX_ARM_FILE")
        if not arm_path:
            sys.stderr.write("review-pass-arm requires TA_FAKE_CODEX_ARM_FILE\n")
            raise SystemExit(2)
        pathlib.Path(arm_path).write_text("armed\n", encoding="utf-8")
elif action in {"modify", "modify-correction"}:
    target = "file.txt" if action == "modify" else "correction.txt"
    pathlib.Path(target).write_text("implemented by fake codex\n", encoding="utf-8")
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
        executable.write_text(
            '@echo off\nif "%1"=="exec" echo --ephemeral\nexit /b 0\n',
            encoding="utf-8",
        )
        return executable

    executable = directory / name
    executable.write_text("#!/bin/sh\necho --ephemeral\nexit 0\n", encoding="utf-8")
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
        agents=AgentSettings(
            assignments={task_kind: PROVIDER_ID for task_kind in AgentTaskKind},
            providers={
                PROVIDER_ID: {
                    "executable": executable,
                    "model": codex_model,
                    "reasoning_effort": codex_reasoning_effort,
                }
            },
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


class _UnexpectedCodexRunner:
    def run(self, command, *, stdin, timeout_seconds, on_process_start=None):
        del command, stdin, timeout_seconds, on_process_start
        raise AssertionError("This test did not inject a Codex process result.")


def make_agent_executor(config: AppConfig, *, process_runner=None):
    """Compose the configured adapter for characterization tests."""

    raw = config.agents.providers[PROVIDER_ID]
    return CodexCliAgentExecutor(
        CodexCliSettings(
            executable=str(raw["executable"]),
            model=str(raw["model"]),
            reasoning_effort=str(raw["reasoning_effort"]),
        ),
        configuration_directory=config.configuration_directory,
        runner=process_runner or _UnexpectedCodexRunner(),
    )


def make_agent_executors(
    config: AppConfig, *, process_runner=None
) -> AgentExecutorAssignments:
    executor = make_agent_executor(config, process_runner=process_runner)
    return AgentExecutorAssignments(
        implementation=executor,
        review=executor,
        correction=executor,
    )


class _TestCodexRegistration(CodexCliProviderRegistration):
    def __init__(self, executor):
        self._executor = executor

    def create_executor(self, policy):
        self.decode_run_policy(self.encode_run_policy(policy))
        return self._executor


def make_resume_agent_executor_factory(agent_executor):
    """Build the executor factory used to restore a persisted test run."""

    return RegisteredProviderExecutorFactory(
        {PROVIDER_ID: _TestCodexRegistration(agent_executor)}
    )


def make_final_patch_capture():
    """Build the final-patch adapter used by lifecycle tests."""

    return production_final_patch_capture()


def make_report_publisher():
    """Build the filesystem report adapter used by lifecycle tests."""

    return FilesystemTerminalReportPublisher()


def run_test_stage(stage, phase, config, run_dir, **kwargs):
    """Execute one stage with the same controller-owned attempt protocol as production."""

    from ticket_automation.attempts import (
        AttemptMetadata,
        StageAttempt,
        complete_stage_attempt,
        start_attempt,
        update_attempt,
    )
    from ticket_automation.git import GitRepository
    from ticket_automation.git_safety import WorkspaceSnapshot
    from ticket_automation.models import AttemptPhase
    from ticket_automation.runs import RUN_RECORD_FILE, load_run_record

    run_path = Path(run_dir)
    before = None
    if phase in {
        AttemptPhase.PREPARING,
        AttemptPhase.VERIFYING,
        AttemptPhase.REVIEWING,
    }:
        record = load_run_record(run_path / RUN_RECORD_FILE)
        try:
            before = WorkspaceSnapshot.capture(
                GitRepository(Path(record.target_repository_path))
            ).fingerprint
        except (OSError, RuntimeError, ValueError):
            before = None
    clock = kwargs.get("clock")
    attempt = start_attempt(
        run_path,
        phase=phase,
        before_workspace_fingerprint=before,
        clock=clock,
    )
    if phase is AttemptPhase.REVIEWING:
        kwargs["mark_process_started"] = lambda: update_attempt(
            attempt, process_started=True
        )
    result = stage(
        config,
        run_path,
        attempt_record=StageAttempt.from_record(attempt),
        **kwargs,
    )
    complete_stage_attempt(
        run_path,
        attempt,
        stage_outcome=result.outcome,
        after_workspace_fingerprint=result.after_workspace_fingerprint,
        process_started=result.process_started,
        metadata=AttemptMetadata(controller_message=result.controller_message),
        clock=clock,
    )
    return result


def make_run_dependencies(
    config: AppConfig, *, process_runner=None, agent_executor=None
) -> dict[str, object]:
    """Construct the explicit provider dependencies required by a new run."""

    prepared = prepare_production_agents(config)
    executors = (
        AgentExecutorAssignments(
            implementation=agent_executor,
            review=agent_executor,
            correction=agent_executor,
        )
        if agent_executor is not None
        else make_agent_executors(config, process_runner=process_runner)
    )
    factory = make_resume_agent_executor_factory(executors.implementation)
    resolved_policy = prepared.resolve_run_policy(
        config,
        target_repository_path=config.project.repo,
    )
    return {
        "provider_preflight": prepared.run_preflight,
        "resolved_policy": resolved_policy,
        "agent_executor_factory": factory,
        "final_patch_capture": production_final_patch_capture(),
        "report_publisher": make_report_publisher(),
    }


def create_test_run_snapshot(
    config: AppConfig,
    ticket_path: Path | str,
    *,
    runs_dir: Path | str,
    clock: Callable[[], datetime] | None = None,
) -> RunCreationResult:
    from ticket_automation.runs import create_run_snapshot

    prepared = prepare_production_agents(config)
    resolved_policy = prepared.resolve_run_policy(
        config,
        target_repository_path=config.project.repo,
    )
    return create_run_snapshot(
        config,
        ticket_path,
        runs_dir=runs_dir,
        provider_preflight=prepared.run_preflight,
        resolved_policy=resolved_policy,
        clock=clock,
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
    from ticket_automation.models import AttemptPhase, WorkflowState
    from ticket_automation.runs import save_run_record
    from ticket_automation.verification import (
        VerificationProcessResult,
        run_baseline_verification_stage,
    )

    class PassingBaselineRunner:
        def run(self, command, *, timeout_seconds):
            del command, timeout_seconds
            return VerificationProcessResult(returncode=0, stdout="", stderr="")

    snapshot = create_test_run_snapshot(
        config,
        ticket_path,
        runs_dir=runs_dir,
        clock=clock,
    )
    verification = run_test_stage(
        run_baseline_verification_stage,
        AttemptPhase.PREPARING,
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
    model: str = DEFAULT_CODEX_MODEL,
    reasoning_effort: str = DEFAULT_CODEX_REASONING_EFFORT,
    max_correction_rounds: int = 1,
    verification_commands: tuple[VerificationCommand, ...] | None = None,
) -> None:
    executable = codex or str(write_fake_codex_executable(config_dir / "fake-bin"))
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
        "[agents.assignments]",
        'implementation = "codex-cli"',
        'review = "codex-cli"',
        'correction = "codex-cli"',
        "",
        "[agents.providers.codex-cli]",
        f"executable = {json.dumps(executable)}",
        f"model = {json.dumps(model)}",
        f"reasoning_effort = {json.dumps(reasoning_effort)}",
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
