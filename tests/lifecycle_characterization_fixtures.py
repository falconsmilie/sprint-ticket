from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tests.helpers import create_git_repo, make_config
from ticket_automation.attempts import AttemptRecord, load_attempt_records
from ticket_automation.config import AppConfig
from ticket_automation.verification import VerificationProcessResult


@dataclass
class TickingClock:
    current: datetime = datetime(2026, 9, 14, 10, 15, tzinfo=UTC)
    step: timedelta = timedelta(seconds=1)

    def __call__(self) -> datetime:
        value = self.current
        self.current += self.step
        return value


@dataclass(frozen=True)
class LifecycleWorkspace:
    repository: Path
    ticket: Path
    runs_dir: Path
    config: AppConfig
    agent_executable: Path


def build_lifecycle_workspace(
    tmp_path: Path,
    *,
    max_correction_rounds: int = 1,
) -> LifecycleWorkspace:
    repository = create_git_repo(tmp_path / "target")
    ticket = tmp_path / "TA-FND-002.md"
    ticket.write_text("# Characterize lifecycle behavior\n", encoding="utf-8")
    config = make_config(
        repository,
        max_correction_rounds=max_correction_rounds,
    )
    return LifecycleWorkspace(
        repository=repository,
        ticket=ticket,
        runs_dir=tmp_path / "runs",
        config=config,
        agent_executable=Path(config.codex.executable),
    )


def configure_fake_codex_actions(
    monkeypatch,
    directory: Path,
    *actions: str,
) -> Path:
    action_path = directory / "fake-codex-actions.json"
    action_path.write_text(json.dumps(actions), encoding="utf-8")
    monkeypatch.setenv("TA_FAKE_CODEX_ACTION_SEQUENCE", str(action_path))
    return action_path


@dataclass
class ScriptedVerificationRunner:
    returncodes: list[int]
    calls: int = 0

    def run(self, command, *, timeout_seconds):
        del command, timeout_seconds
        if not self.returncodes:
            raise AssertionError("No scripted verification result remains.")
        self.calls += 1
        returncode = self.returncodes.pop(0)
        return VerificationProcessResult(
            returncode=returncode,
            stdout=(
                "verification passed\n"
                if returncode == 0
                else "verification failed\n"
            ),
            stderr="",
        )


def assert_attempt_ledger(
    run_dir: Path,
    expected: list[tuple[str, str]],
) -> tuple[AttemptRecord, ...]:
    attempts = load_attempt_records(run_dir)
    assert [(attempt.phase, attempt.status) for attempt in attempts] == expected
    assert [attempt.sequence for attempt in attempts] == list(
        range(1, len(attempts) + 1)
    )
    for attempt in attempts:
        assert attempt.path.parent == attempt.artifact_directory
        assert attempt.path.is_file()
        assert attempt.artifact_directory.parent == run_dir / "attempts"
        authoritative_artifacts: list[Path] = []
        if attempt.execution_path is not None:
            execution_path = attempt.artifact_directory / attempt.execution_path
            assert execution_path.is_relative_to(attempt.artifact_directory)
            assert execution_path.is_file()
            authoritative_artifacts.append(execution_path)
        if attempt.result_path is not None:
            result_path = attempt.artifact_directory / attempt.result_path
            assert result_path.is_relative_to(attempt.artifact_directory)
            if result_path.is_file():
                authoritative_artifacts.append(result_path)
        if attempt.status != "STARTED":
            assert authoritative_artifacts
    return attempts
