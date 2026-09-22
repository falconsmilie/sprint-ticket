from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tests.helpers import create_git_repo, make_config
from ticket_automation.attempts import AttemptRecord, load_attempt_records
from ticket_automation.config import AppConfig
from ticket_automation.providers.codex_cli import CodexProcessResult
from ticket_automation.providers.codex_cli.identity import PROVIDER_ID
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
        agent_executable=Path(str(config.agents.providers[PROVIDER_ID]["executable"])),
    )


def configure_fake_codex_actions(
    monkeypatch,
    directory: Path,
    *actions: str,
) -> ScriptedCodexRunner:
    action_path = directory / "fake-codex-actions.json"
    action_path.write_text(json.dumps(actions), encoding="utf-8")
    monkeypatch.setenv("TA_FAKE_CODEX_ACTION_SEQUENCE", str(action_path))
    return ScriptedCodexRunner(action_path)


@dataclass(frozen=True)
class ScriptedCodexRunner:
    action_path: Path

    def run(self, command, *, stdin, timeout_seconds):
        del stdin, timeout_seconds
        actions = json.loads(self.action_path.read_text(encoding="utf-8"))
        if not isinstance(actions, list) or not actions:
            raise AssertionError("No scripted Codex result remains.")
        action = actions.pop(0)
        self.action_path.write_text(json.dumps(actions), encoding="utf-8")

        if action == "fail-after-change":
            command.cwd.joinpath("partial.txt").write_text(
                "partial change\n", encoding="utf-8"
            )
            return CodexProcessResult(2, "", "fake codex failed\n")
        if action == "fail":
            return CodexProcessResult(2, "", "fake codex failed\n")
        if action == "missing-result":
            return CodexProcessResult(0, "", "")

        output_path = Path(
            command.argv[command.argv.index("--output-last-message") + 1]
        )
        if action == "malformed-result":
            output_path.write_text("{malformed", encoding="utf-8")
            return CodexProcessResult(0, "", "")

        read_only = command.argv[command.argv.index("--sandbox") + 1] == "read-only"
        if read_only:
            result = _scripted_review_result(action)
            if action == "review-pass-arm":
                arm_path = os.environ.get("TA_FAKE_CODEX_ARM_FILE")
                if not arm_path:
                    raise AssertionError("review-pass-arm requires an arm file.")
                Path(arm_path).write_text("armed\n", encoding="utf-8")
        else:
            result = _scripted_implementation_result(action)
            if action in {"modify", "modify-correction"}:
                target = "file.txt" if action == "modify" else "correction.txt"
                command.cwd.joinpath(target).write_text(
                    "implemented by fake codex\n", encoding="utf-8"
                )
        output_path.write_text(json.dumps(result), encoding="utf-8")
        return CodexProcessResult(0, "", "")


def _scripted_implementation_result(action: str) -> dict[str, object]:
    status = "BLOCKED" if action == "blocked" else "COMPLETED"
    return {
        "status": status,
        "summary": "fake implementation result",
        "tests_run": [{"command": "fake validation", "result": "PASS"}],
        "assumptions": [],
        "known_issues": [] if status == "COMPLETED" else ["blocked by fake codex"],
    }


def _scripted_review_result(action: str) -> dict[str, object]:
    corrections_required = action in {
        "review-corrections",
        "review-unsafe",
        "review-inconsistent",
    }
    result: dict[str, object] = {
        "verdict": "CORRECTIONS_REQUIRED" if corrections_required else "PASS",
        "summary": "fake review result",
        "findings": [],
    }
    if corrections_required:
        result["findings"] = [
            {
                "id": "R1",
                "disposition": "REQUIRED",
                "scope_relation": (
                    "REPOSITORY_AUTHORITY"
                    if action == "review-unsafe"
                    else "IMPLEMENTATION"
                ),
                "title": "Correct the implementation",
                "description": "The implementation needs a correction.",
                "evidence": "The deterministic fake review found the defect.",
                "required_change": "Apply the correction.",
                "acceptance_criteria": ["The corrected verification passes."],
            }
        ]
    if action == "review-inconsistent":
        result["verdict"] = "PASS"
    return result


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
                "verification passed\n" if returncode == 0 else "verification failed\n"
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
