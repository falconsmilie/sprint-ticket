from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

import ticket_automation.git_safety as git_safety_module
from tests.helpers import GIT, create_git_repo, make_config
from ticket_automation import workspace_guard as workspace_guard_module
from ticket_automation.codex import CodexCommand, CodexProcessResult
from ticket_automation.git import GitRepository
from ticket_automation.writable_attempts import load_writable_attempt
from ticket_automation.writable_worker import run_writable_codex

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_SCHEMA = PROJECT_ROOT / "schemas" / "implementation-result.schema.json"


@dataclass
class WritableRunner:
    mutation: Callable[[Path], None] | None = None
    calls: int = 0

    def run(
        self,
        command: CodexCommand,
        *,
        stdin: str,
        timeout_seconds: float | None,
    ) -> CodexProcessResult:
        del stdin, timeout_seconds
        self.calls += 1
        if self.mutation is not None:
            self.mutation(command.cwd)
        Path(command.argv[command.argv.index("--output-last-message") + 1]).write_text(
            json.dumps(_implementation_result()), encoding="utf-8"
        )
        return CodexProcessResult(
            returncode=0,
            stdout=_event_stream(_implementation_result()),
            stderr="",
        )


@pytest.mark.skipif(GIT is None, reason="git executable is required for worker tests")
def test_clean_writable_call_keeps_environment_evidence_in_execution_record(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    config = make_config(repo)
    runner = WritableRunner(
        mutation=lambda path: path.joinpath("file.txt").write_text(
            "implemented\n", encoding="utf-8"
        )
    )

    invocation = _run_worker(repo, tmp_path / "run", config, runner)

    assert invocation.execution is not None
    assert invocation.failure is None
    assert invocation.inspection_complete
    assert not invocation.workspace_guard.requires_human
    assert runner.calls == 1
    assert not (tmp_path / "run" / "workspace-guard" / "implementation.json").exists()
    attempt = json.loads(
        (tmp_path / "run" / "writable-attempts" / "implementation.json").read_text(
            encoding="utf-8"
        )
    )
    assert attempt["process_started"] is True
    assert attempt["before_snapshot"] is not None
    assert attempt["after_snapshot"] is not None
    assert attempt["environment_guard"]["new_environments"] == []


@pytest.mark.skipif(GIT is None, reason="git executable is required for worker tests")
def test_preexisting_conda_environment_is_trusted_and_execution_evidence_reloads(
    tmp_path,
):
    repo = create_git_repo(tmp_path / "repo")
    _create_conda_environment(repo / "project-conda")
    config = make_config(repo)
    runner = WritableRunner(
        mutation=lambda path: path.joinpath("file.txt").write_text(
            "implemented\n", encoding="utf-8"
        )
    )

    invocation = _run_worker(repo, tmp_path / "run", config, runner)

    assert invocation.execution is not None
    assert not invocation.workspace_guard.requires_human
    assert not (tmp_path / "run" / "workspace-guard" / "implementation.json").exists()
    attempt_path = tmp_path / "run" / "writable-attempts" / "implementation.json"
    reloaded = load_writable_attempt(attempt_path)
    assert reloaded is not None
    assert reloaded.environment_guard is not None
    assert [item.kind for item in reloaded.environment_guard.before.environments] == [
        "conda"
    ]
    assert reloaded.environment_guard.new_environments == ()


@pytest.mark.skipif(GIT is None, reason="git executable is required for worker tests")
def test_new_python_environment_requires_human_without_cleanup(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    config = make_config(repo)
    runner = WritableRunner(mutation=lambda path: _create_pyvenv(path / ".venv"))

    invocation = _run_worker(repo, tmp_path / "run", config, runner)

    assert invocation.workspace_guard.has_violation
    assert invocation.workspace_guard.requires_human
    assert repo.joinpath(".venv", "pyvenv.cfg").is_file()
    artifact = tmp_path / "run" / "workspace-guard" / "implementation.json"
    assert artifact.is_file()
    assert (
        json.loads(artifact.read_text(encoding="utf-8"))["new_environments"][0]["kind"]
        == "python_venv"
    )


@pytest.mark.skipif(GIT is None, reason="git executable is required for worker tests")
def test_new_conda_environment_requires_human_without_cleanup(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    config = make_config(repo)
    runner = WritableRunner(
        mutation=lambda path: _create_conda_environment(path / "conda")
    )

    invocation = _run_worker(repo, tmp_path / "run", config, runner)

    assert invocation.workspace_guard.has_violation
    assert invocation.workspace_guard.requires_human
    assert repo.joinpath("conda", "conda-meta", "history").is_file()
    artifact = tmp_path / "run" / "workspace-guard" / "implementation.json"
    assert (
        json.loads(artifact.read_text(encoding="utf-8"))["new_environments"][0]["kind"]
        == "conda"
    )


@pytest.mark.skipif(GIT is None, reason="git executable is required for worker tests")
def test_environment_scanner_failure_blocks_before_codex_and_writes_error_evidence(
    monkeypatch,
    tmp_path,
):
    repo = create_git_repo(tmp_path / "repo")
    config = make_config(repo)
    runner = WritableRunner()

    def fail_scandir(_path: Path):
        raise OSError("synthetic scanner failure")

    monkeypatch.setattr(workspace_guard_module, "_scandir", fail_scandir)
    invocation = _run_worker(repo, tmp_path / "run", config, runner)

    assert not invocation.invocation_permitted
    assert invocation.workspace_guard.has_inspection_failure
    assert invocation.workspace_guard.requires_human
    assert runner.calls == 0
    artifact = tmp_path / "run" / "workspace-guard" / "implementation.json"
    data = json.loads(artifact.read_text(encoding="utf-8"))
    assert "synthetic scanner failure" in " ".join(data["inspection_errors_before"])


@pytest.mark.skipif(GIT is None, reason="git executable is required for worker tests")
def test_canonical_snapshot_scanner_failure_is_guarded_and_persisted(
    monkeypatch,
    tmp_path,
):
    repo = create_git_repo(tmp_path / "repo")
    config = make_config(repo)
    runner = WritableRunner()

    def fail_canonical_environment_scan(_repository_path: Path):
        raise OSError("synthetic canonical scanner failure")

    monkeypatch.setattr(
        git_safety_module,
        "capture_workspace_environment_snapshot",
        fail_canonical_environment_scan,
    )
    invocation = _run_worker(repo, tmp_path / "run", config, runner)

    assert not invocation.invocation_permitted
    assert invocation.workspace_guard.has_inspection_failure
    assert runner.calls == 0
    artifact = tmp_path / "run" / "workspace-guard" / "implementation.json"
    data = json.loads(artifact.read_text(encoding="utf-8"))
    assert "canonical scanner failure" in " ".join(data["inspection_errors_before"])


def _run_worker(repo: Path, run_dir: Path, config, runner: WritableRunner):
    return run_writable_codex(
        repository=GitRepository(repo),
        run_dir=run_dir,
        operation="implementation",
        phase="IMPLEMENTING",
        prompt="implement the ticket",
        output_schema=IMPLEMENTATION_SCHEMA,
        artifact_directory=run_dir / "implementation",
        executable=config.codex.executable,
        execution_config=config.codex.execution,
        runner=runner,
    )


def _implementation_result() -> dict[str, object]:
    return {
        "status": "COMPLETED",
        "summary": "implemented",
        "tests_run": [],
        "assumptions": [],
        "known_issues": [],
    }


def _event_stream(result: dict[str, object]) -> str:
    return "\n".join(
        (
            json.dumps({"type": "turn.started"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "agent_message",
                        "text": json.dumps(result),
                    },
                }
            ),
            json.dumps({"type": "turn.completed"}),
            "",
        )
    )


def _create_pyvenv(path: Path) -> None:
    path.mkdir(parents=True)
    path.joinpath("pyvenv.cfg").write_text("home = python\n", encoding="utf-8")


def _create_conda_environment(path: Path) -> None:
    path.joinpath("conda-meta").mkdir(parents=True)
    path.joinpath("conda-meta", "history").write_text("", encoding="utf-8")
