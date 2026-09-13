from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.helpers import GIT, PROJECT_ROOT, create_git_repo, make_config
from ticket_automation.codex import CodexCommand, CodexProcessResult
from ticket_automation.config import VerificationCommand
from ticket_automation.locking import (
    RepositoryLockError,
    acquire_repository_run_lock,
    active_repository_locks,
)
from ticket_automation.models import WorkflowState
from ticket_automation.runs import RunPreflightError, create_run_snapshot
from ticket_automation.verification import (
    VerificationProcessCommand,
    VerificationProcessResult,
)
from ticket_automation.workflow import resume_ticket_lifecycle, run_ticket_lifecycle

LOCK_HOLDER_SCRIPT = """
from __future__ import annotations

import sys
from pathlib import Path

from ticket_automation.locking import acquire_repository_run_lock

target = Path(sys.argv[1])
run_id = sys.argv[2]
state = sys.argv[3]

with acquire_repository_run_lock(
    target,
    run_id=run_id,
    current_state=state,
):
    print("LOCKED", flush=True)
    sys.stdin.readline()
"""


COMPETING_LOCK_SCRIPT = """
from __future__ import annotations

import sys
import time
from pathlib import Path

from ticket_automation.locking import RepositoryLockError, acquire_repository_run_lock

target = Path(sys.argv[1])
barrier = Path(sys.argv[2])
deadline = time.monotonic() + 5
while not barrier.exists():
    if time.monotonic() > deadline:
        print("TIMEOUT", flush=True)
        raise SystemExit(3)
    time.sleep(0.005)

try:
    with acquire_repository_run_lock(
        target,
        run_id=sys.argv[3],
        current_state="IMPLEMENTING",
    ):
        print("ACQUIRED", flush=True)
        time.sleep(1.5)
except RepositoryLockError:
    print("BLOCKED", flush=True)
"""


def fixed_clock() -> datetime:
    return datetime(2026, 9, 11, 13, 5, 19, tzinfo=UTC)


@dataclass(frozen=True)
class CodexStep:
    result: dict[str, object] | None = None
    mutation: Callable[[Path], None] | None = None
    error: BaseException | None = None


@dataclass
class SequencedCodexRunner:
    steps: list[CodexStep]

    def run(
        self,
        command: CodexCommand,
        *,
        stdin: str,
        timeout_seconds: float | None,
    ) -> CodexProcessResult:
        del stdin, timeout_seconds
        assert self.steps, "Unexpected Codex invocation."
        step = self.steps.pop(0)
        if step.error is not None:
            raise step.error
        if step.mutation is not None:
            step.mutation(command.cwd)
        assert step.result is not None
        return CodexProcessResult(
            returncode=0,
            stdout=_event_stream(step.result),
            stderr="fake codex progress\n",
        )


@dataclass
class PassingVerificationRunner:
    calls: int = 0

    def run(
        self,
        command: VerificationProcessCommand,
        *,
        timeout_seconds: float | None,
    ) -> VerificationProcessResult:
        del command, timeout_seconds
        self.calls += 1
        return VerificationProcessResult(
            returncode=0,
            stdout="verification passed\n",
            stderr="",
        )


@pytest.mark.skipif(GIT is None, reason="git executable is required for locking tests")
def test_second_run_against_same_repository_is_blocked(tmp_path):
    repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    holder = start_lock_holder(repo, run_id="owner-run", state="IMPLEMENTING")
    try:
        with pytest.raises(RepositoryLockError) as raised:
            run_ticket_lifecycle(
                config,
                ticket,
                runs_dir=runs_dir,
                codex_runner=SequencedCodexRunner([]),
                verification_runner=PassingVerificationRunner(),
                clock=fixed_clock,
            )
    finally:
        stop_lock_holder(holder)

    message = str(raised.value)
    assert "Target repository is already owned" in message
    assert "Owning run ID: owner-run" in message
    assert "Owning state: IMPLEMENTING" in message
    assert str(repo.resolve()) in message


@pytest.mark.skipif(GIT is None, reason="git executable is required for locking tests")
def test_second_resume_against_same_repository_is_blocked(tmp_path):
    repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    snapshot = create_run_snapshot(config, ticket, runs_dir=runs_dir, clock=fixed_clock)
    holder = start_lock_holder(repo, run_id="owner-run", state="VERIFYING")
    try:
        with pytest.raises(RepositoryLockError) as raised:
            resume_ticket_lifecycle(
                config,
                snapshot.run_record.run_id,
                runs_dir=runs_dir,
                codex_runner=SequencedCodexRunner([]),
                verification_runner=PassingVerificationRunner(),
                clock=fixed_clock,
            )
    finally:
        stop_lock_holder(holder)

    message = str(raised.value)
    assert "Owning run ID: owner-run" in message
    assert "Owning state: VERIFYING" in message


@pytest.mark.skipif(GIT is None, reason="git executable is required for locking tests")
def test_same_repository_identity_uses_git_root_for_subdirectories(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    nested = repo / "nested"
    nested.mkdir()
    holder = start_lock_holder(repo, run_id="owner-run", state="IMPLEMENTING")
    try:
        with pytest.raises(RepositoryLockError) as raised:
            acquire_repository_run_lock(
                nested,
                run_id="competing-run",
                current_state="PREPARING",
                clock=fixed_clock,
            )
    finally:
        stop_lock_holder(holder)

    assert "Owning run ID: owner-run" in str(raised.value)


@pytest.mark.skipif(GIT is None, reason="git executable is required for locking tests")
def test_same_repository_is_blocked_across_distinct_run_roots(tmp_path):
    repo, ticket, config = workflow_inputs(tmp_path)
    second_runs_dir = tmp_path / "second-runs"
    holder = start_lock_holder(
        repo,
        run_id="first-config-owner",
        state="IMPLEMENTING",
    )
    try:
        with pytest.raises(RepositoryLockError) as raised:
            run_ticket_lifecycle(
                config,
                ticket,
                runs_dir=second_runs_dir,
                codex_runner=SequencedCodexRunner([]),
                verification_runner=PassingVerificationRunner(),
                clock=fixed_clock,
            )
    finally:
        stop_lock_holder(holder)

    message = str(raised.value)
    assert "Owning run ID: first-config-owner" in message
    assert "Owning state: IMPLEMENTING" in message


@pytest.mark.skipif(GIT is None, reason="git executable is required for locking tests")
def test_different_repositories_can_run_independently(tmp_path):
    repo_a = create_git_repo(tmp_path / "repo-a")
    repo_b, ticket_b, config_b = workflow_inputs(tmp_path, repo_name="repo-b")
    runs_dir = tmp_path / "runs"
    holder = start_lock_holder(repo_a, run_id="owner-run", state="IMPLEMENTING")
    try:
        result = run_ticket_lifecycle(
            config_b,
            ticket_b,
            runs_dir=runs_dir,
            codex_runner=SequencedCodexRunner(
                [
                    CodexStep(
                        result=implementation_result(),
                        mutation=write_file("implemented in repo b\n"),
                    ),
                    CodexStep(result=review_result()),
                ]
            ),
            verification_runner=PassingVerificationRunner(),
            clock=fixed_clock,
        )
    finally:
        stop_lock_holder(holder)

    assert result.run_record.state == WorkflowState.READY_FOR_HUMAN
    assert repo_b.joinpath("file.txt").read_text(encoding="utf-8") == (
        "implemented in repo b\n"
    )


@pytest.mark.skipif(GIT is None, reason="git executable is required for locking tests")
def test_lock_releases_after_ready_for_human(tmp_path):
    repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"

    result = run_ticket_lifecycle(
        config,
        ticket,
        runs_dir=runs_dir,
        codex_runner=SequencedCodexRunner(
            [
                CodexStep(
                    result=implementation_result(),
                    mutation=write_file("implemented\n"),
                ),
                CodexStep(result=review_result()),
            ]
        ),
        verification_runner=PassingVerificationRunner(),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.READY_FOR_HUMAN
    assert_can_acquire(repo)


@pytest.mark.skipif(GIT is None, reason="git executable is required for locking tests")
def test_lock_releases_after_human_required(tmp_path):
    repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"

    result = run_ticket_lifecycle(
        config,
        ticket,
        runs_dir=runs_dir,
        codex_runner=SequencedCodexRunner([CodexStep(result=implementation_result())]),
        verification_runner=PassingVerificationRunner(),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert "without repository changes" in result.run_record.terminal_reason
    assert_can_acquire(repo)


@pytest.mark.skipif(GIT is None, reason="git executable is required for locking tests")
def test_lock_releases_after_failed(tmp_path):
    repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"

    result = run_ticket_lifecycle(
        config,
        ticket,
        runs_dir=runs_dir,
        codex_runner=SequencedCodexRunner(
            [CodexStep(error=FileNotFoundError("missing codex"))]
        ),
        verification_runner=PassingVerificationRunner(),
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.FAILED
    assert_can_acquire(repo)


@pytest.mark.skipif(GIT is None, reason="git executable is required for locking tests")
def test_lock_releases_after_preflight_failure(tmp_path):
    repo, ticket, config = workflow_inputs(tmp_path)
    runs_dir = tmp_path / "runs"
    repo.joinpath("file.txt").write_text("dirty before preflight\n", encoding="utf-8")

    with pytest.raises(RunPreflightError):
        run_ticket_lifecycle(
            config,
            ticket,
            runs_dir=runs_dir,
            codex_runner=SequencedCodexRunner([]),
            verification_runner=PassingVerificationRunner(),
            clock=fixed_clock,
        )

    assert_can_acquire(repo)


@pytest.mark.skipif(GIT is None, reason="git executable is required for locking tests")
def test_concurrent_acquisition_allows_only_one_owner(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    barrier = tmp_path / "start.lock-test"
    first = start_competing_process(repo, barrier, "first")
    second = start_competing_process(repo, barrier, "second")

    barrier.write_text("go\n", encoding="utf-8")
    first_output = finish_competing_process(first)
    second_output = finish_competing_process(second)

    assert sorted((first_output, second_output)) == ["ACQUIRED", "BLOCKED"]


@pytest.mark.skipif(GIT is None, reason="git executable is required for locking tests")
def test_process_death_does_not_permanently_strand_repository(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    holder = start_lock_holder(repo, run_id="dead-owner", state="IMPLEMENTING")
    assert active_repository_locks()[0].run_id == "dead-owner"

    holder.terminate()
    holder.wait(timeout=10)

    with acquire_repository_run_lock(
        repo,
        run_id="new-owner",
        current_state="PREPARING",
        clock=fixed_clock,
    ) as repository_lock:
        assert repository_lock.metadata.run_id == "new-owner"
    assert active_repository_locks() == ()


def workflow_inputs(
    tmp_path: Path,
    *,
    repo_name: str = "repo",
) -> tuple[Path, Path, object]:
    repo = create_git_repo(tmp_path / repo_name)
    ticket = tmp_path / f"{repo_name}-QDEB-003.md"
    ticket.write_text("# QDEB-003\n\nImplement the ticket.\n", encoding="utf-8")
    config = make_config(
        repo,
        max_correction_rounds=1,
        verification_commands=(
            VerificationCommand(
                name="tests",
                argv=(Path(sys.executable).as_posix(), "-c", "raise SystemExit(0)"),
                timeout_seconds=1800,
            ),
        ),
    )
    return repo, ticket, config


def start_lock_holder(
    repo: Path,
    *,
    run_id: str,
    state: str,
) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            LOCK_HOLDER_SCRIPT,
            str(repo),
            run_id,
            state,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=python_subprocess_env(),
    )
    assert process.stdout is not None
    line = process.stdout.readline()
    if line.strip() != "LOCKED":
        process.terminate()
        stdout, stderr = process.communicate(timeout=10)
        pytest.fail(
            f"lock holder did not start: line={line!r} stdout={stdout!r} stderr={stderr!r}"
        )
    return process


def stop_lock_holder(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    assert process.stdin is not None
    process.stdin.write("\n")
    process.stdin.flush()
    try:
        process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.communicate(timeout=10)


def start_competing_process(
    repo: Path,
    barrier: Path,
    run_id: str,
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            COMPETING_LOCK_SCRIPT,
            str(repo),
            str(barrier),
            run_id,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=python_subprocess_env(),
    )


def finish_competing_process(process: subprocess.Popen[str]) -> str:
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 0, stderr
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    assert len(lines) == 1, stdout
    return lines[0]


def assert_can_acquire(repo: Path) -> None:
    with acquire_repository_run_lock(
        repo,
        run_id="probe-run",
        current_state="PREPARING",
        clock=fixed_clock,
    ):
        pass


def implementation_result() -> dict[str, object]:
    return {
        "status": "COMPLETED",
        "summary": "implemented",
        "tests_run": [{"command": "synthetic", "result": "PASS"}],
        "assumptions": [],
        "known_issues": [],
    }


def review_result() -> dict[str, object]:
    return {
        "verdict": "PASS",
        "summary": "accepted",
        "confidence": "HIGH",
        "findings": [],
    }


def write_file(contents: str) -> Callable[[Path], None]:
    def mutate(cwd: Path) -> None:
        cwd.joinpath("file.txt").write_text(contents, encoding="utf-8")

    return mutate


def _event_stream(result: dict[str, object]) -> str:
    import json

    return (
        "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "thread"}),
                json.dumps({"type": "turn.started"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "item_1",
                            "type": "agent_message",
                            "text": json.dumps(result),
                        },
                    }
                ),
                json.dumps({"type": "turn.completed"}),
            ]
        )
        + "\n"
    )


def python_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return env
