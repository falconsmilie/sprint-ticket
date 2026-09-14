from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.helpers import (
    GIT,
    create_git_repo,
    make_config,
)
from tests.helpers import create_trusted_prepared_run as create_run_snapshot
from ticket_automation.config import VerificationCommand
from ticket_automation.corrections import (
    CorrectionReasonKind,
    ReviewFinding,
    VerificationFailure,
)
from ticket_automation.git import GitRepository
from ticket_automation.git_safety import WorkspaceSnapshot
from ticket_automation.models import (
    StageOutcome,
    StopCategory,
    StopReason,
    WorkflowState,
)
from ticket_automation.runs import load_run_record, save_run_record
from ticket_automation.verification import (
    SubprocessVerificationRunner,
    VerificationError,
    VerificationErrorKind,
    VerificationProcessCommand,
    VerificationProcessResult,
    VerificationProcessTimedOut,
    VerificationProcessTimeout,
    VerificationStatus,
    run_verification_stage,
)


def fixed_clock() -> datetime:
    return datetime(2026, 9, 11, 13, 5, 14, tzinfo=UTC)


def test_subprocess_output_decoding_does_not_depend_on_locale(tmp_path):
    command = VerificationProcessCommand(
        argv=(
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'utf8: \\xc4\\x8d\\n')",
        ),
        cwd=tmp_path,
    )

    result = SubprocessVerificationRunner().run(command, timeout_seconds=30)

    assert result.returncode == 0
    assert result.stdout == "utf8: č\n"
    assert isinstance(result.stderr, str)


@dataclass
class TimeoutRunner:
    command: VerificationProcessCommand | None = None
    timeout_seconds: float | None = None

    def run(
        self,
        command: VerificationProcessCommand,
        *,
        timeout_seconds: float | None,
    ) -> VerificationProcessResult:
        self.command = command
        self.timeout_seconds = timeout_seconds
        raise VerificationProcessTimedOut(
            VerificationProcessTimeout(
                stdout="started\n",
                stderr="still running\n",
                timeout_seconds=float(timeout_seconds or 0),
            )
        )


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for verification tests"
)
def test_one_passing_command_moves_run_to_verify_and_writes_round_zero(tmp_path):
    repo, run_dir = implementation_ready_run(tmp_path)
    config = make_config(
        repo,
        verification_commands=(python_gate("tests", "print('verification passed')"),),
    )

    result = run_verification_stage(config, run_dir, clock=fixed_clock)

    assert result.successful
    assert result.outcome == StageOutcome.COMPLETED
    assert load_run_record(run_dir / "run.json").state == WorkflowState.VERIFYING
    assert result.round_result.status == VerificationStatus.PASS
    assert result.round_result.json_path == run_dir / "verification" / "round-0.json"
    assert result.round_result.log_path == run_dir / "verification" / "round-0.log"
    assert result.round_result.json_path.is_file()
    assert result.round_result.log_path.is_file()
    data = json.loads(result.round_result.json_path.read_text(encoding="utf-8"))
    assert data["status"] == "PASS"
    assert data["commands"][0]["stdout"] == "verification passed\n"
    assert data["checkpoint"]["schema_version"] == 2
    assert (
        data["checkpoint"]["source_fingerprint"]
        == WorkspaceSnapshot.capture(GitRepository(repo)).fingerprint
    )


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for verification tests"
)
@pytest.mark.parametrize(
    "state",
    [
        WorkflowState.PREPARED,
        WorkflowState.CORRECTION_PENDING,
        WorkflowState.HUMAN_REQUIRED,
    ],
)
def test_verification_rejects_untrusted_starting_states(tmp_path, state):
    repo, run_dir = implementation_ready_run(tmp_path)
    run_record_path = run_dir / "run.json"
    stop_reason = (
        StopReason(
            category=StopCategory.HUMAN_JUDGMENT_REQUIRED,
            message="Synthetic terminal state for validation.",
            retryable=False,
        )
        if state == WorkflowState.HUMAN_REQUIRED
        else None
    )
    run_record = replace(
        load_run_record(run_record_path),
        state=state,
        updated_timestamp="2026-09-11T13:05:15Z",
        terminal_reason=None if stop_reason is None else stop_reason.message,
        stop_reason=stop_reason,
    )
    save_run_record(run_record, run_record_path)

    with pytest.raises(VerificationError, match="requires run state VERIFYING"):
        run_verification_stage(make_config(repo), run_dir)


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for verification tests"
)
def test_multiple_passing_commands_all_run(tmp_path):
    repo, run_dir = implementation_ready_run(tmp_path)
    config = make_config(
        repo,
        verification_commands=(
            python_gate("tests", "print('tests pass')"),
            python_gate("typing", "print('typing pass')"),
        ),
    )

    result = run_verification_stage(config, run_dir)

    assert result.outcome == StageOutcome.COMPLETED
    assert [command.status for command in result.round_result.commands] == [
        VerificationStatus.PASS,
        VerificationStatus.PASS,
    ]
    assert [command.stdout for command in result.round_result.commands] == [
        "tests pass\n",
        "typing pass\n",
    ]


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for verification tests"
)
@pytest.mark.parametrize(
    ("failing_indexes", "expected_failed"),
    [
        ((0,), ["first"]),
        ((1,), ["second"]),
        ((0, 1), ["first", "second"]),
    ],
)
def test_failing_gates_move_run_to_correct_and_preserve_all_failures(
    tmp_path,
    failing_indexes,
    expected_failed,
):
    repo, run_dir = implementation_ready_run(tmp_path)
    commands = tuple(
        failing_gate(name, exit_code=index + 2)
        if index in failing_indexes
        else python_gate(name, "print('pass')")
        for index, name in enumerate(("first", "second", "third"))
    )
    config = make_config(repo, verification_commands=commands)

    result = run_verification_stage(config, run_dir)

    assert result.outcome == StageOutcome.CORRECTION_REQUIRED
    assert result.round_result.status == VerificationStatus.FAIL
    assert [
        command.name for command in result.round_result.failed_commands
    ] == expected_failed
    assert len(result.round_result.commands) == 3
    assert [
        reason.to_dict()["kind"] for reason in result.round_result.correction_reasons
    ] == ["VerificationFailure"] * len(expected_failed)
    assert [
        reason.gate_name
        for reason in result.round_result.correction_reasons
        if isinstance(reason, VerificationFailure)
    ] == expected_failed


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for verification tests"
)
def test_command_timeout_is_an_error_and_preserves_partial_output(tmp_path):
    repo, run_dir = implementation_ready_run(tmp_path)
    config = make_config(
        repo,
        verification_commands=(
            VerificationCommand(
                name="slow",
                argv=("slow-tool",),
                timeout_seconds=3,
            ),
        ),
    )
    runner = TimeoutRunner()

    result = run_verification_stage(config, run_dir, process_runner=runner)

    command = result.round_result.commands[0]
    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert result.round_result.status == VerificationStatus.ERROR
    assert command.status == VerificationStatus.ERROR
    assert command.error_kind == VerificationErrorKind.TIMEOUT
    assert command.exit_code is None
    assert command.stdout == "started\n"
    assert command.stderr == "still running\n"
    assert runner.timeout_seconds == 3


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for verification tests"
)
def test_passing_command_that_mutates_worktree_becomes_human_required(tmp_path):
    repo, run_dir = implementation_ready_run(tmp_path)
    config = make_config(
        repo,
        verification_commands=(
            python_gate(
                "mutating-pass",
                "import pathlib; pathlib.Path('file.txt').write_text('changed\\n')",
            ),
        ),
    )

    result = run_verification_stage(config, run_dir)

    command = result.round_result.commands[0]
    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert result.round_result.status == VerificationStatus.ERROR
    assert command.status == VerificationStatus.PASS
    assert result.round_result.correction_reasons == ()
    assert [violation.name for violation in result.round_result.safety_violations] == [
        "tracked-diff"
    ]
    data = json.loads(result.round_result.json_path.read_text(encoding="utf-8"))
    assert data["commands"][0]["status"] == "PASS"
    assert data["safety_violations"][0]["name"] == "tracked-diff"
    assert "Tracked worktree diff changed" in result.round_result.log_path.read_text(
        encoding="utf-8"
    )


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for verification tests"
)
def test_executable_unavailable_is_an_error_not_a_correction_reason(tmp_path):
    repo, run_dir = implementation_ready_run(tmp_path)
    config = make_config(
        repo,
        verification_commands=(
            VerificationCommand(
                name="missing",
                argv=("ticket-automation-missing-verifier-006",),
                timeout_seconds=1800,
            ),
        ),
    )

    result = run_verification_stage(config, run_dir)

    command = result.round_result.commands[0]
    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert command.status == VerificationStatus.ERROR
    assert command.error_kind == VerificationErrorKind.EXECUTABLE_UNAVAILABLE
    assert command.exit_code is None
    assert result.round_result.correction_reasons == ()


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for verification tests"
)
def test_stdout_and_stderr_are_captured_for_successful_commands(tmp_path):
    repo, run_dir = implementation_ready_run(tmp_path)
    config = make_config(
        repo,
        verification_commands=(
            python_gate(
                "output",
                "import sys; print('out'); print('err', file=sys.stderr)",
            ),
        ),
    )

    result = run_verification_stage(config, run_dir)

    command = result.round_result.commands[0]
    assert command.status == VerificationStatus.PASS
    assert command.stdout == "out\n"
    assert command.stderr == "err\n"
    data = json.loads(result.round_result.json_path.read_text(encoding="utf-8"))
    assert data["commands"][0]["stdout"] == "out\n"
    assert data["commands"][0]["stderr"] == "err\n"
    assert "out" in result.round_result.log_path.read_text(encoding="utf-8")
    assert "err" in result.round_result.log_path.read_text(encoding="utf-8")


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for verification tests"
)
def test_arguments_are_preserved_without_shell_interpretation(tmp_path):
    repo, run_dir = implementation_ready_run(tmp_path)
    config = make_config(
        repo,
        verification_commands=(
            VerificationCommand(
                name="argv",
                argv=(
                    sys.executable,
                    "-c",
                    (
                        "import sys; "
                        "assert sys.argv[1:] == ['two words', '&&', '$HOME']; "
                        "print('|'.join(sys.argv[1:]))"
                    ),
                    "two words",
                    "&&",
                    "$HOME",
                ),
                timeout_seconds=1800,
            ),
        ),
    )

    result = run_verification_stage(config, run_dir)

    command = result.round_result.commands[0]
    assert command.status == VerificationStatus.PASS
    assert command.argv[-3:] == ("two words", "&&", "$HOME")
    assert command.stdout == "two words|&&|$HOME\n"


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for verification tests"
)
def test_repository_is_used_as_command_cwd(tmp_path):
    repo, run_dir = implementation_ready_run(tmp_path)
    config = make_config(
        repo,
        verification_commands=(
            VerificationCommand(
                name="cwd",
                argv=(
                    sys.executable,
                    "-c",
                    (
                        "import pathlib, sys; "
                        "assert pathlib.Path.cwd().resolve() == "
                        "pathlib.Path(sys.argv[1]).resolve(); "
                        "print(pathlib.Path.cwd().name)"
                    ),
                    str(repo),
                ),
                timeout_seconds=1800,
            ),
        ),
    )

    result = run_verification_stage(config, run_dir)

    command = result.round_result.commands[0]
    assert command.status == VerificationStatus.PASS
    assert command.cwd == repo
    assert command.stdout == f"{repo.name}\n"


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for verification tests"
)
def test_correction_reason_contains_gate_command_summary_output_and_exit_code(tmp_path):
    repo, run_dir = implementation_ready_run(tmp_path)
    config = make_config(
        repo,
        verification_commands=(failing_gate("tests", exit_code=9),),
    )

    result = run_verification_stage(config, run_dir)

    assert result.outcome == StageOutcome.CORRECTION_REQUIRED
    reason = result.round_result.correction_reasons[0]
    assert isinstance(reason, VerificationFailure)
    assert reason.kind == CorrectionReasonKind.VERIFICATION_FAILURE
    assert reason.gate_name == "tests"
    assert reason.command == config.verification.commands[0].argv
    assert reason.exit_code == 9
    assert "exited with code 9" in reason.failure_summary
    assert "failing stdout" in reason.stdout_excerpt
    assert "failing stderr" in reason.stderr_excerpt
    assert reason.log_path == run_dir / "verification" / "round-0.log"
    data = json.loads(result.round_result.json_path.read_text(encoding="utf-8"))
    assert data["correction_reasons"][0]["kind"] == "VerificationFailure"
    assert "ReviewFinding" not in {item["kind"] for item in data["correction_reasons"]}


def test_correction_reason_kinds_distinguish_review_findings():
    finding = ReviewFinding(
        finding_id="review-1",
        summary="Reviewer found a bug.",
        details="Future review details.",
        disposition="REQUIRED",
        scope_relation="TICKET",
    )

    assert finding.to_dict()["kind"] == "ReviewFinding"
    assert (
        VerificationFailure(
            gate_name="tests",
            command=("pytest",),
            failure_summary="tests failed",
            stdout_excerpt="",
            stderr_excerpt="",
            exit_code=1,
            log_path=Path("round-0.log"),
        ).to_dict()["kind"]
        == "VerificationFailure"
    )


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for verification tests"
)
def test_unexpected_process_runner_bug_still_propagates(tmp_path):
    repo, run_dir = implementation_ready_run(tmp_path)
    config = make_config(repo)

    class BrokenRunner:
        def run(self, command, *, timeout_seconds):
            del command, timeout_seconds
            raise RuntimeError("runner contract bug")

    with pytest.raises(RuntimeError, match="runner contract bug"):
        run_verification_stage(
            config,
            run_dir,
            process_runner=BrokenRunner(),
            clock=fixed_clock,
        )


@pytest.mark.skipif(
    GIT is None, reason="git executable is required for verification tests"
)
def test_post_change_repository_inspection_bug_still_propagates(
    tmp_path,
    monkeypatch,
):
    repo, run_dir = implementation_ready_run(tmp_path)
    config = make_config(repo)

    class BreakEndingInspectionRunner:
        def run(self, command, *, timeout_seconds):
            del command, timeout_seconds

            def fail_capture(repository):
                del repository
                raise RuntimeError("inspection contract bug")

            monkeypatch.setattr(WorkspaceSnapshot, "capture", fail_capture)
            return VerificationProcessResult(returncode=0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="inspection contract bug"):
        run_verification_stage(
            config,
            run_dir,
            process_runner=BreakEndingInspectionRunner(),
            clock=fixed_clock,
        )


def implementation_ready_run(tmp_path):
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "TA-006.md"
    ticket.write_text("# TA-006\n\nVerify the implementation.\n", encoding="utf-8")
    config = make_config(repo)
    snapshot = create_run_snapshot(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    run_record_path = snapshot.run_dir / "run.json"
    run_record = load_run_record(run_record_path).transition_to(
        WorkflowState.IMPLEMENTING,
        updated_timestamp="2026-09-11T13:05:14Z",
    )
    run_record = run_record.transition_to(
        WorkflowState.VERIFYING,
        updated_timestamp="2026-09-11T13:05:14Z",
    )
    save_run_record(run_record, run_record_path)
    return repo, snapshot.run_dir


def python_gate(
    name: str,
    code: str,
    *,
    timeout_seconds: int = 1800,
) -> VerificationCommand:
    return VerificationCommand(
        name=name,
        argv=(sys.executable, "-c", code),
        timeout_seconds=timeout_seconds,
    )


def failing_gate(name: str, *, exit_code: int) -> VerificationCommand:
    return python_gate(
        name,
        (
            "import sys; "
            "print('failing stdout'); "
            "print('failing stderr', file=sys.stderr); "
            f"raise SystemExit({exit_code})"
        ),
    )
