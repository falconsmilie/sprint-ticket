from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import ticket_automation.verification as verification_module
from tests.helpers import (
    GIT,
    create_git_repo,
    create_trusted_prepared_run,
    make_config,
)
from ticket_automation.codex import CodexCommand, CodexProcessResult
from ticket_automation.config import VerificationCommand
from ticket_automation.corrections import run_correction_stage
from ticket_automation.git_safety import WorkspaceSnapshot
from ticket_automation.implementation import run_implementation_stage
from ticket_automation.locking import active_repository_locks
from ticket_automation.models import StageOutcome, WorkflowState
from ticket_automation.runs import (
    create_run_snapshot,
    load_baseline_record,
    load_run_record,
    save_run_record,
)
from ticket_automation.verification import (
    VerificationError,
    VerificationProcessCommand,
    VerificationProcessResult,
    VerificationProcessTimedOut,
    VerificationProcessTimeout,
    VerificationStatus,
    _run_baseline_verification_stage,
    run_verification_stage,
)
from ticket_automation.workflow import resume_ticket_lifecycle, run_ticket_lifecycle


def fixed_clock() -> datetime:
    return datetime(2026, 9, 13, 10, 15, 0, tzinfo=UTC)


@dataclass(frozen=True)
class VerificationStep:
    returncode: int = 0
    stdout: str = "baseline output\n"
    stderr: str = ""
    error: BaseException | None = None
    mutation: str | None = None


@dataclass
class RecordingVerificationRunner:
    steps: list[VerificationStep]
    commands: list[VerificationProcessCommand] = field(default_factory=list)

    @property
    def calls(self) -> int:
        return len(self.commands)

    def run(
        self,
        command: VerificationProcessCommand,
        *,
        timeout_seconds: float | None,
    ) -> VerificationProcessResult:
        del timeout_seconds
        self.commands.append(command)
        assert self.steps, "Unexpected verification invocation."
        step = self.steps.pop(0)
        if step.mutation is not None:
            command.cwd.joinpath("file.txt").write_text(
                step.mutation,
                encoding="utf-8",
            )
        if step.error is not None:
            raise step.error
        return VerificationProcessResult(
            returncode=step.returncode,
            stdout=step.stdout,
            stderr=step.stderr,
        )


@dataclass
class RejectingCodexRunner:
    calls: int = 0

    def run(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        raise AssertionError("Codex must not run after baseline verification failure.")


@dataclass
class BlockingCodexRunner:
    calls: int = 0

    def run(
        self,
        command: CodexCommand,
        *,
        stdin: str,
        timeout_seconds: float | None,
    ) -> CodexProcessResult:
        del command, stdin, timeout_seconds
        self.calls += 1
        result = {
            "status": "BLOCKED",
            "summary": "Stopped after proving the baseline gate passed.",
            "tests_run": [],
            "assumptions": [],
            "known_issues": ["intentional test stop"],
        }
        return CodexProcessResult(
            returncode=0,
            stdout=codex_event_stream(result),
            stderr="",
        )


@dataclass
class RejectingVerificationRunner:
    calls: int = 0

    def run(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        raise AssertionError("Completed baseline verification must not be rerun.")


@dataclass
class LockInspectingVerificationRunner(RecordingVerificationRunner):
    observed_run_id: str | None = None
    observed_state: str | None = None

    def run(
        self,
        command: VerificationProcessCommand,
        *,
        timeout_seconds: float | None,
    ) -> VerificationProcessResult:
        locks = active_repository_locks()
        assert len(locks) == 1
        self.observed_run_id = locks[0].run_id
        self.observed_state = locks[0].current_state
        return super().run(command, timeout_seconds=timeout_seconds)


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_run_creation_persists_preparing_before_baseline_verification(tmp_path):
    repo, ticket = baseline_inputs(tmp_path)

    result = create_run_snapshot(
        make_config(repo),
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.PREPARING
    assert load_run_record(result.run_dir / "run.json").state == WorkflowState.PREPARING
    assert not (result.run_dir / "baseline-verification").exists()


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_clean_baseline_pass_persists_evidence_before_codex_runs(tmp_path):
    repo, ticket = baseline_inputs(tmp_path)
    runner = RecordingVerificationRunner(
        steps=[VerificationStep(stdout="tests passed\n")]
    )
    codex = BlockingCodexRunner()

    result = run_ticket_lifecycle(
        make_config(repo),
        ticket,
        runs_dir=tmp_path / "runs",
        verification_runner=runner,
        codex_runner=codex,
        clock=fixed_clock,
    )

    assert runner.calls == 1
    assert codex.calls == 1
    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert (result.run_dir / "implementation").is_dir()

    baseline = load_baseline_record(result.run_dir / "baseline.json")
    evidence_path = result.run_dir / "baseline-verification" / "verification.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["status"] == VerificationStatus.PASS.value
    assert evidence["commands"][0]["stdout"] == "tests passed\n"
    assert evidence["checkpoint"]["stage"] == WorkflowState.PREPARING.value
    assert (
        evidence["checkpoint"]["source_fingerprint"] == baseline.workspace_fingerprint
    )
    assert evidence["checkpoint"]["verification_commands_fingerprint"] == (
        baseline.verification_commands_fingerprint
    )
    baseline_log = result.run_dir / "baseline-verification" / "verification.log"
    assert baseline_log.is_file()
    assert "tests passed" not in baseline_log.read_text(encoding="utf-8")
    baseline_data = json.loads(
        (result.run_dir / "baseline.json").read_text(encoding="utf-8")
    )
    assert "stdout" not in baseline_data
    assert "stderr" not in baseline_data
    report = (result.run_dir / "final-report.md").read_text(encoding="utf-8")
    assert "Clean baseline verification: PASS" in report


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_baseline_verification_runs_under_the_created_run_lock(tmp_path):
    repo, ticket = baseline_inputs(tmp_path)
    runner = LockInspectingVerificationRunner(steps=[VerificationStep()])

    result = run_ticket_lifecycle(
        make_config(repo),
        ticket,
        runs_dir=tmp_path / "runs",
        verification_runner=runner,
        codex_runner=BlockingCodexRunner(),
        clock=fixed_clock,
    )

    assert runner.observed_run_id == result.run_record.run_id
    assert runner.observed_state == WorkflowState.PREPARING.value


@pytest.mark.skipif(GIT is None, reason="git executable is required")
@pytest.mark.parametrize("failed_gate", ["tests", "typing", "lint"])
def test_any_configured_baseline_gate_failure_requires_human_without_codex(
    tmp_path,
    failed_gate,
):
    repo, ticket = baseline_inputs(tmp_path)
    gate_names = ("tests", "typing", "lint")
    config = make_config(
        repo,
        verification_commands=tuple(python_gate(name) for name in gate_names),
    )
    runner = RecordingVerificationRunner(
        steps=[
            VerificationStep(
                returncode=7 if name == failed_gate else 0,
                stdout=f"{name} {'failed' if name == failed_gate else 'passed'}\n",
                stderr="failure detail\n" if name == failed_gate else "",
            )
            for name in gate_names
        ]
    )
    codex = RejectingCodexRunner()

    result = run_ticket_lifecycle(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        codex_runner=codex,
        verification_runner=runner,
        clock=fixed_clock,
    )

    assert [command.argv for command in runner.commands] == [
        command.argv for command in config.verification.commands
    ]
    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert result.implementation_result is None
    assert result.verification_results == ()
    assert result.correction_results == ()
    assert codex.calls == 0
    assert not (result.run_dir / "implementation").exists()
    assert not (result.run_dir / "corrections").exists()
    evidence = baseline_evidence(result.run_dir)
    assert evidence["status"] == VerificationStatus.FAIL.value
    assert evidence["correction_reasons"] == []
    assert [command["name"] for command in evidence["commands"]] == list(gate_names)
    failed = next(
        command for command in evidence["commands"] if command["name"] == failed_gate
    )
    assert failed["status"] == VerificationStatus.FAIL.value
    report = (result.run_dir / "final-report.md").read_text(encoding="utf-8")
    assert "Clean baseline verification: FAIL" in report


@pytest.mark.skipif(GIT is None, reason="git executable is required")
@pytest.mark.parametrize(
    ("error", "expected_kind"),
    [
        (FileNotFoundError("missing verifier"), "EXECUTABLE_UNAVAILABLE"),
        (OSError("cannot launch verifier"), "PROCESS_START_FAILED"),
        (
            VerificationProcessTimedOut(
                VerificationProcessTimeout(
                    stdout="started\n",
                    stderr="timed out\n",
                    timeout_seconds=30,
                )
            ),
            "TIMEOUT",
        ),
    ],
)
def test_clean_baseline_command_error_stops_before_codex_or_correction(
    tmp_path,
    error,
    expected_kind,
):
    repo, ticket = baseline_inputs(tmp_path)
    runner = RecordingVerificationRunner(steps=[VerificationStep(error=error)])
    codex = RejectingCodexRunner()

    result = run_ticket_lifecycle(
        make_config(repo),
        ticket,
        runs_dir=tmp_path / "runs",
        verification_runner=runner,
        codex_runner=codex,
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert result.implementation_result is None
    assert result.correction_results == ()
    assert codex.calls == 0
    evidence = baseline_evidence(result.run_dir)
    assert evidence["status"] == VerificationStatus.ERROR.value
    assert evidence["commands"][0]["error_kind"] == expected_kind
    assert evidence["correction_reasons"] == []


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_clean_baseline_repository_inspection_error_is_persisted_without_codex(
    tmp_path,
    monkeypatch,
):
    repo, ticket = baseline_inputs(tmp_path)

    class BreakEndingInspectionRunner(RecordingVerificationRunner):
        def run(self, command, *, timeout_seconds):
            result = super().run(command, timeout_seconds=timeout_seconds)

            def fail_capture(repository):
                del repository
                raise OSError("inspection unavailable")

            monkeypatch.setattr(WorkspaceSnapshot, "capture", fail_capture)
            return result

    runner = BreakEndingInspectionRunner(steps=[VerificationStep()])
    codex = RejectingCodexRunner()
    result = run_ticket_lifecycle(
        make_config(repo),
        ticket,
        runs_dir=tmp_path / "runs",
        verification_runner=runner,
        codex_runner=codex,
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert codex.calls == 0
    evidence = baseline_evidence(result.run_dir)
    assert evidence["status"] == VerificationStatus.ERROR.value
    assert evidence["safety_violations"][0]["name"] == "repository-inspection"
    assert evidence["correction_reasons"] == []


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_baseline_evidence_persistence_error_requires_human_without_codex(
    tmp_path,
    monkeypatch,
):
    repo, ticket = baseline_inputs(tmp_path)
    codex = RejectingCodexRunner()

    def fail_write(path, data):
        del path, data
        raise OSError("artifact storage unavailable")

    monkeypatch.setattr(verification_module, "_write_json", fail_write)
    result = run_ticket_lifecycle(
        make_config(repo),
        ticket,
        runs_dir=tmp_path / "runs",
        verification_runner=RecordingVerificationRunner(steps=[VerificationStep()]),
        codex_runner=codex,
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert codex.calls == 0
    assert "artifact storage unavailable" in (result.run_record.terminal_reason or "")


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_clean_baseline_mutation_requires_human_and_is_not_cleaned(tmp_path):
    repo, ticket = baseline_inputs(tmp_path)
    runner = RecordingVerificationRunner(
        steps=[VerificationStep(mutation="mutated by gate\n")]
    )
    codex = RejectingCodexRunner()

    result = run_ticket_lifecycle(
        make_config(repo),
        ticket,
        runs_dir=tmp_path / "runs",
        verification_runner=runner,
        codex_runner=codex,
        clock=fixed_clock,
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert result.implementation_result is None
    assert result.correction_results == ()
    assert codex.calls == 0
    assert repo.joinpath("file.txt").read_text(encoding="utf-8") == "mutated by gate\n"
    evidence = baseline_evidence(result.run_dir)
    assert evidence["status"] == VerificationStatus.ERROR.value
    assert "tracked-diff" in {
        violation["name"] for violation in evidence["safety_violations"]
    }
    assert evidence["correction_reasons"] == []


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_baseline_and_post_change_verification_are_distinct(tmp_path):
    repo, ticket = baseline_inputs(tmp_path)
    config = make_config(repo)
    snapshot = create_trusted_prepared_run(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        verification_runner=RecordingVerificationRunner(
            steps=[VerificationStep(stdout="baseline output\n")]
        ),
        clock=fixed_clock,
    )
    record_path = snapshot.run_dir / "run.json"
    record = load_run_record(record_path).transition_to(
        WorkflowState.IMPLEMENTING,
        updated_timestamp="2026-09-13T10:15:01Z",
    )
    repo.joinpath("file.txt").write_text("implemented\n", encoding="utf-8")
    record = record.transition_to(
        WorkflowState.VERIFYING,
        updated_timestamp="2026-09-13T10:15:02Z",
    )
    save_run_record(record, record_path)

    run_verification_stage(
        config,
        snapshot.run_dir,
        process_runner=RecordingVerificationRunner(
            steps=[VerificationStep(stdout="post-change output\n")]
        ),
        clock=fixed_clock,
    )

    baseline_path = snapshot.run_dir / "baseline-verification" / "verification.json"
    post_change_path = snapshot.run_dir / "verification" / "round-0.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    post_change = json.loads(post_change_path.read_text(encoding="utf-8"))
    assert baseline_path != post_change_path
    assert baseline["checkpoint"]["stage"] == WorkflowState.PREPARING.value
    assert post_change["checkpoint"]["stage"] == WorkflowState.VERIFYING.value
    assert baseline["commands"][0]["stdout"] == "baseline output\n"
    assert post_change["commands"][0]["stdout"] == "post-change output\n"


@pytest.mark.skipif(GIT is None, reason="git executable is required")
@pytest.mark.parametrize("tamper", ["missing_evidence", "changed_ticket"])
def test_implementation_rejects_untrusted_baseline_construction(tmp_path, tamper):
    repo, ticket = baseline_inputs(tmp_path)
    config = make_config(repo)
    snapshot = create_trusted_prepared_run(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    if tamper == "missing_evidence":
        (snapshot.run_dir / "baseline-verification" / "verification.json").unlink()
    else:
        snapshot.run_dir.joinpath("ticket.md").write_text(
            "# changed ticket\n",
            encoding="utf-8",
        )
    record_path = snapshot.run_dir / "run.json"
    save_run_record(
        load_run_record(record_path).transition_to(
            WorkflowState.IMPLEMENTING,
            updated_timestamp="2026-09-13T10:15:01Z",
        ),
        record_path,
    )
    codex = RejectingCodexRunner()

    result = run_implementation_stage(
        config,
        snapshot.run_dir,
        codex_runner=codex,
        clock=fixed_clock,
    )

    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert codex.calls == 0
    assert result.codex_execution is None
    assert result.safety_violations[0].name == "baseline-verification"


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_correction_entry_point_rejects_missing_baseline_evidence(tmp_path):
    repo, ticket = baseline_inputs(tmp_path)
    config = make_config(repo)
    snapshot = create_trusted_prepared_run(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    (snapshot.run_dir / "baseline-verification" / "verification.json").unlink()
    record = snapshot.run_record
    for state in (
        WorkflowState.IMPLEMENTING,
        WorkflowState.VERIFYING,
        WorkflowState.CORRECTION_PENDING,
        WorkflowState.CORRECTING,
    ):
        record = record.transition_to(
            state,
            updated_timestamp="2026-09-13T10:15:01Z",
        )
    save_run_record(record, snapshot.run_dir / "run.json")
    codex = RejectingCodexRunner()

    result = run_correction_stage(
        config,
        snapshot.run_dir,
        codex_runner=codex,
        clock=fixed_clock,
    )

    assert result.outcome == StageOutcome.HUMAN_REQUIRED
    assert result.ticket_path is None
    assert result.codex_execution is None
    assert codex.calls == 0
    assert result.safety_violations[0].name == "baseline-verification"
    assert not (snapshot.run_dir / "corrections").exists()


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_resume_reruns_missing_baseline_evidence_and_can_advance(tmp_path):
    repo, ticket = baseline_inputs(tmp_path)
    config = make_config(repo)
    snapshot = create_run_snapshot(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    rerun = RecordingVerificationRunner(steps=[VerificationStep()])
    codex = BlockingCodexRunner()

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=tmp_path / "runs",
        verification_runner=rerun,
        codex_runner=codex,
        clock=fixed_clock,
    )

    assert rerun.calls == 1
    assert codex.calls == 1
    assert baseline_evidence(result.run_dir)["status"] == VerificationStatus.PASS.value


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_resume_retries_partial_evidence_only_from_matching_baseline(tmp_path):
    repo, ticket = baseline_inputs(tmp_path)
    config = make_config(repo)
    snapshot = create_run_snapshot(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    artifact_dir = snapshot.run_dir / "baseline-verification"
    artifact_dir.mkdir()
    artifact_dir.joinpath("verification.json").write_text("{\n", encoding="utf-8")
    rerun = RecordingVerificationRunner(steps=[VerificationStep()])
    codex = BlockingCodexRunner()

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=tmp_path / "runs",
        verification_runner=rerun,
        codex_runner=codex,
        clock=fixed_clock,
    )

    assert rerun.calls == 1
    assert codex.calls == 1
    assert baseline_evidence(result.run_dir)["status"] == VerificationStatus.PASS.value
    assert (artifact_dir / "_incomplete" / "baseline" / "verification.json").read_text(
        encoding="utf-8"
    ) == "{\n"


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_resume_adopts_complete_passing_evidence_without_rerunning(tmp_path):
    repo, ticket = baseline_inputs(tmp_path)
    config = make_config(repo)
    snapshot = create_run_snapshot(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    completed = _run_baseline_verification_stage(
        config,
        snapshot.run_dir,
        process_runner=RecordingVerificationRunner(steps=[VerificationStep()]),
        clock=fixed_clock,
    )
    assert completed.successful
    assert (
        load_run_record(snapshot.run_dir / "run.json").state == WorkflowState.PREPARING
    )
    rerun = RejectingVerificationRunner()
    codex = BlockingCodexRunner()

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=tmp_path / "runs",
        verification_runner=rerun,
        codex_runner=codex,
        clock=fixed_clock,
    )

    assert rerun.calls == 0
    assert codex.calls == 1
    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_resume_adopts_complete_failing_evidence_without_rerunning_or_codex(tmp_path):
    repo, ticket = baseline_inputs(tmp_path)
    config = make_config(repo)
    snapshot = create_run_snapshot(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    completed = _run_baseline_verification_stage(
        config,
        snapshot.run_dir,
        process_runner=RecordingVerificationRunner(
            steps=[VerificationStep(returncode=3)]
        ),
        clock=fixed_clock,
    )
    assert completed.outcome == StageOutcome.HUMAN_REQUIRED
    rerun = RejectingVerificationRunner()
    codex = RejectingCodexRunner()

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=tmp_path / "runs",
        verification_runner=rerun,
        codex_runner=codex,
        clock=fixed_clock,
    )

    assert rerun.calls == 0
    assert codex.calls == 0
    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert not (result.run_dir / "corrections").exists()


@pytest.mark.skipif(GIT is None, reason="git executable is required")
@pytest.mark.parametrize("corruption", ["status", "commands"])
def test_resume_rejects_corrupt_complete_baseline_evidence(tmp_path, corruption):
    repo, ticket = baseline_inputs(tmp_path)
    config = make_config(repo)
    snapshot = create_run_snapshot(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    completed = _run_baseline_verification_stage(
        config,
        snapshot.run_dir,
        process_runner=RecordingVerificationRunner(steps=[VerificationStep()]),
        clock=fixed_clock,
    )
    assert completed.successful
    evidence_path = snapshot.run_dir / "baseline-verification" / "verification.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if corruption == "status":
        evidence["status"] = VerificationStatus.FAIL.value
    else:
        evidence["commands"] = []
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    rerun = RejectingVerificationRunner()
    codex = RejectingCodexRunner()

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=tmp_path / "runs",
        verification_runner=rerun,
        codex_runner=codex,
        clock=fixed_clock,
    )

    assert rerun.calls == 0
    assert codex.calls == 0
    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert "baseline verification" in (result.run_record.terminal_reason or "").lower()


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_resume_does_not_rerun_preparing_after_workspace_changes(tmp_path):
    repo, ticket = baseline_inputs(tmp_path)
    config = make_config(repo)
    snapshot = create_run_snapshot(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    repo.joinpath("file.txt").write_text(
        "changed after interruption\n",
        encoding="utf-8",
    )
    rerun = RejectingVerificationRunner()
    codex = RejectingCodexRunner()

    result = resume_ticket_lifecycle(
        config,
        snapshot.run_record.run_id,
        runs_dir=tmp_path / "runs",
        verification_runner=rerun,
        codex_runner=codex,
        clock=fixed_clock,
    )

    assert rerun.calls == 0
    assert codex.calls == 0
    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert "recorded clean baseline" in (result.run_record.terminal_reason or "")
    assert repo.joinpath("file.txt").read_text() == "changed after interruption\n"


@pytest.mark.skipif(GIT is None, reason="git executable is required")
def test_baseline_stage_rejects_any_state_other_than_preparing(tmp_path):
    repo, ticket = baseline_inputs(tmp_path)
    config = make_config(repo)
    snapshot = create_trusted_prepared_run(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )

    with pytest.raises(VerificationError, match="requires run state PREPARING"):
        _run_baseline_verification_stage(
            config,
            snapshot.run_dir,
            process_runner=RejectingVerificationRunner(),
            clock=fixed_clock,
        )


def baseline_inputs(tmp_path: Path) -> tuple[Path, Path]:
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "TA-ARCH-003.md"
    ticket.write_text("# TA-ARCH-003\n\nVerify the clean baseline.\n", encoding="utf-8")
    return repo, ticket


def python_gate(name: str) -> VerificationCommand:
    return VerificationCommand(
        name=name,
        argv=(sys.executable, "-c", "raise SystemExit(0)"),
        timeout_seconds=30,
    )


def baseline_evidence(run_dir: Path) -> dict[str, Any]:
    return json.loads(
        (run_dir / "baseline-verification" / "verification.json").read_text(
            encoding="utf-8"
        )
    )


def codex_event_stream(result: dict[str, object]) -> str:
    return "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
            json.dumps({"type": "turn.started"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "id": "item-1",
                        "type": "agent_message",
                        "text": json.dumps(result),
                    },
                }
            ),
            json.dumps({"type": "turn.completed"}),
        ]
    )
