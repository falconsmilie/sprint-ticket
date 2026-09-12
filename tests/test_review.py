from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.helpers import GIT, create_git_repo, make_config, run_git
from ticket_automation.codex import (
    CodexCommand,
    CodexProcessResult,
    CodexResultValidationError,
    Sandbox,
    validate_json_schema,
)
from ticket_automation.config import AppConfig
from ticket_automation.models import WorkflowState
from ticket_automation.review import (
    REVIEW_DIR_NAME,
    ReviewError,
    ReviewResultConsistencyError,
    ReviewVerdict,
    run_review_stage,
    validate_review_result_semantics,
)
from ticket_automation.runs import create_run_snapshot, load_run_record, save_run_record

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUND_1 = "round-1"


def fixed_clock() -> datetime:
    return datetime(2026, 9, 11, 13, 5, 15, tzinfo=UTC)


@dataclass
class ReviewRunner:
    result: dict[str, object]
    mutation: Callable[[Path], None] | None = None
    command: CodexCommand | None = None
    stdin: str | None = None
    timeout_seconds: float | None = None
    calls: int = 0

    def run(
        self,
        command: CodexCommand,
        *,
        stdin: str,
        timeout_seconds: float | None,
    ) -> CodexProcessResult:
        self.calls += 1
        self.command = command
        self.stdin = stdin
        self.timeout_seconds = timeout_seconds
        if self.mutation is not None:
            self.mutation(command.cwd)
        return CodexProcessResult(
            returncode=0,
            stdout=event_stream(self.result),
            stderr="review progress\n",
        )


@pytest.mark.skipif(GIT is None, reason="git executable is required for review tests")
def test_review_prompt_contains_full_ticket_baseline_verification_and_scope(tmp_path):
    ticket_text = (
        "# Synthetic Ticket\r\n\r\n"
        "Implement all requirements.\r\n\r\n"
        "```text\r\n"
        "$HOME stays literal\r\n"
        "```\r\n"
    )
    repo, run_dir, config = verified_run(tmp_path, ticket_text=ticket_text)
    runner = ReviewRunner(result=review_result())

    run_review_stage(config, run_dir, codex_runner=runner, clock=fixed_clock)

    assert runner.stdin is not None
    prompt = runner.stdin
    baseline_sha = load_run_record(run_dir / "run.json").baseline_sha
    assert ticket_text in prompt
    assert f"Baseline SHA: {baseline_sha}" in prompt
    assert "Starting branch: feature/example" in prompt
    assert "Current branch: feature/example" in prompt
    assert "deterministic passed" in prompt
    assert "implemented synthetic ticket" in prompt
    assert "baseline SHA -> complete current working tree" in prompt
    assert "Do not limit review scope to the latest changed file" in prompt
    assert repo.joinpath("file.txt").read_text(encoding="utf-8") == "implemented\n"


@pytest.mark.skipif(GIT is None, reason="git executable is required for review tests")
def test_review_uses_read_only_sandbox_and_writes_round_one_artifacts(tmp_path):
    _repo, run_dir, config = verified_run(tmp_path)
    runner = ReviewRunner(result=review_result())

    result = run_review_stage(config, run_dir, codex_runner=runner, clock=fixed_clock)

    assert result.successful
    assert result.run_record.state == WorkflowState.REPORT
    assert runner.command is not None
    assert sandbox_value(runner.command.argv) == Sandbox.READ_ONLY.value
    review_dir = run_dir / REVIEW_DIR_NAME / ROUND_1
    assert result.artifact_directory == review_dir
    assert review_dir.joinpath("prompt.md").is_file()
    assert review_dir.joinpath("events.jsonl").is_file()
    assert review_dir.joinpath("stderr.log").is_file()
    assert (
        json.loads(review_dir.joinpath("result.json").read_text())["verdict"] == "PASS"
    )
    assert review_dir.joinpath("stderr.log").read_text() == "review progress\n"


@pytest.mark.skipif(GIT is None, reason="git executable is required for review tests")
def test_review_rejects_non_verify_starting_state(tmp_path):
    _repo, run_dir, config = verified_run(tmp_path)
    record_path = run_dir / "run.json"
    save_run_record(
        load_run_record(record_path).with_state(
            WorkflowState.IMPLEMENT,
            updated_timestamp="2026-09-11T13:05:15Z",
        ),
        record_path,
    )

    with pytest.raises(ReviewError, match="requires run state VERIFY"):
        run_review_stage(config, run_dir, codex_runner=ReviewRunner(review_result()))


@pytest.mark.skipif(GIT is None, reason="git executable is required for review tests")
@pytest.mark.parametrize("verification_status", ["FAIL", "ERROR"])
def test_review_rejects_non_passing_verification_results_without_invoking_codex(
    tmp_path,
    verification_status,
):
    _repo, run_dir, config = verified_run(
        tmp_path,
        verification_status=verification_status,
    )
    runner = ReviewRunner(result=review_result())

    with pytest.raises(ReviewError, match="must have PASS status before review"):
        run_review_stage(config, run_dir, codex_runner=runner)

    assert runner.calls == 0
    assert load_run_record(run_dir / "run.json").state == WorkflowState.VERIFY
    assert not run_dir.joinpath(REVIEW_DIR_NAME, ROUND_1, "prompt.md").exists()


@pytest.mark.skipif(GIT is None, reason="git executable is required for review tests")
def test_pass_review_accepts_advisory_and_follow_up_observations(tmp_path):
    _repo, run_dir, config = verified_run(tmp_path)
    result_payload = review_result(
        findings=[
            finding("R1-F1", disposition="ADVISORY"),
            finding("R1-F2", disposition="FOLLOW_UP"),
        ]
    )

    result = run_review_stage(
        config,
        run_dir,
        codex_runner=ReviewRunner(result=result_payload),
    )

    assert result.run_record.state == WorkflowState.REPORT
    assert result.required_findings == ()
    assert [item["disposition"] for item in result.review_result["findings"]] == [
        "ADVISORY",
        "FOLLOW_UP",
    ]


@pytest.mark.skipif(GIT is None, reason="git executable is required for review tests")
def test_corrections_required_moves_to_correct_and_exposes_required_findings(tmp_path):
    _repo, run_dir, config = verified_run(tmp_path)
    required = finding("R1-F1", disposition="REQUIRED")

    result = run_review_stage(
        config,
        run_dir,
        codex_runner=ReviewRunner(
            result=review_result(
                verdict="CORRECTIONS_REQUIRED",
                findings=[required],
            )
        ),
    )

    assert result.run_record.state == WorkflowState.CORRECT
    assert result.required_findings == (required,)
    assert "required corrections" in result.controller_message


@pytest.mark.skipif(GIT is None, reason="git executable is required for review tests")
def test_human_review_required_moves_to_human_required(tmp_path):
    _repo, run_dir, config = verified_run(tmp_path)

    result = run_review_stage(
        config,
        run_dir,
        codex_runner=ReviewRunner(
            result=review_result(
                verdict="HUMAN_REVIEW_REQUIRED",
                summary="Repository authority is ambiguous.",
                confidence="LOW",
            )
        ),
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert result.review_result["verdict"] == "HUMAN_REVIEW_REQUIRED"


@pytest.mark.skipif(GIT is None, reason="git executable is required for review tests")
def test_pass_with_required_finding_is_rejected_without_changing_verdict(tmp_path):
    _repo, run_dir, config = verified_run(tmp_path)

    result = run_review_stage(
        config,
        run_dir,
        codex_runner=ReviewRunner(
            result=review_result(findings=[finding("R1-F1", disposition="REQUIRED")])
        ),
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert result.processing_error == "PASS results must not contain REQUIRED findings."
    assert result.review_result["verdict"] == "PASS"
    saved = json.loads(
        run_dir.joinpath(REVIEW_DIR_NAME, ROUND_1, "result.json").read_text()
    )
    assert saved["verdict"] == "PASS"


@pytest.mark.skipif(GIT is None, reason="git executable is required for review tests")
def test_corrections_required_without_required_findings_is_rejected(tmp_path):
    _repo, run_dir, config = verified_run(tmp_path)

    result = run_review_stage(
        config,
        run_dir,
        codex_runner=ReviewRunner(
            result=review_result(
                verdict="CORRECTIONS_REQUIRED",
                findings=[finding("R1-F1", disposition="ADVISORY")],
            )
        ),
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert (
        result.processing_error
        == "CORRECTIONS_REQUIRED results must contain at least one REQUIRED finding."
    )


@pytest.mark.skipif(GIT is None, reason="git executable is required for review tests")
def test_branch_invariant_is_rechecked_after_review(tmp_path):
    repo, run_dir, config = verified_run(tmp_path)

    result = run_review_stage(
        config,
        run_dir,
        codex_runner=ReviewRunner(
            result=review_result(),
            mutation=lambda cwd: run_git(cwd, "checkout", "-b", "review-branch"),
        ),
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert [violation.name for violation in result.safety_violations] == ["branch"]
    assert run_git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "review-branch"


@pytest.mark.skipif(GIT is None, reason="git executable is required for review tests")
def test_head_invariant_is_rechecked_after_review(tmp_path):
    repo, run_dir, config = verified_run(tmp_path)

    def commit_change(cwd: Path) -> None:
        cwd.joinpath("file.txt").write_text("review committed\n", encoding="utf-8")
        run_git(cwd, "add", "file.txt")
        run_git(cwd, "commit", "-m", "review changed head")

    result = run_review_stage(
        config,
        run_dir,
        codex_runner=ReviewRunner(result=review_result(), mutation=commit_change),
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert [violation.name for violation in result.safety_violations] == ["HEAD"]
    assert run_git(repo, "rev-parse", "HEAD") != result.run_record.baseline_sha


@pytest.mark.skipif(GIT is None, reason="git executable is required for review tests")
def test_staging_invariant_is_rechecked_after_review(tmp_path):
    _repo, run_dir, config = verified_run(tmp_path)

    def stage_change(cwd: Path) -> None:
        cwd.joinpath("file.txt").write_text("review staged\n", encoding="utf-8")
        run_git(cwd, "add", "file.txt")

    result = run_review_stage(
        config,
        run_dir,
        codex_runner=ReviewRunner(result=review_result(), mutation=stage_change),
    )

    assert result.run_record.state == WorkflowState.HUMAN_REQUIRED
    assert [violation.name for violation in result.safety_violations] == ["staging"]


@pytest.mark.skipif(GIT is None, reason="git executable is required for review tests")
def test_disposable_repository_does_not_require_earlier_ticket_history(tmp_path):
    repo, run_dir, config = verified_run(
        tmp_path,
        ticket_text="# Synthetic Review\n\nNo earlier TicketAutomation ticket exists.\n",
    )

    result = run_review_stage(
        config,
        run_dir,
        codex_runner=ReviewRunner(result=review_result()),
    )

    assert result.run_record.state == WorkflowState.REPORT
    assert run_git(repo, "log", "--oneline").count("initial") == 1


def test_review_result_schema_accepts_supported_verdicts_and_finding_dispositions():
    schema = review_schema()

    validate_json_schema(review_result(), schema)
    validate_json_schema(
        review_result(
            verdict="CORRECTIONS_REQUIRED",
            findings=[finding("R1-F1", disposition="REQUIRED")],
        ),
        schema,
    )
    validate_json_schema(review_result(verdict="HUMAN_REVIEW_REQUIRED"), schema)
    validate_json_schema(
        review_result(findings=[finding("R1-F2", disposition="FOLLOW_UP")]),
        schema,
    )


@pytest.mark.parametrize(
    "patch",
    [
        {"verdict": "BLOCKED"},
        {"confidence": "CERTAIN"},
        {"findings": [{"id": "R1-F1"}]},
        {"summary": ""},
    ],
)
def test_review_result_schema_rejects_unsupported_results(patch):
    schema = review_schema()
    result = review_result()
    result.update(patch)

    with pytest.raises(CodexResultValidationError):
        validate_json_schema(result, schema)


def test_review_result_semantics_reject_contradictions():
    with pytest.raises(ReviewResultConsistencyError):
        validate_review_result_semantics(
            review_result(findings=[finding("R1-F1", disposition="REQUIRED")])
        )
    with pytest.raises(ReviewResultConsistencyError):
        validate_review_result_semantics(review_result(verdict="CORRECTIONS_REQUIRED"))


def verified_run(
    tmp_path,
    *,
    ticket_text: str = "# Synthetic Review Ticket\n\nReview this implementation.\n",
    verification_status: str = "PASS",
) -> tuple[Path, Path, AppConfig]:
    repo = create_git_repo(tmp_path / "repo")
    ticket = tmp_path / "synthetic-review-ticket.md"
    ticket.write_bytes(ticket_text.encode("utf-8"))
    config = make_config(repo)
    snapshot = create_run_snapshot(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        clock=fixed_clock,
    )
    repo.joinpath("file.txt").write_text("implemented\n", encoding="utf-8")
    implementation_dir = snapshot.run_dir / "implementation"
    implementation_dir.mkdir()
    implementation_dir.joinpath("result.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED",
                "summary": "implemented synthetic ticket",
                "tests_run": [{"command": "fake", "result": "PASS"}],
                "assumptions": [],
                "known_issues": [],
            }
        ),
        encoding="utf-8",
    )
    verification_dir = snapshot.run_dir / "verification"
    verification_dir.mkdir()
    verification_dir.joinpath("round-0.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "ticket_automation.verification_round",
                "round_index": 0,
                "status": verification_status,
                "commands": [
                    {
                        "name": "tests",
                        "argv": ["fake", "test"],
                        "cwd": str(repo),
                        "status": verification_status,
                        "stdout": (
                            "deterministic passed\n"
                            if verification_status == "PASS"
                            else f"deterministic {verification_status.lower()}\n"
                        ),
                        "stderr": "",
                        "exit_code": 0 if verification_status == "PASS" else 1,
                    }
                ],
                "safety_violations": [],
                "correction_reasons": [],
            }
        ),
        encoding="utf-8",
    )
    verification_dir.joinpath("round-0.log").write_text(
        "deterministic passed\n",
        encoding="utf-8",
    )
    run_record_path = snapshot.run_dir / "run.json"
    save_run_record(
        load_run_record(run_record_path).with_state(
            WorkflowState.VERIFY,
            updated_timestamp="2026-09-11T13:05:15Z",
        ),
        run_record_path,
    )
    return repo, snapshot.run_dir, config


def review_result(
    *,
    verdict: str = ReviewVerdict.PASS.value,
    summary: str = "review passed",
    confidence: str = "HIGH",
    findings: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "verdict": verdict,
        "summary": summary,
        "confidence": confidence,
        "findings": [] if findings is None else findings,
    }


def finding(
    finding_id: str,
    *,
    disposition: str,
    severity: str = "MEDIUM",
) -> dict[str, object]:
    return {
        "id": finding_id,
        "severity": severity,
        "category": "CORRECTNESS",
        "disposition": disposition,
        "scope_relation": "TICKET",
        "title": "Synthetic finding",
        "description": "Synthetic review finding description.",
        "evidence": "Synthetic evidence.",
        "required_change": "Make the required synthetic change.",
        "acceptance_criteria": ["Synthetic acceptance criterion."],
    }


def event_stream(result: dict[str, object]) -> str:
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


def sandbox_value(argv: tuple[str, ...]) -> str:
    sandbox_index = argv.index("--sandbox")
    return argv[sandbox_index + 1]


def review_schema() -> dict[str, object]:
    schema_path = PROJECT_ROOT / "schemas" / "review-result.schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))
