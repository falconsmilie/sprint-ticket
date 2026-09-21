from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tests.architecture_fitness import parse_imports
from ticket_automation.attempts import complete_attempt, start_attempt
from ticket_automation.correction_planner import (
    plan_pending_correction,
    plan_review_correction,
    plan_verification_correction,
)
from ticket_automation.corrections import (
    CorrectionCauseSet,
    CorrectionError,
    ReviewCorrectionCause,
    ReviewCorrectionScope,
    VerificationCorrectionCause,
    render_correction_ticket,
)
from ticket_automation.domain.task_results import (
    FindingDisposition,
    FindingScopeRelation,
    ReviewFinding,
    ReviewResult,
    ReviewVerdict,
)
from ticket_automation.task_result_codecs import encode_review_result
from ticket_automation.verification import VerificationError, VerificationFailure


def _finding(
    *,
    finding_id: str = "REV-1",
    scope_relation: FindingScopeRelation = FindingScopeRelation.IMPLEMENTATION,
) -> ReviewFinding:
    return ReviewFinding(
        id=finding_id,
        disposition=FindingDisposition.REQUIRED,
        scope_relation=scope_relation,
        title="Correct the parser",
        description="The parser drops a required field.",
        evidence="A focused test demonstrates the missing field.",
        required_change="Preserve the required field.",
        acceptance_criteria=("The focused test passes.",),
    )


def _failure(tmp_path: Path, *, exit_code: int | None = 1) -> VerificationFailure:
    return VerificationFailure(
        gate_name="unit",
        command=("python", "-m", "pytest"),
        failure_summary="Verification gate 'unit' exited with code 1.",
        stdout_excerpt="one failed",
        stderr_excerpt="traceback",
        exit_code=exit_code,
        result_path=tmp_path / "verification" / "result.json",
    )


def _completed_attempt(run_dir: Path, *, phase: str, payload: object) -> None:
    attempt = start_attempt(
        run_dir,
        phase=phase,
        before_workspace_fingerprint="before",
    )
    attempt.artifact_directory.joinpath("result.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    complete_attempt(
        attempt,
        status="COMPLETED",
        after_workspace_fingerprint="after",
    )


def test_verification_failures_map_to_typed_causes_and_complete_ticket(
    tmp_path: Path,
) -> None:
    cause_set = plan_verification_correction((_failure(tmp_path),))

    assert isinstance(cause_set.causes[0], VerificationCorrectionCause)
    ticket = render_correction_ticket(
        ticket_id="TA-1",
        round_number=1,
        cause_set=cause_set,
        run_dir=tmp_path,
    )
    for expected in (
        "# TA-1 - Corrective Round 1",
        "This corrective ticket addresses deterministic verification failures",
        "### Verification failure - unit",
        "python -m pytest",
        "Exit code:\n1",
        "Verification gate 'unit' exited with code 1.",
        "Stdout:\none failed",
        "Stderr:\ntraceback",
        "verification",
        "result.json",
        "## Constraints",
        "## Validation",
    ):
        assert expected in ticket


@pytest.mark.parametrize(("exit_code", "display"), [(None, "n/a"), (0, "0")])
def test_existing_verification_exit_code_states_remain_eligible(
    tmp_path: Path, exit_code: int | None, display: str
) -> None:
    cause_set = plan_verification_correction((_failure(tmp_path, exit_code=exit_code),))

    ticket = render_correction_ticket(
        ticket_id="TA-1",
        round_number=1,
        cause_set=cause_set,
    )
    assert f"Exit code:\n{display}" in ticket


@pytest.mark.parametrize(
    "scope_relation",
    [FindingScopeRelation.TICKET, FindingScopeRelation.IMPLEMENTATION],
)
def test_supported_review_findings_map_to_typed_causes_and_complete_ticket(
    scope_relation: FindingScopeRelation,
) -> None:
    result = ReviewResult(
        verdict=ReviewVerdict.CORRECTIONS_REQUIRED,
        summary="A correction is required.",
        findings=(_finding(scope_relation=scope_relation),),
    )

    cause_set = plan_review_correction(result)

    assert isinstance(cause_set.causes[0], ReviewCorrectionCause)
    ticket = render_correction_ticket(
        ticket_id="TA-1",
        round_number=2,
        cause_set=cause_set,
    )
    for expected in (
        "# TA-1 - Corrective Round 2",
        "independent review of TA-1",
        "### REV-1 - Correct the parser",
        scope_relation.value.replace("_", " ").title(),
        "The parser drops a required field.",
        "A focused test demonstrates the missing field.",
        "Preserve the required field.",
        "- The focused test passes.",
        "## Constraints",
        "## Validation",
    ):
        assert expected in ticket


@pytest.mark.parametrize(
    "scope_relation",
    [
        FindingScopeRelation.REPOSITORY_AUTHORITY,
        FindingScopeRelation.OUT_OF_SCOPE,
        FindingScopeRelation.AMBIGUOUS,
    ],
)
def test_unsafe_review_scopes_remain_human_required(
    scope_relation: FindingScopeRelation,
) -> None:
    result = ReviewResult(
        verdict=ReviewVerdict.CORRECTIONS_REQUIRED,
        summary="Human judgment is required.",
        findings=(_finding(scope_relation=scope_relation),),
    )

    assert result.required_findings[0].correction_eligible is False
    with pytest.raises(CorrectionError, match="not eligible"):
        plan_review_correction(result)


def test_mixed_eligible_and_ineligible_review_findings_are_rejected() -> None:
    result = ReviewResult(
        verdict=ReviewVerdict.CORRECTIONS_REQUIRED,
        summary="The findings require different owners.",
        findings=(
            _finding(finding_id="REV-1"),
            _finding(
                finding_id="REV-2",
                scope_relation=FindingScopeRelation.REPOSITORY_AUTHORITY,
            ),
        ),
    )

    with pytest.raises(CorrectionError, match="not eligible"):
        plan_review_correction(result)


def test_empty_mixed_and_unsupported_cause_sets_are_rejected(tmp_path: Path) -> None:
    verification = VerificationCorrectionCause(
        gate_name="unit",
        command=("pytest",),
        failure_summary="failed",
        stdout_excerpt="",
        stderr_excerpt="",
        exit_code=1,
        result_path=tmp_path / "result.json",
    )
    review = ReviewCorrectionCause(
        finding_id="REV-1",
        summary="Correct it",
        details="A defect exists.",
        evidence="test failure",
        required_change="Fix the defect.",
        acceptance_criteria=("Test passes.",),
        scope_relation=ReviewCorrectionScope.IMPLEMENTATION,
    )

    with pytest.raises(CorrectionError, match="at least one eligible cause"):
        CorrectionCauseSet(())
    with pytest.raises(CorrectionError, match="one source type"):
        CorrectionCauseSet((verification, review))
    with pytest.raises(CorrectionError, match="must be a tuple"):
        CorrectionCauseSet([verification])  # type: ignore[arg-type]
    with pytest.raises(CorrectionError, match="Unsupported correction cause"):
        CorrectionCauseSet((object(),))  # type: ignore[arg-type]


def test_invalid_review_cause_is_rejected() -> None:
    with pytest.raises(CorrectionError, match="scope_relation is not supported"):
        ReviewCorrectionCause(
            finding_id="REV-1",
            summary="Correct it",
            details="A defect exists.",
            evidence="test failure",
            required_change="Fix the defect.",
            acceptance_criteria=("Test passes.",),
            scope_relation="IMPLEMENTATION",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"gate_name": ""}, "gate_name"),
        ({"command": ()}, "command"),
        ({"command": ("pytest", "")}, "command"),
        ({"failure_summary": ""}, "failure_summary"),
        ({"stdout_excerpt": None}, "excerpts"),
        ({"exit_code": True}, "exit_code"),
        ({"result_path": "result.json"}, "result_path"),
    ],
)
def test_verification_failure_rejects_invalid_trusted_construction(
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "gate_name": "unit",
        "command": ("pytest",),
        "failure_summary": "failed",
        "stdout_excerpt": "",
        "stderr_excerpt": "",
        "exit_code": 1,
        "result_path": tmp_path / "result.json",
    }
    values.update(changes)

    with pytest.raises(VerificationError, match=message):
        VerificationFailure(**values)  # type: ignore[arg-type]


def test_typed_values_are_immutable(tmp_path: Path) -> None:
    failure = _failure(tmp_path)
    cause_set = plan_verification_correction((failure,))

    with pytest.raises(FrozenInstanceError):
        failure.gate_name = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        cause_set.causes = ()  # type: ignore[misc]


def test_plan_pending_correction_reconstructs_owned_verification_failure(
    tmp_path: Path,
) -> None:
    failure = _failure(tmp_path, exit_code=None)
    _completed_attempt(
        tmp_path,
        phase="VERIFYING",
        payload={"status": "FAIL", "correction_reasons": [failure.to_dict()]},
    )

    cause_set = plan_pending_correction(tmp_path)

    assert cause_set == plan_verification_correction((failure,))


def test_plan_pending_correction_maps_persisted_review_result(tmp_path: Path) -> None:
    result = ReviewResult(
        verdict=ReviewVerdict.CORRECTIONS_REQUIRED,
        summary="A correction is required.",
        findings=(_finding(),),
    )
    _completed_attempt(
        tmp_path,
        phase="REVIEWING",
        payload=encode_review_result(result),
    )

    assert plan_pending_correction(tmp_path) == plan_review_correction(result)


def test_plan_pending_correction_rejects_malformed_review_evidence(
    tmp_path: Path,
) -> None:
    _completed_attempt(
        tmp_path,
        phase="REVIEWING",
        payload={
            "verdict": "CORRECTIONS_REQUIRED",
            "summary": "A correction is required.",
            "findings": [{"id": "REV-1"}],
        },
    )

    with pytest.raises(CorrectionError, match="Review correction source is invalid"):
        plan_pending_correction(tmp_path)


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "FAIL", "correction_reasons": []},
        {
            "status": "FAIL",
            "correction_reasons": [{"kind": "VerificationFailure"}],
        },
        {
            "status": "FAIL",
            "correction_reasons": [
                {
                    "kind": "ReviewFinding",
                    "gate_name": "unit",
                    "command": ["pytest"],
                    "failure_summary": "failed",
                    "stdout_excerpt": "",
                    "stderr_excerpt": "",
                    "exit_code": 1,
                    "result_path": "result.json",
                }
            ],
        },
    ],
)
def test_plan_pending_correction_rejects_malformed_verification_evidence(
    tmp_path: Path,
    payload: object,
) -> None:
    _completed_attempt(tmp_path, phase="VERIFYING", payload=payload)

    with pytest.raises(CorrectionError):
        plan_pending_correction(tmp_path)


@pytest.mark.parametrize(
    "verdict", [ReviewVerdict.PASS, ReviewVerdict.HUMAN_REVIEW_REQUIRED]
)
def test_non_correction_review_results_are_rejected(verdict: ReviewVerdict) -> None:
    result = ReviewResult(verdict=verdict, summary="Review complete.", findings=())

    with pytest.raises(CorrectionError, match="CORRECTIONS_REQUIRED"):
        plan_review_correction(result)


def test_planner_rejects_wrong_typed_inputs() -> None:
    with pytest.raises(CorrectionError, match="must be a tuple"):
        plan_verification_correction([])  # type: ignore[arg-type]
    with pytest.raises(CorrectionError, match="VerificationFailure"):
        plan_verification_correction((object(),))  # type: ignore[arg-type]
    with pytest.raises(CorrectionError, match="ReviewResult"):
        plan_review_correction(object())  # type: ignore[arg-type]


def test_verification_and_review_do_not_import_correction_modules() -> None:
    package_root = Path(__file__).parents[1] / "ticket_automation"
    forbidden = {
        "ticket_automation.corrections",
        "ticket_automation.correction_planner",
    }
    for module_name in ("verification", "review"):
        path = package_root / f"{module_name}.py"
        edges = parse_imports(
            path.read_text(encoding="utf-8"),
            f"ticket_automation.{module_name}",
        )
        assert not forbidden.intersection(edge.imported_module for edge in edges)
