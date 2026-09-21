from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from ticket_automation.attempts import complete_attempt, start_attempt
from ticket_automation.domain.task_results import (
    FindingDisposition,
    FindingScopeRelation,
    ImplementationResult,
    ImplementationStatus,
    ImplementationTestResult,
    ResultValidationError,
    ReviewFinding,
    ReviewResult,
    ReviewVerdict,
)
from ticket_automation.models import AttemptPhase, AttemptStatus
from ticket_automation.reporting import latest_review_result
from ticket_automation.task_result_codecs import (
    decode_implementation_result,
    decode_review_result,
    encode_implementation_result,
    encode_review_result,
)


def implementation_json(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "status": "COMPLETED",
        "summary": "Implemented the ticket.",
        "tests_run": [{"command": "python -m pytest", "result": "passed"}],
        "assumptions": ["The configured interpreter is authoritative."],
        "known_issues": [],
    }
    value.update(changes)
    return value


def finding_json(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "id": "F-1",
        "disposition": "REQUIRED",
        "scope_relation": "IMPLEMENTATION",
        "title": "Handle invalid input",
        "description": "Invalid input is accepted.",
        "evidence": "The malformed fixture completed successfully.",
        "required_change": "Reject the malformed fixture.",
        "acceptance_criteria": ["The malformed fixture is rejected."],
    }
    value.update(changes)
    return value


def review_json(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "verdict": "CORRECTIONS_REQUIRED",
        "summary": "One required correction was found.",
        "findings": [finding_json()],
    }
    value.update(changes)
    return value


def review_finding(**changes: object) -> ReviewFinding:
    values: dict[str, object] = {
        "id": "F-1",
        "disposition": FindingDisposition.REQUIRED,
        "scope_relation": FindingScopeRelation.IMPLEMENTATION,
        "title": "Handle invalid input",
        "description": "Invalid input is accepted.",
        "evidence": "The malformed fixture completed successfully.",
        "required_change": "Reject the malformed fixture.",
        "acceptance_criteria": ("The malformed fixture is rejected.",),
    }
    values.update(changes)
    return ReviewFinding(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("member", "transport_value"),
    [
        (ImplementationStatus.COMPLETED, "COMPLETED"),
        (ImplementationStatus.BLOCKED, "BLOCKED"),
        (ReviewVerdict.PASS, "PASS"),
        (ReviewVerdict.CORRECTIONS_REQUIRED, "CORRECTIONS_REQUIRED"),
        (ReviewVerdict.HUMAN_REVIEW_REQUIRED, "HUMAN_REVIEW_REQUIRED"),
        (FindingDisposition.REQUIRED, "REQUIRED"),
        (FindingDisposition.ADVISORY, "ADVISORY"),
        (FindingDisposition.FOLLOW_UP, "FOLLOW_UP"),
        (FindingScopeRelation.TICKET, "TICKET"),
        (FindingScopeRelation.IMPLEMENTATION, "IMPLEMENTATION"),
        (FindingScopeRelation.REPOSITORY_AUTHORITY, "REPOSITORY_AUTHORITY"),
        (FindingScopeRelation.OUT_OF_SCOPE, "OUT_OF_SCOPE"),
        (FindingScopeRelation.AMBIGUOUS, "AMBIGUOUS"),
    ],
)
def test_task_result_enum_values_are_the_transport_values(
    member: object,
    transport_value: str,
) -> None:
    assert member.value == transport_value  # type: ignore[union-attr]


def test_value_objects_support_valid_frozen_trusted_construction() -> None:
    test_result = ImplementationTestResult(command="pytest", result="passed")
    implementation = ImplementationResult(
        status=ImplementationStatus.COMPLETED,
        summary="Implemented the ticket.",
        tests_run=(test_result,),
        assumptions=("The interpreter is configured.",),
        known_issues=(),
    )
    finding = review_finding()
    review = ReviewResult(
        verdict=ReviewVerdict.CORRECTIONS_REQUIRED,
        summary="A correction is required.",
        findings=(finding,),
    )

    assert implementation.tests_run == (test_result,)
    assert review.required_findings == (finding,)
    for value, field in (
        (test_result, "command"),
        (implementation, "summary"),
        (finding, "title"),
        (review, "summary"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(value, field, "changed")


@pytest.mark.parametrize(("field", "value"), [("command", ""), ("result", "  ")])
def test_implementation_test_result_requires_text(field: str, value: str) -> None:
    values = {"command": "pytest", "result": "passed", field: value}
    with pytest.raises(ResultValidationError, match="non-empty string"):
        ImplementationTestResult(**values)


def test_implementation_result_is_frozen_and_requires_domain_values() -> None:
    result = decode_implementation_result(implementation_json())
    mutable_result: Any = result
    with pytest.raises(FrozenInstanceError):
        mutable_result.summary = "changed"
    with pytest.raises(ResultValidationError, match="implementation status"):
        ImplementationResult(
            status="COMPLETED",  # type: ignore[arg-type]
            summary="summary",
            tests_run=(),
            assumptions=(),
            known_issues=(),
        )
    with pytest.raises(ResultValidationError, match="implementation summary"):
        ImplementationResult(
            status=ImplementationStatus.COMPLETED,
            summary="",
            tests_run=(),
            assumptions=(),
            known_issues=(),
        )
    with pytest.raises(ResultValidationError, match="tests_run must be a tuple"):
        ImplementationResult(
            status=ImplementationStatus.COMPLETED,
            summary="summary",
            tests_run=[],  # type: ignore[arg-type]
            assumptions=(),
            known_issues=(),
        )
    with pytest.raises(ResultValidationError, match=r"tests_run\[0\]"):
        ImplementationResult(
            status=ImplementationStatus.COMPLETED,
            summary="summary",
            tests_run=("pytest",),  # type: ignore[arg-type]
            assumptions=(),
            known_issues=(),
        )
    with pytest.raises(ResultValidationError, match=r"assumptions\[0\]"):
        ImplementationResult(
            status=ImplementationStatus.COMPLETED,
            summary="summary",
            tests_run=(),
            assumptions=(" ",),
            known_issues=(),
        )
    with pytest.raises(ResultValidationError, match=r"known_issues\[0\]"):
        ImplementationResult(
            status=ImplementationStatus.COMPLETED,
            summary="summary",
            tests_run=(),
            assumptions=(),
            known_issues=("",),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", "", "review finding id"),
        ("disposition", "REQUIRED", "review finding disposition"),
        ("scope_relation", "TICKET", "review finding scope_relation"),
        ("title", "", "review finding title"),
        ("description", "", "review finding description"),
        ("evidence", "", "review finding evidence"),
        ("required_change", "", "review finding required_change"),
        ("acceptance_criteria", (), "at least 1"),
        ("acceptance_criteria", ["criterion"], "must be a tuple"),
    ],
)
def test_review_finding_enforces_each_invariant(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(ResultValidationError, match=message):
        review_finding(**{field: value})


def test_review_result_enforces_verdict_finding_consistency() -> None:
    required = review_finding()
    with pytest.raises(ResultValidationError, match="PASS results"):
        ReviewResult(
            verdict=ReviewVerdict.PASS,
            summary="Contradictory pass.",
            findings=(required,),
        )
    with pytest.raises(ResultValidationError, match="at least one REQUIRED"):
        ReviewResult(
            verdict=ReviewVerdict.CORRECTIONS_REQUIRED,
            summary="Contradictory correction request.",
            findings=(
                review_finding(disposition=FindingDisposition.ADVISORY),
            ),
        )
    with pytest.raises(ResultValidationError, match="review verdict"):
        ReviewResult(
            verdict="PASS",  # type: ignore[arg-type]
            summary="Wrong enum type.",
            findings=(),
        )
    with pytest.raises(ResultValidationError, match="review summary"):
        ReviewResult(
            verdict=ReviewVerdict.PASS,
            summary="",
            findings=(),
        )
    with pytest.raises(ResultValidationError, match="findings must be a tuple"):
        ReviewResult(
            verdict=ReviewVerdict.PASS,
            summary="Wrong findings container.",
            findings=[],  # type: ignore[arg-type]
        )
    with pytest.raises(ResultValidationError, match=r"findings\[0\]"):
        ReviewResult(
            verdict=ReviewVerdict.HUMAN_REVIEW_REQUIRED,
            summary="Wrong finding value.",
            findings=("finding",),  # type: ignore[arg-type]
        )


def test_implementation_codec_round_trips_through_an_immutable_result() -> None:
    raw = implementation_json()
    result = decode_implementation_result(raw)
    encoded = encode_implementation_result(result)

    assert result.status is ImplementationStatus.COMPLETED
    assert isinstance(result.tests_run, tuple)
    assert encoded == raw
    assert encoded is not raw
    assert encoded["tests_run"] is not raw["tests_run"]


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        implementation_json(status="UNKNOWN"),
        implementation_json(summary=3),
        implementation_json(tests_run={}),
        implementation_json(tests_run=[{"command": "pytest"}]),
        implementation_json(assumptions="none"),
        implementation_json(known_issues=[1]),
    ],
)
def test_implementation_codec_rejects_malformed_data(value: object) -> None:
    with pytest.raises(ResultValidationError):
        decode_implementation_result(value)


def test_implementation_codec_rejects_absent_fields() -> None:
    value = implementation_json()
    del value["summary"]
    with pytest.raises(ResultValidationError, match="missing required fields: summary"):
        decode_implementation_result(value)


def test_review_codec_round_trips_through_an_immutable_result() -> None:
    raw = review_json()
    result = decode_review_result(raw)
    encoded = encode_review_result(result)

    assert result.verdict is ReviewVerdict.CORRECTIONS_REQUIRED
    assert result.required_findings == result.findings
    assert encoded == raw
    assert encoded is not raw
    assert encoded["findings"] is not raw["findings"]


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        review_json(verdict="UNKNOWN"),
        review_json(summary=False),
        review_json(findings={}),
        review_json(findings=[finding_json(disposition="UNKNOWN")]),
        review_json(findings=[finding_json(scope_relation="UNKNOWN")]),
        review_json(findings=[finding_json(acceptance_criteria=[])]),
        review_json(verdict="PASS"),
        review_json(verdict="CORRECTIONS_REQUIRED", findings=[]),
    ],
)
def test_review_codec_rejects_malformed_or_inconsistent_data(value: object) -> None:
    with pytest.raises(ResultValidationError):
        decode_review_result(value)


def test_review_codec_rejects_absent_fields() -> None:
    value = review_json()
    del value["findings"]
    with pytest.raises(ResultValidationError, match="missing required fields: findings"):
        decode_review_result(value)


def test_review_codec_rejects_absent_nested_finding_fields() -> None:
    finding = finding_json()
    del finding["evidence"]
    with pytest.raises(ResultValidationError, match="missing required fields: evidence"):
        decode_review_result(review_json(findings=[finding]))


def test_reporting_returns_a_typed_review_result(tmp_path: Path) -> None:
    attempt = start_attempt(
        tmp_path,
        phase=AttemptPhase.REVIEWING,
        before_workspace_fingerprint="before",
    )
    attempt.artifact_directory.joinpath("result.json").write_text(
        json.dumps(review_json()),
        encoding="utf-8",
    )
    complete_attempt(
        attempt,
        status=AttemptStatus.COMPLETED,
        after_workspace_fingerprint="after",
    )

    result = latest_review_result(tmp_path)

    assert isinstance(result, ReviewResult)
    assert result.verdict is ReviewVerdict.CORRECTIONS_REQUIRED


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (review_json(verdict="PASS"), "PASS results"),
        ([], "review result must be an object"),
    ],
)
def test_reporting_does_not_hide_invalid_review_evidence(
    tmp_path: Path,
    payload: object,
    message: str,
) -> None:
    attempt = start_attempt(
        tmp_path,
        phase=AttemptPhase.REVIEWING,
        before_workspace_fingerprint="before",
    )
    attempt.artifact_directory.joinpath("result.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    complete_attempt(
        attempt,
        status=AttemptStatus.HUMAN_REQUIRED,
        after_workspace_fingerprint="after",
    )

    with pytest.raises(ResultValidationError, match=message):
        latest_review_result(tmp_path)
