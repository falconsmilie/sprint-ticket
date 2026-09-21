"""JSON boundary codecs for immutable task results."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypeVar

from .domain.task_results import (
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


def decode_implementation_result(value: object) -> ImplementationResult:
    result = _require_exact_object(
        value,
        required=("status", "summary", "tests_run", "assumptions", "known_issues"),
        source="implementation result",
    )
    tests_value = result["tests_run"]
    if not isinstance(tests_value, list):
        raise ResultValidationError("implementation result.tests_run must be an array.")
    tests: list[ImplementationTestResult] = []
    for index, item in enumerate(tests_value):
        source = f"implementation result.tests_run[{index}]"
        test = _require_exact_object(
            item,
            required=("command", "result"),
            source=source,
        )
        tests.append(
            ImplementationTestResult(
                command=_require_string(test, "command", source=source),
                result=_require_string(test, "result", source=source),
            )
        )
    return ImplementationResult(
        status=_require_enum(
            result,
            "status",
            enum_type=ImplementationStatus,
            source="implementation result",
        ),
        summary=_require_string(result, "summary", source="implementation result"),
        tests_run=tuple(tests),
        assumptions=_require_string_tuple(
            result["assumptions"], source="implementation result.assumptions"
        ),
        known_issues=_require_string_tuple(
            result["known_issues"], source="implementation result.known_issues"
        ),
    )


def encode_implementation_result(result: ImplementationResult) -> dict[str, Any]:
    if not isinstance(result, ImplementationResult):
        raise ResultValidationError("value must be an ImplementationResult.")
    return {
        "status": result.status.value,
        "summary": result.summary,
        "tests_run": [
            {"command": test.command, "result": test.result}
            for test in result.tests_run
        ],
        "assumptions": list(result.assumptions),
        "known_issues": list(result.known_issues),
    }


def decode_review_result(value: object) -> ReviewResult:
    result = _require_exact_object(
        value,
        required=("verdict", "summary", "findings"),
        source="review result",
    )
    findings_value = result["findings"]
    if not isinstance(findings_value, list):
        raise ResultValidationError("review result.findings must be an array.")
    findings = tuple(
        _decode_review_finding(item, index=index)
        for index, item in enumerate(findings_value)
    )
    return ReviewResult(
        verdict=_require_enum(
            result,
            "verdict",
            enum_type=ReviewVerdict,
            source="review result",
        ),
        summary=_require_string(result, "summary", source="review result"),
        findings=findings,
    )


def encode_review_result(result: ReviewResult) -> dict[str, Any]:
    if not isinstance(result, ReviewResult):
        raise ResultValidationError("value must be a ReviewResult.")
    return {
        "verdict": result.verdict.value,
        "summary": result.summary,
        "findings": [
            {
                "id": finding.id,
                "disposition": finding.disposition.value,
                "scope_relation": finding.scope_relation.value,
                "title": finding.title,
                "description": finding.description,
                "evidence": finding.evidence,
                "required_change": finding.required_change,
                "acceptance_criteria": list(finding.acceptance_criteria),
            }
            for finding in result.findings
        ],
    }


def _decode_review_finding(value: object, *, index: int) -> ReviewFinding:
    source = f"review result.findings[{index}]"
    finding = _require_exact_object(
        value,
        required=(
            "id",
            "disposition",
            "scope_relation",
            "title",
            "description",
            "evidence",
            "required_change",
            "acceptance_criteria",
        ),
        source=source,
    )
    return ReviewFinding(
        id=_require_string(finding, "id", source=source),
        disposition=_require_enum(
            finding,
            "disposition",
            enum_type=FindingDisposition,
            source=source,
        ),
        scope_relation=_require_enum(
            finding,
            "scope_relation",
            enum_type=FindingScopeRelation,
            source=source,
        ),
        title=_require_string(finding, "title", source=source),
        description=_require_string(finding, "description", source=source),
        evidence=_require_string(finding, "evidence", source=source),
        required_change=_require_string(finding, "required_change", source=source),
        acceptance_criteria=_require_string_tuple(
            finding["acceptance_criteria"],
            source=f"{source}.acceptance_criteria",
            minimum_items=1,
        ),
    )


def _require_exact_object(
    value: object,
    *,
    required: tuple[str, ...],
    source: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ResultValidationError(f"{source} must be an object.")
    if any(not isinstance(key, str) for key in value):
        raise ResultValidationError(f"{source} field names must be strings.")
    missing = [field for field in required if field not in value]
    if missing:
        raise ResultValidationError(
            f"{source} is missing required fields: {', '.join(missing)}."
        )
    extra = sorted(set(value) - set(required))
    if extra:
        raise ResultValidationError(
            f"{source} contains unsupported fields: {', '.join(extra)}."
        )
    return value


def _require_string(data: dict[str, object], field: str, *, source: str) -> str:
    value = data[field]
    if not isinstance(value, str) or not value.strip():
        raise ResultValidationError(f"{source}.{field} must be a non-empty string.")
    return value


_EnumType = TypeVar("_EnumType", bound=StrEnum)


def _require_enum(
    data: dict[str, object],
    field: str,
    *,
    enum_type: type[_EnumType],
    source: str,
) -> _EnumType:
    value = _require_string(data, field, source=source)
    try:
        return enum_type(value)
    except ValueError as error:
        supported = ", ".join(item.value for item in enum_type)
        raise ResultValidationError(
            f"{source}.{field} must be one of: {supported}."
        ) from error


def _require_string_tuple(
    value: object,
    *,
    source: str,
    minimum_items: int = 0,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ResultValidationError(f"{source} must be an array.")
    if len(value) < minimum_items:
        raise ResultValidationError(
            f"{source} must contain at least {minimum_items} item(s)."
        )
    strings: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ResultValidationError(
                f"{source}[{index}] must be a non-empty string."
            )
        strings.append(item)
    return tuple(strings)


__all__ = [
    "decode_implementation_result",
    "decode_review_result",
    "encode_implementation_result",
    "encode_review_result",
]
