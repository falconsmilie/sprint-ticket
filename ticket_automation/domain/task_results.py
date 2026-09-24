"""Immutable task result concepts shared by application stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias


class ResultValidationError(ValueError):
    """Raised when task result data violates the domain contract."""


class ReviewResultConsistencyError(ResultValidationError):
    """Raised when a review verdict contradicts its required findings."""


class ImplementationStatus(StrEnum):
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"


class ReviewVerdict(StrEnum):
    PASS = "PASS"
    CORRECTIONS_REQUIRED = "CORRECTIONS_REQUIRED"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"


class FindingDisposition(StrEnum):
    REQUIRED = "REQUIRED"
    ADVISORY = "ADVISORY"
    FOLLOW_UP = "FOLLOW_UP"


class FindingScopeRelation(StrEnum):
    TICKET = "TICKET"
    IMPLEMENTATION = "IMPLEMENTATION"
    REPOSITORY_AUTHORITY = "REPOSITORY_AUTHORITY"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    AMBIGUOUS = "AMBIGUOUS"


_AUTOMATIC_CORRECTION_SCOPE_RELATIONS = frozenset(
    {FindingScopeRelation.TICKET, FindingScopeRelation.IMPLEMENTATION}
)


def _require_non_empty_text(value: object, *, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ResultValidationError(f"{field} must be a non-empty string.")


def _require_enum(value: object, enum_type: type[StrEnum], *, field: str) -> None:
    if not isinstance(value, enum_type):
        supported = ", ".join(item.value for item in enum_type)
        raise ResultValidationError(f"{field} must be one of: {supported}.")


def _require_text_tuple(value: object, *, field: str, minimum_items: int = 0) -> None:
    if not isinstance(value, tuple):
        raise ResultValidationError(f"{field} must be a tuple.")
    if len(value) < minimum_items:
        raise ResultValidationError(
            f"{field} must contain at least {minimum_items} item(s)."
        )
    for index, item in enumerate(value):
        _require_non_empty_text(item, field=f"{field}[{index}]")


@dataclass(frozen=True)
class ImplementationTestResult:
    command: str
    result: str

    def __post_init__(self) -> None:
        _require_non_empty_text(self.command, field="implementation test command")
        _require_non_empty_text(self.result, field="implementation test result")


@dataclass(frozen=True)
class ImplementationResult:
    status: ImplementationStatus
    summary: str
    tests_run: tuple[ImplementationTestResult, ...]
    assumptions: tuple[str, ...]
    known_issues: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_enum(self.status, ImplementationStatus, field="implementation status")
        _require_non_empty_text(self.summary, field="implementation summary")
        if not isinstance(self.tests_run, tuple):
            raise ResultValidationError("implementation tests_run must be a tuple.")
        for index, test in enumerate(self.tests_run):
            if not isinstance(test, ImplementationTestResult):
                raise ResultValidationError(
                    f"implementation tests_run[{index}] must be an "
                    "ImplementationTestResult."
                )
        _require_text_tuple(self.assumptions, field="implementation assumptions")
        _require_text_tuple(self.known_issues, field="implementation known_issues")


@dataclass(frozen=True)
class ReviewFinding:
    id: str
    disposition: FindingDisposition
    scope_relation: FindingScopeRelation
    title: str
    description: str
    evidence: str
    required_change: str
    acceptance_criteria: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_non_empty_text(self.id, field="review finding id")
        _require_enum(
            self.disposition,
            FindingDisposition,
            field="review finding disposition",
        )
        _require_enum(
            self.scope_relation,
            FindingScopeRelation,
            field="review finding scope_relation",
        )
        _require_non_empty_text(self.title, field="review finding title")
        _require_non_empty_text(self.description, field="review finding description")
        _require_non_empty_text(self.evidence, field="review finding evidence")
        _require_non_empty_text(
            self.required_change,
            field="review finding required_change",
        )
        _require_text_tuple(
            self.acceptance_criteria,
            field="review finding acceptance_criteria",
            minimum_items=1,
        )

    @property
    def correction_eligible(self) -> bool:
        """Whether application policy permits an automatic correction round."""

        return (
            self.disposition is FindingDisposition.REQUIRED
            and self.scope_relation in _AUTOMATIC_CORRECTION_SCOPE_RELATIONS
        )


@dataclass(frozen=True)
class ReviewResult:
    verdict: ReviewVerdict
    summary: str
    findings: tuple[ReviewFinding, ...]

    def __post_init__(self) -> None:
        _require_enum(self.verdict, ReviewVerdict, field="review verdict")
        _require_non_empty_text(self.summary, field="review summary")
        if not isinstance(self.findings, tuple):
            raise ResultValidationError("review findings must be a tuple.")
        for index, finding in enumerate(self.findings):
            if not isinstance(finding, ReviewFinding):
                raise ResultValidationError(
                    f"review findings[{index}] must be a ReviewFinding."
                )
        required_count = len(self.required_findings)
        if self.verdict is ReviewVerdict.PASS and required_count:
            raise ReviewResultConsistencyError(
                "PASS results must not contain REQUIRED findings."
            )
        if self.verdict is ReviewVerdict.CORRECTIONS_REQUIRED and required_count == 0:
            raise ReviewResultConsistencyError(
                "CORRECTIONS_REQUIRED results must contain at least one REQUIRED "
                "finding."
            )

    @property
    def required_findings(self) -> tuple[ReviewFinding, ...]:
        return tuple(
            finding
            for finding in self.findings
            if finding.disposition is FindingDisposition.REQUIRED
        )


TaskResult: TypeAlias = ImplementationResult | ReviewResult


__all__ = [
    "FindingDisposition",
    "FindingScopeRelation",
    "ImplementationResult",
    "ImplementationStatus",
    "ImplementationTestResult",
    "ResultValidationError",
    "ReviewFinding",
    "ReviewResult",
    "ReviewResultConsistencyError",
    "ReviewVerdict",
    "TaskResult",
]
