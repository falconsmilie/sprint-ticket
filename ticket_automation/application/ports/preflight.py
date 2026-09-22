"""Provider-neutral preflight result contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class PreflightStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: PreflightStatus
    message: str = ""

    @property
    def passed(self) -> bool:
        return self.status == PreflightStatus.PASS


@dataclass(frozen=True)
class PreflightResult:
    checks: tuple[PreflightCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> tuple[PreflightCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


class ProviderPreflight(Protocol):
    """Run checks owned by the providers assigned to the current run."""

    def __call__(self, *, repository_path: Path) -> PreflightResult: ...


__all__ = [
    "PreflightCheck",
    "PreflightResult",
    "PreflightStatus",
    "ProviderPreflight",
]
