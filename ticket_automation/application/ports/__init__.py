"""Application-owned ports shared with concrete adapters."""

from .preflight import (
    PreflightCheck,
    PreflightResult,
    PreflightStatus,
    ProviderPreflight,
)

__all__ = [
    "PreflightCheck",
    "PreflightResult",
    "PreflightStatus",
    "ProviderPreflight",
]
