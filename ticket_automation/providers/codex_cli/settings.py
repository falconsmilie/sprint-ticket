"""Codex CLI settings and validation."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CODEX_EXECUTABLE = "codex"
DEFAULT_CODEX_MODEL = "gpt-5.5"
DEFAULT_CODEX_REASONING_EFFORT = "xhigh"
DEFAULT_TIMEOUT_SECONDS = 60 * 60
SUPPORTED_CODEX_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh"}
)


class CodexSettingsError(ValueError):
    """Raised when Codex CLI settings cannot be executed safely."""


@dataclass(frozen=True)
class CodexExecutionSettings:
    model: str
    reasoning_effort: str


@dataclass(frozen=True)
class CodexExecutionOverrides:
    model: str | None = None
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class CodexSettings:
    executable: str
    model: str
    reasoning_effort: str

    @property
    def execution(self) -> CodexExecutionSettings:
        return CodexExecutionSettings(self.model, self.reasoning_effort)


def validate_codex_execution_settings(
    settings: CodexExecutionSettings,
    *,
    model_name: str = "codex.model",
    reasoning_name: str = "codex.reasoning_effort",
) -> CodexExecutionSettings:
    if not isinstance(settings, CodexExecutionSettings):
        raise CodexSettingsError("settings must be CodexExecutionSettings.")
    model = _non_empty(settings.model, model_name)
    effort = _non_empty(settings.reasoning_effort, reasoning_name)
    if effort not in SUPPORTED_CODEX_REASONING_EFFORTS:
        supported = ", ".join(sorted(SUPPORTED_CODEX_REASONING_EFFORTS))
        raise CodexSettingsError(f"{reasoning_name} must be one of: {supported}.")
    return CodexExecutionSettings(model=model, reasoning_effort=effort)


def validate_codex_settings(settings: CodexSettings) -> CodexSettings:
    if not isinstance(settings, CodexSettings):
        raise CodexSettingsError("settings must be CodexSettings.")
    executable = _non_empty(settings.executable, "codex.executable")
    execution = validate_codex_execution_settings(settings.execution)
    return CodexSettings(
        executable=executable,
        model=execution.model,
        reasoning_effort=execution.reasoning_effort,
    )


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CodexSettingsError(f"Missing required non-empty string: {name}.")
    return value.strip()


__all__ = [
    "DEFAULT_CODEX_EXECUTABLE",
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_CODEX_REASONING_EFFORT",
    "DEFAULT_TIMEOUT_SECONDS",
    "SUPPORTED_CODEX_REASONING_EFFORTS",
    "CodexExecutionOverrides",
    "CodexExecutionSettings",
    "CodexSettings",
    "CodexSettingsError",
    "validate_codex_execution_settings",
    "validate_codex_settings",
]
