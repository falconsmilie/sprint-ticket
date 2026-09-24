"""Codex CLI settings and validation."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CODEX_MODEL = "gpt-5.5"
DEFAULT_CODEX_REASONING_EFFORT = "xhigh"
SUPPORTED_CODEX_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh"}
)


class CodexCliSettingsError(ValueError):
    """Raised when Codex CLI settings cannot be executed safely."""


@dataclass(frozen=True)
class CodexCliSettings:
    executable: str
    model: str
    reasoning_effort: str


def validate_codex_cli_settings(
    settings: CodexCliSettings,
    *,
    executable_name: str = "agents.providers.codex-cli.executable",
    model_name: str = "agents.providers.codex-cli.model",
    reasoning_name: str = "agents.providers.codex-cli.reasoning_effort",
) -> CodexCliSettings:
    if not isinstance(settings, CodexCliSettings):
        raise CodexCliSettingsError("settings must be CodexCliSettings.")
    executable = _non_empty(settings.executable, executable_name)
    model = _non_empty(settings.model, model_name)
    effort = _non_empty(settings.reasoning_effort, reasoning_name)
    if effort not in SUPPORTED_CODEX_REASONING_EFFORTS:
        supported = ", ".join(sorted(SUPPORTED_CODEX_REASONING_EFFORTS))
        raise CodexCliSettingsError(f"{reasoning_name} must be one of: {supported}.")
    return CodexCliSettings(
        executable=executable,
        model=model,
        reasoning_effort=effort,
    )


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CodexCliSettingsError(f"Missing required non-empty string: {name}.")
    return value.strip()


__all__ = [
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_CODEX_REASONING_EFFORT",
    "SUPPORTED_CODEX_REASONING_EFFORTS",
    "CodexCliSettings",
    "CodexCliSettingsError",
    "validate_codex_cli_settings",
]
