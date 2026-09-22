"""Composition contract implementation for the Codex CLI provider."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ...application.agent_execution import AgentCapability, AgentExecutor, ProviderId
from ...application.ports.preflight import PreflightCheck, PreflightStatus
from .adapter import CodexCliAgentExecutor
from .executable import resolve_executable
from .identity import CAPABILITIES, PROVIDER_ID
from .settings import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING_EFFORT,
    CodexExecutionSettings,
    CodexSettings,
    CodexSettingsError,
    validate_codex_execution_settings,
)
from .validation import CodexCliValidationError, cli_version, supports_ephemeral


@dataclass(frozen=True)
class CodexCliConfiguredSettings:
    executable: str
    execution: CodexExecutionSettings
    configuration_directory: Path


@dataclass(frozen=True)
class CodexCliRunPolicy:
    executable: Path
    model: str
    reasoning_effort: str
    cli_version: str


class CodexCliProviderRegistration:
    provider_id: ProviderId = PROVIDER_ID
    capabilities: frozenset[AgentCapability] = CAPABILITIES

    def resolve_settings(
        self,
        raw_settings: Mapping[str, object],
        *,
        configuration_directory: Path,
    ) -> CodexCliConfiguredSettings:
        allowed = {"executable", "model", "reasoning_effort"}
        unknown = sorted(set(raw_settings) - allowed)
        if unknown:
            raise CodexSettingsError(
                "Unknown setting(s): " + ", ".join(unknown) + "."
            )
        executable = _non_empty(
            raw_settings.get("executable"),
            "agents.providers.codex-cli.executable",
        )
        execution = validate_codex_execution_settings(
            CodexExecutionSettings(
                model=raw_settings.get("model", DEFAULT_CODEX_MODEL),  # type: ignore[arg-type]
                reasoning_effort=raw_settings.get(
                    "reasoning_effort", DEFAULT_CODEX_REASONING_EFFORT
                ),  # type: ignore[arg-type]
            ),
            model_name="agents.providers.codex-cli.model",
            reasoning_name="agents.providers.codex-cli.reasoning_effort",
        )
        return CodexCliConfiguredSettings(
            executable=executable,
            execution=execution,
            configuration_directory=configuration_directory,
        )

    def run_preflight(
        self,
        settings: object,
        *,
        repository_path: Path,
    ) -> tuple[PreflightCheck, ...]:
        configured = _configured_settings(settings)
        checks: list[PreflightCheck] = []
        project_config = repository_path / ".codex" / "config.toml"
        if project_config.is_file():
            checks.append(
                _fail(
                    "project configuration",
                    "Target repository contains .codex/config.toml; its execution "
                    "policy cannot be isolated reliably.",
                )
            )
        else:
            checks.append(_pass("project configuration"))

        executable = resolve_executable(
            configured.executable,
            config_dir=configured.configuration_directory,
        )
        if executable is None:
            checks.append(
                _fail("executable", f"Executable not found: {configured.executable}")
            )
            return tuple(checks)
        checks.append(_pass("executable", str(executable)))
        if supports_ephemeral(executable, cwd=repository_path):
            checks.append(_pass("isolated invocation"))
        else:
            checks.append(
                _fail(
                    "isolated invocation",
                    "Codex CLI does not advertise required --ephemeral support.",
                )
            )
        return tuple(checks)

    def resolve_run_policy(self, settings: object) -> CodexCliRunPolicy:
        configured = _configured_settings(settings)
        executable = resolve_executable(
            configured.executable,
            config_dir=configured.configuration_directory,
        )
        if executable is None:
            raise CodexSettingsError(
                f"Configured executable is unavailable: {configured.executable}"
            )
        if not supports_ephemeral(executable, cwd=configured.configuration_directory):
            raise CodexSettingsError(
                "Configured executable does not support required --ephemeral mode."
            )
        try:
            version = cli_version(executable, cwd=configured.configuration_directory)
        except CodexCliValidationError as error:
            raise CodexSettingsError(str(error)) from error
        return CodexCliRunPolicy(
            executable=executable,
            model=configured.execution.model,
            reasoning_effort=configured.execution.reasoning_effort,
            cli_version=version,
        )

    def runtime_compatibility_problem(self, policy: object) -> str | None:
        resolved = _run_policy(policy)
        if not resolved.executable.is_file():
            return f"Resolved executable is unavailable: {resolved.executable}"
        try:
            current_version = cli_version(
                resolved.executable, cwd=resolved.executable.parent
            )
        except CodexCliValidationError as error:
            return str(error)
        if current_version != resolved.cli_version:
            return (
                "Codex CLI version changed after policy resolution: expected "
                f"{resolved.cli_version!r}, got {current_version!r}."
            )
        if not supports_ephemeral(resolved.executable, cwd=resolved.executable.parent):
            return "Codex CLI no longer supports required --ephemeral mode."
        return None

    def create_executor(self, policy: object) -> AgentExecutor:
        resolved = _run_policy(policy)
        return CodexCliAgentExecutor(
            CodexSettings(
                executable=str(resolved.executable),
                model=resolved.model,
                reasoning_effort=resolved.reasoning_effort,
            ),
            configuration_directory=resolved.executable.parent,
        )


def _configured_settings(settings: object) -> CodexCliConfiguredSettings:
    if not isinstance(settings, CodexCliConfiguredSettings):
        raise CodexSettingsError("Codex CLI configured settings have the wrong type.")
    return settings


def _run_policy(policy: object) -> CodexCliRunPolicy:
    if not isinstance(policy, CodexCliRunPolicy):
        raise CodexSettingsError("Codex CLI run policy has the wrong type.")
    return policy


def _non_empty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CodexSettingsError(f"Missing required non-empty string: {name}.")
    return value.strip()


def _pass(name: str, message: str = "") -> PreflightCheck:
    return PreflightCheck(name=name, status=PreflightStatus.PASS, message=message)


def _fail(name: str, message: str) -> PreflightCheck:
    return PreflightCheck(name=name, status=PreflightStatus.FAIL, message=message)


__all__ = [
    "CodexCliProviderRegistration",
]
