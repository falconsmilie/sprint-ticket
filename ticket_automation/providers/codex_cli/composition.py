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
    CodexCliSettings,
    CodexCliSettingsError,
    validate_codex_cli_settings,
)
from .validation import CodexCliValidationError, cli_version, supports_ephemeral


@dataclass(frozen=True)
class CodexCliConfiguredSettings:
    settings: CodexCliSettings
    configuration_directory: Path


@dataclass(frozen=True)
class CodexCliRunPolicy:
    executable: Path
    model: str
    reasoning_effort: str
    cli_version: str


class CodexCliProviderRegistration:
    provider_id: ProviderId = PROVIDER_ID
    policy_version = "1"
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
            raise CodexCliSettingsError(
                "Unknown setting(s): " + ", ".join(unknown) + "."
            )
        settings = validate_codex_cli_settings(
            CodexCliSettings(
                executable=raw_settings.get("executable"),  # type: ignore[arg-type]
                model=raw_settings.get("model", DEFAULT_CODEX_MODEL),  # type: ignore[arg-type]
                reasoning_effort=raw_settings.get(
                    "reasoning_effort", DEFAULT_CODEX_REASONING_EFFORT
                ),  # type: ignore[arg-type]
            )
        )
        return CodexCliConfiguredSettings(
            settings=settings,
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
            configured.settings.executable,
            config_dir=configured.configuration_directory,
        )
        if executable is None:
            checks.append(
                _fail(
                    "executable",
                    f"Executable not found: {configured.settings.executable}",
                )
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

    def display_settings(self, settings: object) -> Mapping[str, object]:
        configured = _configured_settings(settings)
        return {
            "executable": configured.settings.executable,
            "model": configured.settings.model,
            "reasoning_effort": configured.settings.reasoning_effort,
        }

    def resolve_run_policy(self, settings: object) -> CodexCliRunPolicy:
        configured = _configured_settings(settings)
        executable = resolve_executable(
            configured.settings.executable,
            config_dir=configured.configuration_directory,
        )
        if executable is None:
            raise CodexCliSettingsError(
                "Configured executable is unavailable: "
                f"{configured.settings.executable}"
            )
        if not supports_ephemeral(executable, cwd=configured.configuration_directory):
            raise CodexCliSettingsError(
                "Configured executable does not support required --ephemeral mode."
            )
        try:
            version = cli_version(executable, cwd=configured.configuration_directory)
        except CodexCliValidationError as error:
            raise CodexCliSettingsError(str(error)) from error
        return CodexCliRunPolicy(
            executable=executable,
            model=configured.settings.model,
            reasoning_effort=configured.settings.reasoning_effort,
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

    def encode_run_policy(self, policy: object) -> object:
        resolved = _run_policy(policy)
        return {
            "executable": str(resolved.executable),
            "model": resolved.model,
            "reasoning_effort": resolved.reasoning_effort,
            "cli_version": resolved.cli_version,
            "ephemeral": True,
        }

    def decode_run_policy(self, payload: object) -> CodexCliRunPolicy:
        if not isinstance(payload, dict):
            raise CodexCliSettingsError("Codex CLI resolved payload must be an object.")
        expected = {
            "executable",
            "model",
            "reasoning_effort",
            "cli_version",
            "ephemeral",
        }
        if set(payload) != expected:
            raise CodexCliSettingsError(
                "Codex CLI resolved payload fields are incomplete or unsupported."
            )
        if payload.get("ephemeral") is not True:
            raise CodexCliSettingsError(
                "Codex CLI resolved payload requires ephemeral invocation."
            )
        executable = Path(_payload_string(payload, "executable"))
        if not executable.is_absolute():
            raise CodexCliSettingsError(
                "Codex CLI resolved executable must be an absolute path."
            )
        settings = validate_codex_cli_settings(
            CodexCliSettings(
                executable=str(executable),
                model=_payload_string(payload, "model"),
                reasoning_effort=_payload_string(payload, "reasoning_effort"),
            )
        )
        return CodexCliRunPolicy(
            executable=executable,
            model=settings.model,
            reasoning_effort=settings.reasoning_effort,
            cli_version=_payload_string(payload, "cli_version"),
        )

    def create_executor(self, policy: object) -> AgentExecutor:
        resolved = _run_policy(policy)
        return CodexCliAgentExecutor(
            CodexCliSettings(
                executable=str(resolved.executable),
                model=resolved.model,
                reasoning_effort=resolved.reasoning_effort,
            ),
            configuration_directory=resolved.executable.parent,
        )


def _configured_settings(settings: object) -> CodexCliConfiguredSettings:
    if not isinstance(settings, CodexCliConfiguredSettings):
        raise CodexCliSettingsError(
            "Codex CLI configured settings have the wrong type."
        )
    return settings


def _run_policy(policy: object) -> CodexCliRunPolicy:
    if not isinstance(policy, CodexCliRunPolicy):
        raise CodexCliSettingsError("Codex CLI run policy has the wrong type.")
    return policy


def _payload_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CodexCliSettingsError(
            f"Codex CLI resolved payload field must be a non-empty string: {key}."
        )
    return value.strip()


def _pass(name: str, message: str = "") -> PreflightCheck:
    return PreflightCheck(name=name, status=PreflightStatus.PASS, message=message)


def _fail(name: str, message: str) -> PreflightCheck:
    return PreflightCheck(name=name, status=PreflightStatus.FAIL, message=message)


__all__ = [
    "CodexCliProviderRegistration",
]
