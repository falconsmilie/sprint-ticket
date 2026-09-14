"""Immutable execution policy captured when a ticket run is created."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .config import (
    IMPLEMENTATION_SANDBOX_POLICY,
    REVIEW_SANDBOX_POLICY,
    AppConfig,
    CodexExecutionSettings,
    CodexSettings,
    ProjectSettings,
    RunnerSettings,
    VerificationCommand,
    VerificationSettings,
    validate_codex_execution_settings,
)
from .executable_resolution import resolve_executable

RESOLVED_RUN_CONFIG_SCHEMA_VERSION = 1
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_POLICY_ASSETS = {
    "implementation_prompt": _PROJECT_ROOT / "prompts" / "implement.md",
    "correction_prompt": _PROJECT_ROOT / "prompts" / "correct.md",
    "review_prompt": _PROJECT_ROOT / "prompts" / "review.md",
    "implementation_result_schema": (
        _PROJECT_ROOT / "schemas" / "implementation-result.schema.json"
    ),
    "review_result_schema": _PROJECT_ROOT / "schemas" / "review-result.schema.json",
}


class ResolvedRunConfigError(ValueError):
    """Raised when immutable execution policy cannot be captured or trusted."""


@dataclass(frozen=True)
class ResolvedRunConfig:
    """The complete V1 policy input set for a single run."""

    target_repository_path: str
    protected_branches: tuple[str, ...]
    codex_model: str
    codex_reasoning_effort: str
    codex_executable: str
    codex_cli_version: str
    implementation_sandbox: str
    review_sandbox: str
    verification_commands: tuple[VerificationCommand, ...]
    max_correction_rounds: int
    ticket_automation_package: str
    ticket_automation_version: str
    ticket_automation_git_sha: str | None
    policy_asset_sha256: tuple[tuple[str, str], ...]
    schema_version: int = RESOLVED_RUN_CONFIG_SCHEMA_VERSION

    @property
    def codex_execution(self) -> CodexExecutionSettings:
        return CodexExecutionSettings(
            model=self.codex_model,
            reasoning_effort=self.codex_reasoning_effort,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_repository_path": self.target_repository_path,
            "protected_branches": list(self.protected_branches),
            "codex": {
                "model": self.codex_model,
                "reasoning_effort": self.codex_reasoning_effort,
                "executable": self.codex_executable,
                "cli_version": self.codex_cli_version,
                "ephemeral": True,
            },
            "sandbox_policy": {
                "implementation": self.implementation_sandbox,
                "review": self.review_sandbox,
            },
            "verification": {
                "commands": [
                    {
                        "name": command.name,
                        "argv": list(command.argv),
                        "timeout_seconds": command.timeout_seconds,
                    }
                    for command in self.verification_commands
                ]
            },
            "max_correction_rounds": self.max_correction_rounds,
            "ticket_automation": {
                "package": self.ticket_automation_package,
                "version": self.ticket_automation_version,
                "git_sha": self.ticket_automation_git_sha,
            },
            "prompt_schema_versions": dict(self.policy_asset_sha256),
        }

    @classmethod
    def from_dict(cls, data: Any) -> ResolvedRunConfig:
        if not isinstance(data, dict):
            raise ResolvedRunConfigError("resolved_config must be an object.")
        if data.get("schema_version") != RESOLVED_RUN_CONFIG_SCHEMA_VERSION:
            raise ResolvedRunConfigError(
                "resolved_config has an unsupported schema version."
            )
        codex = _require_table(data, "codex")
        sandbox = _require_table(data, "sandbox_policy")
        verification = _require_table(data, "verification")
        package = _require_table(data, "ticket_automation")
        asset_versions = _require_table(data, "prompt_schema_versions")
        if codex.get("ephemeral") is not True:
            raise ResolvedRunConfigError(
                "resolved_config requires ephemeral Codex invocation policy."
            )
        implementation_sandbox = _require_string(sandbox, "implementation")
        review_sandbox = _require_string(sandbox, "review")
        if implementation_sandbox != IMPLEMENTATION_SANDBOX_POLICY:
            raise ResolvedRunConfigError(
                "resolved_config implementation sandbox is incompatible with V1 policy."
            )
        if review_sandbox != REVIEW_SANDBOX_POLICY:
            raise ResolvedRunConfigError(
                "resolved_config review sandbox is incompatible with V1 policy."
            )
        target_repository_path = _require_string(data, "target_repository_path")
        if not Path(target_repository_path).is_absolute():
            raise ResolvedRunConfigError(
                "resolved_config target_repository_path must be absolute."
            )
        codex_executable = _require_string(codex, "executable")
        if not Path(codex_executable).is_absolute():
            raise ResolvedRunConfigError(
                "resolved_config codex.executable must be absolute."
            )
        try:
            execution = validate_codex_execution_settings(
                CodexExecutionSettings(
                    model=_require_string(codex, "model"),
                    reasoning_effort=_require_string(codex, "reasoning_effort"),
                )
            )
        except ValueError as error:
            raise ResolvedRunConfigError(
                f"resolved_config Codex execution settings are invalid: {error}"
            ) from error
        git_sha = package.get("git_sha")
        if git_sha is not None and not _is_sha(git_sha):
            raise ResolvedRunConfigError(
                "resolved_config ticket_automation.git_sha must be a SHA or null."
            )
        assets = tuple(sorted(asset_versions.items()))
        if set(dict(assets)) != set(_POLICY_ASSETS) or not all(
            isinstance(name, str) and _is_sha(digest) for name, digest in assets
        ):
            raise ResolvedRunConfigError(
                "resolved_config prompt_schema_versions is incomplete or invalid."
            )
        return cls(
            schema_version=RESOLVED_RUN_CONFIG_SCHEMA_VERSION,
            target_repository_path=target_repository_path,
            protected_branches=_require_string_tuple(data, "protected_branches"),
            codex_model=execution.model,
            codex_reasoning_effort=execution.reasoning_effort,
            codex_executable=codex_executable,
            codex_cli_version=_require_string(codex, "cli_version"),
            implementation_sandbox=implementation_sandbox,
            review_sandbox=review_sandbox,
            verification_commands=_parse_verification_commands(verification),
            max_correction_rounds=_require_positive_int(data, "max_correction_rounds"),
            ticket_automation_package=_require_package_name(package),
            ticket_automation_version=_require_string(package, "version"),
            ticket_automation_git_sha=git_sha,
            policy_asset_sha256=assets,
        )

    def runtime_compatibility_problem(self) -> str | None:
        expected_assets = dict(self.policy_asset_sha256)
        try:
            actual_assets = _policy_asset_sha256()
        except ResolvedRunConfigError as error:
            return str(error)
        for name, actual in actual_assets.items():
            if expected_assets.get(name) != actual:
                return (
                    "Persisted prompt/schema version is incompatible with this "
                    f"TicketAutomation installation: {name}."
                )
        if self.ticket_automation_version != __version__:
            return (
                "Persisted TicketAutomation version is incompatible with this "
                f"installation: expected {self.ticket_automation_version!r}, "
                f"got {__version__!r}."
            )
        current_git_sha = _ticket_automation_git_sha()
        if (
            self.ticket_automation_git_sha is not None
            and current_git_sha != self.ticket_automation_git_sha
        ):
            return "Persisted TicketAutomation Git SHA is incompatible with this installation."
        executable = Path(self.codex_executable)
        if not executable.is_file():
            return f"Persisted Codex executable is unavailable: {executable}"
        try:
            version = codex_cli_version(executable)
        except ResolvedRunConfigError as error:
            return str(error)
        if version != self.codex_cli_version:
            return (
                "Persisted Codex CLI version no longer matches the selected "
                f"executable: expected {self.codex_cli_version!r}, got {version!r}."
            )
        if not _codex_cli_supports_ephemeral(executable):
            return (
                "Persisted Codex executable does not support required --ephemeral mode."
            )
        return None


def resolve_run_config(
    config: AppConfig,
    *,
    target_repository_path: Path | str,
) -> ResolvedRunConfig:
    executable = resolve_executable(
        config.codex.executable,
        config_dir=config.configuration_directory,
    )
    if executable is None:
        raise ResolvedRunConfigError(
            f"Configured Codex executable is unavailable: {config.codex.executable}"
        )
    if not _codex_cli_supports_ephemeral(executable):
        raise ResolvedRunConfigError(
            "Configured Codex executable does not support required --ephemeral mode."
        )
    return ResolvedRunConfig(
        target_repository_path=str(Path(target_repository_path).resolve()),
        protected_branches=config.project.protected_branches,
        codex_model=config.codex.model,
        codex_reasoning_effort=config.codex.reasoning_effort,
        codex_executable=str(executable),
        codex_cli_version=codex_cli_version(executable),
        implementation_sandbox=IMPLEMENTATION_SANDBOX_POLICY,
        review_sandbox=REVIEW_SANDBOX_POLICY,
        verification_commands=config.verification.commands,
        max_correction_rounds=config.runner.max_correction_rounds,
        ticket_automation_package="ticket-automation",
        ticket_automation_version=__version__,
        ticket_automation_git_sha=_ticket_automation_git_sha(),
        policy_asset_sha256=tuple(sorted(_policy_asset_sha256().items())),
    )


def config_from_resolved_run_config(resolved: ResolvedRunConfig) -> AppConfig:
    return AppConfig(
        project=ProjectSettings(
            name="Persisted run configuration",
            repo=Path(resolved.target_repository_path),
            protected_branches=resolved.protected_branches,
        ),
        runner=RunnerSettings(max_correction_rounds=resolved.max_correction_rounds),
        codex=CodexSettings(
            executable=resolved.codex_executable,
            model=resolved.codex_model,
            reasoning_effort=resolved.codex_reasoning_effort,
        ),
        verification=VerificationSettings(commands=resolved.verification_commands),
        source_files=(),
        configuration_directory=Path(resolved.codex_executable).parent,
    )


def codex_cli_version(executable: Path | str) -> str:
    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ResolvedRunConfigError(
            f"Could not determine Codex CLI version: {error}"
        ) from error
    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0 or not output:
        raise ResolvedRunConfigError(
            "Could not determine Codex CLI version from the resolved executable."
        )
    return " ".join(output.splitlines()[0].split())


def _codex_cli_supports_ephemeral(executable: Path | str) -> bool:
    """Return whether this Codex executable advertises ephemeral exec support."""

    try:
        completed = subprocess.run(
            [str(executable), "exec", "--help"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and "--ephemeral" in (
        (completed.stdout or "") + (completed.stderr or "")
    )


def _policy_asset_sha256() -> dict[str, str]:
    try:
        return {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in _POLICY_ASSETS.items()
        }
    except OSError as error:
        raise ResolvedRunConfigError(
            f"Could not read TicketAutomation policy asset: {error}"
        ) from error


def _ticket_automation_git_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(_PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    sha = completed.stdout.strip()
    return sha if completed.returncode == 0 and _is_sha(sha) else None


def _require_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ResolvedRunConfigError(f"resolved_config field must be an object: {key}")
    return value


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ResolvedRunConfigError(
            f"resolved_config field must be a non-empty string: {key}"
        )
    return value.strip()


def _require_string_tuple(data: dict[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ResolvedRunConfigError(
            f"resolved_config field must be a non-empty string list: {key}"
        )
    parsed = tuple(_require_string({key: item}, key) for item in value)
    return parsed


def _require_positive_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ResolvedRunConfigError(
            f"resolved_config field must be a positive integer: {key}"
        )
    return value


def _require_package_name(data: dict[str, Any]) -> str:
    package = _require_string(data, "package")
    if package != "ticket-automation":
        raise ResolvedRunConfigError(
            "resolved_config ticket_automation.package is unsupported."
        )
    return package


def _parse_verification_commands(
    verification: dict[str, Any],
) -> tuple[VerificationCommand, ...]:
    commands = verification.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ResolvedRunConfigError(
            "resolved_config verification.commands must be a non-empty list."
        )
    parsed: list[VerificationCommand] = []
    for index, command in enumerate(commands, start=1):
        if not isinstance(command, dict):
            raise ResolvedRunConfigError(
                f"resolved_config verification.commands[{index}] must be an object."
            )
        argv = command.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(item, str) and item.strip() for item in argv)
        ):
            raise ResolvedRunConfigError(
                f"resolved_config verification.commands[{index}].argv is invalid."
            )
        parsed.append(
            VerificationCommand(
                name=_require_string(command, "name"),
                argv=tuple(argv),
                timeout_seconds=_require_positive_int(command, "timeout_seconds"),
            )
        )
    return tuple(parsed)


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "RESOLVED_RUN_CONFIG_SCHEMA_VERSION",
    "ResolvedRunConfig",
    "ResolvedRunConfigError",
    "codex_cli_version",
    "config_from_resolved_run_config",
    "resolve_run_config",
]
