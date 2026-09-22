from __future__ import annotations

import copy
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .application.agent_execution import AgentContractError, AgentTaskKind, ProviderId

DEFAULT_MAX_CORRECTION_ROUNDS = 1
IMPLEMENTATION_SANDBOX_POLICY = "workspace-write"
REVIEW_SANDBOX_POLICY = "read-only"
SECRET_FIELD_MARKERS = ("secret", "password", "token", "api_key", "apikey")


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class ProjectSettings:
    name: str
    repo: Path
    protected_branches: tuple[str, ...]


@dataclass(frozen=True)
class RunnerSettings:
    max_correction_rounds: int


@dataclass(frozen=True)
class VerificationCommand:
    name: str
    argv: tuple[str, ...]
    timeout_seconds: int


@dataclass(frozen=True)
class VerificationSettings:
    commands: tuple[VerificationCommand, ...]


@dataclass(frozen=True)
class AgentSettings:
    assignments: Mapping[AgentTaskKind, ProviderId]
    providers: Mapping[ProviderId, Mapping[str, object]]

    def __post_init__(self) -> None:
        if not isinstance(self.assignments, Mapping):
            raise ConfigError("agents.assignments must be a mapping.")
        if not all(
            isinstance(task_kind, AgentTaskKind)
            and isinstance(provider_id, ProviderId)
            for task_kind, provider_id in self.assignments.items()
        ):
            raise ConfigError(
                "agents.assignments must map AgentTaskKind values to ProviderId values."
            )
        expected_tasks = set(AgentTaskKind)
        actual_tasks = set(self.assignments)
        if actual_tasks != expected_tasks:
            missing = sorted(task.value for task in expected_tasks - actual_tasks)
            unexpected = sorted(str(task) for task in actual_tasks - expected_tasks)
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if unexpected:
                details.append("unexpected: " + ", ".join(unexpected))
            raise ConfigError(
                "agents.assignments must contain implementation, review, and correction"
                + (f" ({'; '.join(details)})" if details else "")
                + "."
            )
        if not isinstance(self.providers, Mapping):
            raise ConfigError("agents.providers must be a mapping.")
        if not all(
            isinstance(provider_id, ProviderId) and isinstance(settings, Mapping)
            for provider_id, settings in self.providers.items()
        ):
            raise ConfigError(
                "agents.providers must map ProviderId values to settings mappings."
            )
        for task_kind, provider_id in self.assignments.items():
            if provider_id not in self.providers:
                raise ConfigError(
                    f"{task_kind.value} is assigned to unconfigured provider {provider_id}."
                )

        object.__setattr__(
            self, "assignments", MappingProxyType(dict(self.assignments))
        )
        object.__setattr__(
            self,
            "providers",
            MappingProxyType(
                {
                    provider_id: _freeze_mapping(settings)
                    for provider_id, settings in self.providers.items()
                }
            ),
        )


@dataclass(frozen=True)
class AppConfig:
    project: ProjectSettings
    runner: RunnerSettings
    agents: AgentSettings
    verification: VerificationSettings
    source_files: tuple[Path, ...]
    configuration_directory: Path


def load_config(config_dir: Path | str | None = None) -> AppConfig:
    base_dir = (Path.cwd() if config_dir is None else Path(config_dir)).resolve()
    local_path = base_dir / "config.local.toml"
    raw_config = _application_defaults()
    source_files: tuple[Path, ...] = ()
    if local_path.is_file():
        raw_config = _deep_merge(raw_config, _read_toml_file(local_path))
        source_files = (local_path,)
    return parse_config(
        raw_config,
        source_files=source_files,
        configuration_directory=base_dir,
    )


def parse_config(
    raw_config: dict[str, Any],
    *,
    source_files: tuple[Path, ...] = (),
    configuration_directory: Path,
) -> AppConfig:
    if "codex" in raw_config:
        raise ConfigError(
            "Unsupported [codex] configuration section; configure explicit "
            "[agents.assignments] and [agents.providers.<provider-id>] sections."
        )
    project = _require_table(raw_config, "project")
    runner = _require_table(raw_config, "runner")
    agents = _require_table(raw_config, "agents")
    verification = _require_table(raw_config, "verification")
    return AppConfig(
        project=ProjectSettings(
            name=_require_non_empty_string(project, "project.name"),
            repo=Path(_require_non_empty_string(project, "project.repo")),
            protected_branches=_require_string_tuple(
                project,
                "protected_branches",
                "project.protected_branches",
                allow_empty=False,
            ),
        ),
        runner=RunnerSettings(
            max_correction_rounds=_require_positive_int(
                runner, "max_correction_rounds", "runner.max_correction_rounds"
            ),
        ),
        agents=_parse_agent_settings(agents),
        verification=VerificationSettings(
            commands=_parse_verification_commands(verification)
        ),
        source_files=source_files,
        configuration_directory=configuration_directory,
    )


def format_config_summary(
    config: AppConfig,
    *,
    provider_settings: Mapping[ProviderId, Mapping[str, object]],
) -> str:
    if set(provider_settings) != set(config.agents.providers):
        raise ConfigError(
            "Effective provider settings must cover every configured provider."
        )
    source_files = (
        ", ".join(str(path) for path in config.source_files) or "in-memory config"
    )
    verification = "\n".join(
        f"  - {command.name} ({command.timeout_seconds}s): {_format_argv(command.argv)}"
        for command in config.verification.commands
    )
    assignments = "\n".join(
        f"  {task_kind.value}: {config.agents.assignments[task_kind]}"
        for task_kind in AgentTaskKind
    )
    providers: list[str] = []
    for provider_id, settings in provider_settings.items():
        providers.append(f"  {provider_id}")
        providers.extend(
            f"    {key}: {_redact_if_secret(key, str(value))}"
            for key, value in sorted(settings.items())
        )
    return "\n".join(
        [
            "TicketAutomation configuration",
            f"Sources: {source_files}",
            "",
            "Project",
            f"  name: {_redact_if_secret('project.name', config.project.name)}",
            f"  repo: {_redact_if_secret('project.repo', str(config.project.repo))}",
            f"  protected branches: {', '.join(config.project.protected_branches)}",
            "",
            "Runner",
            f"  max correction rounds: {config.runner.max_correction_rounds}",
            "",
            "Agent assignments",
            assignments,
            "",
            "Agent providers",
            *providers,
            "",
            "Execution policy",
            f"  implementation/correction access: {IMPLEMENTATION_SANDBOX_POLICY}",
            f"  review access: {REVIEW_SANDBOX_POLICY}",
            "",
            "Verification commands",
            verification,
        ]
    )


def _read_toml_file(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        data = tomllib.load(config_file)
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file did not contain a TOML table: {path}")
    return data


def _application_defaults() -> dict[str, Any]:
    return {"runner": {"max_correction_rounds": DEFAULT_MAX_CORRECTION_ROUNDS}}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _parse_agent_settings(agents: dict[str, Any]) -> AgentSettings:
    assignments_table = _require_nested_table(
        agents, "assignments", "agents.assignments"
    )
    providers_table = _require_nested_table(agents, "providers", "agents.providers")
    unknown_tasks = sorted(set(assignments_table) - {kind.value for kind in AgentTaskKind})
    if unknown_tasks:
        raise ConfigError(
            "Unknown agent task assignment(s): " + ", ".join(unknown_tasks) + "."
        )
    assignments: dict[AgentTaskKind, ProviderId] = {}
    for task_kind in AgentTaskKind:
        dotted_name = f"agents.assignments.{task_kind.value}"
        value = _require_non_empty_string(assignments_table, dotted_name)
        assignments[task_kind] = _provider_id(value, dotted_name)
    providers: dict[ProviderId, Mapping[str, object]] = {}
    for raw_provider_id, settings in providers_table.items():
        if not isinstance(raw_provider_id, str) or not raw_provider_id.strip():
            raise ConfigError("agents.providers keys must be non-empty provider IDs.")
        provider_id = _provider_id(raw_provider_id, f"agents.providers.{raw_provider_id}")
        if not isinstance(settings, dict):
            raise ConfigError(
                f"agents.providers.{provider_id} must be a configuration table."
            )
        providers[provider_id] = settings
    for task_kind, provider_id in assignments.items():
        if provider_id not in providers:
            raise ConfigError(
                f"{task_kind.value} is assigned to unconfigured provider "
                f"{provider_id}; add [agents.providers.{provider_id}]."
            )
    return AgentSettings(assignments=assignments, providers=providers)


def _provider_id(value: str, dotted_name: str) -> ProviderId:
    try:
        return ProviderId(value)
    except AgentContractError as error:
        raise ConfigError(f"Invalid provider ID at {dotted_name}: {error}") from error


def _require_table(raw_config: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing required [{key}] configuration section.")
    return value


def _require_nested_table(
    table: dict[str, Any], key: str, dotted_name: str
) -> dict[str, Any]:
    value = table.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing required [{dotted_name}] configuration section.")
    return value


def _require_non_empty_string(table: dict[str, Any], dotted_name: str) -> str:
    key = dotted_name.rsplit(".", maxsplit=1)[-1]
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Missing required non-empty string: {dotted_name}.")
    return value.strip()


def _require_positive_int(table: dict[str, Any], key: str, dotted_name: str) -> int:
    value = table.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigError(f"{dotted_name} must be a positive integer.")
    return value


def _require_string_tuple(
    table: dict[str, Any],
    key: str,
    dotted_name: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    value = table.get(key)
    if not isinstance(value, list):
        raise ConfigError(f"{dotted_name} must be a list of strings.")
    if not allow_empty and not value:
        raise ConfigError(f"{dotted_name} must contain at least one value.")
    strings: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{dotted_name} must contain only non-empty strings.")
        strings.append(item)
    return tuple(strings)


def _parse_verification_commands(
    verification: dict[str, Any],
) -> tuple[VerificationCommand, ...]:
    commands = verification.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ConfigError("verification.commands must contain at least one command.")
    parsed: list[VerificationCommand] = []
    for index, command in enumerate(commands, start=1):
        if not isinstance(command, dict):
            raise ConfigError(f"verification.commands[{index}] must be a table.")
        parsed.append(
            VerificationCommand(
                name=_require_non_empty_string(
                    command, f"verification.commands[{index}].name"
                ),
                argv=_require_string_tuple(
                    command,
                    "argv",
                    f"verification.commands[{index}].argv",
                    allow_empty=False,
                ),
                timeout_seconds=_require_positive_int(
                    command,
                    "timeout_seconds",
                    f"verification.commands[{index}].timeout_seconds",
                ),
            )
        )
    return tuple(parsed)


def _freeze_mapping(
    value: Mapping[str, object],
    *,
    active: set[int] | None = None,
) -> Mapping[str, object]:
    if not all(isinstance(key, str) for key in value):
        raise ConfigError("Provider settings keys must be strings.")
    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise ConfigError("Provider settings must not contain recursive values.")
    active.add(identity)
    try:
        return MappingProxyType(
            {key: _freeze_value(item, active=active) for key, item in value.items()}
        )
    finally:
        active.remove(identity)


def _freeze_value(value: object, *, active: set[int]) -> object:
    if isinstance(value, Mapping):
        return _freeze_mapping(value, active=active)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise ConfigError("Provider settings must not contain recursive values.")
        active.add(identity)
        try:
            return tuple(_freeze_value(item, active=active) for item in value)
        finally:
            active.remove(identity)
    if value is None or isinstance(
        value,
        (str, bool, int, float, date, datetime, time),
    ):
        return value
    raise ConfigError(
        "Provider settings values must use immutable TOML-compatible types."
    )


def _format_argv(argv: tuple[str, ...]) -> str:
    return " ".join(_quote_arg(argument) for argument in argv)


def _quote_arg(argument: str) -> str:
    if not argument or any(character.isspace() for character in argument):
        return repr(argument)
    return argument


def _redact_if_secret(key: str, value: str) -> str:
    if any(marker in key.lower() for marker in SECRET_FIELD_MARKERS):
        return "<redacted>"
    return value


__all__ = [
    "IMPLEMENTATION_SANDBOX_POLICY",
    "REVIEW_SANDBOX_POLICY",
    "AgentSettings",
    "AppConfig",
    "ConfigError",
    "ProjectSettings",
    "RunnerSettings",
    "VerificationCommand",
    "VerificationSettings",
    "format_config_summary",
    "load_config",
    "parse_config",
]
