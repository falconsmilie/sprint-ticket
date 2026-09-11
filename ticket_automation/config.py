from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_SANDBOXES = frozenset({"read-only", "workspace-write"})
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
class CodexSettings:
    executable: str
    implementation_sandbox: str
    review_sandbox: str


@dataclass(frozen=True)
class VerificationCommand:
    name: str
    argv: tuple[str, ...]
    timeout_seconds: int


@dataclass(frozen=True)
class VerificationSettings:
    commands: tuple[VerificationCommand, ...]


@dataclass(frozen=True)
class AppConfig:
    project: ProjectSettings
    runner: RunnerSettings
    codex: CodexSettings
    verification: VerificationSettings
    source_files: tuple[Path, ...]


def load_config(config_dir: Path | str | None = None) -> AppConfig:
    base_dir = Path.cwd() if config_dir is None else Path(config_dir)
    example_path = base_dir / "config.example.toml"
    local_path = base_dir / "config.local.toml"

    if not example_path.is_file():
        raise ConfigError(f"Missing required configuration file: {example_path}")

    raw_config = _read_toml_file(example_path)
    source_files = [example_path]

    if local_path.is_file():
        raw_config = _deep_merge(raw_config, _read_toml_file(local_path))
        source_files.append(local_path)

    return parse_config(raw_config, source_files=tuple(source_files))


def parse_config(raw_config: dict[str, Any], *, source_files: tuple[Path, ...] = ()) -> AppConfig:
    project = _require_table(raw_config, "project")
    runner = _require_table(raw_config, "runner")
    codex = _require_table(raw_config, "codex")
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
                runner,
                "max_correction_rounds",
                "runner.max_correction_rounds",
            ),
        ),
        codex=CodexSettings(
            executable=_require_non_empty_string(codex, "codex.executable"),
            implementation_sandbox=_require_sandbox(
                codex,
                "codex.implementation_sandbox",
            ),
            review_sandbox=_require_sandbox(codex, "codex.review_sandbox"),
        ),
        verification=VerificationSettings(
            commands=_parse_verification_commands(verification),
        ),
        source_files=source_files,
    )


def format_config_summary(config: AppConfig) -> str:
    source_files = ", ".join(str(path) for path in config.source_files) or "in-memory config"
    verification = "\n".join(
        f"  - {command.name} ({command.timeout_seconds}s): {_format_argv(command.argv)}"
        for command in config.verification.commands
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
            "Codex",
            f"  executable: {_redact_if_secret('codex.executable', config.codex.executable)}",
            f"  implementation sandbox: {config.codex.implementation_sandbox}",
            f"  review sandbox: {config.codex.review_sandbox}",
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


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _require_table(raw_config: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw_config.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing required [{key}] configuration section.")
    return value


def _require_non_empty_string(table: dict[str, Any], dotted_name: str) -> str:
    key = dotted_name.rsplit(".", maxsplit=1)[-1]
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Missing required non-empty string: {dotted_name}.")
    return value


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


def _require_sandbox(table: dict[str, Any], dotted_name: str) -> str:
    value = _require_non_empty_string(table, dotted_name)
    if value not in SUPPORTED_SANDBOXES:
        supported = ", ".join(sorted(SUPPORTED_SANDBOXES))
        raise ConfigError(f"{dotted_name} must be one of: {supported}.")
    return value


def _parse_verification_commands(verification: dict[str, Any]) -> tuple[VerificationCommand, ...]:
    commands = verification.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ConfigError("verification.commands must contain at least one command.")

    parsed_commands: list[VerificationCommand] = []
    for index, command in enumerate(commands, start=1):
        if not isinstance(command, dict):
            raise ConfigError(f"verification.commands[{index}] must be a table.")
        name = _require_non_empty_string(command, f"verification.commands[{index}].name")
        argv = _require_string_tuple(
            command,
            "argv",
            f"verification.commands[{index}].argv",
            allow_empty=False,
        )
        timeout_seconds = _require_positive_int(
            command,
            "timeout_seconds",
            f"verification.commands[{index}].timeout_seconds",
        )
        parsed_commands.append(
            VerificationCommand(
                name=name,
                argv=argv,
                timeout_seconds=timeout_seconds,
            )
        )
    return tuple(parsed_commands)


def _format_argv(argv: tuple[str, ...]) -> str:
    return " ".join(_quote_arg(argument) for argument in argv)


def _quote_arg(argument: str) -> str:
    if not argument or any(character.isspace() for character in argument):
        return repr(argument)
    return argument


def _redact_if_secret(key: str, value: str) -> str:
    lowered_key = key.lower()
    if any(marker in lowered_key for marker in SECRET_FIELD_MARKERS):
        return "<redacted>"
    return value
