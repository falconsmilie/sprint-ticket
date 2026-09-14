from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

DEFAULT_CODEX_MODEL = "gpt-5.5"
DEFAULT_CODEX_REASONING_EFFORT = "xhigh"
DEFAULT_MAX_CORRECTION_ROUNDS = 1
IMPLEMENTATION_SANDBOX_POLICY = "workspace-write"
REVIEW_SANDBOX_POLICY = "read-only"
SUPPORTED_CODEX_REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh"}
)
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
        return CodexExecutionSettings(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
        )


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
    project = _require_table(raw_config, "project")
    runner = _require_table(raw_config, "runner")
    codex = _require_table(raw_config, "codex")
    verification = _require_table(raw_config, "verification")
    _reject_configurable_sandbox_policy(codex)

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
            model=_optional_codex_model(codex, "codex.model"),
            reasoning_effort=_optional_reasoning_effort(
                codex,
                "codex.reasoning_effort",
            ),
        ),
        verification=VerificationSettings(
            commands=_parse_verification_commands(verification),
        ),
        source_files=source_files,
        configuration_directory=configuration_directory,
    )


def apply_codex_execution_overrides(
    config: AppConfig,
    overrides: CodexExecutionOverrides,
) -> AppConfig:
    execution = resolve_codex_execution_config(config.codex, overrides)
    return with_codex_execution_settings(config, execution)


def with_codex_execution_settings(
    config: AppConfig,
    execution: CodexExecutionSettings,
) -> AppConfig:
    validated = validate_codex_execution_settings(execution)
    return replace(
        config,
        codex=replace(
            config.codex,
            model=validated.model,
            reasoning_effort=validated.reasoning_effort,
        ),
    )


def resolve_codex_execution_config(
    codex: CodexSettings,
    overrides: CodexExecutionOverrides | None = None,
) -> CodexExecutionSettings:
    overrides = overrides or CodexExecutionOverrides()
    model_name = "--model" if overrides.model is not None else "codex.model"
    reasoning_name = (
        "--reasoning-effort"
        if overrides.reasoning_effort is not None
        else "codex.reasoning_effort"
    )
    return _codex_execution_settings(
        overrides.model if overrides.model is not None else codex.model,
        overrides.reasoning_effort
        if overrides.reasoning_effort is not None
        else codex.reasoning_effort,
        model_name=model_name,
        reasoning_name=reasoning_name,
    )


def validate_codex_execution_settings(
    execution: CodexExecutionSettings,
) -> CodexExecutionSettings:
    return _codex_execution_settings(
        execution.model,
        execution.reasoning_effort,
        model_name="codex.model",
        reasoning_name="codex.reasoning_effort",
    )


def format_config_summary(config: AppConfig) -> str:
    source_files = (
        ", ".join(str(path) for path in config.source_files) or "in-memory config"
    )
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
            f"  model: {_redact_if_secret('codex.model', config.codex.model)}",
            f"  reasoning effort: {config.codex.reasoning_effort}",
            f"  implementation sandbox: {IMPLEMENTATION_SANDBOX_POLICY}",
            f"  review sandbox: {REVIEW_SANDBOX_POLICY}",
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
    return {
        "runner": {"max_correction_rounds": DEFAULT_MAX_CORRECTION_ROUNDS},
        "codex": {
            "model": DEFAULT_CODEX_MODEL,
            "reasoning_effort": DEFAULT_CODEX_REASONING_EFFORT,
        },
    }


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


def _optional_codex_model(table: dict[str, Any], dotted_name: str) -> str:
    key = dotted_name.rsplit(".", maxsplit=1)[-1]
    value = table.get(key, DEFAULT_CODEX_MODEL)
    return _require_non_empty_value(value, dotted_name)


def _optional_reasoning_effort(table: dict[str, Any], dotted_name: str) -> str:
    key = dotted_name.rsplit(".", maxsplit=1)[-1]
    value = table.get(key, DEFAULT_CODEX_REASONING_EFFORT)
    return _require_supported_reasoning_effort(value, dotted_name)


def _codex_execution_settings(
    model: Any,
    reasoning_effort: Any,
    *,
    model_name: str,
    reasoning_name: str,
) -> CodexExecutionSettings:
    return CodexExecutionSettings(
        model=_require_non_empty_value(model, model_name),
        reasoning_effort=_require_supported_reasoning_effort(
            reasoning_effort,
            reasoning_name,
        ),
    )


def _require_non_empty_value(value: Any, dotted_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Missing required non-empty string: {dotted_name}.")
    return value.strip()


def _require_supported_reasoning_effort(value: Any, dotted_name: str) -> str:
    effort = _require_non_empty_value(value, dotted_name)
    if effort not in SUPPORTED_CODEX_REASONING_EFFORTS:
        supported = ", ".join(sorted(SUPPORTED_CODEX_REASONING_EFFORTS))
        raise ConfigError(f"{dotted_name} must be one of: {supported}.")
    return effort


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


def _reject_configurable_sandbox_policy(codex: dict[str, Any]) -> None:
    forbidden = tuple(
        key for key in ("implementation_sandbox", "review_sandbox") if key in codex
    )
    if forbidden:
        names = ", ".join(f"codex.{key}" for key in forbidden)
        raise ConfigError(
            "Phase sandbox policy is controller-owned and cannot be configured: "
            f"{names}."
        )


def _parse_verification_commands(
    verification: dict[str, Any],
) -> tuple[VerificationCommand, ...]:
    commands = verification.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ConfigError("verification.commands must contain at least one command.")

    parsed_commands: list[VerificationCommand] = []
    for index, command in enumerate(commands, start=1):
        if not isinstance(command, dict):
            raise ConfigError(f"verification.commands[{index}] must be a table.")
        name = _require_non_empty_string(
            command, f"verification.commands[{index}].name"
        )
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
