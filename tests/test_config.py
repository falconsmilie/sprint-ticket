from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import copy_example_config
from ticket_automation.application.agent_execution import AgentTaskKind, ProviderId
from ticket_automation.composition import (
    apply_codex_execution_overrides,
    prepare_production_agents,
)
from ticket_automation.config import ConfigError, load_config, parse_config
from ticket_automation.providers.codex_cli import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING_EFFORT,
    CodexExecutionOverrides,
)


def test_example_config_is_not_loaded_at_runtime(tmp_path):
    copy_example_config(tmp_path)
    with pytest.raises(ConfigError, match=r"Missing required \[project\]"):
        load_config(tmp_path)


def test_application_and_provider_defaults_work_without_example_file(tmp_path):
    local_config = tmp_path / "config.local.toml"
    local_config.write_text(_config_toml(), encoding="utf-8")

    config = load_config(tmp_path)
    effective = apply_codex_execution_overrides(config, CodexExecutionOverrides())
    prepare_production_agents(effective)
    provider = effective.agents.providers[ProviderId("codex-cli")]

    assert config.project.name == "Local project"
    assert config.project.repo.as_posix() == "D:/work/local-project"
    assert config.runner.max_correction_rounds == 1
    assert all(
        config.agents.assignments[kind] == ProviderId("codex-cli")
        for kind in AgentTaskKind
    )
    assert provider["model"] == DEFAULT_CODEX_MODEL
    assert provider["reasoning_effort"] == DEFAULT_CODEX_REASONING_EFFORT
    assert config.source_files == (local_config,)


def test_local_config_overrides_application_defaults(tmp_path):
    local_config = tmp_path / "config.local.toml"
    local_config.write_text(
        _config_toml(
            runner="\n[runner]\nmax_correction_rounds = 2\n",
            provider='executable = "codex"\nmodel = "local-model"\nreasoning_effort = "high"',
        ),
        encoding="utf-8",
    )
    config = load_config(tmp_path)
    effective = apply_codex_execution_overrides(config, CodexExecutionOverrides())
    prepare_production_agents(effective)
    settings = effective.agents.providers[ProviderId("codex-cli")]

    assert effective.runner.max_correction_rounds == 2
    assert settings["model"] == "local-model"
    assert settings["reasoning_effort"] == "high"


def test_cli_codex_overrides_take_precedence_independently():
    raw = base_raw_config()
    raw["agents"]["providers"]["codex-cli"].update(  # type: ignore[index]
        {"model": "local-model", "reasoning_effort": "medium"}
    )
    config = parse_config(raw, configuration_directory=Path.cwd())

    model_override = apply_codex_execution_overrides(
        config, CodexExecutionOverrides(model="cli-model")
    )
    reasoning_override = apply_codex_execution_overrides(
        config, CodexExecutionOverrides(reasoning_effort="high")
    )

    model = model_override.agents.providers[ProviderId("codex-cli")]
    reasoning = reasoning_override.agents.providers[ProviderId("codex-cli")]
    assert model["model"] == "cli-model"
    assert model["reasoning_effort"] == "medium"
    assert reasoning["model"] == "local-model"
    assert reasoning["reasoning_effort"] == "high"


def test_old_codex_table_is_rejected_explicitly():
    raw = base_raw_config()
    raw["codex"] = {"executable": "codex"}
    with pytest.raises(ConfigError, match=r"Unsupported \[codex\]"):
        parse_config(raw, configuration_directory=Path.cwd())


@pytest.mark.parametrize("missing", list(AgentTaskKind))
def test_every_task_assignment_is_required(missing):
    raw = base_raw_config()
    del raw["agents"]["assignments"][missing.value]  # type: ignore[index]
    with pytest.raises(ConfigError, match=f"agents.assignments.{missing.value}"):
        parse_config(raw, configuration_directory=Path.cwd())


def test_assignment_to_unconfigured_provider_is_rejected():
    raw = base_raw_config()
    raw["agents"]["assignments"]["review"] = "missing"  # type: ignore[index]
    with pytest.raises(ConfigError, match="unconfigured provider missing"):
        parse_config(raw, configuration_directory=Path.cwd())


def test_assigned_unregistered_provider_is_rejected_by_composition():
    raw = base_raw_config()
    raw["agents"]["assignments"]["review"] = "stub"  # type: ignore[index]
    raw["agents"]["providers"]["stub"] = {}  # type: ignore[index]
    config = parse_config(raw, configuration_directory=Path.cwd())
    with pytest.raises(ConfigError, match="stub is not registered"):
        prepare_production_agents(config)


def test_missing_provider_settings_are_rejected_by_composition():
    raw = base_raw_config()
    raw["agents"]["providers"]["codex-cli"] = {}  # type: ignore[index]
    config = parse_config(raw, configuration_directory=Path.cwd())
    with pytest.raises(ConfigError, match="codex-cli.executable"):
        prepare_production_agents(config)


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        (("project", "repo", ""), "project.repo"),
        (("runner", "max_correction_rounds", 0), "positive integer"),
        (("provider", "model", ""), "codex-cli.model"),
        (("provider", "reasoning_effort", "extreme"), "codex-cli.reasoning_effort"),
    ],
)
def test_invalid_required_values_are_rejected(patch, message):
    raw = base_raw_config()
    section, key, value = patch
    if section == "provider":
        raw["agents"]["providers"]["codex-cli"][key] = value  # type: ignore[index]
    else:
        raw[section][key] = value  # type: ignore[index]
    if section == "provider":
        config = parse_config(raw, configuration_directory=Path.cwd())
        with pytest.raises(ConfigError, match=message):
            prepare_production_agents(config)
    else:
        with pytest.raises(ConfigError, match=message):
            parse_config(raw, configuration_directory=Path.cwd())


@pytest.mark.parametrize(
    ("command", "message"),
    [
        (
            {"name": "tests", "argv": [], "timeout_seconds": 1800},
            r"verification.commands\[1\].argv",
        ),
        (
            {"name": "tests", "argv": ["python", "-m", "pytest"]},
            r"verification.commands\[1\].timeout_seconds",
        ),
        (
            {
                "name": "tests",
                "argv": ["python", "-m", "pytest"],
                "timeout_seconds": 0,
            },
            r"verification.commands\[1\].timeout_seconds",
        ),
    ],
)
def test_invalid_verification_command_values_are_rejected(command, message):
    raw = base_raw_config()
    raw["verification"]["commands"] = [command]  # type: ignore[index]

    with pytest.raises(ConfigError, match=message):
        parse_config(raw, configuration_directory=Path.cwd())


def test_verification_commands_retain_argument_boundaries():
    raw = base_raw_config()
    raw["verification"]["commands"][0]["argv"] = [  # type: ignore[index]
        "python", "-m", "pytest", "tests/unit/test file.py"
    ]
    config = parse_config(raw, configuration_directory=Path.cwd())
    assert config.verification.commands[0].argv == (
        "python", "-m", "pytest", "tests/unit/test file.py"
    )


def base_raw_config() -> dict[str, object]:
    return {
        "project": {
            "name": "PhosPy",
            "repo": "C:/Projects/phospy",
            "protected_branches": ["main"],
        },
        "runner": {"max_correction_rounds": 3},
        "agents": {
            "assignments": {kind.value: "codex-cli" for kind in AgentTaskKind},
            "providers": {"codex-cli": {"executable": "codex"}},
        },
        "verification": {
            "commands": [
                {
                    "name": "tests",
                    "argv": ["python", "-m", "pytest"],
                    "timeout_seconds": 1800,
                }
            ]
        },
    }


def _config_toml(*, runner: str = "", provider: str = 'executable = "codex"') -> str:
    return f'''[project]
name = "Local project"
repo = "D:/work/local-project"
protected_branches = ["main"]
{runner}
[agents.assignments]
implementation = "codex-cli"
review = "codex-cli"
correction = "codex-cli"

[agents.providers.codex-cli]
{provider}

[[verification.commands]]
name = "tests"
argv = ["python", "-m", "pytest"]
timeout_seconds = 1800
'''.strip()
