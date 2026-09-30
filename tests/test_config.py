from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import copy_example_config
from ticket_automation.application.agent_execution import AgentTaskKind, ProviderId
from ticket_automation.composition import prepare_production_agents
from ticket_automation.config import (
    ConfigError,
    format_config_summary,
    load_config,
    parse_config,
)
from ticket_automation.providers.codex_cli import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING_EFFORT,
)


def test_example_config_is_not_loaded_at_runtime(tmp_path):
    copy_example_config(tmp_path)
    with pytest.raises(ConfigError, match=r"Missing required \[project\]"):
        load_config(tmp_path)


def test_application_and_provider_defaults_work_without_example_file(tmp_path):
    local_config = tmp_path / "config.local.toml"
    local_config.write_text(_config_toml(), encoding="utf-8")

    config = load_config(tmp_path)
    providers = prepare_production_agents(config)
    provider = providers.display_settings()[ProviderId("codex-cli")]

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
    assert config.agents.timeouts == {kind: 3600 for kind in AgentTaskKind}


def test_agent_timeout_partial_overrides_and_summary():
    raw = base_raw_config()
    raw["agents"]["timeouts"] = {"review_seconds": 91}  # type: ignore[index]

    config = parse_config(raw, configuration_directory=Path.cwd())
    summary = format_config_summary(
        config,
        provider_settings={ProviderId("codex-cli"): {"executable": "codex"}},
    )

    assert config.agents.timeouts == {
        AgentTaskKind.IMPLEMENTATION: 3600,
        AgentTaskKind.REVIEW: 91,
        AgentTaskKind.CORRECTION: 3600,
    }
    assert "implementation: 3600s" in summary
    assert "review: 91s" in summary
    assert "correction: 3600s" in summary


def test_all_agent_task_timeouts_are_independent():
    raw = base_raw_config()
    raw["agents"]["timeouts"] = {  # type: ignore[index]
        "implementation_seconds": 120,
        "review_seconds": 90,
        "correction_seconds": 121,
    }

    config = parse_config(raw, configuration_directory=Path.cwd())

    assert tuple(config.agents.timeouts.values()) == (120, 90, 121)


@pytest.mark.parametrize("value", [0, -1, True, "60", 1.5, None])
def test_invalid_agent_timeout_values_are_rejected(value):
    raw = base_raw_config()
    raw["agents"]["timeouts"] = {"implementation_seconds": value}  # type: ignore[index]

    with pytest.raises(ConfigError, match="agents.timeouts.implementation_seconds"):
        parse_config(raw, configuration_directory=Path.cwd())


def test_malformed_and_unknown_agent_timeout_tables_are_rejected():
    raw = base_raw_config()
    raw["agents"]["timeouts"] = "one hour"  # type: ignore[index]
    with pytest.raises(ConfigError, match="agents.timeouts must be a configuration table"):
        parse_config(raw, configuration_directory=Path.cwd())

    raw = base_raw_config()
    raw["agents"]["timeouts"] = {"idle_seconds": 10}  # type: ignore[index]
    with pytest.raises(ConfigError, match="agents.timeouts.*idle_seconds"):
        parse_config(raw, configuration_directory=Path.cwd())


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
    providers = prepare_production_agents(config)
    settings = providers.display_settings()[ProviderId("codex-cli")]

    assert config.runner.max_correction_rounds == 2
    assert settings["model"] == "local-model"
    assert settings["reasoning_effort"] == "high"


def test_unknown_top_level_configuration_section_is_rejected():
    raw = base_raw_config()
    raw["obsolete-provider"] = {"executable": "tool"}
    with pytest.raises(ConfigError, match="Unknown top-level configuration section"):
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
        "python",
        "-m",
        "pytest",
        "tests/unit/test file.py",
    ]
    config = parse_config(raw, configuration_directory=Path.cwd())
    assert config.verification.commands[0].argv == (
        "python",
        "-m",
        "pytest",
        "tests/unit/test file.py",
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
    return f"""[project]
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
""".strip()
