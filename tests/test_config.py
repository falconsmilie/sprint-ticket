from __future__ import annotations

import pytest

from tests.helpers import copy_example_config
from ticket_automation.config import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING_EFFORT,
    CodexExecutionOverrides,
    ConfigError,
    apply_codex_execution_overrides,
    load_config,
    parse_config,
)


def test_loads_example_config(tmp_path):
    copy_example_config(tmp_path)

    config = load_config(tmp_path)

    assert config.project.name == "PhosPy"
    assert config.project.repo.as_posix() == "C:/Projects/phospy"
    assert config.project.protected_branches == ("main", "master")
    assert config.runner.max_correction_rounds == 1
    assert config.codex.executable == "codex"
    assert config.codex.model == DEFAULT_CODEX_MODEL
    assert config.codex.reasoning_effort == DEFAULT_CODEX_REASONING_EFFORT
    assert config.codex.implementation_sandbox == "workspace-write"
    assert config.codex.review_sandbox == "read-only"
    assert [command.name for command in config.verification.commands] == [
        "tests",
        "typing",
    ]
    assert [command.timeout_seconds for command in config.verification.commands] == [
        1800,
        1800,
    ]


def test_local_config_overrides_example_config(tmp_path):
    copy_example_config(tmp_path)
    local_config = tmp_path / "config.local.toml"
    local_config.write_text(
        """
[project]
repo = "D:/work/phospy-local"

[runner]
max_correction_rounds = 2

[codex]
model = "local-model"
reasoning_effort = "high"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(tmp_path)

    assert config.project.name == "PhosPy"
    assert config.project.repo.as_posix() == "D:/work/phospy-local"
    assert config.runner.max_correction_rounds == 2
    assert config.codex.model == "local-model"
    assert config.codex.reasoning_effort == "high"
    assert config.source_files == (tmp_path / "config.example.toml", local_config)


def test_missing_local_config_is_acceptable(tmp_path):
    copy_example_config(tmp_path)

    config = load_config(tmp_path)

    assert config.source_files == (tmp_path / "config.example.toml",)


def test_codex_execution_defaults_are_application_owned():
    config = parse_config(base_raw_config())

    assert config.codex.model == DEFAULT_CODEX_MODEL
    assert config.codex.reasoning_effort == DEFAULT_CODEX_REASONING_EFFORT


def test_cli_codex_overrides_take_precedence_independently():
    raw_config = base_raw_config()
    raw_config["codex"]["model"] = "local-model"
    raw_config["codex"]["reasoning_effort"] = "medium"
    config = parse_config(raw_config)

    model_override = apply_codex_execution_overrides(
        config,
        CodexExecutionOverrides(model="cli-model"),
    )
    reasoning_override = apply_codex_execution_overrides(
        config,
        CodexExecutionOverrides(reasoning_effort="high"),
    )

    assert model_override.codex.model == "cli-model"
    assert model_override.codex.reasoning_effort == "medium"
    assert reasoning_override.codex.model == "local-model"
    assert reasoning_override.codex.reasoning_effort == "high"


@pytest.mark.parametrize(
    ("config_patch", "message"),
    [
        ({"project": {"repo": ""}}, "project.repo"),
        ({"runner": {"max_correction_rounds": 0}}, "positive integer"),
        (
            {"codex": {"implementation_sandbox": "danger-full-access"}},
            "codex.implementation_sandbox",
        ),
        ({"codex": {"model": ""}}, "codex.model"),
        ({"codex": {"reasoning_effort": ""}}, "codex.reasoning_effort"),
        (
            {"codex": {"reasoning_effort": "extreme"}},
            "codex.reasoning_effort",
        ),
        (
            {"verification": {"commands": [{"name": "tests", "argv": []}]}},
            r"verification.commands\[1\].argv",
        ),
        (
            {
                "verification": {
                    "commands": [{"name": "tests", "argv": ["python", "-m", "pytest"]}]
                }
            },
            r"verification.commands\[1\].timeout_seconds",
        ),
        (
            {
                "verification": {
                    "commands": [
                        {
                            "name": "tests",
                            "argv": ["python", "-m", "pytest"],
                            "timeout_seconds": 0,
                        }
                    ]
                }
            },
            r"verification.commands\[1\].timeout_seconds",
        ),
    ],
)
def test_invalid_required_values_are_rejected(config_patch, message):
    raw_config = base_raw_config()
    for section, values in config_patch.items():
        raw_config[section].update(values)

    with pytest.raises(ConfigError, match=message):
        parse_config(raw_config)


def test_verification_commands_retain_argument_boundaries():
    config = parse_config(
        base_raw_config(
            verification_commands=[
                {
                    "name": "targeted tests",
                    "argv": ["python", "-m", "pytest", "tests/unit/test file.py"],
                    "timeout_seconds": 1800,
                }
            ],
        )
    )

    assert config.verification.commands[0].argv == (
        "python",
        "-m",
        "pytest",
        "tests/unit/test file.py",
    )
    assert config.verification.commands[0].timeout_seconds == 1800


def base_raw_config(
    *,
    verification_commands: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "project": {
            "name": "PhosPy",
            "repo": "C:/Projects/phospy",
            "protected_branches": ["main"],
        },
        "runner": {"max_correction_rounds": 3},
        "codex": {
            "executable": "codex",
            "implementation_sandbox": "workspace-write",
            "review_sandbox": "read-only",
        },
        "verification": {
            "commands": verification_commands
            or [
                {
                    "name": "tests",
                    "argv": ["python", "-m", "pytest"],
                    "timeout_seconds": 1800,
                }
            ]
        },
    }
