from __future__ import annotations

import pytest

from tests.helpers import copy_example_config
from ticket_automation.config import ConfigError, load_config, parse_config


def test_loads_example_config(tmp_path):
    copy_example_config(tmp_path)

    config = load_config(tmp_path)

    assert config.project.name == "PhosPy"
    assert config.project.repo.as_posix() == "C:/Projects/phospy"
    assert config.project.protected_branches == ("main", "master")
    assert config.runner.max_correction_rounds == 3
    assert config.codex.executable == "codex"
    assert config.codex.implementation_sandbox == "workspace-write"
    assert config.codex.review_sandbox == "read-only"
    assert [command.name for command in config.verification.commands] == ["tests", "typing"]


def test_local_config_overrides_example_config(tmp_path):
    copy_example_config(tmp_path)
    local_config = tmp_path / "config.local.toml"
    local_config.write_text(
        """
[project]
repo = "D:/work/phospy-local"

[runner]
max_correction_rounds = 2
""".strip(),
        encoding="utf-8",
    )

    config = load_config(tmp_path)

    assert config.project.name == "PhosPy"
    assert config.project.repo.as_posix() == "D:/work/phospy-local"
    assert config.runner.max_correction_rounds == 2
    assert config.source_files == (tmp_path / "config.example.toml", local_config)


def test_missing_local_config_is_acceptable(tmp_path):
    copy_example_config(tmp_path)

    config = load_config(tmp_path)

    assert config.source_files == (tmp_path / "config.example.toml",)


@pytest.mark.parametrize(
    ("config_patch", "message"),
    [
        ({"project": {"repo": ""}}, "project.repo"),
        ({"runner": {"max_correction_rounds": 0}}, "positive integer"),
        (
            {"codex": {"implementation_sandbox": "danger-full-access"}},
            "codex.implementation_sandbox",
        ),
        (
            {"verification": {"commands": [{"name": "tests", "argv": []}]}},
            r"verification.commands\[1\].argv",
        ),
    ],
)
def test_invalid_required_values_are_rejected(config_patch, message):
    raw_config = {
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
            "commands": [{"name": "tests", "argv": ["python", "-m", "pytest"]}]
        },
    }
    for section, values in config_patch.items():
        raw_config[section].update(values)

    with pytest.raises(ConfigError, match=message):
        parse_config(raw_config)


def test_verification_commands_retain_argument_boundaries():
    config = parse_config(
        {
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
                "commands": [
                    {
                        "name": "targeted tests",
                        "argv": ["python", "-m", "pytest", "tests/unit/test file.py"],
                    }
                ]
            },
        }
    )

    assert config.verification.commands[0].argv == (
        "python",
        "-m",
        "pytest",
        "tests/unit/test file.py",
    )
