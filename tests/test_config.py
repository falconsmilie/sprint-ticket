from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from ticket_automation.config import ConfigError, load_config, parse_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def copy_example_config(config_dir: Path) -> Path:
    source = PROJECT_ROOT / "config.example.toml"
    destination = config_dir / source.name
    shutil.copyfile(source, destination)
    return destination


class ConfigTests(unittest.TestCase):
    def test_loads_example_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_dir = Path(tmp_dir)
            copy_example_config(config_dir)

            config = load_config(config_dir)

        self.assertEqual(config.project.name, "PhosPy")
        self.assertEqual(config.project.repo.as_posix(), "C:/Projects/phospy")
        self.assertEqual(config.project.protected_branches, ("main", "master"))
        self.assertEqual(config.runner.max_correction_rounds, 3)
        self.assertEqual(config.codex.executable, "codex")
        self.assertEqual(config.codex.implementation_sandbox, "workspace-write")
        self.assertEqual(config.codex.review_sandbox, "read-only")
        self.assertEqual(
            [command.name for command in config.verification.commands],
            ["tests", "typing"],
        )

    def test_local_config_overrides_example_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_dir = Path(tmp_dir)
            copy_example_config(config_dir)
            local_config = config_dir / "config.local.toml"
            local_config.write_text(
                """
[project]
repo = "D:/work/phospy-local"

[runner]
max_correction_rounds = 2
""".strip(),
                encoding="utf-8",
            )

            config = load_config(config_dir)

        self.assertEqual(config.project.name, "PhosPy")
        self.assertEqual(config.project.repo.as_posix(), "D:/work/phospy-local")
        self.assertEqual(config.runner.max_correction_rounds, 2)
        self.assertEqual(
            config.source_files,
            (config_dir / "config.example.toml", local_config),
        )

    def test_missing_local_config_is_acceptable(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_dir = Path(tmp_dir)
            copy_example_config(config_dir)

            config = load_config(config_dir)

        self.assertEqual(config.source_files, (config_dir / "config.example.toml",))

    def test_invalid_required_values_are_rejected(self):
        test_cases = [
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
        ]
        for config_patch, message in test_cases:
            with self.subTest(message=message):
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

                with self.assertRaisesRegex(ConfigError, message):
                    parse_config(raw_config)

    def test_verification_commands_retain_argument_boundaries(self):
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

        self.assertEqual(
            config.verification.commands[0].argv,
            (
                "python",
                "-m",
                "pytest",
                "tests/unit/test file.py",
            ),
        )

