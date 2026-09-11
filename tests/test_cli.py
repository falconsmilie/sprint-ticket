from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "ticket_automation", *args],
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


class CliTests(unittest.TestCase):
    def test_cli_help_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_cli("--help", cwd=Path(tmp_dir))

        self.assertEqual(result.returncode, 0)
        self.assertIn("config", result.stdout)

    def test_cli_config_output(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_dir = Path(tmp_dir)
            shutil.copyfile(PROJECT_ROOT / "config.example.toml", config_dir / "config.example.toml")

            result = run_cli("config", cwd=config_dir)

        self.assertEqual(result.returncode, 0)
        self.assertIn("TicketAutomation configuration", result.stdout)
        self.assertIn("PhosPy", result.stdout)
        self.assertIn("C:\\Projects\\phospy", result.stdout)
        self.assertIn("tests: python -m pytest", result.stdout)

