from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ticket_automation import workspace_guard as workspace_guard_module
from ticket_automation.workspace_guard import (
    LocalEnvironment,
    WorkspaceEnvironmentSnapshot,
    capture_workspace_environment_snapshot,
    new_environments,
)


def fixed_clock() -> datetime:
    return datetime(2026, 9, 12, 10, 15, 30, tzinfo=UTC)


def test_detects_python_virtual_environment_with_windows_structure(tmp_path):
    repo = tmp_path / "repo with spaces"
    environment = repo / ".venv-correction"
    environment.joinpath("Scripts").mkdir(parents=True)
    environment.joinpath("pyvenv.cfg").write_text("home = python\n", encoding="utf-8")
    environment.joinpath("Scripts", "python.exe").write_text("", encoding="utf-8")

    snapshot = capture_workspace_environment_snapshot(repo)

    assert [(item.kind, item.root_path.name) for item in snapshot.environments] == [
        ("python_venv", ".venv-correction")
    ]
    assert snapshot.environments[0].primary_marker_path == environment / "pyvenv.cfg"


def test_detects_python_virtual_environment_with_posix_structure(tmp_path):
    repo = tmp_path / "repo"
    environment = repo / "venv-fix"
    environment.joinpath("bin").mkdir(parents=True)
    environment.joinpath("pyvenv.cfg").write_text("home = python\n", encoding="utf-8")
    environment.joinpath("bin", "python").write_text("", encoding="utf-8")

    snapshot = capture_workspace_environment_snapshot(repo)

    assert [(item.kind, item.root_path.name) for item in snapshot.environments] == [
        ("python_venv", "venv-fix")
    ]
    assert snapshot.environments[0].primary_marker_path == environment / "pyvenv.cfg"


def test_detects_conda_environment(tmp_path):
    repo = tmp_path / "repo"
    environment = repo / "some-env"
    environment.joinpath("conda-meta").mkdir(parents=True)
    environment.joinpath("conda-meta", "history").write_text("", encoding="utf-8")

    snapshot = capture_workspace_environment_snapshot(repo)

    assert [(item.kind, item.root_path.name) for item in snapshot.environments] == [
        ("conda", "some-env")
    ]
    assert (
        snapshot.environments[0].primary_marker_path
        == environment / "conda-meta" / "history"
    )


def test_suspicious_directory_name_without_markers_is_not_detected(tmp_path):
    repo = tmp_path / "repo"
    notes = repo / ".venv-notes"
    notes.mkdir(parents=True)
    notes.joinpath("README.md").write_text("not an environment\n", encoding="utf-8")

    snapshot = capture_workspace_environment_snapshot(repo)

    assert snapshot.environments == ()


def test_existing_environment_is_not_new_between_snapshots(tmp_path):
    repo = tmp_path / "repo"
    create_pyvenv(repo / ".venv")
    before = capture_workspace_environment_snapshot(repo)

    after = capture_workspace_environment_snapshot(repo)

    assert [environment.root_path.name for environment in before.environments] == [
        ".venv"
    ]
    assert new_environments(before, after) == ()


def test_new_ignored_environment_is_detected_independently_of_git_status(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    repo.joinpath(".gitignore").write_text(".venv/\n", encoding="utf-8")
    before = capture_workspace_environment_snapshot(repo)
    create_pyvenv(repo / ".venv")

    after = capture_workspace_environment_snapshot(repo)

    assert [
        environment.root_path.name for environment in new_environments(before, after)
    ] == [".venv"]


def test_external_runner_environment_is_not_considered_part_of_target_repo(tmp_path):
    external_environment = tmp_path / "runner" / ".venv"
    create_pyvenv(external_environment)
    repo = tmp_path / "target-repo"
    repo.mkdir()

    snapshot = capture_workspace_environment_snapshot(repo)

    assert snapshot.environments == ()


def test_detected_environment_subtree_is_pruned(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    environment = repo / ".venv-correction"
    create_pyvenv(environment)
    environment.joinpath("Lib", "site-packages", "package").mkdir(parents=True)
    environment.joinpath("Lib", "site-packages", "package", "module.py").write_text(
        "value = 1\n",
        encoding="utf-8",
    )
    visited: list[Path] = []
    real_scandir = workspace_guard_module._scandir

    def tracking_scandir(path: Path):
        visited.append(Path(path).resolve(strict=False))
        return real_scandir(path)

    monkeypatch.setattr(workspace_guard_module, "_scandir", tracking_scandir)

    snapshot = capture_workspace_environment_snapshot(repo)

    environment_root = environment.resolve(strict=False)
    assert [item.root_path for item in snapshot.environments] == [environment_root]
    assert environment_root not in visited
    assert not any(environment_root in path.parents for path in visited)


@pytest.mark.skipif(os.name != "nt", reason="Windows path identity is case-insensitive")
def test_windows_path_identity_is_case_insensitive(tmp_path):
    repo = (tmp_path / "Repo With Spaces").resolve(strict=False)
    environment = repo / ".VENV"
    marker = environment / "pyvenv.cfg"
    before = WorkspaceEnvironmentSnapshot(
        repository_path=repo,
        environments=(
            LocalEnvironment(
                kind="python_venv",
                root_path=environment,
                marker_paths=(marker,),
            ),
        ),
    )
    after = WorkspaceEnvironmentSnapshot(
        repository_path=repo,
        environments=(
            LocalEnvironment(
                kind="python_venv",
                root_path=Path(str(environment).lower()),
                marker_paths=(Path(str(marker).lower()),),
            ),
        ),
    )

    assert new_environments(before, after) == ()


def create_pyvenv(path: Path) -> None:
    path.mkdir(parents=True)
    path.joinpath("pyvenv.cfg").write_text("home = python\n", encoding="utf-8")
