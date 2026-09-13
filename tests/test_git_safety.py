from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from tests.helpers import GIT, create_git_repo, run_git
from ticket_automation.git import GitCommandError, GitRepository
from ticket_automation.git_safety import WorkspaceSnapshot
from ticket_automation.workspace_guard import WorkspaceEnvironmentSnapshot

pytestmark = pytest.mark.skipif(
    GIT is None,
    reason="git executable is required for workspace snapshot tests",
)


def test_identical_workspace_has_deterministic_fingerprint(tmp_path):
    repository = GitRepository(create_git_repo(tmp_path / "repo"))

    first = WorkspaceSnapshot.capture(repository)
    second = WorkspaceSnapshot.capture(repository)

    assert first.inspection_complete
    assert first.fingerprint == second.fingerprint
    assert first.canonical_json() == second.canonical_json()
    assert first.matches(second)


def test_tracked_modification_changes_fingerprint(tmp_path):
    repository = GitRepository(create_git_repo(tmp_path / "repo"))
    before = WorkspaceSnapshot.capture(repository)

    repository.path.joinpath("file.txt").write_text("changed\n", encoding="utf-8")
    after = WorkspaceSnapshot.capture(repository)

    assert before.fingerprint != after.fingerprint
    assert {change.name for change in before.compare(after)} == {"tracked-diff"}


def test_untracked_path_addition_and_removal_change_fingerprint(tmp_path):
    repository = GitRepository(create_git_repo(tmp_path / "repo"))
    before = WorkspaceSnapshot.capture(repository)
    untracked = repository.path / "new.txt"

    untracked.write_text("new\n", encoding="utf-8")
    added = WorkspaceSnapshot.capture(repository)
    untracked.unlink()
    removed = WorkspaceSnapshot.capture(repository)

    assert before.fingerprint != added.fingerprint
    assert {change.name for change in before.compare(added)} == {"untracked-files"}
    assert removed.fingerprint == before.fingerprint


def test_untracked_content_change_changes_fingerprint(tmp_path):
    repository = GitRepository(create_git_repo(tmp_path / "repo"))
    untracked = repository.path / "new.txt"
    untracked.write_text("one\n", encoding="utf-8")
    before = WorkspaceSnapshot.capture(repository)

    untracked.write_text("two\n", encoding="utf-8")
    after = WorkspaceSnapshot.capture(repository)

    assert before.fingerprint != after.fingerprint
    assert {change.name for change in before.compare(after)} == {"untracked-content"}


def test_untracked_path_and_existing_content_changes_are_both_reported(tmp_path):
    repository = GitRepository(create_git_repo(tmp_path / "repo"))
    existing = repository.path / "existing.txt"
    existing.write_text("before\n", encoding="utf-8")
    before = WorkspaceSnapshot.capture(repository)

    existing.write_text("after\n", encoding="utf-8")
    repository.path.joinpath("added.txt").write_text("added\n", encoding="utf-8")
    after = WorkspaceSnapshot.capture(repository)

    assert {change.name for change in before.compare(after)} == {
        "untracked-content",
        "untracked-files",
    }


def test_staging_change_changes_fingerprint(tmp_path):
    repository = GitRepository(create_git_repo(tmp_path / "repo"))
    repository.path.joinpath("file.txt").write_text("changed\n", encoding="utf-8")
    before = WorkspaceSnapshot.capture(repository)

    run_git(repository.path, "add", "file.txt")
    after = WorkspaceSnapshot.capture(repository)

    assert before.fingerprint != after.fingerprint
    assert "staging" in {change.name for change in before.compare(after)}


def test_staged_content_change_with_same_paths_changes_fingerprint(tmp_path):
    repository = GitRepository(create_git_repo(tmp_path / "repo"))
    tracked = repository.path / "file.txt"
    tracked.write_text("first staged version\n", encoding="utf-8")
    run_git(repository.path, "add", "file.txt")
    tracked.write_text("same worktree version\n", encoding="utf-8")
    before = WorkspaceSnapshot.capture(repository)

    run_git(repository.path, "add", "file.txt")
    after = WorkspaceSnapshot.capture(repository)

    assert before.staged_paths == after.staged_paths
    assert before.tracked_diff_sha256 == after.tracked_diff_sha256
    assert before.fingerprint != after.fingerprint
    assert "staging" in {change.name for change in before.compare(after)}


def test_branch_and_head_changes_change_fingerprint(tmp_path):
    repository = GitRepository(create_git_repo(tmp_path / "repo"))
    before = WorkspaceSnapshot.capture(repository)

    run_git(repository.path, "checkout", "-b", "snapshot-branch")
    branch_changed = WorkspaceSnapshot.capture(repository)
    repository.path.joinpath("file.txt").write_text("committed\n", encoding="utf-8")
    run_git(repository.path, "add", "file.txt")
    run_git(repository.path, "commit", "-m", "snapshot head")
    head_changed = WorkspaceSnapshot.capture(repository)

    assert before.fingerprint != branch_changed.fingerprint
    assert "branch" in {change.name for change in before.compare(branch_changed)}
    assert branch_changed.fingerprint != head_changed.fingerprint
    assert "HEAD" in {change.name for change in branch_changed.compare(head_changed)}


def test_environment_root_change_changes_fingerprint(tmp_path):
    repository = GitRepository(create_git_repo(tmp_path / "repo"))
    before = WorkspaceSnapshot.capture(repository)

    environment = repository.path / ".venv"
    environment.mkdir()
    environment.joinpath("pyvenv.cfg").write_text(
        "home = synthetic\n", encoding="utf-8"
    )
    after = WorkspaceSnapshot.capture(repository)

    assert before.fingerprint != after.fingerprint
    assert "environment-roots" in {change.name for change in before.compare(after)}


def test_inspection_error_is_explicit_and_never_matches(monkeypatch, tmp_path):
    repository = GitRepository(create_git_repo(tmp_path / "repo"))

    def fail_head() -> str:
        raise GitCommandError("synthetic inspection failure")

    monkeypatch.setattr(repository, "head_sha", fail_head)
    first = WorkspaceSnapshot.capture(repository)
    second = WorkspaceSnapshot.capture(repository)

    assert not first.inspection_complete
    assert "synthetic inspection failure" in " ".join(first.inspection_errors)
    assert not first.matches(second)
    assert not first.matches_fingerprint(first.fingerprint)
    assert "inspection-incomplete" in {change.name for change in first.compare(second)}


@pytest.mark.parametrize(
    "method_name",
    [
        "current_branch",
        "head_sha",
        "staged_files",
        "staged_diff",
        "tracked_diff",
        "untracked_files",
    ],
)
def test_each_git_inspection_failure_marks_snapshot_incomplete(
    monkeypatch,
    tmp_path,
    method_name,
):
    repository = GitRepository(create_git_repo(tmp_path / "repo"))

    def fail_inspection():
        raise GitCommandError(f"synthetic {method_name} failure")

    monkeypatch.setattr(repository, method_name, fail_inspection)

    snapshot = WorkspaceSnapshot.capture(repository)

    assert not snapshot.inspection_complete
    assert any(method_name in error for error in snapshot.inspection_errors)


def test_untracked_content_read_failure_marks_snapshot_incomplete(
    monkeypatch,
    tmp_path,
):
    repository = GitRepository(create_git_repo(tmp_path / "repo"))
    repository.path.joinpath("untracked.txt").write_text("content\n", encoding="utf-8")

    def fail_hash(path):
        raise PermissionError(f"denied: {path}")

    monkeypatch.setattr("ticket_automation.git_safety._file_sha256", fail_hash)

    snapshot = WorkspaceSnapshot.capture(repository)

    assert not snapshot.inspection_complete
    assert any(
        "untracked-content:untracked.txt" in error
        for error in snapshot.inspection_errors
    )


def test_environment_inspection_error_marks_snapshot_incomplete(
    monkeypatch,
    tmp_path,
):
    repository = GitRepository(create_git_repo(tmp_path / "repo"))
    monkeypatch.setattr(
        "ticket_automation.git_safety.capture_workspace_environment_snapshot",
        lambda path: WorkspaceEnvironmentSnapshot(
            repository_path=path,
            environments=(),
            inspection_errors=("denied environment directory",),
        ),
    )

    snapshot = WorkspaceSnapshot.capture(repository)

    assert not snapshot.inspection_complete
    assert any(
        "denied environment directory" in error for error in snapshot.inspection_errors
    )


def test_snapshot_paths_are_sorted_deterministically(tmp_path):
    repository = GitRepository(create_git_repo(tmp_path / "repo"))
    repository.path.joinpath("z.txt").write_text("z\n", encoding="utf-8")
    repository.path.joinpath("a.txt").write_text("a\n", encoding="utf-8")

    snapshot = WorkspaceSnapshot.capture(repository)
    reordered = replace(
        snapshot,
        untracked_paths=tuple(reversed(snapshot.untracked_paths)),
        untracked_file_hashes=tuple(reversed(snapshot.untracked_file_hashes)),
    )

    assert snapshot.untracked_paths == ("a.txt", "z.txt")
    assert snapshot.fingerprint == reordered.fingerprint


def test_trusted_snapshot_construction_is_canonical_and_immutable(tmp_path):
    empty_hash = hashlib.sha256(b"").hexdigest()
    a_hash = hashlib.sha256(b"a").hexdigest()
    z_hash = hashlib.sha256(b"z").hexdigest()
    snapshot = WorkspaceSnapshot(
        repository_path=tmp_path / "repo",
        branch="feature/example",
        head_sha="a" * 40,
        staged_paths=("z.py", "a.py"),
        staged_diff_sha256=empty_hash,
        tracked_diff_sha256=empty_hash,
        untracked_paths=("z.txt", "a.txt"),
        untracked_file_hashes=(("z.txt", z_hash), ("a.txt", a_hash)),
        environment_roots=("z-env", "a-env"),
        inspection_complete=True,
    )

    assert snapshot.inspection_complete
    assert snapshot.staged_paths == ("a.py", "z.py")
    assert snapshot.untracked_paths == ("a.txt", "z.txt")
    assert snapshot.untracked_file_hashes == (("a.txt", a_hash), ("z.txt", z_hash))
    assert snapshot.environment_roots == ("a-env", "z-env")
    expected_json = (
        '{"branch":"feature/example","environment_roots":["a-env","z-env"],'
        '"format":"ticket_automation.workspace_snapshot",'
        f'"head_sha":"{"a" * 40}","inspection_complete":true,'
        '"inspection_errors":[],"repository_path":'
        f"{json.dumps(snapshot.repository_path.as_posix())},"
        '"schema_version":1,'
        f'"staged_diff_sha256":"{empty_hash}",'
        '"staged_paths":["a.py","z.py"],'
        f'"tracked_diff_sha256":"{empty_hash}",'
        f'"untracked_file_hashes":[["a.txt","{a_hash}"],["z.txt","{z_hash}"]],'
        '"untracked_paths":["a.txt","z.txt"]}'
    )
    assert snapshot.canonical_json() == expected_json
    assert snapshot.fingerprint == hashlib.sha256(expected_json.encode()).hexdigest()
    with pytest.raises(FrozenInstanceError):
        snapshot.__setattr__("branch", "other")


@pytest.mark.parametrize(
    ("changes", "error_fragment"),
    [
        ({"head_sha": None}, "HEAD"),
        ({"staged_diff_sha256": None}, "staged-diff"),
        ({"tracked_diff_sha256": "invalid"}, "tracked-worktree"),
        ({"untracked_file_hashes": ()}, "paths and content hashes"),
    ],
)
def test_invalid_trusted_construction_is_forced_incomplete(
    tmp_path,
    changes,
    error_fragment,
):
    content_hash = hashlib.sha256(b"content").hexdigest()
    empty_hash = hashlib.sha256(b"").hexdigest()
    valid = WorkspaceSnapshot(
        repository_path=tmp_path / "repo",
        branch="feature/example",
        head_sha="a" * 40,
        staged_paths=(),
        staged_diff_sha256=empty_hash,
        tracked_diff_sha256=empty_hash,
        untracked_paths=("file.txt",),
        untracked_file_hashes=(("file.txt", content_hash),),
        environment_roots=(),
        inspection_complete=True,
    )

    invalid = replace(
        valid,
        **changes,
        inspection_complete=True,
        inspection_errors=(),
    )

    assert not invalid.inspection_complete
    assert any(error_fragment in error for error in invalid.inspection_errors)
    assert not invalid.matches(invalid)


def test_snapshot_capture_does_not_mutate_git(tmp_path):
    repository = GitRepository(create_git_repo(tmp_path / "repo"))
    repository.path.joinpath("file.txt").write_text("changed\n", encoding="utf-8")
    repository.path.joinpath("untracked.txt").write_text("new\n", encoding="utf-8")
    before_status = run_git(repository.path, "status", "--porcelain=v1")
    before_head = run_git(repository.path, "rev-parse", "HEAD")

    WorkspaceSnapshot.capture(repository)

    assert run_git(repository.path, "status", "--porcelain=v1") == before_status
    assert run_git(repository.path, "rev-parse", "HEAD") == before_head
