"""Small, append-only records for work performed during one run.

The controller owns state transitions. Attempt records only describe an
invocation that was started while the controller was in a particular state;
they are deliberately not checkpoints that can advance a run on their own.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .application.agent_execution import (
    EXECUTION_EVIDENCE_FILE,
    AgentContractError,
    AttemptArtifactLayout,
)
from .models import (
    ATTEMPT_STATUS_BY_STAGE_OUTCOME,
    ATTEMPTS_DIR_NAME,
    PHASE_DEFINITIONS,
    AttemptPhase,
    AttemptStatus,
    StageOutcome,
)
from .persistence import (
    CodecError,
    atomic_write_json,
    parse_timestamp,
    read_json_object,
    timestamp_now,
)
from .run_ownership import (
    RunOwnership,
    RunOwnershipError,
    validate_unlinked_run_directory,
)

ATTEMPT_RECORD_FILE = "attempt.json"
ATTEMPT_RECORD_FORMAT = "ticket_automation.attempt"
ATTEMPT_RECORD_SCHEMA_VERSION = 1
_ATTEMPT_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "format",
        "sequence",
        "phase",
        "status",
        "before_workspace_fingerprint",
        "after_workspace_fingerprint",
        "process_started",
        "started_at",
        "ended_at",
        "result_path",
        "execution_path",
        "metadata",
    }
)

_ATTEMPT_DIRECTORY_PATTERN = re.compile(r"^(0*[1-9][0-9]*)-(.+)$")
_ATTEMPT_COMMIT_FENCES: dict[str, threading.Lock] = {}
_ATTEMPT_COMMIT_FENCES_LOCK = threading.Lock()


def _attempt_commit_fence(path: Path) -> threading.Lock:
    """Return the process-local serialization fence for one attempt ledger."""

    key = str(path.absolute())
    if os.name == "nt":
        key = key.casefold()
    with _ATTEMPT_COMMIT_FENCES_LOCK:
        return _ATTEMPT_COMMIT_FENCES.setdefault(key, threading.Lock())


@dataclass(frozen=True)
class AttemptMetadata:
    controller_message: str | None = None
    before_workspace_error: str | None = None
    after_workspace_error: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("controller_message", self.controller_message),
            ("before_workspace_error", self.before_workspace_error),
            ("after_workspace_error", self.after_workspace_error),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise AttemptError(f"Attempt metadata {field_name} must be non-empty.")

    def to_dict(self) -> dict[str, str]:
        return {
            key: value
            for key, value in (
                ("controller_message", self.controller_message),
                ("before_workspace_error", self.before_workspace_error),
                ("after_workspace_error", self.after_workspace_error),
            )
            if value is not None
        }


@dataclass(frozen=True)
class AttemptRecord:
    sequence: int
    phase: AttemptPhase
    status: AttemptStatus
    before_workspace_fingerprint: str | None
    after_workspace_fingerprint: str | None
    process_started: bool
    started_at: str
    ended_at: str | None
    result_path: str
    execution_path: str | None
    metadata: AttemptMetadata
    artifact_directory: Path
    artifact_layout: AttemptArtifactLayout | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    schema_version: int = ATTEMPT_RECORD_SCHEMA_VERSION
    format: str = ATTEMPT_RECORD_FORMAT

    def __post_init__(self) -> None:
        _validate_record(self)

    @property
    def path(self) -> Path:
        return self.artifact_directory / ATTEMPT_RECORD_FILE

    @property
    def completed(self) -> bool:
        return self.status is AttemptStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "format": self.format,
            "sequence": self.sequence,
            "phase": self.phase.value,
            "status": self.status.value,
            "before_workspace_fingerprint": self.before_workspace_fingerprint,
            "after_workspace_fingerprint": self.after_workspace_fingerprint,
            "process_started": self.process_started,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "result_path": self.result_path,
            "execution_path": self.execution_path,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True)
class StageAttempt:
    """Immutable attempt identity supplied to stage execution."""

    sequence: int
    phase: AttemptPhase
    before_workspace_fingerprint: str | None
    result_path: str
    artifact_directory: Path
    artifact_layout: AttemptArtifactLayout | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _validate_stage_attempt(self)

    @classmethod
    def from_record(
        cls,
        record: AttemptRecord,
        *,
        run_ownership: RunOwnership | None = None,
    ) -> StageAttempt:
        layout = attempt_artifact_layout(
            record.artifact_directory.parent.parent,
            record,
            run_ownership=run_ownership,
        )
        return cls(
            sequence=record.sequence,
            phase=record.phase,
            before_workspace_fingerprint=record.before_workspace_fingerprint,
            result_path=record.result_path,
            artifact_directory=record.artifact_directory,
            artifact_layout=layout,
        )


def require_stage_attempt(
    run_dir: Path | str,
    attempt: StageAttempt,
    *,
    phase: AttemptPhase,
    run_ownership: RunOwnership | None = None,
) -> StageAttempt:
    """Validate a stage capability against its controller-owned persisted record."""

    if not isinstance(attempt, StageAttempt):
        raise AttemptError("attempt must be a StageAttempt.")
    _validate_phase_value(phase, field="phase")
    if attempt.phase is not phase:
        raise AttemptError(
            f"Stage received {attempt.phase.value} attempt evidence; "
            f"expected {phase.value}."
        )
    attempt_layout = attempt_artifact_layout(
        run_dir,
        attempt,
        run_ownership=run_ownership,
    )
    persisted = _load_attempt(
        attempt_layout.path(ATTEMPT_RECORD_FILE),
        run_ownership=run_ownership,
    )
    persisted = replace(persisted, artifact_layout=attempt_layout)
    persisted_layout = attempt_artifact_layout(
        run_dir,
        persisted,
        run_ownership=run_ownership,
    )
    if persisted.status is not AttemptStatus.STARTED:
        raise AttemptError("Stage execution requires an active started attempt.")
    if (
        persisted.sequence != attempt.sequence
        or persisted.phase is not attempt.phase
        or persisted.before_workspace_fingerprint
        != attempt.before_workspace_fingerprint
        or persisted.result_path != attempt.result_path
        or persisted_layout.attempt_root != attempt_layout.attempt_root
    ):
        raise AttemptError("Stage attempt identity does not match persisted evidence.")
    return replace(attempt, artifact_layout=attempt_layout)


class AttemptError(RuntimeError):
    pass


def start_attempt(
    run_dir: Path | str,
    *,
    phase: AttemptPhase,
    before_workspace_fingerprint: str | None,
    execution_path: str | None = None,
    clock: Callable[[], datetime] | None = None,
    run_ownership: RunOwnership | None = None,
) -> AttemptRecord:
    """Create a fully recorded attempt before exposing its final directory.

    The initial record is written in a private temporary directory and then
    renamed into ``attempts/<sequence>-<phase>``. A crash cannot leave a final
    attempt directory without its authoritative record, and any old orphaned
    directory still reserves its sequence number as diagnostic history.
    """

    _validate_phase_value(phase, field="phase")
    run_path = Path(run_dir)
    try:
        if run_ownership is None:
            run_path.mkdir(parents=True, exist_ok=True)
        run_path = _validated_run_path(run_path, run_ownership=run_ownership)
        attempts_root = AttemptArtifactLayout.attempts_root(run_path)
        attempts_root.mkdir(parents=True, exist_ok=True)
        attempts_root = AttemptArtifactLayout.attempts_root(run_path)
    except RunOwnershipError as error:
        if run_ownership is not None:
            raise
        raise AttemptError(
            f"Could not prepare the owning run attempts directory safely: {error}"
        ) from error
    except (AgentContractError, OSError, RuntimeError) as error:
        raise AttemptError(
            f"Could not prepare the owning run attempts directory safely: {error}"
        ) from error
    while True:
        sequence = _next_sequence(run_path, attempts_root)
        directory = attempts_root / f"{sequence:03d}-{_phase_slug(phase)}"
        temporary_directory = Path(
            tempfile.mkdtemp(
                dir=attempts_root,
                prefix=f".{sequence:03d}-{_phase_slug(phase)}-",
                suffix=".tmp",
            )
        )
        record: AttemptRecord
        try:
            record = AttemptRecord(
                sequence=sequence,
                phase=phase,
                status=AttemptStatus.STARTED,
                before_workspace_fingerprint=before_workspace_fingerprint,
                after_workspace_fingerprint=None,
                process_started=False,
                started_at=timestamp_now(clock),
                ended_at=None,
                result_path=PHASE_DEFINITIONS[phase].result_artifact_name,
                execution_path=execution_path,
                metadata=AttemptMetadata(),
                artifact_directory=directory,
            )
            atomic_write_json(
                temporary_directory / ATTEMPT_RECORD_FILE, record.to_dict()
            )
            try:
                temporary_directory.rename(directory)
            except FileExistsError:
                _remove_temporary_attempt_directory(temporary_directory)
                continue
        except Exception:
            _remove_temporary_attempt_directory(temporary_directory)
            raise
        return replace(
            record,
            artifact_layout=AttemptArtifactLayout.for_attempt(
                run_path,
                directory,
                run_ownership=run_ownership,
            ),
        )


def save_attempt(
    record: AttemptRecord,
    *,
    expected_record: AttemptRecord | None = None,
    run_ownership: RunOwnership | None = None,
) -> None:
    _validate_record(record)
    run_dir = record.artifact_directory.parent.parent
    layout = attempt_artifact_layout(
        run_dir,
        record,
        run_ownership=run_ownership,
    )
    try:
        path = layout.path(ATTEMPT_RECORD_FILE)
    except AgentContractError as error:
        raise AttemptError(f"Attempt record confinement failed: {error}") from error
    if not layout.artifact_file_exists(ATTEMPT_RECORD_FILE):
        raise AttemptError(
            "Attempt update requires the original persisted attempt record."
        )

    def require_expected_persisted_record() -> None:
        persisted = _load_attempt(path, run_ownership=run_ownership)
        if not _same_attempt_identity(persisted, record):
            raise AttemptError("Persisted attempt identity changed before update.")
        if expected_record is not None and persisted != expected_record:
            raise AttemptError(
                "Persisted attempt changed after this update was prepared."
            )

    # Retain the early diagnostic check, then repeat it inside the short commit
    # fence. A delayed start recorder can prepare and fsync a stale temporary
    # payload without holding up lifecycle finalisation; once released, it must
    # revalidate after the finaliser and cannot replace the newer record.
    require_expected_persisted_record()
    fence = _attempt_commit_fence(path)

    def commit(replace_destination: Callable[[], None]) -> None:
        with fence:
            require_expected_persisted_record()
            replace_destination()

    atomic_write_json(path, record.to_dict(), commit=commit)


def update_attempt(
    record: AttemptRecord,
    *,
    status: AttemptStatus | None = None,
    before_workspace_fingerprint: str | None = None,
    after_workspace_fingerprint: str | None = None,
    process_started: bool | None = None,
    execution_path: str | None = None,
    metadata: AttemptMetadata | None = None,
    ended: bool = False,
    clock: Callable[[], datetime] | None = None,
    run_ownership: RunOwnership | None = None,
) -> AttemptRecord:
    updated = replace(
        record,
        status=record.status if status is None else status,
        before_workspace_fingerprint=(
            record.before_workspace_fingerprint
            if before_workspace_fingerprint is None
            else before_workspace_fingerprint
        ),
        after_workspace_fingerprint=(
            record.after_workspace_fingerprint
            if after_workspace_fingerprint is None
            else after_workspace_fingerprint
        ),
        process_started=(
            record.process_started if process_started is None else process_started
        ),
        execution_path=(
            record.execution_path if execution_path is None else execution_path
        ),
        metadata=record.metadata if metadata is None else metadata,
        ended_at=timestamp_now(clock) if ended else record.ended_at,
    )
    save_attempt(
        updated,
        expected_record=record,
        run_ownership=run_ownership,
    )
    return updated


def complete_attempt(
    record: AttemptRecord,
    *,
    status: AttemptStatus,
    after_workspace_fingerprint: str | None,
    process_started: bool | None = None,
    execution_path: str | None = None,
    metadata: AttemptMetadata | None = None,
    clock: Callable[[], datetime] | None = None,
    run_ownership: RunOwnership | None = None,
) -> AttemptRecord:
    return update_attempt(
        record,
        status=status,
        after_workspace_fingerprint=after_workspace_fingerprint,
        process_started=process_started,
        execution_path=execution_path,
        metadata=metadata,
        ended=True,
        clock=clock,
        run_ownership=run_ownership,
    )


def load_attempt_records(
    run_dir: Path | str,
    *,
    run_ownership: RunOwnership | None = None,
) -> tuple[AttemptRecord, ...]:
    """Load trusted records, rejecting malformed final attempt directories.

    Directories without ``attempt.json`` are harmless crash diagnostics. They
    are never adopted and reserve their sequence number for later attempts.
    A directory that claims to contain an authoritative record must validate
    completely; otherwise callers must stop rather than infer state from it.
    """

    try:
        run_path = _validated_run_path(
            Path(run_dir),
            run_ownership=run_ownership,
        )
    except RunOwnershipError as error:
        if run_ownership is not None:
            raise
        raise AttemptError(
            f"Could not inspect the owning run attempts directory safely: {error}"
        ) from error
    try:
        root = AttemptArtifactLayout.attempts_root(run_path)
        if not AttemptArtifactLayout.path_is_directory(
            root,
            description="attempts root",
        ):
            if run_ownership is not None:
                run_ownership.validate_run_path(run_path)
            return ()
        directories = (
            sorted(root.iterdir(), key=lambda item: item.name)
            if run_ownership is None
            else run_ownership.read_descendant(
                root,
                lambda source: sorted(source.iterdir(), key=lambda item: item.name),
            )
        )
    except (AgentContractError, OSError, RuntimeError) as error:
        raise AttemptError(
            f"Could not inspect the owning run attempts directory safely: {error}"
        ) from error
    records: list[AttemptRecord] = []
    for directory in directories:
        if directory.name.startswith("."):
            continue
        try:
            layout = AttemptArtifactLayout.for_attempt(
                run_path,
                directory,
                run_ownership=run_ownership,
            )
            if not AttemptArtifactLayout.path_is_directory(
                layout.attempt_root,
                description="attempt root",
            ):
                continue
            path = layout.path(ATTEMPT_RECORD_FILE)
            if not layout.artifact_file_exists(ATTEMPT_RECORD_FILE):
                continue
            record = _load_attempt(path, run_ownership=run_ownership)
            record = replace(record, artifact_layout=layout)
            attempt_artifact_layout(
                run_path,
                record,
                run_ownership=run_ownership,
            )
        except RunOwnershipError:
            raise
        except (AgentContractError, OSError, RuntimeError) as error:
            raise AttemptError(
                f"Could not inspect attempt directory {directory} safely: {error}"
            ) from error
        records.append(record)
    ordered = tuple(sorted(records, key=lambda item: item.sequence))
    sequences = [record.sequence for record in ordered]
    if len(sequences) != len(set(sequences)):
        raise AttemptError("Attempt records contain duplicate sequence numbers.")
    return ordered


def latest_attempt(
    run_dir: Path | str,
    *,
    phases: Iterable[AttemptPhase] | None = None,
    statuses: Iterable[AttemptStatus] | None = None,
    run_ownership: RunOwnership | None = None,
) -> AttemptRecord | None:
    phase_set = None if phases is None else _phase_filter(phases)
    status_set = None if statuses is None else _status_filter(statuses)
    for record in reversed(load_attempt_records(run_dir, run_ownership=run_ownership)):
        if phase_set is not None and record.phase not in phase_set:
            continue
        if status_set is not None and record.status not in status_set:
            continue
        return record
    return None


def latest_writable_attempt(
    run_dir: Path | str,
    *,
    run_ownership: RunOwnership | None = None,
) -> AttemptRecord | None:
    return latest_attempt(
        run_dir,
        phases=(
            phase
            for phase, definition in PHASE_DEFINITIONS.items()
            if definition.writes_target_repository
        ),
        statuses=(AttemptStatus.COMPLETED,),
        run_ownership=run_ownership,
    )


def complete_stage_attempt(
    run_dir: Path | str,
    attempt: AttemptRecord | StageAttempt,
    *,
    stage_outcome: StageOutcome,
    after_workspace_fingerprint: str | None,
    process_started: bool | None,
    metadata: AttemptMetadata | None = None,
    clock: Callable[[], datetime] | None = None,
    run_ownership: RunOwnership | None = None,
) -> AttemptRecord:
    """Complete the exact trusted attempt started by lifecycle orchestration."""

    if not isinstance(attempt, AttemptRecord | StageAttempt):
        raise AttemptError("attempt must be an AttemptRecord or StageAttempt.")
    if not isinstance(stage_outcome, StageOutcome):
        raise AttemptError("stage_outcome must be a StageOutcome value.")
    attempt_layout = attempt_artifact_layout(
        run_dir,
        attempt,
        run_ownership=run_ownership,
    )
    persisted = _load_attempt(
        attempt_layout.path(ATTEMPT_RECORD_FILE),
        run_ownership=run_ownership,
    )
    persisted = replace(persisted, artifact_layout=attempt_layout)
    persisted_layout = attempt_artifact_layout(
        run_dir,
        persisted,
        run_ownership=run_ownership,
    )
    if (
        persisted.sequence != attempt.sequence
        or persisted.phase is not attempt.phase
        or persisted.result_path != attempt.result_path
        or persisted_layout.attempt_root != attempt_layout.attempt_root
    ):
        raise AttemptError("Persisted attempt identity changed during stage dispatch.")
    if persisted.status is not AttemptStatus.STARTED:
        raise AttemptError("Lifecycle can only complete its active started attempt.")
    status = ATTEMPT_STATUS_BY_STAGE_OUTCOME[stage_outcome]
    return complete_attempt(
        persisted,
        status=status,
        after_workspace_fingerprint=after_workspace_fingerprint,
        process_started=process_started,
        execution_path=(
            EXECUTION_EVIDENCE_FILE
            if attempt_artifact_file_exists(
                run_dir,
                persisted,
                EXECUTION_EVIDENCE_FILE,
                run_ownership=run_ownership,
            )
            else persisted.execution_path
        ),
        metadata=persisted.metadata if metadata is None else metadata,
        clock=clock,
        run_ownership=run_ownership,
    )


def attempt_result_path(
    run_dir: Path | str,
    record: AttemptRecord | StageAttempt,
    *,
    run_ownership: RunOwnership | None = None,
) -> Path:
    return attempt_artifact_path(
        run_dir,
        record,
        record.result_path,
        run_ownership=run_ownership,
    )


def attempt_artifact_layout(
    run_dir: Path | str,
    record: AttemptRecord | StageAttempt,
    *,
    run_ownership: RunOwnership | None = None,
) -> AttemptArtifactLayout:
    """Return the one physically confined path owner for an attempt record."""

    if not isinstance(record, AttemptRecord | StageAttempt):
        raise AttemptError("record must be an AttemptRecord or StageAttempt.")
    requested_path = Path(run_dir).absolute()
    expected_record_directory = (
        requested_path / ATTEMPTS_DIR_NAME / _directory_name(record)
    )
    if record.artifact_directory.absolute() != expected_record_directory:
        raise AttemptError(
            "Attempt record does not belong to the supplied run directory."
        )
    try:
        run_path = _validated_run_path(
            requested_path,
            run_ownership=run_ownership,
        )
    except RunOwnershipError as error:
        if run_ownership is not None:
            raise
        raise AttemptError(f"Attempt artifact confinement failed: {error}") from error
    expected_directory = run_path / ATTEMPTS_DIR_NAME / _directory_name(record)
    bound_layout = record.artifact_layout
    if bound_layout is None:
        raise AttemptError(
            "Attempt artifact access requires the original bound artifact layout."
        )
    if not isinstance(bound_layout, AttemptArtifactLayout):
        raise AttemptError("Attempt artifact layout has the wrong type.")
    try:
        bound_layout.revalidate()
    except AgentContractError as error:
        raise AttemptError(
            f"Attempt artifact confinement in the owning run was lost: {error}"
        ) from error
    if (
        bound_layout.run_root != run_path
        or bound_layout.attempt_root != expected_directory
    ):
        raise AttemptError(
            "Attempt artifact layout does not match the persisted attempt identity."
        )
    return bound_layout


def _validated_run_path(
    run_dir: Path,
    *,
    run_ownership: RunOwnership | None = None,
) -> Path:
    if run_ownership is None:
        return validate_unlinked_run_directory(run_dir)
    if not isinstance(run_ownership, RunOwnership):
        raise TypeError("run_ownership must be a RunOwnership or None.")
    return run_ownership.validate_run_path(run_dir)


def attempt_artifact_path(
    run_dir: Path | str,
    record: AttemptRecord | StageAttempt,
    relative_path: str,
    *,
    run_ownership: RunOwnership | None = None,
) -> Path:
    layout = attempt_artifact_layout(
        run_dir,
        record,
        run_ownership=run_ownership,
    )
    _validate_relative_artifact_path(relative_path, field="artifact path")
    try:
        return layout.path(relative_path)
    except AgentContractError as error:
        raise AttemptError(f"Attempt artifact confinement failed: {error}") from error


def attempt_artifact_file_exists(
    run_dir: Path | str,
    record: AttemptRecord | StageAttempt,
    relative_path: str,
    *,
    run_ownership: RunOwnership | None = None,
) -> bool:
    """Return false for a missing artifact and reject inspection failures."""

    layout = attempt_artifact_layout(
        run_dir,
        record,
        run_ownership=run_ownership,
    )
    _validate_relative_artifact_path(relative_path, field="artifact path")
    try:
        return layout.artifact_file_exists(relative_path)
    except AgentContractError as error:
        raise AttemptError(f"Attempt artifact inspection failed: {error}") from error


def _load_attempt(
    path: Path,
    *,
    run_ownership: RunOwnership | None = None,
) -> AttemptRecord:
    try:
        data = (
            read_json_object(path)
            if run_ownership is None
            else run_ownership.read_descendant(path, read_json_object)
        )
    except CodecError as error:
        raise AttemptError(f"Could not read attempt record {path}: {error}") from error
    if not isinstance(data, dict):
        raise AttemptError(f"Attempt record {path} must be a JSON object.")
    try:
        if (
            type(data.get("schema_version")) is not int
            or data["schema_version"] != ATTEMPT_RECORD_SCHEMA_VERSION
        ):
            raise AttemptError("Attempt record has an unsupported schema version.")
        if data.get("format") != ATTEMPT_RECORD_FORMAT:
            raise AttemptError("Attempt record has an unsupported format.")
        _require_exact_record_fields(data)
        record = AttemptRecord(
            sequence=_required_positive_int(data, "sequence"),
            phase=_required_phase(data),
            status=_required_status(data),
            before_workspace_fingerprint=_nullable_string(
                data["before_workspace_fingerprint"]
            ),
            after_workspace_fingerprint=_nullable_string(
                data["after_workspace_fingerprint"]
            ),
            process_started=_required_bool(data, "process_started"),
            started_at=_required_timestamp(data, "started_at"),
            ended_at=_nullable_timestamp(data["ended_at"]),
            result_path=_required_relative_artifact_path(data, "result_path"),
            execution_path=_nullable_relative_artifact_path(
                data["execution_path"], "execution_path"
            ),
            metadata=_required_metadata(data),
            artifact_directory=path.parent,
            schema_version=data["schema_version"],
            format=data["format"],
        )
        if run_ownership is not None:
            run_ownership.validate_descendant(path)
        return record
    except (KeyError, TypeError, ValueError, AttemptError) as error:
        raise AttemptError(f"Invalid attempt record {path}: {error}") from error


def _next_sequence(run_dir: Path, root: Path) -> int:
    sequences: list[int] = []
    for directory in root.iterdir():
        if directory.name.startswith("."):
            continue
        try:
            layout = AttemptArtifactLayout.for_attempt(run_dir, directory)
            if not AttemptArtifactLayout.path_is_directory(
                layout.attempt_root,
                description="attempt sequence entry",
            ):
                continue
        except AgentContractError as error:
            raise AttemptError(
                f"Could not inspect attempt sequence entries safely: {error}"
            ) from error
        match = _ATTEMPT_DIRECTORY_PATTERN.fullmatch(directory.name)
        if match is not None:
            sequences.append(int(match.group(1)))
    return max(sequences, default=0) + 1


def _validate_record(record: AttemptRecord) -> None:
    if (
        type(record.schema_version) is not int
        or record.schema_version != ATTEMPT_RECORD_SCHEMA_VERSION
    ):
        raise AttemptError("Attempt record has an unsupported schema version.")
    if record.format != ATTEMPT_RECORD_FORMAT:
        raise AttemptError("Attempt record has an unsupported format.")
    if (
        not isinstance(record.sequence, int)
        or isinstance(record.sequence, bool)
        or record.sequence < 1
    ):
        raise AttemptError("Attempt sequence must be a positive integer.")
    if not isinstance(record.phase, AttemptPhase):
        raise AttemptError(f"Attempt phase is unsupported: {record.phase!r}.")
    if not isinstance(record.status, AttemptStatus):
        raise AttemptError(f"Attempt status is unsupported: {record.status!r}.")
    if not isinstance(record.process_started, bool):
        raise AttemptError("Attempt process_started must be a boolean.")
    _validate_timestamp(record.started_at, field="started_at")
    if record.ended_at is None:
        if record.status is not AttemptStatus.STARTED:
            raise AttemptError("Completed attempt records must have ended_at.")
    else:
        _validate_timestamp(record.ended_at, field="ended_at")
        if record.status is AttemptStatus.STARTED:
            raise AttemptError("Started attempt records cannot have ended_at.")
    _validate_nullable_string(
        record.before_workspace_fingerprint, "before_workspace_fingerprint"
    )
    _validate_nullable_string(
        record.after_workspace_fingerprint, "after_workspace_fingerprint"
    )
    _validate_relative_artifact_path(record.result_path, field="result_path")
    if record.result_path != PHASE_DEFINITIONS[record.phase].result_artifact_name:
        raise AttemptError("Attempt result_path does not match its phase definition.")
    if record.execution_path is not None:
        _validate_relative_artifact_path(record.execution_path, field="execution_path")
    if not isinstance(record.metadata, AttemptMetadata):
        raise AttemptError("Attempt metadata must be an AttemptMetadata value.")
    if record.artifact_directory.name != _directory_name(record):
        raise AttemptError("Attempt directory does not match its sequence and phase.")
    _validate_bound_layout(record)


def _validate_stage_attempt(attempt: StageAttempt) -> None:
    if (
        not isinstance(attempt.sequence, int)
        or isinstance(attempt.sequence, bool)
        or attempt.sequence < 1
    ):
        raise AttemptError("Attempt sequence must be a positive integer.")
    _validate_phase_value(attempt.phase, field="phase")
    _validate_nullable_string(
        attempt.before_workspace_fingerprint,
        "before_workspace_fingerprint",
    )
    _validate_relative_artifact_path(attempt.result_path, field="result_path")
    if attempt.result_path != PHASE_DEFINITIONS[attempt.phase].result_artifact_name:
        raise AttemptError("Attempt result_path does not match its phase definition.")
    if not isinstance(attempt.artifact_directory, Path):
        raise AttemptError("Attempt artifact_directory must be a Path.")
    if attempt.artifact_directory.name != _directory_name(attempt):
        raise AttemptError("Attempt directory does not match its sequence and phase.")


def _validate_bound_layout(record: AttemptRecord | StageAttempt) -> None:
    layout = record.artifact_layout
    if layout is None:
        return
    if not isinstance(layout, AttemptArtifactLayout):
        raise AttemptError("Attempt artifact_layout must be an AttemptArtifactLayout.")
    expected_run = record.artifact_directory.parent.parent.absolute()
    if (
        layout.run_root != expected_run
        or layout.attempt_root != record.artifact_directory.absolute()
    ):
        raise AttemptError("Attempt artifact_layout does not match its directory.")


def _same_attempt_identity(left: AttemptRecord, right: AttemptRecord) -> bool:
    return (
        left.sequence == right.sequence
        and left.phase is right.phase
        and left.started_at == right.started_at
        and left.result_path == right.result_path
        and left.artifact_directory == right.artifact_directory
    )


def _directory_name(record: AttemptRecord | StageAttempt) -> str:
    return f"{record.sequence:03d}-{_phase_slug(record.phase)}"


def _phase_slug(phase: AttemptPhase) -> str:
    _validate_phase_value(phase, field="phase")
    return PHASE_DEFINITIONS[phase].slug


def _validate_phase_value(phase: object, *, field: str) -> None:
    if not isinstance(phase, AttemptPhase):
        raise AttemptError(f"{field} must be an AttemptPhase value.")


def _phase_filter(phases: Iterable[AttemptPhase]) -> frozenset[AttemptPhase]:
    values = frozenset(phases)
    for phase in values:
        _validate_phase_value(phase, field="phase filter")
    return values


def _status_filter(statuses: Iterable[AttemptStatus]) -> frozenset[AttemptStatus]:
    values = frozenset(statuses)
    if any(not isinstance(status, AttemptStatus) for status in values):
        raise AttemptError("status filter must contain only AttemptStatus values.")
    return values


def _required_positive_int(data: dict[str, Any], key: str) -> int:
    value = data[key]
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _require_exact_record_fields(data: dict[str, Any]) -> None:
    actual = set(data)
    missing = sorted(_ATTEMPT_RECORD_FIELDS - actual)
    unexpected = sorted(actual - _ATTEMPT_RECORD_FIELDS)
    if not missing and not unexpected:
        return
    details: list[str] = []
    if missing:
        details.append("missing " + ", ".join(missing))
    if unexpected:
        details.append("unexpected " + ", ".join(unexpected))
    raise AttemptError("Attempt record fields are invalid: " + "; ".join(details))


def _required_phase(data: dict[str, Any]) -> AttemptPhase:
    value = _required_string(data, "phase")
    try:
        return AttemptPhase(value)
    except ValueError as error:
        raise ValueError(f"phase is unsupported: {value!r}") from error


def _required_status(data: dict[str, Any]) -> AttemptStatus:
    value = _required_string(data, "status")
    try:
        return AttemptStatus(value)
    except ValueError as error:
        raise ValueError(f"status is unsupported: {value!r}") from error


def _required_bool(data: dict[str, Any], key: str) -> bool:
    value = data[key]
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean")
    return value


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_timestamp(data: dict[str, Any], key: str) -> str:
    value = _required_string(data, key)
    _validate_timestamp(value, field=key)
    return value


def _nullable_timestamp(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("ended_at must be a string or null")
    _validate_timestamp(value, field="ended_at")
    return value


def _validate_timestamp(value: str, *, field: str) -> None:
    try:
        parse_timestamp(value, field=field)
    except CodecError as error:
        raise AttemptError(str(error)) from error


def _nullable_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("expected a string or null")
    return value


def _validate_nullable_string(value: str | None, field: str) -> None:
    if value is not None and not isinstance(value, str):
        raise AttemptError(f"{field} must be a string or null.")


def _nullable_relative_artifact_path(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string or null")
    _validate_relative_artifact_path(value, field=field)
    return value


def _required_relative_artifact_path(data: dict[str, Any], field: str) -> str:
    value = _required_string(data, field)
    _validate_relative_artifact_path(value, field=field)
    return value


def _validate_relative_artifact_path(value: str, *, field: str) -> None:
    if not value or "\\" in value:
        raise AttemptError(f"{field} must be a non-empty POSIX relative path.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise AttemptError(f"{field} must remain inside the attempt directory.")


def _required_metadata(data: dict[str, Any]) -> AttemptMetadata:
    value = data["metadata"]
    if not isinstance(value, dict):
        raise TypeError("metadata must be an object")
    supported = {
        "controller_message",
        "before_workspace_error",
        "after_workspace_error",
    }
    extra = set(value) - supported
    if extra:
        raise ValueError("metadata has unsupported fields: " + ", ".join(sorted(extra)))
    return AttemptMetadata(
        controller_message=_metadata_string(value, "controller_message"),
        before_workspace_error=_metadata_string(value, "before_workspace_error"),
        after_workspace_error=_metadata_string(value, "after_workspace_error"),
    )


def _metadata_string(data: dict[str, Any], field: str) -> str | None:
    value = data.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"metadata.{field} must be a non-empty string")
    return value


def _remove_temporary_attempt_directory(path: Path) -> None:
    try:
        for item in path.iterdir():
            item.unlink(missing_ok=True)
        path.rmdir()
    except OSError:
        # A leftover hidden temporary directory is diagnostic only and does not
        # participate in sequence allocation or result loading.
        pass


__all__ = [
    "ATTEMPTS_DIR_NAME",
    "ATTEMPT_RECORD_FILE",
    "AttemptError",
    "AttemptMetadata",
    "AttemptRecord",
    "StageAttempt",
    "attempt_artifact_file_exists",
    "attempt_artifact_layout",
    "attempt_artifact_path",
    "attempt_result_path",
    "complete_attempt",
    "complete_stage_attempt",
    "latest_attempt",
    "latest_writable_attempt",
    "load_attempt_records",
    "require_stage_attempt",
    "save_attempt",
    "start_attempt",
    "update_attempt",
]
