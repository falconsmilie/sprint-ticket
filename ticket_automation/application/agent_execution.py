"""Provider-neutral contracts for agent-backed application work."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from math import isclose, isfinite
from pathlib import Path, PurePosixPath, PureWindowsPath
from stat import S_ISDIR, S_ISREG
from types import MappingProxyType
from typing import Generic, Protocol, TypeAlias, TypeVar

from ..domain.task_results import ImplementationResult, ReviewResult, TaskResult
from ..models import ATTEMPT_RESULT_ARTIFACT_NAME, ATTEMPTS_DIR_NAME
from ..run_ownership import RunOwnership


class AgentContractError(ValueError):
    """Raised when an execution contract is internally inconsistent."""


@dataclass(frozen=True, order=True)
class ProviderId:
    """Stable identity of the provider that performed an invocation."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value.strip():
            raise AgentContractError("provider id must be a non-empty string.")
        if self.value != self.value.strip():
            raise AgentContractError(
                "provider id must not contain surrounding whitespace."
            )

    def __str__(self) -> str:
        return self.value


class AgentTaskKind(StrEnum):
    IMPLEMENTATION = "implementation"
    REVIEW = "review"
    CORRECTION = "correction"


class RepositoryAccess(StrEnum):
    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"


class AgentCapability(StrEnum):
    READ_ONLY_EXECUTION = "read-only-execution"
    WORKSPACE_WRITE_EXECUTION = "workspace-write-execution"
    STRUCTURED_RESULT = "structured-result"
    ISOLATED_INVOCATION = "isolated-invocation"
    DIAGNOSTIC_ARTIFACT_CAPTURE = "diagnostic-artifact-capture"
    NETWORK_POLICY_CONTROL = "network-policy-control"


class NetworkAccess(StrEnum):
    DENIED = "denied"
    ALLOWED = "allowed"


class AgentExecutionStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"


class InvocationStart(StrEnum):
    """What is known about whether provider-controlled work began."""

    NOT_STARTED = "not-started"
    STARTED = "started"
    UNKNOWN = "unknown"


class AgentFailureCategory(StrEnum):
    PROVIDER_UNAVAILABLE = "provider-unavailable"
    INVOCATION_START_FAILURE = "invocation-start-failure"
    TIMEOUT = "timeout"
    PROVIDER_REJECTION_OR_SERVICE_FAILURE = "provider-rejection-or-service-failure"
    NON_SUCCESSFUL_EXECUTION = "non-successful-execution"
    MISSING_RESULT = "missing-result"
    INVALID_RESULT = "invalid-result"
    CAPABILITY_OR_CONFIGURATION_FAILURE = "capability-or-configuration-failure"


def validate_execution_semantics(
    *,
    status: AgentExecutionStatus,
    invocation_start: InvocationStart,
    failure_category: AgentFailureCategory | None,
    failure_message: str | None,
) -> None:
    """Validate the shared runtime and persisted execution outcome vocabulary."""

    if not isinstance(status, AgentExecutionStatus):
        raise AgentContractError("status must be an AgentExecutionStatus value.")
    if not isinstance(invocation_start, InvocationStart):
        raise AgentContractError("invocation_start must be an InvocationStart value.")
    if status is AgentExecutionStatus.SUCCESS:
        if invocation_start is not InvocationStart.STARTED:
            raise AgentContractError("successful execution must have started.")
        if failure_category is not None or failure_message is not None:
            raise AgentContractError(
                "successful execution must not contain failure details."
            )
        return
    if not isinstance(failure_category, AgentFailureCategory):
        raise AgentContractError("failed execution must contain a failure category.")
    if not isinstance(failure_message, str) or not failure_message.strip():
        raise AgentContractError("failed execution must contain a failure message.")
    pre_invocation_failures = {
        AgentFailureCategory.PROVIDER_UNAVAILABLE,
        AgentFailureCategory.INVOCATION_START_FAILURE,
        AgentFailureCategory.CAPABILITY_OR_CONFIGURATION_FAILURE,
    }
    if (
        failure_category in pre_invocation_failures
        and invocation_start is not InvocationStart.NOT_STARTED
    ):
        raise AgentContractError(
            f"{failure_category.value} must be recorded before invocation start."
        )
    if (
        failure_category not in {*pre_invocation_failures, AgentFailureCategory.TIMEOUT}
        and invocation_start is InvocationStart.NOT_STARTED
    ):
        raise AgentContractError(
            f"{failure_category.value} requires a started or uncertain invocation."
        )


class ArtifactRole(StrEnum):
    """Stable provider-neutral meanings for attempt-owned artifacts."""

    EXECUTION_EVIDENCE = "execution-evidence"
    TYPED_RESULT = "typed-result"
    PROMPT = "prompt"
    PROVIDER_EVENTS = "provider-events"
    STANDARD_ERROR = "standard-error"
    PROVIDER_EXECUTION_DETAILS = "provider-execution-details"
    PROVIDER_DIAGNOSTIC_RESULT = "provider-diagnostic-result"
    WORKSPACE_GUARD = "workspace-guard"
    CORRECTION_TICKET = "correction-ticket"


EXECUTION_EVIDENCE_FILE = "execution.json"
WORKSPACE_GUARD_FILE = "workspace-guard.json"
CORRECTION_TICKET_FILE = "correction-ticket.md"


@dataclass(frozen=True)
class AttemptArtifactLayout:
    """Confine all artifact paths to one trusted run and attempt root."""

    run_root: Path
    attempt_root: Path
    run_ownership: RunOwnership | None = None
    _run_root_identity: tuple[int, int] | None = field(
        init=False,
        repr=False,
        compare=False,
    )
    _attempts_root_identity: tuple[int, int] | None = field(
        init=False,
        repr=False,
        compare=False,
    )
    _attempt_root_identity: tuple[int, int] | None = field(
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def attempts_root(cls, run_root: Path) -> Path:
        """Resolve the canonical attempts root without allowing a run escape."""

        resolved_run = _resolve_artifact_path(run_root, description="run root")
        attempts_root = _resolve_artifact_path(
            resolved_run / ATTEMPTS_DIR_NAME,
            description="attempts root",
        )
        try:
            attempts_relative = attempts_root.relative_to(resolved_run)
        except ValueError as error:
            raise AgentContractError(
                "attempts root must remain inside the owning run."
            ) from error
        if not attempts_relative.parts:
            raise AgentContractError("attempts root must remain inside the owning run.")
        return attempts_root

    @classmethod
    def for_attempt(
        cls,
        run_root: Path,
        attempt_root: Path,
        *,
        run_ownership: RunOwnership | None = None,
    ) -> AttemptArtifactLayout:
        """Build the canonical layout for a direct child of ``run/attempts``."""

        resolved_run = _resolve_artifact_path(run_root, description="run root")
        attempts_root = cls.attempts_root(resolved_run)
        layout = cls(resolved_run, attempt_root, run_ownership)
        try:
            attempt_relative = layout.attempt_root.relative_to(attempts_root)
        except ValueError as error:
            raise AgentContractError(
                "attempt artifact root must remain inside the owning run attempts root."
            ) from error
        if len(attempt_relative.parts) != 1:
            raise AgentContractError(
                "attempt artifact root must be a direct child of the owning run attempts root."
            )
        if (
            layout._attempts_root_identity is None
            or layout._attempt_root_identity is None
        ):
            raise AgentContractError(
                "attempts root and attempt artifact root must already exist."
            )
        return layout

    def __post_init__(self) -> None:
        if not isinstance(self.run_root, Path) or not isinstance(
            self.attempt_root, Path
        ):
            raise AgentContractError("artifact layout roots must be Path values.")
        if self.run_ownership is not None and not isinstance(
            self.run_ownership, RunOwnership
        ):
            raise AgentContractError("run_ownership must be a RunOwnership or None.")
        if self.run_ownership is not None:
            self.run_ownership.validate_run_path(self.run_root)
            self.run_ownership.validate_descendant(self.attempt_root)
        run_root = _resolve_artifact_path(self.run_root, description="run root")
        attempt_root = _resolve_artifact_path(
            self.attempt_root,
            description="attempt artifact root",
        )
        try:
            relative = attempt_root.relative_to(run_root)
        except ValueError as error:
            raise AgentContractError(
                "attempt artifact root must remain inside the owning run."
            ) from error
        if not relative.parts:
            raise AgentContractError(
                "attempt artifact root must be below the run root."
            )
        object.__setattr__(self, "run_root", run_root)
        object.__setattr__(self, "attempt_root", attempt_root)
        object.__setattr__(
            self,
            "_run_root_identity",
            _directory_identity(run_root, description="run root", allow_missing=True),
        )
        object.__setattr__(
            self,
            "_attempts_root_identity",
            _directory_identity(
                attempt_root.parent,
                description="attempts root",
                allow_missing=True,
            ),
        )
        object.__setattr__(
            self,
            "_attempt_root_identity",
            _directory_identity(
                attempt_root,
                description="attempt artifact root",
                allow_missing=True,
            ),
        )

    def revalidate(self) -> None:
        """Revalidate controller-owned roots after an external or long-running call."""

        if self.run_ownership is not None:
            self.run_ownership.validate_run_path(self.run_root)
            self.run_ownership.validate_descendant(self.attempt_root)
        current_run = _resolve_artifact_path(self.run_root, description="run root")
        current_attempts = _resolve_artifact_path(
            self.attempt_root.parent,
            description="attempts root",
        )
        current_attempt = _resolve_artifact_path(
            self.attempt_root,
            description="attempt artifact root",
        )
        if current_run != self.run_root:
            raise AgentContractError("run root was retargeted after layout binding.")
        if current_attempts != self.attempt_root.parent:
            raise AgentContractError(
                "attempts root was retargeted after layout binding."
            )
        if current_attempt != self.attempt_root:
            raise AgentContractError(
                "attempt artifact root was retargeted after layout binding."
            )
        self._revalidate_identity(
            current_run,
            field_name="_run_root_identity",
            description="run root",
            allow_missing=True,
        )
        self._revalidate_identity(
            current_attempts,
            field_name="_attempts_root_identity",
            description="attempts root",
            allow_missing=True,
        )
        self._revalidate_identity(
            current_attempt,
            field_name="_attempt_root_identity",
            description="attempt artifact root",
            allow_missing=True,
        )

    def _revalidate_identity(
        self,
        path: Path,
        *,
        field_name: str,
        description: str,
        allow_missing: bool,
    ) -> None:
        current = _directory_identity(
            path,
            description=description,
            allow_missing=allow_missing,
        )
        expected = getattr(self, field_name)
        if expected is None:
            if current is not None:
                object.__setattr__(self, field_name, current)
            return
        if current != expected:
            state = "missing" if current is None else "replaced"
            raise AgentContractError(
                f"{description} ownership was lost; the directory was {state}."
            )

    @property
    def attempt_id(self) -> str:
        return self.attempt_root.name

    @property
    def artifact_directory(self) -> Path:
        return self.attempt_root

    @staticmethod
    def path_is_directory(path: Path, *, description: str) -> bool:
        """Inspect a directory path without suppressing filesystem failures."""

        resolved = _resolve_artifact_path(path, description=description)
        details = _inspect_artifact_path(resolved, description=description)
        return details is not None and S_ISDIR(details.st_mode)

    @staticmethod
    def path_is_file(path: Path, *, description: str) -> bool:
        """Inspect a file path, returning false only for missing/non-file paths."""

        resolved = _resolve_artifact_path(path, description=description)
        details = _inspect_artifact_path(resolved, description=description)
        return details is not None and S_ISREG(details.st_mode)

    def artifact_file_exists(self, relative_path: str | PurePosixPath) -> bool:
        """Inspect an attempt-owned file while preserving controlled failures."""

        path = self.path(relative_path)
        exists = self.path_is_file(path, description="attempt artifact")
        self.revalidate()
        return exists

    def path(self, relative_path: str | PurePosixPath) -> Path:
        self.revalidate()
        relative = _validated_artifact_path(relative_path)
        candidate = _resolve_artifact_path(
            self.attempt_root / Path(*relative.parts),
            description="artifact path",
        )
        try:
            candidate.relative_to(self.attempt_root)
        except ValueError as error:
            raise AgentContractError(
                "artifact path must remain inside the owning attempt."
            ) from error
        return candidate

    def named_path(self, role: ArtifactRole) -> Path:
        filenames = {
            ArtifactRole.EXECUTION_EVIDENCE: EXECUTION_EVIDENCE_FILE,
            ArtifactRole.TYPED_RESULT: ATTEMPT_RESULT_ARTIFACT_NAME,
            ArtifactRole.WORKSPACE_GUARD: WORKSPACE_GUARD_FILE,
            ArtifactRole.CORRECTION_TICKET: CORRECTION_TICKET_FILE,
        }
        try:
            return self.path(filenames[role])
        except KeyError as error:
            raise AgentContractError(
                f"artifact role {role.value!r} has no provider-neutral filename."
            ) from error

    def reference(
        self,
        role: ArtifactRole,
        path: Path,
        media_type: str | None = None,
        *,
        require_exists: bool = False,
    ) -> ArtifactReference:
        self.revalidate()
        if not isinstance(role, ArtifactRole):
            raise AgentContractError("artifact role must be an ArtifactRole value.")
        resolved = _resolve_artifact_path(path, description="artifact reference")
        try:
            resolved.relative_to(self.attempt_root)
            run_relative = resolved.relative_to(self.run_root).as_posix()
        except ValueError as error:
            raise AgentContractError(
                "artifact reference must remain inside the owning attempt."
            ) from error
        if require_exists and not self.path_is_file(
            resolved,
            description="artifact reference",
        ):
            raise AgentContractError(f"referenced artifact does not exist: {resolved}")
        return ArtifactReference(role, run_relative, media_type)

    def resolve(
        self, reference: ArtifactReference, *, require_exists: bool = False
    ) -> Path:
        self.revalidate()
        if not isinstance(reference, ArtifactReference):
            raise AgentContractError("reference must be an ArtifactReference.")
        relative = _validated_artifact_path(reference.run_relative_path)
        resolved = _resolve_artifact_path(
            self.run_root / Path(*relative.parts),
            description="artifact reference",
        )
        try:
            resolved.relative_to(self.attempt_root)
        except ValueError as error:
            raise AgentContractError(
                "artifact reference does not belong to the owning attempt."
            ) from error
        if require_exists and not self.path_is_file(
            resolved,
            description="artifact reference",
        ):
            raise AgentContractError(f"referenced artifact does not exist: {resolved}")
        return resolved


def _result_type_for(
    task_kind: AgentTaskKind,
) -> type[ImplementationResult | ReviewResult]:
    if task_kind in {AgentTaskKind.IMPLEMENTATION, AgentTaskKind.CORRECTION}:
        return ImplementationResult
    if task_kind is AgentTaskKind.REVIEW:
        return ReviewResult
    raise AgentContractError("task_kind must be an AgentTaskKind value.")


@dataclass(frozen=True)
class AgentExecutionPolicy:
    """Application policy values an adapter must enforce for one invocation."""

    timeout_seconds: float | None
    network_access: NetworkAccess

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None:
            if isinstance(self.timeout_seconds, bool) or not isinstance(
                self.timeout_seconds, (int, float)
            ):
                raise AgentContractError("timeout_seconds must be a number or None.")
            if not isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
                raise AgentContractError(
                    "timeout_seconds must be finite and greater than zero."
                )
            object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        if not isinstance(self.network_access, NetworkAccess):
            raise AgentContractError("network_access must be a NetworkAccess value.")


ResultT_co = TypeVar("ResultT_co", bound=TaskResult, covariant=True)
ResultT = TypeVar("ResultT", bound=TaskResult)


@dataclass(frozen=True)
class AgentResultContract(Generic[ResultT_co]):
    """Names the domain type expected for a particular application task."""

    task_kind: AgentTaskKind
    result_type: type[ResultT_co]

    def __post_init__(self) -> None:
        if not isinstance(self.task_kind, AgentTaskKind):
            raise AgentContractError("result contract task_kind is invalid.")
        expected = _result_type_for(self.task_kind)
        if self.result_type is not expected:
            raise AgentContractError(
                f"{self.task_kind.value} tasks require {expected.__name__} results."
            )

    def accepts(self, result: object) -> bool:
        return type(result) is self.result_type


IMPLEMENTATION_RESULT_CONTRACT: AgentResultContract[ImplementationResult] = (
    AgentResultContract(AgentTaskKind.IMPLEMENTATION, ImplementationResult)
)
REVIEW_RESULT_CONTRACT: AgentResultContract[ReviewResult] = AgentResultContract(
    AgentTaskKind.REVIEW, ReviewResult
)
CORRECTION_RESULT_CONTRACT: AgentResultContract[ImplementationResult] = (
    AgentResultContract(AgentTaskKind.CORRECTION, ImplementationResult)
)


@dataclass(frozen=True)
class AgentExecutionRequest(Generic[ResultT_co]):
    task_kind: AgentTaskKind
    repository_path: Path
    repository_access: RepositoryAccess
    prompt: str
    result_contract: AgentResultContract[ResultT_co]
    artifact_directory: Path
    policy: AgentExecutionPolicy
    required_capabilities: frozenset[AgentCapability]
    artifact_layout: AttemptArtifactLayout | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_kind, AgentTaskKind):
            raise AgentContractError("task_kind must be an AgentTaskKind value.")
        if not isinstance(self.repository_path, Path):
            raise AgentContractError("repository_path must be a Path.")
        if not isinstance(self.repository_access, RepositoryAccess):
            raise AgentContractError(
                "repository_access must be a RepositoryAccess value."
            )
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise AgentContractError("prompt must be a non-empty string.")
        if not isinstance(self.result_contract, AgentResultContract):
            raise AgentContractError("result_contract must be an AgentResultContract.")
        if self.result_contract.task_kind is not self.task_kind:
            raise AgentContractError(
                "result_contract task kind must match the requested task kind."
            )
        if not isinstance(self.artifact_directory, Path):
            raise AgentContractError("artifact_directory must be a Path.")
        layout = self.artifact_layout
        if layout is None:
            inferred_run_root = self.artifact_directory.parent
            if inferred_run_root.name == ATTEMPTS_DIR_NAME:
                inferred_run_root = inferred_run_root.parent
            layout = AttemptArtifactLayout(
                inferred_run_root,
                self.artifact_directory,
            )
            object.__setattr__(self, "artifact_layout", layout)
        elif not isinstance(layout, AttemptArtifactLayout):
            raise AgentContractError(
                "artifact_layout must be an AttemptArtifactLayout."
            )
        elif layout.attempt_root != _resolve_artifact_path(
            self.artifact_directory,
            description="artifact directory",
        ):
            raise AgentContractError(
                "artifact_directory must match the artifact layout attempt root."
            )
        if not isinstance(self.policy, AgentExecutionPolicy):
            raise AgentContractError("policy must be an AgentExecutionPolicy.")
        capabilities = _capability_set(self.required_capabilities)
        object.__setattr__(self, "required_capabilities", capabilities)
        required_access = _capability_for_access(self.repository_access)
        if required_access not in capabilities:
            raise AgentContractError(
                f"{required_access.value} is required for "
                f"{self.repository_access.value} repository access."
            )
        if AgentCapability.STRUCTURED_RESULT not in capabilities:
            raise AgentContractError(
                "structured-result is required for a typed agent result contract."
            )
        if AgentCapability.NETWORK_POLICY_CONTROL not in capabilities:
            raise AgentContractError(
                "network-policy-control is required for an explicit network policy."
            )

    def missing_capabilities(
        self, available: Iterable[AgentCapability]
    ) -> frozenset[AgentCapability]:
        """Return unmet requirements so executors can fail before invocation."""

        return self.required_capabilities - _capability_set(available)


@dataclass(frozen=True)
class ArtifactReference:
    """A provider-neutral pointer to evidence captured for an invocation."""

    role: ArtifactRole
    run_relative_path: str
    media_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, ArtifactRole):
            raise AgentContractError("artifact role must be an ArtifactRole value.")
        normalized = _validated_artifact_path(self.run_relative_path).as_posix()
        object.__setattr__(self, "run_relative_path", normalized)
        if self.media_type is not None and (
            not isinstance(self.media_type, str) or not self.media_type.strip()
        ):
            raise AgentContractError("artifact media_type must be non-empty when set.")

    @property
    def name(self) -> str:
        return self.role.value


ProviderMetadataScalar: TypeAlias = str | int | float | bool | None
ProviderMetadataValue: TypeAlias = (
    ProviderMetadataScalar
    | tuple["ProviderMetadataValue", ...]
    | Mapping[str, "ProviderMetadataValue"]
)


@dataclass(frozen=True)
class AgentExecution(Generic[ResultT_co]):
    provider_id: ProviderId
    task_kind: AgentTaskKind
    status: AgentExecutionStatus
    invocation_start: InvocationStart
    started_at: datetime
    ended_at: datetime
    duration_seconds: float
    result: ResultT_co | None = None
    failure_category: AgentFailureCategory | None = None
    failure_message: str | None = None
    artifacts: tuple[ArtifactReference, ...] = ()
    provider_metadata: Mapping[str, ProviderMetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.provider_id, ProviderId):
            raise AgentContractError("provider_id must be a ProviderId.")
        if not isinstance(self.task_kind, AgentTaskKind):
            raise AgentContractError("task_kind must be an AgentTaskKind value.")
        if not isinstance(self.status, AgentExecutionStatus):
            raise AgentContractError("status must be an AgentExecutionStatus value.")
        if not isinstance(self.invocation_start, InvocationStart):
            raise AgentContractError(
                "invocation_start must be an InvocationStart value."
            )
        _validate_timestamp(self.started_at, field_name="started_at")
        _validate_timestamp(self.ended_at, field_name="ended_at")
        if self.ended_at < self.started_at:
            raise AgentContractError("ended_at must not be earlier than started_at.")
        if isinstance(self.duration_seconds, bool) or not isinstance(
            self.duration_seconds, (int, float)
        ):
            raise AgentContractError("duration_seconds must be a number.")
        if not isfinite(self.duration_seconds) or self.duration_seconds < 0:
            raise AgentContractError(
                "duration_seconds must be finite and not negative."
            )
        elapsed_seconds = (self.ended_at - self.started_at).total_seconds()
        if not isclose(
            float(self.duration_seconds), elapsed_seconds, rel_tol=0, abs_tol=1e-6
        ):
            raise AgentContractError(
                "duration_seconds must match the execution timestamps."
            )
        object.__setattr__(self, "duration_seconds", float(self.duration_seconds))
        if not isinstance(self.artifacts, tuple) or not all(
            isinstance(artifact, ArtifactReference) for artifact in self.artifacts
        ):
            raise AgentContractError(
                "artifacts must be a tuple of ArtifactReference values."
            )
        object.__setattr__(
            self, "provider_metadata", _freeze_metadata(self.provider_metadata)
        )
        if self.status is AgentExecutionStatus.SUCCESS:
            self._validate_success()
        else:
            self._validate_failure()
        validate_execution_semantics(
            status=self.status,
            invocation_start=self.invocation_start,
            failure_category=self.failure_category,
            failure_message=self.failure_message,
        )

    @property
    def successful(self) -> bool:
        return self.status is AgentExecutionStatus.SUCCESS

    def _validate_success(self) -> None:
        if self.result is None:
            raise AgentContractError(
                "successful execution must contain a typed result."
            )
        if type(self.result) is not _result_type_for(self.task_kind):
            raise AgentContractError(
                "successful execution result does not match its task kind."
            )

    def _validate_failure(self) -> None:
        if self.result is not None:
            raise AgentContractError("failed execution must not contain a result.")


class AgentExecutor(Protocol):
    @property
    def capabilities(self) -> frozenset[AgentCapability]: ...

    def execute(
        self,
        request: AgentExecutionRequest[ResultT],
        *,
        on_invocation_start: Callable[[], None] | None = None,
    ) -> AgentExecution[ResultT]: ...


@dataclass(frozen=True)
class AgentExecutorAssignments:
    """Constructed executor ports selected for each application task kind."""

    implementation: AgentExecutor
    review: AgentExecutor
    correction: AgentExecutor


class PersistedAgentExecutorFactory(Protocol):
    """Construct executors from an already resolved, persisted-shape policy."""

    def compatibility_problem(self, resolved_policy: object) -> str | None: ...

    def create_executors(self, resolved_policy: object) -> AgentExecutorAssignments: ...


def required_execution_capabilities(
    access: RepositoryAccess,
) -> frozenset[AgentCapability]:
    """Return the application requirements shared by all agent-backed stages."""

    return frozenset(
        {
            _capability_for_access(access),
            AgentCapability.STRUCTURED_RESULT,
            AgentCapability.ISOLATED_INVOCATION,
            AgentCapability.DIAGNOSTIC_ARTIFACT_CAPTURE,
            AgentCapability.NETWORK_POLICY_CONTROL,
        }
    )


def _capability_for_access(access: RepositoryAccess) -> AgentCapability:
    if access is RepositoryAccess.READ_ONLY:
        return AgentCapability.READ_ONLY_EXECUTION
    if access is RepositoryAccess.WORKSPACE_WRITE:
        return AgentCapability.WORKSPACE_WRITE_EXECUTION
    raise AgentContractError("access must be a RepositoryAccess value.")


def _capability_set(values: Iterable[AgentCapability]) -> frozenset[AgentCapability]:
    try:
        capabilities = frozenset(values)
    except TypeError as error:
        raise AgentContractError("capabilities must be an iterable.") from error
    if not all(isinstance(value, AgentCapability) for value in capabilities):
        raise AgentContractError(
            "capabilities must contain only AgentCapability values."
        )
    return capabilities


def _validate_timestamp(value: object, *, field_name: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise AgentContractError(f"{field_name} must be a timezone-aware datetime.")


def _resolve_artifact_path(path: Path, *, description: str) -> Path:
    """Resolve existing components while distinguishing absence from OS errors."""

    try:
        absolute = Path(os.path.abspath(path))
    except (OSError, RuntimeError, ValueError) as error:
        raise AgentContractError(
            f"Could not resolve {description} safely: {error}"
        ) from error
    try:
        return absolute.resolve(strict=True)
    except FileNotFoundError as error:
        # ``Path.resolve(strict=False)`` suppresses all OSErrors on supported
        # Python versions. Walk to the nearest missing component explicitly so
        # only genuine absence is tolerated and dangling links still fail.
        try:
            absolute.lstat()
        except FileNotFoundError:
            parent = absolute.parent
            if parent == absolute:
                raise AgentContractError(
                    f"Could not resolve {description} safely: {error}"
                ) from error
            return (
                _resolve_artifact_path(
                    parent,
                    description=description,
                )
                / absolute.name
            )
        except OSError as inspection_error:
            raise AgentContractError(
                f"Could not inspect {description} while resolving it safely: "
                f"{inspection_error}"
            ) from inspection_error
        raise AgentContractError(
            f"Could not resolve {description} safely: an existing path component "
            "has a missing target."
        ) from error
    except (OSError, RuntimeError) as error:
        raise AgentContractError(
            f"Could not resolve {description} safely: {error}"
        ) from error


def _inspect_artifact_path(path: Path, *, description: str) -> os.stat_result | None:
    try:
        return path.stat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise AgentContractError(
            f"Could not inspect {description} safely: {error}"
        ) from error


def _directory_identity(
    path: Path,
    *,
    description: str,
    allow_missing: bool,
) -> tuple[int, int] | None:
    try:
        details = path.stat()
    except FileNotFoundError as error:
        if allow_missing:
            return None
        raise AgentContractError(f"{description} does not exist: {path}") from error
    except OSError as error:
        raise AgentContractError(
            f"Could not inspect {description} identity safely: {error}"
        ) from error
    if not S_ISDIR(details.st_mode):
        raise AgentContractError(f"{description} must be a directory: {path}")
    return details.st_dev, details.st_ino


def _validated_artifact_path(value: object) -> PurePosixPath:
    if not isinstance(value, str | PurePosixPath):
        raise AgentContractError("artifact path must be a POSIX relative path.")
    text = str(value)
    if not text or "\\" in text:
        raise AgentContractError(
            "artifact path must be a non-empty POSIX relative path."
        )
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or PureWindowsPath(text).is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise AgentContractError("artifact path must be relative without traversal.")
    return path


def _freeze_metadata(
    value: Mapping[str, ProviderMetadataValue],
) -> Mapping[str, ProviderMetadataValue]:
    if not isinstance(value, Mapping):
        raise AgentContractError("provider_metadata must be a mapping.")
    if not all(isinstance(key, str) for key in value):
        raise AgentContractError("provider_metadata keys must be strings.")
    return MappingProxyType(
        {key: _freeze_metadata_value(item) for key, item in value.items()}
    )


def _freeze_metadata_value(value: object) -> ProviderMetadataValue:
    if isinstance(value, Mapping):
        return _freeze_metadata(value)
    if isinstance(value, tuple):
        return tuple(_freeze_metadata_value(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    raise AgentContractError(
        "provider_metadata values must be immutable JSON-compatible values."
    )


__all__ = [
    "CORRECTION_RESULT_CONTRACT",
    "CORRECTION_TICKET_FILE",
    "EXECUTION_EVIDENCE_FILE",
    "IMPLEMENTATION_RESULT_CONTRACT",
    "REVIEW_RESULT_CONTRACT",
    "WORKSPACE_GUARD_FILE",
    "AgentCapability",
    "AgentContractError",
    "AgentExecution",
    "AgentExecutionPolicy",
    "AgentExecutionRequest",
    "AgentExecutionStatus",
    "AgentExecutor",
    "AgentExecutorAssignments",
    "AgentFailureCategory",
    "AgentResultContract",
    "AgentTaskKind",
    "ArtifactReference",
    "ArtifactRole",
    "AttemptArtifactLayout",
    "InvocationStart",
    "NetworkAccess",
    "PersistedAgentExecutorFactory",
    "ProviderId",
    "ProviderMetadataScalar",
    "ProviderMetadataValue",
    "RepositoryAccess",
    "required_execution_capabilities",
    "validate_execution_semantics",
]
