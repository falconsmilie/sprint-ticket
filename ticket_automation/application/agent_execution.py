"""Provider-neutral contracts for agent-backed application work."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from math import isclose, isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Generic, Protocol, TypeAlias, TypeVar

from ..domain.task_results import ImplementationResult, ReviewResult, TaskResult


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

    name: str
    path: Path
    media_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise AgentContractError("artifact name must be a non-empty string.")
        if not isinstance(self.path, Path):
            raise AgentContractError("artifact path must be a Path.")
        if self.media_type is not None and (
            not isinstance(self.media_type, str) or not self.media_type.strip()
        ):
            raise AgentContractError("artifact media_type must be non-empty when set.")


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

    @property
    def successful(self) -> bool:
        return self.status is AgentExecutionStatus.SUCCESS

    @property
    def invocation_started(self) -> bool | None:
        if self.invocation_start is InvocationStart.UNKNOWN:
            return None
        return self.invocation_start is InvocationStart.STARTED

    def _validate_success(self) -> None:
        if self.invocation_start is not InvocationStart.STARTED:
            raise AgentContractError("successful execution must have started.")
        if self.result is None:
            raise AgentContractError(
                "successful execution must contain a typed result."
            )
        if type(self.result) is not _result_type_for(self.task_kind):
            raise AgentContractError(
                "successful execution result does not match its task kind."
            )
        if self.failure_category is not None or self.failure_message is not None:
            raise AgentContractError(
                "successful execution must not contain failure details."
            )

    def _validate_failure(self) -> None:
        if self.result is not None:
            raise AgentContractError("failed execution must not contain a result.")
        if not isinstance(self.failure_category, AgentFailureCategory):
            raise AgentContractError(
                "failed execution must contain a failure category."
            )
        if (
            not isinstance(self.failure_message, str)
            or not self.failure_message.strip()
        ):
            raise AgentContractError("failed execution must contain a failure message.")
        if (
            self.failure_category
            in {
                AgentFailureCategory.PROVIDER_UNAVAILABLE,
                AgentFailureCategory.INVOCATION_START_FAILURE,
                AgentFailureCategory.CAPABILITY_OR_CONFIGURATION_FAILURE,
            }
            and self.invocation_start is not InvocationStart.NOT_STARTED
        ):
            raise AgentContractError(
                f"{self.failure_category.value} must be recorded before invocation start."
            )
        if (
            self.failure_category
            not in {
                AgentFailureCategory.PROVIDER_UNAVAILABLE,
                AgentFailureCategory.INVOCATION_START_FAILURE,
                AgentFailureCategory.CAPABILITY_OR_CONFIGURATION_FAILURE,
            }
            and self.invocation_start is InvocationStart.NOT_STARTED
        ):
            raise AgentContractError(
                f"{self.failure_category.value} requires a started or uncertain invocation."
            )


class AgentExecutor(Protocol):
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
    "IMPLEMENTATION_RESULT_CONTRACT",
    "REVIEW_RESULT_CONTRACT",
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
    "InvocationStart",
    "NetworkAccess",
    "PersistedAgentExecutorFactory",
    "ProviderId",
    "ProviderMetadataScalar",
    "ProviderMetadataValue",
    "RepositoryAccess",
    "required_execution_capabilities",
]
