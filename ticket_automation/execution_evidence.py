"""Codec for the provider-neutral execution evidence envelope."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType

from .application.agent_execution import (
    AgentCapability,
    AgentExecution,
    AgentExecutionRequest,
    AgentExecutionStatus,
    AgentFailureCategory,
    AgentTaskKind,
    ArtifactReference,
    ArtifactRole,
    AttemptArtifactLayout,
    InvocationStart,
    ProviderId,
    ProviderMetadataValue,
    RepositoryAccess,
)
from .domain.task_results import TaskResult
from .persistence import (
    CodecError,
    JsonObject,
    JsonValue,
    atomic_write_json,
    format_timestamp,
    parse_timestamp,
    read_json_object,
)

EXECUTION_EVIDENCE_SCHEMA_VERSION = 1
EXECUTION_EVIDENCE_FORMAT = "ticket_automation.execution_evidence"


@dataclass(frozen=True)
class ExecutionEvidence:
    execution_id: str
    attempt_id: str
    provider_id: ProviderId
    task_kind: AgentTaskKind
    repository_access: RepositoryAccess
    required_capabilities: frozenset[AgentCapability]
    started_at: datetime
    ended_at: datetime
    duration_seconds: float
    status: AgentExecutionStatus
    invocation_start: InvocationStart
    failure_category: AgentFailureCategory | None = None
    failure_message: str | None = None
    typed_result_artifact: ArtifactReference | None = None
    artifacts: tuple[ArtifactReference, ...] = ()
    provider_metadata: Mapping[str, ProviderMetadataValue] = field(default_factory=dict)
    schema_version: int = EXECUTION_EVIDENCE_SCHEMA_VERSION
    format: str = EXECUTION_EVIDENCE_FORMAT

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTION_EVIDENCE_SCHEMA_VERSION:
            raise CodecError("Execution evidence has an unsupported schema version.")
        if self.format != EXECUTION_EVIDENCE_FORMAT:
            raise CodecError("Execution evidence has an unsupported format.")
        for name, value in (
            ("execution_id", self.execution_id),
            ("attempt_id", self.attempt_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise CodecError(f"{name} must be a non-empty string.")
        if not isinstance(self.provider_id, ProviderId):
            raise CodecError("provider_id must be a ProviderId.")
        if not isinstance(self.task_kind, AgentTaskKind):
            raise CodecError("task_kind must be an AgentTaskKind.")
        if not isinstance(self.repository_access, RepositoryAccess):
            raise CodecError("repository_access must be a RepositoryAccess.")
        if not isinstance(self.required_capabilities, frozenset) or not all(
            isinstance(item, AgentCapability) for item in self.required_capabilities
        ):
            raise CodecError("required_capabilities contains an unsupported value.")
        if not isinstance(self.status, AgentExecutionStatus):
            raise CodecError("status must be an AgentExecutionStatus.")
        if not isinstance(self.invocation_start, InvocationStart):
            raise CodecError("invocation_start must be an InvocationStart.")
        if self.ended_at < self.started_at:
            raise CodecError("ended_at must not precede started_at.")
        if (
            abs(
                (self.ended_at - self.started_at).total_seconds()
                - self.duration_seconds
            )
            > 1e-6
        ):
            raise CodecError("duration_seconds must match the execution timestamps.")
        if self.typed_result_artifact is not None and (
            not isinstance(self.typed_result_artifact, ArtifactReference)
            or self.typed_result_artifact.role is not ArtifactRole.TYPED_RESULT
        ):
            raise CodecError("typed_result_artifact must have the typed-result role.")
        if not isinstance(self.artifacts, tuple) or not all(
            isinstance(item, ArtifactReference) for item in self.artifacts
        ):
            raise CodecError("artifacts must be ArtifactReference values.")
        if self.status is AgentExecutionStatus.SUCCESS:
            if self.invocation_start is not InvocationStart.STARTED:
                raise CodecError(
                    "Successful evidence must record a started invocation."
                )
            if self.failure_category is not None or self.failure_message is not None:
                raise CodecError("Successful evidence cannot contain failure details.")
        elif (
            not isinstance(self.failure_category, AgentFailureCategory)
            or not isinstance(self.failure_message, str)
            or not self.failure_message.strip()
        ):
            raise CodecError("Failed evidence requires a category and message.")
        object.__setattr__(
            self,
            "provider_metadata",
            _freeze_metadata(self.provider_metadata),
        )


def evidence_from_execution(
    request: AgentExecutionRequest[TaskResult],
    execution: AgentExecution[TaskResult],
) -> ExecutionEvidence:
    layout = request.artifact_layout
    assert layout is not None
    typed = next(
        (
            item
            for item in execution.artifacts
            if item.role is ArtifactRole.TYPED_RESULT
        ),
        None,
    )
    return ExecutionEvidence(
        execution_id=f"{layout.attempt_id}:{request.task_kind.value}",
        attempt_id=layout.attempt_id,
        provider_id=execution.provider_id,
        task_kind=execution.task_kind,
        repository_access=request.repository_access,
        required_capabilities=request.required_capabilities,
        started_at=execution.started_at,
        ended_at=execution.ended_at,
        duration_seconds=execution.duration_seconds,
        status=execution.status,
        invocation_start=execution.invocation_start,
        failure_category=execution.failure_category,
        failure_message=execution.failure_message,
        typed_result_artifact=typed,
        artifacts=tuple(item for item in execution.artifacts if item is not typed),
        provider_metadata=execution.provider_metadata,
    )


def write_execution_evidence(
    layout: AttemptArtifactLayout,
    evidence: ExecutionEvidence,
) -> ArtifactReference:
    path = layout.named_path(ArtifactRole.EXECUTION_EVIDENCE)
    atomic_write_json(path, encode_execution_evidence(evidence))
    return layout.reference(
        ArtifactRole.EXECUTION_EVIDENCE,
        path,
        "application/json",
        require_exists=True,
    )


def read_execution_evidence(layout: AttemptArtifactLayout) -> ExecutionEvidence:
    return decode_execution_evidence(
        read_json_object(layout.named_path(ArtifactRole.EXECUTION_EVIDENCE)),
        layout=layout,
    )


def encode_execution_evidence(evidence: ExecutionEvidence) -> JsonObject:
    if not isinstance(evidence, ExecutionEvidence):
        raise CodecError("value must be ExecutionEvidence.")
    return {
        "schema_version": evidence.schema_version,
        "format": evidence.format,
        "execution_id": evidence.execution_id,
        "attempt_id": evidence.attempt_id,
        "provider_id": evidence.provider_id.value,
        "task_kind": evidence.task_kind.value,
        "requested_repository_access": evidence.repository_access.value,
        "required_capabilities": sorted(
            item.value for item in evidence.required_capabilities
        ),
        "started_at": format_timestamp(evidence.started_at),
        "ended_at": format_timestamp(evidence.ended_at),
        "duration_seconds": evidence.duration_seconds,
        "status": evidence.status.value,
        "invocation_start": evidence.invocation_start.value,
        "failure": None
        if evidence.failure_category is None
        else {
            "category": evidence.failure_category.value,
            "message": evidence.failure_message,
        },
        "typed_result_artifact": _encode_artifact(evidence.typed_result_artifact),
        "artifacts": [_encode_artifact(item) for item in evidence.artifacts],
        "provider_metadata": _metadata_to_json(evidence.provider_metadata),
    }


def decode_execution_evidence(
    value: object,
    *,
    layout: AttemptArtifactLayout | None = None,
) -> ExecutionEvidence:
    data = _exact_object(
        value,
        fields={
            "schema_version",
            "format",
            "execution_id",
            "attempt_id",
            "provider_id",
            "task_kind",
            "requested_repository_access",
            "required_capabilities",
            "started_at",
            "ended_at",
            "duration_seconds",
            "status",
            "invocation_start",
            "failure",
            "typed_result_artifact",
            "artifacts",
            "provider_metadata",
        },
        source="execution evidence",
    )
    _require_exact(data, "schema_version", EXECUTION_EVIDENCE_SCHEMA_VERSION)
    _require_exact(data, "format", EXECUTION_EVIDENCE_FORMAT)
    capabilities_value = data["required_capabilities"]
    if not isinstance(capabilities_value, list):
        raise CodecError("execution evidence.required_capabilities must be an array.")
    capabilities = frozenset(
        _enum(AgentCapability, item, "required_capabilities")
        for item in capabilities_value
    )
    artifacts_value = data["artifacts"]
    if not isinstance(artifacts_value, list):
        raise CodecError("execution evidence.artifacts must be an array.")
    artifacts = tuple(_decode_artifact(item) for item in artifacts_value)
    typed_value = data["typed_result_artifact"]
    typed = None if typed_value is None else _decode_artifact(typed_value)
    failure_value = data["failure"]
    failure_category = None
    failure_message = None
    if failure_value is not None:
        failure = _exact_object(
            failure_value,
            fields={"category", "message"},
            source="execution evidence.failure",
        )
        failure_category = _enum(
            AgentFailureCategory, failure["category"], "failure.category"
        )
        failure_message = _string(failure["message"], "failure.message")
    metadata_value = data["provider_metadata"]
    if not isinstance(metadata_value, dict):
        raise CodecError("execution evidence.provider_metadata must be an object.")
    try:
        provider_id = ProviderId(_string(data["provider_id"], "provider_id"))
    except ValueError as error:
        raise CodecError(str(error)) from error
    evidence = ExecutionEvidence(
        schema_version=data["schema_version"],
        format=data["format"],
        execution_id=_string(data["execution_id"], "execution_id"),
        attempt_id=_string(data["attempt_id"], "attempt_id"),
        provider_id=provider_id,
        task_kind=_enum(AgentTaskKind, data["task_kind"], "task_kind"),
        repository_access=_enum(
            RepositoryAccess,
            data["requested_repository_access"],
            "requested_repository_access",
        ),
        required_capabilities=capabilities,
        started_at=parse_timestamp(data["started_at"], field="started_at"),
        ended_at=parse_timestamp(data["ended_at"], field="ended_at"),
        duration_seconds=_number(data["duration_seconds"], "duration_seconds"),
        status=_enum(AgentExecutionStatus, data["status"], "status"),
        invocation_start=_enum(
            InvocationStart, data["invocation_start"], "invocation_start"
        ),
        failure_category=failure_category,
        failure_message=failure_message,
        typed_result_artifact=typed,
        artifacts=artifacts,
        provider_metadata=_decode_metadata(metadata_value),
    )
    if layout is not None:
        if evidence.attempt_id != layout.attempt_id:
            raise CodecError("Execution evidence belongs to a different attempt.")
        for reference in (
            *(
                (evidence.typed_result_artifact,)
                if evidence.typed_result_artifact
                else ()
            ),
            *evidence.artifacts,
        ):
            try:
                layout.resolve(reference)
            except ValueError as error:
                raise CodecError(str(error)) from error
    return evidence


def _encode_artifact(reference: ArtifactReference | None) -> JsonObject | None:
    if reference is None:
        return None
    return {
        "role": reference.role.value,
        "path": reference.run_relative_path,
        "media_type": reference.media_type,
    }


def _decode_artifact(value: object) -> ArtifactReference:
    data = _exact_object(
        value, fields={"role", "path", "media_type"}, source="artifact"
    )
    media_type = data["media_type"]
    if media_type is not None and not isinstance(media_type, str):
        raise CodecError("artifact.media_type must be a string or null.")
    try:
        return ArtifactReference(
            _enum(ArtifactRole, data["role"], "artifact.role"),
            _string(data["path"], "artifact.path"),
            media_type,
        )
    except ValueError as error:
        raise CodecError(str(error)) from error


def _metadata_to_json(value: Mapping[str, ProviderMetadataValue]) -> JsonObject:
    return {key: _metadata_value_to_json(item) for key, item in value.items()}


def _freeze_metadata(
    value: Mapping[str, ProviderMetadataValue],
) -> Mapping[str, ProviderMetadataValue]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise CodecError("provider_metadata must be an object with string keys.")
    return MappingProxyType(
        {key: _freeze_metadata_value(item) for key, item in value.items()}
    )


def _freeze_metadata_value(value: object) -> ProviderMetadataValue:
    if isinstance(value, Mapping):
        return _freeze_metadata(value)
    if isinstance(value, tuple):
        return tuple(_freeze_metadata_value(item) for item in value)
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float) and -float("inf") < value < float("inf"):
        return value
    raise CodecError("provider_metadata contains a non-JSON value.")


def _metadata_value_to_json(value: ProviderMetadataValue) -> JsonValue:
    if isinstance(value, Mapping):
        return {key: _metadata_value_to_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_metadata_value_to_json(item) for item in value]
    return value


def _decode_metadata(value: JsonObject) -> dict[str, ProviderMetadataValue]:
    return {key: _decode_metadata_value(item) for key, item in value.items()}


def _decode_metadata_value(value: JsonValue) -> ProviderMetadataValue:
    if isinstance(value, dict):
        return {key: _decode_metadata_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(_decode_metadata_value(item) for item in value)
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float) and -float("inf") < value < float("inf"):
        return value
    raise CodecError("provider_metadata contains a non-JSON value.")


def _exact_object(value: object, *, fields: set[str], source: str) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CodecError(f"{source} must be an object.")
    missing = fields - set(value)
    extra = set(value) - fields
    if missing:
        raise CodecError(f"{source} is missing fields: {', '.join(sorted(missing))}.")
    if extra:
        raise CodecError(
            f"{source} has unsupported fields: {', '.join(sorted(extra))}."
        )
    return value


def _require_exact(data: JsonObject, field: str, expected: object) -> None:
    if data[field] != expected or type(data[field]) is not type(expected):
        raise CodecError(f"Execution evidence has unsupported {field}.")


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CodecError(f"{field} must be a non-empty string.")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise CodecError(f"{field} must be a non-negative number.")
    return float(value)


def _enum(enum_type, value: object, field: str):
    text = _string(value, field)
    try:
        return enum_type(text)
    except ValueError as error:
        raise CodecError(f"{field} has an unsupported value: {text!r}.") from error


__all__ = [
    "EXECUTION_EVIDENCE_FORMAT",
    "EXECUTION_EVIDENCE_SCHEMA_VERSION",
    "ExecutionEvidence",
    "decode_execution_evidence",
    "encode_execution_evidence",
    "evidence_from_execution",
    "read_execution_evidence",
    "write_execution_evidence",
]
