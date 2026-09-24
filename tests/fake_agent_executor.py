"""Deterministic test support for the provider-neutral execution port."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import TypeVar, cast

from ticket_automation.application.agent_execution import (
    AgentCapability,
    AgentExecution,
    AgentExecutionRequest,
    AgentExecutionStatus,
    AgentFailureCategory,
    AgentTaskKind,
    ArtifactReference,
    ArtifactRole,
    InvocationStart,
    ProviderId,
)
from ticket_automation.domain.task_results import TaskResult
from ticket_automation.task_result_codecs import (
    encode_implementation_result,
    encode_review_result,
)

_IN_MEMORY_PROVIDER = ProviderId("in-memory")
ResultT = TypeVar("ResultT", bound=TaskResult)


class InMemoryAgentExecutor:
    def __init__(
        self,
        results: Mapping[AgentTaskKind, TaskResult],
        *,
        capabilities: frozenset[AgentCapability],
        provider_id: ProviderId = _IN_MEMORY_PROVIDER,
    ) -> None:
        self._results = dict(results)
        self._capabilities = capabilities
        self._provider_id = provider_id
        self.requests: list[AgentExecutionRequest[TaskResult]] = []

    @property
    def capabilities(self) -> frozenset[AgentCapability]:
        return self._capabilities

    def execute(
        self,
        request: AgentExecutionRequest[ResultT],
        *,
        on_invocation_start: Callable[[], None] | None = None,
    ) -> AgentExecution[ResultT]:
        missing = request.missing_capabilities(self._capabilities)
        if missing:
            now = datetime.now(UTC)
            names = ", ".join(sorted(capability.value for capability in missing))
            artifacts = _artifacts(request)
            return AgentExecution(
                provider_id=self._provider_id,
                task_kind=request.task_kind,
                status=AgentExecutionStatus.FAILED,
                invocation_start=InvocationStart.NOT_STARTED,
                started_at=now,
                ended_at=now,
                duration_seconds=0,
                failure_category=(
                    AgentFailureCategory.CAPABILITY_OR_CONFIGURATION_FAILURE
                ),
                failure_message=f"Unsupported required capabilities: {names}.",
                artifacts=artifacts,
                provider_metadata={
                    "missing_capabilities": tuple(
                        sorted(capability.value for capability in missing)
                    )
                },
            )
        result = self._results[request.task_kind]
        if not request.result_contract.accepts(result):
            raise TypeError(
                "Scripted result does not satisfy the requested result contract."
            )
        if on_invocation_start is not None:
            on_invocation_start()
        self.requests.append(request)
        artifacts = _artifacts(request, result)
        now = datetime.now(UTC)
        return AgentExecution(
            provider_id=self._provider_id,
            task_kind=request.task_kind,
            status=AgentExecutionStatus.SUCCESS,
            invocation_start=InvocationStart.STARTED,
            started_at=now,
            ended_at=now,
            duration_seconds=0,
            result=cast(ResultT, result),
            artifacts=artifacts,
        )


def _artifacts(
    request: AgentExecutionRequest[TaskResult],
    result: TaskResult | None = None,
) -> tuple[ArtifactReference, ...]:
    directory = request.artifact_directory
    paths = {
        ArtifactRole.PROMPT: directory / "request.txt",
        ArtifactRole.PROVIDER_EVENTS: directory / "memory.log",
        ArtifactRole.PROVIDER_EXECUTION_DETAILS: directory / "memory-execution.json",
        ArtifactRole.TYPED_RESULT: directory / "memory-result.json",
    }
    directory.mkdir(parents=True, exist_ok=True)
    paths[ArtifactRole.PROMPT].write_text(request.prompt, encoding="utf-8")
    paths[ArtifactRole.PROVIDER_EVENTS].write_text("", encoding="utf-8")
    paths[ArtifactRole.PROVIDER_EXECUTION_DETAILS].write_text("{}\n", encoding="utf-8")
    roles = [
        ArtifactRole.PROMPT,
        ArtifactRole.PROVIDER_EVENTS,
        ArtifactRole.PROVIDER_EXECUTION_DETAILS,
    ]
    if result is not None:
        encoder = (
            encode_review_result
            if request.task_kind is AgentTaskKind.REVIEW
            else encode_implementation_result
        )
        paths[ArtifactRole.TYPED_RESULT].write_text(
            json.dumps(encoder(result)), encoding="utf-8"
        )
        roles.append(ArtifactRole.TYPED_RESULT)
    assert request.artifact_layout is not None
    return tuple(request.artifact_layout.reference(role, paths[role]) for role in roles)


__all__ = ["InMemoryAgentExecutor"]
