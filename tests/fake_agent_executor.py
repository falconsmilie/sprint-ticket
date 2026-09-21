"""Deterministic test support for the provider-neutral execution port."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TypeVar, cast

from ticket_automation.application.agent_execution import (
    AgentCapability,
    AgentExecution,
    AgentExecutionRequest,
    AgentExecutionStatus,
    AgentFailureCategory,
    AgentTaskKind,
    InvocationStart,
    ProviderId,
)
from ticket_automation.domain.task_results import TaskResult

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

    def execute(
        self, request: AgentExecutionRequest[ResultT]
    ) -> AgentExecution[ResultT]:
        missing = request.missing_capabilities(self._capabilities)
        if missing:
            now = datetime.now(UTC)
            names = ", ".join(sorted(capability.value for capability in missing))
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
        self.requests.append(request)
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
        )


__all__ = ["InMemoryAgentExecutor"]
