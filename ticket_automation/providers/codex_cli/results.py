"""Strict adaptation of Codex structured output into domain results."""

from __future__ import annotations

from typing import TypeVar

from ...application.agent_execution import AgentExecutionRequest
from ...domain.task_results import ImplementationResult, ReviewResult, TaskResult
from ...task_result_codecs import decode_implementation_result, decode_review_result

ResultT = TypeVar("ResultT", bound=TaskResult)


def decode_result(
    value: object,
    request: AgentExecutionRequest[ResultT],
) -> ResultT:
    expected = request.result_contract.result_type
    if expected is ReviewResult:
        decoder = decode_review_result
    elif expected is ImplementationResult:
        decoder = decode_implementation_result
    else:
        raise TypeError("Requested result contract is not supported by Codex CLI.")
    result = decoder(value)
    if not request.result_contract.accepts(result):
        raise TypeError(
            "Decoded result does not satisfy the requested result contract."
        )
    return result


__all__ = ["decode_result"]
