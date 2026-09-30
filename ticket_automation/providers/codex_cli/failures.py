"""One-time mapping from Codex-native failures to neutral categories."""

from enum import StrEnum

from ...application.agent_execution import AgentFailureCategory, InvocationStart


class CodexFailureReason(StrEnum):
    EXECUTABLE_UNAVAILABLE = "EXECUTABLE_UNAVAILABLE"
    PROCESS_START_FAILED = "PROCESS_START_FAILED"
    TIMEOUT = "TIMEOUT"
    NON_ZERO_EXIT = "NON_ZERO_EXIT"
    TRANSPORT_FAILURE = "TRANSPORT_FAILURE"
    FINALIZATION_FAILED = "FINALIZATION_FAILED"
    MISSING_STRUCTURED_RESULT = "MISSING_STRUCTURED_RESULT"
    INVALID_STRUCTURED_RESULT = "INVALID_STRUCTURED_RESULT"
    AUTHENTICATION_OR_SERVICE = "AUTHENTICATION_OR_SERVICE"
    PROJECT_CONFIGURATION_REJECTED = "PROJECT_CONFIGURATION_REJECTED"
    CAPABILITY_REJECTED = "CAPABILITY_REJECTED"
    POLICY_REJECTED = "POLICY_REJECTED"


_FAILURE_CATEGORIES = {
    CodexFailureReason.EXECUTABLE_UNAVAILABLE: AgentFailureCategory.PROVIDER_UNAVAILABLE,
    CodexFailureReason.PROCESS_START_FAILED: AgentFailureCategory.INVOCATION_START_FAILURE,
    CodexFailureReason.TIMEOUT: AgentFailureCategory.TIMEOUT,
    CodexFailureReason.NON_ZERO_EXIT: AgentFailureCategory.NON_SUCCESSFUL_EXECUTION,
    CodexFailureReason.TRANSPORT_FAILURE: AgentFailureCategory.NON_SUCCESSFUL_EXECUTION,
    CodexFailureReason.FINALIZATION_FAILED: AgentFailureCategory.NON_SUCCESSFUL_EXECUTION,
    CodexFailureReason.MISSING_STRUCTURED_RESULT: AgentFailureCategory.MISSING_RESULT,
    CodexFailureReason.INVALID_STRUCTURED_RESULT: AgentFailureCategory.INVALID_RESULT,
    CodexFailureReason.AUTHENTICATION_OR_SERVICE: (
        AgentFailureCategory.PROVIDER_REJECTION_OR_SERVICE_FAILURE
    ),
    CodexFailureReason.PROJECT_CONFIGURATION_REJECTED: (
        AgentFailureCategory.CAPABILITY_OR_CONFIGURATION_FAILURE
    ),
    CodexFailureReason.CAPABILITY_REJECTED: (
        AgentFailureCategory.CAPABILITY_OR_CONFIGURATION_FAILURE
    ),
    CodexFailureReason.POLICY_REJECTED: (
        AgentFailureCategory.CAPABILITY_OR_CONFIGURATION_FAILURE
    ),
}


def map_failure(reason: CodexFailureReason) -> AgentFailureCategory:
    return _FAILURE_CATEGORIES[reason]


def invocation_start_for(reason: CodexFailureReason) -> InvocationStart:
    if reason in {
        CodexFailureReason.EXECUTABLE_UNAVAILABLE,
        CodexFailureReason.PROCESS_START_FAILED,
        CodexFailureReason.PROJECT_CONFIGURATION_REJECTED,
        CodexFailureReason.CAPABILITY_REJECTED,
        CodexFailureReason.POLICY_REJECTED,
    }:
        return InvocationStart.NOT_STARTED
    return InvocationStart.STARTED


def looks_like_authentication_or_service_failure(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(
        marker in lowered
        for marker in (
            "auth",
            "api key",
            "unauthorized",
            "forbidden",
            "rate limit",
            "service unavailable",
            "temporarily unavailable",
            "network",
        )
    )


__all__ = [
    "CodexFailureReason",
    "invocation_start_for",
    "looks_like_authentication_or_service_failure",
    "map_failure",
]
