"""Stable provider identity and declared execution capabilities."""

from ...application.agent_execution import AgentCapability, ProviderId

PROVIDER_ID = ProviderId("codex-cli")

CAPABILITIES = frozenset(
    {
        AgentCapability.READ_ONLY_EXECUTION,
        AgentCapability.WORKSPACE_WRITE_EXECUTION,
        AgentCapability.STRUCTURED_RESULT,
        AgentCapability.ISOLATED_INVOCATION,
        AgentCapability.DIAGNOSTIC_ARTIFACT_CAPTURE,
        AgentCapability.NETWORK_POLICY_CONTROL,
    }
)

__all__ = ["CAPABILITIES", "PROVIDER_ID"]
