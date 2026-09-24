"""Offline provider used only to prove the provider seam in tests."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
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
    RepositoryAccess,
)
from ticket_automation.application.ports.preflight import (
    PreflightCheck,
    PreflightStatus,
)
from ticket_automation.domain.task_results import (
    FindingDisposition,
    FindingScopeRelation,
    ImplementationResult,
    ImplementationStatus,
    ImplementationTestResult,
    ReviewFinding,
    ReviewResult,
    ReviewVerdict,
    TaskResult,
)
from ticket_automation.task_result_codecs import (
    encode_implementation_result,
    encode_review_result,
)

PROVIDER_ID = ProviderId("scripted-test-provider")
POLICY_VERSION = "scripted-test-v1"
CORE_CAPABILITIES = frozenset(
    {
        AgentCapability.STRUCTURED_RESULT,
        AgentCapability.ISOLATED_INVOCATION,
        AgentCapability.DIAGNOSTIC_ARTIFACT_CAPTURE,
        AgentCapability.NETWORK_POLICY_CONTROL,
    }
)
READ_ONLY_CAPABILITIES = CORE_CAPABILITIES | {AgentCapability.READ_ONLY_EXECUTION}
ALL_CAPABILITIES = READ_ONLY_CAPABILITIES | {AgentCapability.WORKSPACE_WRITE_EXECUTION}

ResultT = TypeVar("ResultT", bound=TaskResult)


class ScriptedOutcome(StrEnum):
    SUCCESS = "success"
    REVIEW_CORRECTIONS = "review-corrections"
    BLOCKED = "blocked"
    PROVIDER_UNAVAILABLE = "provider-unavailable"
    INVOCATION_START_FAILURE = "invocation-start-failure"
    TIMEOUT = "timeout"
    PROVIDER_REJECTION = "provider-rejection"
    NON_SUCCESSFUL_EXECUTION = "non-successful-execution"
    MISSING_RESULT = "missing-result"
    INVALID_RESULT = "invalid-result"


@dataclass(frozen=True)
class ScriptedProviderSettings:
    label: str
    script: tuple[tuple[AgentTaskKind, tuple[ScriptedOutcome, ...]], ...]
    write_workspace: bool
    emit_native_diagnostic: bool


@dataclass(frozen=True)
class ScriptedProviderPolicy:
    label: str
    script: tuple[tuple[AgentTaskKind, tuple[ScriptedOutcome, ...]], ...]
    write_workspace: bool
    emit_native_diagnostic: bool


@dataclass
class ScriptedProviderObservation:
    invocation_attempts: int = 0
    invocation_starts: int = 0
    prompts: tuple[str, ...] = ()
    requests: list[AgentExecutionRequest[TaskResult]] = field(default_factory=list)

    def record_prompt(self, prompt: str) -> None:
        self.prompts = (*self.prompts, prompt)


class ScriptedProviderRegistration:
    """Explicit test registration; production composition never imports it."""

    provider_id = PROVIDER_ID
    policy_version = POLICY_VERSION

    def __init__(
        self,
        *,
        capabilities: frozenset[AgentCapability] = ALL_CAPABILITIES,
        observation: ScriptedProviderObservation | None = None,
    ) -> None:
        self.capabilities = capabilities
        self.observation = observation or ScriptedProviderObservation()
        self.created_executors: list[ScriptedProviderExecutor] = []

    def resolve_settings(
        self,
        raw_settings: Mapping[str, object],
        *,
        configuration_directory: Path,
    ) -> ScriptedProviderSettings:
        del configuration_directory
        allowed = {
            "label",
            "script",
            "write_workspace",
            "emit_native_diagnostic",
        }
        unknown = sorted(set(raw_settings) - allowed)
        if unknown:
            raise ValueError(
                "Unknown scripted provider setting(s): " + ", ".join(unknown)
            )
        label = raw_settings.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("scripted provider label must be a non-empty string")
        write_workspace = raw_settings.get("write_workspace", False)
        emit_native_diagnostic = raw_settings.get("emit_native_diagnostic", False)
        if not isinstance(write_workspace, bool):
            raise TypeError("write_workspace must be a boolean")
        if not isinstance(emit_native_diagnostic, bool):
            raise TypeError("emit_native_diagnostic must be a boolean")
        return ScriptedProviderSettings(
            label=label.strip(),
            script=_parse_script(raw_settings.get("script", {})),
            write_workspace=write_workspace,
            emit_native_diagnostic=emit_native_diagnostic,
        )

    def run_preflight(
        self,
        settings: object,
        *,
        repository_path: Path,
    ) -> tuple[PreflightCheck, ...]:
        _require_settings(settings)
        repository_status = (
            PreflightStatus.PASS if repository_path.is_dir() else PreflightStatus.FAIL
        )
        return (
            PreflightCheck(
                name="repository",
                status=repository_status,
                message=str(repository_path),
            ),
            PreflightCheck(
                name="offline scripted execution",
                status=PreflightStatus.PASS,
                message="No network, credentials, or external process required.",
            ),
        )

    def resolve_run_policy(self, settings: object) -> ScriptedProviderPolicy:
        configured = _require_settings(settings)
        return ScriptedProviderPolicy(
            label=configured.label,
            script=configured.script,
            write_workspace=configured.write_workspace,
            emit_native_diagnostic=configured.emit_native_diagnostic,
        )

    def encode_run_policy(self, policy: object) -> object:
        resolved = _require_policy(policy)
        return {
            "label": resolved.label,
            "script": {
                task_kind.value: [outcome.value for outcome in outcomes]
                for task_kind, outcomes in resolved.script
            },
            "write_workspace": resolved.write_workspace,
            "emit_native_diagnostic": resolved.emit_native_diagnostic,
        }

    def decode_run_policy(self, payload: object) -> ScriptedProviderPolicy:
        if not isinstance(payload, dict):
            raise TypeError("scripted provider policy must be an object")
        settings = self.resolve_settings(payload, configuration_directory=Path.cwd())
        return self.resolve_run_policy(settings)

    def runtime_compatibility_problem(self, policy: object) -> str | None:
        try:
            _require_policy(policy)
        except TypeError as error:
            return str(error)
        return None

    def create_executor(self, policy: object) -> ScriptedProviderExecutor:
        executor = ScriptedProviderExecutor(
            _require_policy(policy),
            capabilities=self.capabilities,
            observation=self.observation,
        )
        self.created_executors.append(executor)
        return executor


class ScriptedProviderExecutor:
    def __init__(
        self,
        policy: ScriptedProviderPolicy,
        *,
        capabilities: frozenset[AgentCapability],
        observation: ScriptedProviderObservation | None = None,
    ) -> None:
        self._policy = policy
        self._capabilities = capabilities
        self.observation = observation or ScriptedProviderObservation()
        self._script = {task: list(outcomes) for task, outcomes in policy.script}

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
        expected_access = (
            RepositoryAccess.READ_ONLY
            if request.task_kind is AgentTaskKind.REVIEW
            else RepositoryAccess.WORKSPACE_WRITE
        )
        if request.repository_access is not expected_access:
            return self._failure(
                request,
                AgentFailureCategory.CAPABILITY_OR_CONFIGURATION_FAILURE,
                InvocationStart.NOT_STARTED,
                "Task kind and repository access do not match.",
            )
        if missing:
            names = ", ".join(sorted(item.value for item in missing))
            return self._failure(
                request,
                AgentFailureCategory.CAPABILITY_OR_CONFIGURATION_FAILURE,
                InvocationStart.NOT_STARTED,
                f"Missing required capabilities: {names}.",
            )

        outcome = self._next_outcome(request.task_kind)
        if outcome is ScriptedOutcome.PROVIDER_UNAVAILABLE:
            return self._failure(
                request,
                AgentFailureCategory.PROVIDER_UNAVAILABLE,
                InvocationStart.NOT_STARTED,
                "Scripted provider is unavailable.",
            )

        self.observation.invocation_attempts += 1
        self.observation.record_prompt(request.prompt)
        if outcome is ScriptedOutcome.INVOCATION_START_FAILURE:
            return self._failure(
                request,
                AgentFailureCategory.INVOCATION_START_FAILURE,
                InvocationStart.NOT_STARTED,
                "Scripted invocation could not start.",
            )

        self.observation.invocation_starts += 1
        self.observation.requests.append(
            cast(AgentExecutionRequest[TaskResult], request)
        )
        if on_invocation_start is not None:
            on_invocation_start()

        failure = {
            ScriptedOutcome.TIMEOUT: AgentFailureCategory.TIMEOUT,
            ScriptedOutcome.PROVIDER_REJECTION: (
                AgentFailureCategory.PROVIDER_REJECTION_OR_SERVICE_FAILURE
            ),
            ScriptedOutcome.NON_SUCCESSFUL_EXECUTION: (
                AgentFailureCategory.NON_SUCCESSFUL_EXECUTION
            ),
            ScriptedOutcome.MISSING_RESULT: AgentFailureCategory.MISSING_RESULT,
            ScriptedOutcome.INVALID_RESULT: AgentFailureCategory.INVALID_RESULT,
        }.get(outcome)
        if failure is not None:
            return self._failure(
                request,
                failure,
                InvocationStart.STARTED,
                f"Scripted outcome: {outcome.value}.",
                artifacts=self._artifacts(request, outcome=outcome),
            )

        result = _result_for(request.task_kind, outcome)
        if not request.result_contract.accepts(result):
            raise TypeError("Scripted result does not satisfy the requested contract.")
        if (
            self._policy.write_workspace
            and request.repository_access is RepositoryAccess.WORKSPACE_WRITE
        ):
            path = request.repository_path / f"scripted-{request.task_kind.value}.txt"
            path.write_text(
                f"{self._policy.label}: {request.task_kind.value}\n",
                encoding="utf-8",
            )
        artifacts = self._artifacts(request, outcome=outcome, result=result)
        now = datetime.now(UTC)
        return AgentExecution(
            provider_id=PROVIDER_ID,
            task_kind=request.task_kind,
            status=AgentExecutionStatus.SUCCESS,
            invocation_start=InvocationStart.STARTED,
            started_at=now,
            ended_at=now,
            duration_seconds=0,
            result=cast(ResultT, result),
            artifacts=artifacts,
            provider_metadata={
                "script_label": self._policy.label,
                "scripted_outcome": outcome.value,
            },
        )

    def _next_outcome(self, task_kind: AgentTaskKind) -> ScriptedOutcome:
        outcomes = self._script.setdefault(task_kind, [])
        return outcomes.pop(0) if outcomes else ScriptedOutcome.SUCCESS

    def _artifacts(
        self,
        request: AgentExecutionRequest[TaskResult],
        *,
        outcome: ScriptedOutcome,
        result: TaskResult | None = None,
    ) -> tuple[ArtifactReference, ...]:
        layout = request.artifact_layout
        assert layout is not None
        layout.attempt_root.mkdir(parents=True, exist_ok=True)
        prompt = layout.path("scripted-provider-prompt.txt")
        prompt.write_text(request.prompt, encoding="utf-8")
        references = [layout.reference(ArtifactRole.PROMPT, prompt, "text/plain")]
        if result is not None:
            typed_result = layout.named_path(ArtifactRole.TYPED_RESULT)
            encoder = (
                encode_review_result
                if isinstance(result, ReviewResult)
                else encode_implementation_result
            )
            typed_result.write_text(
                json.dumps(encoder(result), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            references.append(
                layout.reference(
                    ArtifactRole.TYPED_RESULT,
                    typed_result,
                    "application/json",
                )
            )
        if self._policy.emit_native_diagnostic:
            diagnostic = layout.path("scripted-provider-diagnostic.json")
            diagnostic.write_text(
                json.dumps(
                    {
                        "label": self._policy.label,
                        "outcome": outcome.value,
                        "task": request.task_kind.value,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            references.append(
                layout.reference(
                    ArtifactRole.PROVIDER_EXECUTION_DETAILS,
                    diagnostic,
                    "application/json",
                )
            )
        return tuple(references)

    @staticmethod
    def _failure(
        request: AgentExecutionRequest[ResultT],
        category: AgentFailureCategory,
        invocation_start: InvocationStart,
        message: str,
        *,
        artifacts: tuple[ArtifactReference, ...] = (),
    ) -> AgentExecution[ResultT]:
        now = datetime.now(UTC)
        return AgentExecution(
            provider_id=PROVIDER_ID,
            task_kind=request.task_kind,
            status=AgentExecutionStatus.FAILED,
            invocation_start=invocation_start,
            started_at=now,
            ended_at=now,
            duration_seconds=0,
            failure_category=category,
            failure_message=message,
            artifacts=artifacts,
        )


def policy_for_probe(
    task_kind: AgentTaskKind,
    outcome: ScriptedOutcome,
    *,
    emit_native_diagnostic: bool = True,
) -> ScriptedProviderPolicy:
    return ScriptedProviderPolicy(
        label="contract-probe",
        script=((task_kind, (outcome,)),),
        write_workspace=False,
        emit_native_diagnostic=emit_native_diagnostic,
    )


def _parse_script(
    value: object,
) -> tuple[tuple[AgentTaskKind, tuple[ScriptedOutcome, ...]], ...]:
    if not isinstance(value, Mapping):
        raise TypeError("script must be an object keyed by task kind")
    unknown = sorted(set(value) - {kind.value for kind in AgentTaskKind})
    if unknown:
        raise ValueError("Unknown scripted task kind(s): " + ", ".join(unknown))
    parsed: list[tuple[AgentTaskKind, tuple[ScriptedOutcome, ...]]] = []
    for task_kind in AgentTaskKind:
        raw_outcomes = value.get(task_kind.value, (ScriptedOutcome.SUCCESS.value,))
        if isinstance(raw_outcomes, str):
            raw_outcomes = (raw_outcomes,)
        if not isinstance(raw_outcomes, tuple | list) or not raw_outcomes:
            raise ValueError(f"script.{task_kind.value} must be a non-empty sequence")
        try:
            outcomes = tuple(ScriptedOutcome(item) for item in raw_outcomes)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"script.{task_kind.value} contains an unsupported outcome"
            ) from error
        parsed.append((task_kind, outcomes))
    return tuple(parsed)


def _require_settings(settings: object) -> ScriptedProviderSettings:
    if not isinstance(settings, ScriptedProviderSettings):
        raise TypeError("scripted provider settings have the wrong type")
    return settings


def _require_policy(policy: object) -> ScriptedProviderPolicy:
    if not isinstance(policy, ScriptedProviderPolicy):
        raise TypeError("scripted provider policy has the wrong type")
    return policy


def _result_for(
    task_kind: AgentTaskKind,
    outcome: ScriptedOutcome,
) -> TaskResult:
    if task_kind is AgentTaskKind.REVIEW:
        if outcome is ScriptedOutcome.REVIEW_CORRECTIONS:
            return ReviewResult(
                verdict=ReviewVerdict.CORRECTIONS_REQUIRED,
                summary="The scripted review requires one correction.",
                findings=(
                    ReviewFinding(
                        id="SCRIPT-1",
                        disposition=FindingDisposition.REQUIRED,
                        scope_relation=FindingScopeRelation.IMPLEMENTATION,
                        title="Apply the scripted correction",
                        description="The deterministic test script requested a correction.",
                        evidence="scripted-provider outcome",
                        required_change="Run the correction task.",
                        acceptance_criteria=("The next review passes.",),
                    ),
                ),
            )
        return ReviewResult(
            verdict=ReviewVerdict.PASS,
            summary="The scripted review passed.",
            findings=(),
        )
    blocked = outcome is ScriptedOutcome.BLOCKED
    return ImplementationResult(
        status=(
            ImplementationStatus.BLOCKED if blocked else ImplementationStatus.COMPLETED
        ),
        summary=f"The scripted {task_kind.value} task completed.",
        tests_run=(ImplementationTestResult("scripted-check", "PASS"),),
        assumptions=(),
        known_issues=("Scripted block.",) if blocked else (),
    )


__all__ = [
    "ALL_CAPABILITIES",
    "CORE_CAPABILITIES",
    "POLICY_VERSION",
    "PROVIDER_ID",
    "READ_ONLY_CAPABILITIES",
    "ScriptedOutcome",
    "ScriptedProviderExecutor",
    "ScriptedProviderObservation",
    "ScriptedProviderPolicy",
    "ScriptedProviderRegistration",
    "ScriptedProviderSettings",
    "policy_for_probe",
]
