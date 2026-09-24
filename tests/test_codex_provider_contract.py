from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.provider_contract import (
    AdapterProbe,
    ContractOutcome,
    ProviderContractFixture,
    ProviderContractTests,
    TransportObservation,
)
from ticket_automation.application.agent_execution import AgentCapability, AgentTaskKind
from ticket_automation.providers.codex_cli import (
    CodexCliAgentExecutor,
    CodexCliSettings,
    CodexCommand,
    CodexProcessResult,
    CodexProcessTimedOut,
    CodexProcessTimeout,
)
from ticket_automation.providers.codex_cli import composition as codex_composition
from ticket_automation.providers.codex_cli.composition import (
    CodexCliProviderRegistration,
)
from ticket_automation.providers.codex_cli.identity import CAPABILITIES, PROVIDER_ID


def _result_payload(task_kind: AgentTaskKind) -> dict[str, object]:
    if task_kind is AgentTaskKind.REVIEW:
        return {
            "verdict": "PASS",
            "summary": "Contract review passed.",
            "findings": [],
        }
    return {
        "status": "COMPLETED",
        "summary": "Contract task completed.",
        "tests_run": [],
        "assumptions": [],
        "known_issues": [],
    }


@dataclass
class DeterministicCodexTransport:
    outcome: ContractOutcome
    task_kind: AgentTaskKind
    observation: TransportObservation

    def run(
        self,
        command: CodexCommand,
        *,
        stdin: str,
        timeout_seconds: float | None,
        on_process_start=None,
    ) -> CodexProcessResult:
        self.observation.invocation_attempts += 1
        self.observation.record_prompt(stdin)
        if self.outcome is ContractOutcome.INVOCATION_START_FAILURE:
            raise OSError("deterministic process start failure")

        self.observation.invocation_starts += 1
        if on_process_start is not None:
            on_process_start()
        if self.outcome is ContractOutcome.TIMEOUT:
            raise CodexProcessTimedOut(
                CodexProcessTimeout(
                    stdout='{"type":"contract.timeout"}\n',
                    stderr="deterministic timeout\n",
                    timeout_seconds=timeout_seconds or 2,
                )
            )
        if self.outcome is ContractOutcome.PROVIDER_REJECTION:
            return CodexProcessResult(2, "", "authentication rejected")
        if self.outcome is ContractOutcome.NON_SUCCESSFUL_EXECUTION:
            return CodexProcessResult(2, "", "deterministic execution failure")
        if self.outcome is ContractOutcome.MISSING_RESULT:
            return CodexProcessResult(0, "", "")

        result_path = Path(
            command.argv[command.argv.index("--output-last-message") + 1]
        )
        if self.outcome is ContractOutcome.INVALID_RESULT:
            result_path.write_text("{invalid", encoding="utf-8")
        else:
            result_path.write_text(
                json.dumps(_result_payload(self.task_kind)),
                encoding="utf-8",
            )
        return CodexProcessResult(0, "", "")


@dataclass
class ReadOnlyCodexTransport:
    observation: TransportObservation

    def run(
        self,
        command: CodexCommand,
        *,
        stdin: str,
        timeout_seconds: float | None,
        on_process_start=None,
    ) -> CodexProcessResult:
        del command, stdin, timeout_seconds, on_process_start
        self.observation.invocation_attempts += 1
        raise AssertionError("capability rejection must precede transport invocation")


class CapabilityLimitedCodexExecutor(CodexCliAgentExecutor):
    @property
    def capabilities(self) -> frozenset[AgentCapability]:
        return CAPABILITIES - {AgentCapability.WORKSPACE_WRITE_EXECUTION}


class TestCodexProviderContract(ProviderContractTests):
    @pytest.fixture
    def provider_contract(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> ProviderContractFixture:
        executable = Path(sys.executable).resolve()

        def resolve_executable(
            configured: str,
            *,
            config_dir: Path | None = None,
        ) -> Path:
            del configured, config_dir
            return executable

        def supports_ephemeral(candidate: Path, *, cwd: Path) -> bool:
            del candidate, cwd
            return True

        def cli_version(candidate: Path, *, cwd: Path) -> str:
            del candidate, cwd
            return "codex-contract-fixture 1.0"

        monkeypatch.setattr(codex_composition, "resolve_executable", resolve_executable)
        monkeypatch.setattr(codex_composition, "supports_ephemeral", supports_ephemeral)
        monkeypatch.setattr(codex_composition, "cli_version", cli_version)

        settings = CodexCliSettings(str(executable), "contract-model", "high")

        def make_probe(
            outcome: ContractOutcome,
            task_kind: AgentTaskKind,
        ) -> AdapterProbe:
            observation = TransportObservation()
            transport = DeterministicCodexTransport(
                outcome=outcome,
                task_kind=task_kind,
                observation=observation,
            )
            configured = settings
            if outcome is ContractOutcome.PROVIDER_UNAVAILABLE:
                configured = CodexCliSettings(
                    str(tmp_path / "missing-provider-executable"),
                    settings.model,
                    settings.reasoning_effort,
                )
            return AdapterProbe(
                executor=CodexCliAgentExecutor(configured, runner=transport),
                observation=observation,
            )

        def make_capability_limited_probe(
            task_kind: AgentTaskKind,
        ) -> AdapterProbe:
            assert task_kind is AgentTaskKind.IMPLEMENTATION
            observation = TransportObservation()
            return AdapterProbe(
                executor=CapabilityLimitedCodexExecutor(
                    settings,
                    runner=ReadOnlyCodexTransport(observation),
                ),
                observation=observation,
            )

        repository = tmp_path / "repository"
        repository.mkdir()
        configuration = tmp_path / "configuration"
        configuration.mkdir()
        return ProviderContractFixture(
            provider_id=PROVIDER_ID,
            declared_capabilities=CAPABILITIES,
            registration=CodexCliProviderRegistration(),
            valid_settings={
                "executable": str(executable),
                "model": settings.model,
                "reasoning_effort": settings.reasoning_effort,
            },
            invalid_settings={"unsupported-setting": True},
            configuration_directory=configuration,
            repository_path=repository,
            artifact_root=tmp_path / "contract-artifacts",
            make_probe=make_probe,
            make_capability_limited_probe=make_capability_limited_probe,
            expected_transport_prompt=lambda prompt: prompt,
        )
