from __future__ import annotations

from pathlib import Path

import pytest

from tests.provider_contract import (
    AdapterProbe,
    ContractOutcome,
    ProviderContractFixture,
    ProviderContractTests,
)
from tests.scripted_provider import (
    ALL_CAPABILITIES,
    CORE_CAPABILITIES,
    PROVIDER_ID,
    ScriptedOutcome,
    ScriptedProviderExecutor,
    ScriptedProviderObservation,
    ScriptedProviderRegistration,
    policy_for_probe,
)
from ticket_automation.application.agent_execution import AgentTaskKind


class TestScriptedProviderContract(ProviderContractTests):
    @pytest.fixture
    def provider_contract(self, tmp_path: Path) -> ProviderContractFixture:
        registration = ScriptedProviderRegistration()

        def make_probe(
            outcome: ContractOutcome,
            task_kind: AgentTaskKind,
        ) -> AdapterProbe:
            observation = ScriptedProviderObservation()
            return AdapterProbe(
                executor=ScriptedProviderExecutor(
                    policy_for_probe(task_kind, ScriptedOutcome(outcome.value)),
                    capabilities=ALL_CAPABILITIES,
                    observation=observation,
                ),
                observation=observation,  # type: ignore[arg-type]
            )

        def make_capability_limited_probe(
            task_kind: AgentTaskKind,
        ) -> AdapterProbe:
            observation = ScriptedProviderObservation()
            return AdapterProbe(
                executor=ScriptedProviderExecutor(
                    policy_for_probe(task_kind, ScriptedOutcome.SUCCESS),
                    capabilities=CORE_CAPABILITIES,
                    observation=observation,
                ),
                observation=observation,  # type: ignore[arg-type]
            )

        repository = tmp_path / "repository"
        repository.mkdir()
        configuration = tmp_path / "configuration"
        configuration.mkdir()
        return ProviderContractFixture(
            provider_id=PROVIDER_ID,
            declared_capabilities=ALL_CAPABILITIES,
            registration=registration,
            valid_settings={
                "label": "contract-suite",
                "script": {
                    task_kind.value: [ScriptedOutcome.SUCCESS.value]
                    for task_kind in AgentTaskKind
                },
                "write_workspace": False,
                "emit_native_diagnostic": True,
            },
            invalid_settings={"credential": "must-not-exist"},
            configuration_directory=configuration,
            repository_path=repository,
            artifact_root=tmp_path / "contract-artifacts",
            make_probe=make_probe,
            make_capability_limited_probe=make_capability_limited_probe,
            expected_transport_prompt=lambda prompt: prompt,
        )
