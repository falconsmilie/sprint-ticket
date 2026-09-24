from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.helpers import (
    create_git_repo,
    make_config,
    make_final_patch_capture,
    make_report_publisher,
    run_test_stage,
)
from tests.lifecycle_characterization_fixtures import (
    ScriptedVerificationRunner,
    TickingClock,
    configure_fake_codex_actions,
)
from tests.scripted_provider import (
    PROVIDER_ID as SCRIPTED_PROVIDER_ID,
)
from tests.scripted_provider import (
    READ_ONLY_CAPABILITIES,
    ScriptedProviderRegistration,
)
from ticket_automation.application.agent_execution import (
    AgentTaskKind,
    ProviderId,
)
from ticket_automation.composition.providers import (
    RegisteredProviderExecutorFactory,
    prepare_agent_providers,
)
from ticket_automation.composition.root import production_provider_registry
from ticket_automation.config import AgentSettings, AppConfig, ConfigError
from ticket_automation.models import AttemptPhase, WorkflowState
from ticket_automation.providers.codex_cli import CodexCliAgentExecutor, CodexSettings
from ticket_automation.providers.codex_cli import composition as codex_composition
from ticket_automation.providers.codex_cli.composition import (
    CodexCliProviderRegistration,
    CodexCliRunPolicy,
)
from ticket_automation.providers.codex_cli.identity import (
    PROVIDER_ID as CODEX_PROVIDER_ID,
)
from ticket_automation.runs import create_run_snapshot, load_run_record, save_run_record
from ticket_automation.verification import run_baseline_verification_stage
from ticket_automation.workflow import resume_ticket_lifecycle, run_ticket_lifecycle


class InMemoryCodexRegistration(CodexCliProviderRegistration):
    def __init__(self, runner: object) -> None:
        self._runner = runner

    def create_executor(self, policy: object) -> CodexCliAgentExecutor:
        if not isinstance(policy, CodexCliRunPolicy):
            raise TypeError("Codex policy has the wrong type")
        return CodexCliAgentExecutor(
            CodexSettings(
                executable=str(policy.executable),
                model=policy.model,
                reasoning_effort=policy.reasoning_effort,
            ),
            configuration_directory=policy.executable.parent,
            runner=self._runner,  # type: ignore[arg-type]
        )


def _patch_codex_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        codex_composition,
        "resolve_executable",
        lambda configured, *, config_dir=None: Path(configured).resolve(),
    )
    monkeypatch.setattr(
        codex_composition,
        "supports_ephemeral",
        lambda candidate, *, cwd: True,
    )
    monkeypatch.setattr(
        codex_composition,
        "cli_version",
        lambda candidate, *, cwd: "in-memory-codex 1.0",
    )


def _config_with_assignments(
    base: AppConfig,
    *,
    implementation: ProviderId,
    review: ProviderId,
    correction: ProviderId,
    scripted_settings: dict[str, object],
) -> AppConfig:
    return replace(
        base,
        agents=AgentSettings(
            assignments={
                AgentTaskKind.IMPLEMENTATION: implementation,
                AgentTaskKind.REVIEW: review,
                AgentTaskKind.CORRECTION: correction,
            },
            providers={
                CODEX_PROVIDER_ID: dict(base.agents.providers[CODEX_PROVIDER_ID]),
                SCRIPTED_PROVIDER_ID: scripted_settings,
            },
        ),
    )


def _prepare_run_for_resume(
    config: AppConfig,
    ticket: Path,
    runs_dir: Path,
    *,
    prepared,
    resolved_policy,
    clock: TickingClock,
):
    snapshot = create_run_snapshot(
        config,
        ticket,
        runs_dir=runs_dir,
        provider_preflight=prepared.run_preflight,
        resolved_policy=resolved_policy,
        clock=clock,
    )
    baseline = run_test_stage(
        run_baseline_verification_stage,
        AttemptPhase.PREPARING,
        config,
        snapshot.run_dir,
        process_runner=ScriptedVerificationRunner([0]),
        clock=clock,
    )
    record = baseline.run_record.transition_to(
        WorkflowState.PREPARED,
        updated_timestamp=clock().isoformat(),
    )
    save_run_record(record, snapshot.run_dir / "run.json")
    return replace(snapshot, run_record=record)


def test_mixed_codex_write_and_scripted_review_assignments_survive_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_codex_validation(monkeypatch)
    repository = create_git_repo(tmp_path / "repository")
    ticket = tmp_path / "TA-AGENT-004.md"
    ticket.write_text("# Prove mixed provider resume\n", encoding="utf-8")
    config = _config_with_assignments(
        make_config(repository),
        implementation=CODEX_PROVIDER_ID,
        review=SCRIPTED_PROVIDER_ID,
        correction=CODEX_PROVIDER_ID,
        scripted_settings={
            "label": "resume-review",
            "emit_native_diagnostic": False,
        },
    )
    scripted = ScriptedProviderRegistration()
    codex_runner = configure_fake_codex_actions(monkeypatch, tmp_path, "modify")
    codex = InMemoryCodexRegistration(codex_runner)
    registry = {
        CODEX_PROVIDER_ID: codex,
        SCRIPTED_PROVIDER_ID: scripted,
    }
    prepared = prepare_agent_providers(
        config.agents,
        registry=registry,
        configuration_directory=config.configuration_directory,
    )
    policy = prepared.resolve_run_policy(
        config,
        target_repository_path=repository,
    )
    snapshot = _prepare_run_for_resume(
        config,
        ticket,
        tmp_path / "runs",
        prepared=prepared,
        resolved_policy=policy,
        clock=TickingClock(),
    )

    persisted = load_run_record(snapshot.run_dir / "run.json").resolved_policy
    assert persisted.assignments == {
        AgentTaskKind.IMPLEMENTATION: CODEX_PROVIDER_ID,
        AgentTaskKind.REVIEW: SCRIPTED_PROVIDER_ID,
        AgentTaskKind.CORRECTION: CODEX_PROVIDER_ID,
    }
    assignment_view = persisted.assignments
    assignment_view[AgentTaskKind.REVIEW] = CODEX_PROVIDER_ID
    assert persisted.assignments[AgentTaskKind.REVIEW] == SCRIPTED_PROVIDER_ID

    result = resume_ticket_lifecycle(
        snapshot.run_record.run_id,
        runs_dir=tmp_path / "runs",
        agent_executor_factory=RegisteredProviderExecutorFactory(registry),
        final_patch_capture=make_final_patch_capture(),
        report_publisher=make_report_publisher(),
        verification_runner=ScriptedVerificationRunner([0]),
        clock=TickingClock(),
    )

    assert result.run_record.state is WorkflowState.READY_FOR_HUMAN
    assert [request.task_kind for request in scripted.observation.requests] == [
        AgentTaskKind.REVIEW
    ]
    report = (result.run_dir / "report.md").read_text(encoding="utf-8")
    assert "Provider codex-cli: success" in report
    assert f"Provider {SCRIPTED_PROVIDER_ID}: success" in report
    assert not tuple(result.run_dir.rglob("scripted-provider-diagnostic.json"))


@pytest.mark.parametrize("emit_native_diagnostic", [False, True])
def test_scripted_provider_can_own_both_writable_tasks_and_generic_reporting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    emit_native_diagnostic: bool,
) -> None:
    _patch_codex_validation(monkeypatch)
    repository = create_git_repo(tmp_path / "repository")
    ticket = tmp_path / "TA-AGENT-004.md"
    ticket.write_text("# Prove writable provider swap\n", encoding="utf-8")
    config = _config_with_assignments(
        make_config(repository, max_correction_rounds=1),
        implementation=SCRIPTED_PROVIDER_ID,
        review=CODEX_PROVIDER_ID,
        correction=SCRIPTED_PROVIDER_ID,
        scripted_settings={
            "label": "writable-proof",
            "write_workspace": True,
            "emit_native_diagnostic": emit_native_diagnostic,
        },
    )
    scripted = ScriptedProviderRegistration()
    codex_runner = configure_fake_codex_actions(
        monkeypatch,
        tmp_path,
        "review-corrections",
        "review-pass",
    )
    registry = {
        CODEX_PROVIDER_ID: InMemoryCodexRegistration(codex_runner),
        SCRIPTED_PROVIDER_ID: scripted,
    }
    prepared = prepare_agent_providers(
        config.agents,
        registry=registry,
        configuration_directory=config.configuration_directory,
    )
    policy = prepared.resolve_run_policy(
        config,
        target_repository_path=repository,
    )

    result = run_ticket_lifecycle(
        config,
        ticket,
        runs_dir=tmp_path / "runs",
        provider_preflight=prepared.run_preflight,
        resolved_policy=policy,
        agent_executor_factory=RegisteredProviderExecutorFactory(registry),
        final_patch_capture=make_final_patch_capture(),
        report_publisher=make_report_publisher(),
        verification_runner=ScriptedVerificationRunner([0, 0, 0]),
        clock=TickingClock(),
    )

    assert result.run_record.state is WorkflowState.READY_FOR_HUMAN
    assert [request.task_kind for request in scripted.observation.requests] == [
        AgentTaskKind.IMPLEMENTATION,
        AgentTaskKind.CORRECTION,
    ]
    diagnostics = tuple(result.run_dir.rglob("scripted-provider-diagnostic.json"))
    assert bool(diagnostics) is emit_native_diagnostic
    report = (result.run_dir / "report.md").read_text(encoding="utf-8")
    assert f"Provider {SCRIPTED_PROVIDER_ID}: success" in report
    assert "scripted-provider-diagnostic" not in report


def test_capability_mismatch_is_rejected_before_run_creation_or_execution(
    tmp_path: Path,
) -> None:
    repository = create_git_repo(tmp_path / "repository")
    config = _config_with_assignments(
        make_config(repository),
        implementation=SCRIPTED_PROVIDER_ID,
        review=SCRIPTED_PROVIDER_ID,
        correction=SCRIPTED_PROVIDER_ID,
        scripted_settings={"label": "read-only-mismatch"},
    )
    scripted = ScriptedProviderRegistration(capabilities=READ_ONLY_CAPABILITIES)

    with pytest.raises(
        ConfigError,
        match="cannot serve implementation; missing capabilities: workspace-write-execution",
    ):
        prepare_agent_providers(
            config.agents,
            registry={SCRIPTED_PROVIDER_ID: scripted},
            configuration_directory=config.configuration_directory,
        )

    assert scripted.created_executors == []
    assert not (tmp_path / "runs").exists()


def test_scripted_provider_is_not_in_production_composition_or_example() -> None:
    assert SCRIPTED_PROVIDER_ID not in production_provider_registry()
    project_root = Path(__file__).parents[1]
    assert SCRIPTED_PROVIDER_ID.value not in (
        project_root / "config.example.toml"
    ).read_text(encoding="utf-8")
    for path in (project_root / "ticket_automation").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "tests.scripted_provider" not in source
        assert SCRIPTED_PROVIDER_ID.value not in source
