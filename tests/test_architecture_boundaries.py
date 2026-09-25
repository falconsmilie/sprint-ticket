from __future__ import annotations

from pathlib import Path

import pytest

from tests.architecture_fitness import (
    ARCHITECTURE_DEBT,
    FLAT_APPLICATION_MODULES,
    FLAT_DOMAIN_MODULES,
    module_name_for_path,
    scan_package,
    violations_for_source,
)

PACKAGE_ROOT = Path(__file__).parents[1] / "ticket_automation"


def _rules(source: str, module: str) -> set[str]:
    return {violation.rule for violation in violations_for_source(source, module)}


def test_allowed_dependencies_have_no_violations():
    examples = (
        ("from ..domain import Ticket\n", "ticket_automation.application.service"),
        (
            "from ..application.ports import AgentPort\nfrom ..domain import Ticket\n",
            "ticket_automation.providers.example",
        ),
        (
            "from ..application.ports import RunStore\nfrom ..domain import RunId\n",
            "ticket_automation.infrastructure.storage",
        ),
        (
            "from ..application import service\n",
            "ticket_automation.presentation.cli",
        ),
        (
            "from ..providers import example\nfrom ..infrastructure import storage\n",
            "ticket_automation.composition.root",
        ),
    )

    for source, module in examples:
        assert violations_for_source(source, module) == ()


def test_agent_execution_port_is_application_owned_and_provider_neutral():
    port_path = PACKAGE_ROOT / "application" / "agent_execution.py"
    source = port_path.read_text(encoding="utf-8")

    assert (
        violations_for_source(
            source,
            "ticket_automation.application.agent_execution",
        )
        == ()
    )


@pytest.mark.parametrize("package_name", ["application", "domain"])
def test_inward_packages_contain_no_concrete_provider_vocabulary(
    package_name: str,
):
    package = PACKAGE_ROOT / package_name
    violations = [
        violation
        for path in package.rglob("*.py")
        for violation in violations_for_source(
            path.read_text(encoding="utf-8"),
            module_name_for_path(path, PACKAGE_ROOT),
            is_package=path.name == "__init__.py",
        )
        if violation.rule == "concrete_provider_name"
    ]

    assert violations == []


@pytest.mark.parametrize(
    "source",
    [
        "codex_model = 'provider-model'\n",
        "sandbox = 'workspace-write'\n",
        "options = ('--output-schema', 'schema.json')\n",
    ],
)
def test_provider_transport_vocabulary_fails_in_application_port(source: str):
    assert "concrete_provider_name" in _rules(
        source, "ticket_automation.application.agent_execution"
    )


@pytest.mark.parametrize(
    ("source", "rule"),
    [
        ("from ..application import service\n", "domain_outward_import"),
        ("from ..providers import example\n", "domain_outward_import"),
        ("from ..infrastructure import storage\n", "domain_outward_import"),
        ("from ..presentation import cli\n", "domain_outward_import"),
        ("import subprocess\n", "domain_runtime_boundary_import"),
        ("from pathlib import Path\n", "domain_runtime_boundary_import"),
        ("from ticket_automation import git\n", "domain_runtime_boundary_import"),
        ("import ticket_automation.providers.codex_cli\n", "domain_outward_import"),
    ],
)
def test_domain_outward_imports_fail(source: str, rule: str):
    assert rule in _rules(source, "ticket_automation.domain.ticket")


_AUTHORITATIVE_FLAT_CORE_MODULES = (
    "ticket_automation.models",
    "ticket_automation.resolved_config",
    "ticket_automation.execution_evidence",
    "ticket_automation.run_ownership",
    "ticket_automation.runs",
    "ticket_automation.attempts",
    "ticket_automation.failure_classification",
)


def test_authoritative_flat_core_modules_have_explicit_layer_ownership():
    assert set(_AUTHORITATIVE_FLAT_CORE_MODULES) <= (
        FLAT_DOMAIN_MODULES | FLAT_APPLICATION_MODULES
    )


@pytest.mark.parametrize("module", _AUTHORITATIVE_FLAT_CORE_MODULES)
def test_authoritative_flat_core_modules_reject_concrete_provider_imports(
    module: str,
):
    rules = _rules("from ticket_automation.providers import codex_cli\n", module)

    expected = (
        "domain_outward_import"
        if module in FLAT_DOMAIN_MODULES
        else "application_concrete_adapter_import"
    )
    assert expected in rules


@pytest.mark.parametrize("module", _AUTHORITATIVE_FLAT_CORE_MODULES)
@pytest.mark.parametrize("outward_layer", ["composition", "presentation"])
def test_authoritative_flat_core_modules_reject_outward_imports(
    module: str,
    outward_layer: str,
):
    assert "inward_presentation_import" in _rules(
        f"from ticket_automation.{outward_layer} import injected\n",
        module,
    )


@pytest.mark.parametrize(
    ("source", "rule"),
    [
        (
            "from ticket_automation.providers import codex_cli\n",
            "application_concrete_adapter_import",
        ),
        (
            "from ticket_automation.presentation import reporting\n",
            "inward_presentation_import",
        ),
        (
            "from ticket_automation.composition import root\n",
            "inward_presentation_import",
        ),
    ],
)
def test_run_ownership_rejects_outward_application_dependencies(
    source: str,
    rule: str,
):
    assert rule in _rules(source, "ticket_automation.run_ownership")


@pytest.mark.parametrize(
    ("source", "rule"),
    [
        (
            "from ticket_automation.application import service\n",
            "domain_outward_import",
        ),
        ("from pathlib import Path\n", "domain_runtime_boundary_import"),
        ("import subprocess\n", "domain_runtime_boundary_import"),
        ("from ticket_automation import git\n", "domain_runtime_boundary_import"),
    ],
)
def test_flat_domain_owner_rejects_outward_runtime_dependencies(
    source: str,
    rule: str,
):
    assert rule in _rules(source, "ticket_automation.models")


def test_fake_provider_import_in_application_fixture_fails(tmp_path: Path):
    package = tmp_path / "ticket_automation"
    module = package / "application" / "fake_use_case.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "from ticket_automation.providers import fake\n", encoding="utf-8"
    )

    violations = scan_package(package)

    assert {violation.rule for violation in violations} == {
        "application_concrete_adapter_import"
    }


def test_concrete_provider_import_in_application_fails():
    assert "application_concrete_adapter_import" in _rules(
        "from ticket_automation.providers.codex_cli import CodexProcessRunner\n",
        "ticket_automation.application.use_case",
    )


def test_provider_lifecycle_implementation_import_fails():
    rules = _rules(
        "from ..application.lifecycle import run_stage\n",
        "ticket_automation.providers.fake",
    )

    assert "provider_application_implementation_import" in rules


def test_flat_provider_lifecycle_import_fails():
    rules = _rules(
        "from ticket_automation.review import run_review_stage\n",
        "ticket_automation.providers.fake",
    )

    assert "provider_lifecycle_import" in rules


@pytest.mark.parametrize(
    "source",
    [
        "from ticket_automation.application.lifecycle import run_stage\n",
        "from ticket_automation.workflow import run_ticket_lifecycle\n",
    ],
)
def test_infrastructure_application_implementation_import_fails(source: str):
    assert "infrastructure_application_implementation_import" in _rules(
        source, "ticket_automation.infrastructure.storage"
    )


@pytest.mark.parametrize(
    "module",
    [
        "ticket_automation.application.use_case",
        "ticket_automation.domain.model",
        "ticket_automation.infrastructure.storage",
        "ticket_automation.providers.fake",
    ],
)
def test_presentation_imported_by_inward_code_fails(module: str):
    assert "inward_presentation_import" in _rules(
        "from ticket_automation.presentation import cli\n", module
    )


def test_cross_module_private_symbol_import_fails():
    violations = violations_for_source(
        "from ticket_automation.domain.model import _normalize\n",
        "ticket_automation.domain.service",
    )

    assert [violation.rule for violation in violations] == [
        "cross_module_private_import"
    ]


@pytest.mark.parametrize("layer", ["application", "domain"])
@pytest.mark.parametrize(
    "name", ["CodexExecutor", "CodexExecution", "CodexFailureKind", "CodexSettings"]
)
def test_concrete_provider_names_fail_in_inward_layers(layer: str, name: str):
    assert "concrete_provider_name" in _rules(
        f"value = {name}\n", f"ticket_automation.{layer}.contract"
    )


@pytest.mark.parametrize("layer", ["application", "domain"])
@pytest.mark.parametrize(
    "name", ["CodexExecutor", "CodexExecution", "CodexFailureKind", "CodexSettings"]
)
@pytest.mark.parametrize(
    "source_template",
    ['annotation = "{name}"\n', 'value = getattr(provider, "{name}")\n'],
)
def test_concrete_provider_names_in_executable_strings_fail(
    layer: str, name: str, source_template: str
):
    assert "concrete_provider_name" in _rules(
        source_template.format(name=name), f"ticket_automation.{layer}.contract"
    )


@pytest.mark.parametrize("layer", ["application", "domain"])
def test_provider_neutral_string_references_are_allowed(layer: str):
    assert (
        violations_for_source(
            'annotation = "AgentSettings"\n', f"ticket_automation.{layer}.contract"
        )
        == ()
    )


def test_repository_has_no_architecture_debt():
    assert ARCHITECTURE_DEBT == ()
    assert scan_package(PACKAGE_ROOT) == ()
