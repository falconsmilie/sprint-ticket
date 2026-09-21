from __future__ import annotations

from pathlib import Path

import pytest

from tests.architecture_fitness import (
    ARCHITECTURE_DEBT,
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
        ("import ticket_automation.codex\n", "domain_outward_import"),
    ],
)
def test_domain_outward_imports_fail(source: str, rule: str):
    assert rule in _rules(source, "ticket_automation.domain.ticket")


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


def test_flat_concrete_provider_import_in_application_fails():
    assert "application_concrete_adapter_import" in _rules(
        "from ticket_automation.codex import CodexProcessRunner\n",
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


def test_repository_matches_the_exact_temporary_debt_allowlist():
    violations = scan_package(PACKAGE_ROOT)
    actual = {violation.key: violation for violation in violations}
    allowed = {entry.key: entry for entry in ARCHITECTURE_DEBT}
    duplicate_debt_keys = len(allowed) != len(ARCHITECTURE_DEBT)
    unexpected = [
        actual[key].describe()
        for key in sorted(actual.keys() - allowed.keys(), key=repr)
    ]
    stale = [
        allowed[key].describe()
        for key in sorted(allowed.keys() - actual.keys(), key=repr)
    ]

    assert not duplicate_debt_keys, "architecture debt entries must be unique"
    assert not unexpected, "new architecture violations:\n" + "\n".join(unexpected)
    assert not stale, "remove stale architecture debt entries:\n" + "\n".join(stale)
    assert all(entry.removal_ticket.startswith("TA-") for entry in ARCHITECTURE_DEBT)
    assert all(entry.rationale for entry in ARCHITECTURE_DEBT)
    assert all(
        "*" not in value
        for entry in ARCHITECTURE_DEBT
        for value in (entry.importer, entry.imported_module, entry.symbol)
    )
