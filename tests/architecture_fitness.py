"""AST-based dependency checks for the target package architecture."""

from __future__ import annotations

import ast
import importlib.util
import tokenize
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

PACKAGE = "ticket_automation"
LAYERS = frozenset(
    {
        "application",
        "composition",
        "domain",
        "infrastructure",
        "presentation",
        "providers",
    }
)
CONCRETE_PROVIDER_NAMES = frozenset(
    {
        "Codex",
        "CodexCommand",
        "CodexExecution",
        "CodexExecutionFailure",
        "CodexExecutionSettings",
        "CodexExecutor",
        "CodexFailureKind",
        "CodexProcessResult",
        "CodexProcessRunner",
        "CodexSettings",
        "Sandbox",
        "SubprocessCodexRunner",
        "--ephemeral",
        "--json",
        "--model",
        "--output-schema",
        "--sandbox",
    }
)
FILESYSTEM_MODULES = frozenset(
    {"fnmatch", "glob", "os", "pathlib", "shutil", "tempfile"}
)
FLAT_GIT_MODULES = frozenset(
    {f"{PACKAGE}.git", f"{PACKAGE}.git_safety", f"{PACKAGE}.workspace_guard"}
)
LIFECYCLE_APPLICATION_MODULES = frozenset(
    {
        f"{PACKAGE}.correction_planner",
        f"{PACKAGE}.corrections",
        f"{PACKAGE}.implementation",
        f"{PACKAGE}.preflight",
        f"{PACKAGE}.review",
        f"{PACKAGE}.verification",
        f"{PACKAGE}.workflow",
    }
)


@dataclass(frozen=True)
class ImportEdge:
    importer: str
    imported_module: str
    symbol: str | None
    line: int


@dataclass(frozen=True)
class Violation:
    rule: str
    importer: str
    imported_module: str | None
    symbol: str | None
    line: int
    detail: str

    @property
    def key(self) -> tuple[str, str, str | None, str | None]:
        return (self.rule, self.importer, self.imported_module, self.symbol)

    def describe(self) -> str:
        target = self.imported_module or "<source>"
        if self.symbol is not None:
            target = f"{target}:{self.symbol}"
        return f"{self.importer}:{self.line} -> {target} [{self.rule}] {self.detail}"


ARCHITECTURE_DEBT = ()


def module_name_for_path(path: Path, package_root: Path) -> str:
    relative = path.relative_to(package_root.parent).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_from_module(
    importer: str, module: str | None, level: int, *, is_package: bool
) -> str:
    if level == 0:
        return module or ""
    package = importer if is_package else importer.rpartition(".")[0]
    relative_name = "." * level + (module or "")
    return importlib.util.resolve_name(relative_name, package)


def parse_imports(
    source: str, module: str, *, is_package: bool = False
) -> tuple[ImportEdge, ...]:
    tree = ast.parse(source, filename=module)
    edges: list[ImportEdge] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            edges.extend(
                ImportEdge(module, alias.name, None, node.lineno)
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            imported_module = _resolve_from_module(
                module, node.module, node.level, is_package=is_package
            )
            edges.extend(
                ImportEdge(module, imported_module, alias.name, node.lineno)
                for alias in node.names
            )
    return tuple(edges)


def _target_modules(edge: ImportEdge) -> tuple[str, ...]:
    targets = [edge.imported_module]
    if edge.symbol not in (None, "*"):
        targets.append(f"{edge.imported_module}.{edge.symbol}")
    return tuple(targets)


def _matches_module(target: str, expected: str) -> bool:
    return target == expected or target.startswith(f"{expected}.")


def _targets_layer(edge: ImportEdge, layer: str) -> bool:
    expected = f"{PACKAGE}.{layer}"
    return any(_matches_module(target, expected) for target in _target_modules(edge))


def _targets_any_module(edge: ImportEdge, modules: frozenset[str] | set[str]) -> bool:
    return any(
        _matches_module(target, expected)
        for target in _target_modules(edge)
        for expected in modules
    )


def _source_layer(module: str) -> str | None:
    parts = module.split(".")
    if len(parts) > 1 and parts[0] == PACKAGE and parts[1] in LAYERS:
        return parts[1]
    return None


def _edge_violation(edge: ImportEdge, rule: str, detail: str) -> Violation:
    return Violation(
        rule=rule,
        importer=edge.importer,
        imported_module=edge.imported_module,
        symbol=edge.symbol,
        line=edge.line,
        detail=detail,
    )


def _import_violations(edge: ImportEdge) -> list[Violation]:
    violations: list[Violation] = []
    layer = _source_layer(edge.importer)

    if edge.symbol is not None and edge.symbol.startswith("_"):
        violations.append(
            _edge_violation(
                edge,
                "cross_module_private_import",
                "leading-underscore symbols are private to their defining module",
            )
        )
    elif edge.symbol is None and any(
        part.startswith("_") for part in edge.imported_module.split(".")
    ):
        violations.append(
            _edge_violation(
                edge,
                "cross_module_private_import",
                "leading-underscore modules are private to their containing package",
            )
        )

    if layer == "domain":
        outward = {
            "application",
            "composition",
            "infrastructure",
            "presentation",
            "providers",
        }
        if any(_targets_layer(edge, candidate) for candidate in outward):
            violations.append(
                _edge_violation(
                    edge,
                    "domain_outward_import",
                    "domain cannot import an outward layer",
                )
            )
        root_module = edge.imported_module.split(".", 1)[0]
        if root_module in FILESYSTEM_MODULES | {"subprocess", "git"} or any(
            target in FLAT_GIT_MODULES or _matches_module(target, f"{PACKAGE}.cli")
            for target in _target_modules(edge)
        ):
            violations.append(
                _edge_violation(
                    edge,
                    "domain_runtime_boundary_import",
                    "domain cannot import CLI, subprocess, filesystem, or Git modules",
                )
            )

    if (layer == "application" or edge.importer in LIFECYCLE_APPLICATION_MODULES) and (
        _targets_layer(edge, "providers") or _targets_layer(edge, "infrastructure")
    ):
        violations.append(
            _edge_violation(
                edge,
                "application_concrete_adapter_import",
                "application cannot import concrete providers or infrastructure",
            )
        )

    if layer in {"infrastructure", "providers"}:
        application_targets = [
            target
            for target in _target_modules(edge)
            if _matches_module(target, f"{PACKAGE}.application")
        ]
        if application_targets and not any(
            _matches_module(target, f"{PACKAGE}.application.ports")
            or _matches_module(target, f"{PACKAGE}.application.agent_execution")
            for target in application_targets
        ):
            rule = (
                "provider_application_implementation_import"
                if layer == "providers"
                else "infrastructure_application_implementation_import"
            )
            violations.append(
                _edge_violation(
                    edge,
                    rule,
                    f"{layer} may import application ports, not use-case implementations",
                )
            )
        if _targets_any_module(edge, LIFECYCLE_APPLICATION_MODULES):
            rule = (
                "provider_lifecycle_import"
                if layer == "providers"
                else "infrastructure_application_implementation_import"
            )
            violations.append(
                _edge_violation(
                    edge,
                    rule,
                    f"{layer} cannot import lifecycle application implementations",
                )
            )

    if (
        layer in {"application", "domain", "infrastructure", "providers"}
        or edge.importer in LIFECYCLE_APPLICATION_MODULES
    ) and (_targets_layer(edge, "presentation") or _targets_layer(edge, "composition")):
        violations.append(
            _edge_violation(
                edge,
                "inward_presentation_import",
                "inward layers and adapters cannot import presentation or composition",
            )
        )

    return violations


def _concrete_provider_name_lines(source: str) -> dict[str, int]:
    occurrences: dict[str, int] = {}
    for token in tokenize.generate_tokens(StringIO(source).readline):
        if token.type not in {tokenize.NAME, tokenize.STRING}:
            continue
        token_value = token.string.casefold()
        candidates = {
            name for name in CONCRETE_PROVIDER_NAMES if name.casefold() in token_value
        }
        for name in candidates:
            occurrences.setdefault(name, token.start[0])
    return occurrences


def violations_for_source(
    source: str, module: str, *, is_package: bool = False
) -> tuple[Violation, ...]:
    violations = [
        violation
        for edge in parse_imports(source, module, is_package=is_package)
        for violation in _import_violations(edge)
    ]
    if _source_layer(module) in {"application", "domain"}:
        for name, line in sorted(_concrete_provider_name_lines(source).items()):
            violations.append(
                Violation(
                    rule="concrete_provider_name",
                    importer=module,
                    imported_module=None,
                    symbol=name,
                    line=line,
                    detail="domain and application code must use provider-neutral names",
                )
            )
    return tuple(violations)


def scan_package(package_root: Path) -> tuple[Violation, ...]:
    violations: list[Violation] = []
    for path in sorted(package_root.rglob("*.py")):
        module = module_name_for_path(path, package_root)
        source = path.read_text(encoding="utf-8")
        violations.extend(
            violations_for_source(source, module, is_package=path.name == "__init__.py")
        )
    return tuple(violations)
