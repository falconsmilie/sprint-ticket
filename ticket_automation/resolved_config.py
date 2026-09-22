"""Immutable, provider-neutral policy captured when a run is created."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from . import __version__
from .application.agent_execution import (
    AgentCapability,
    AgentTaskKind,
    ProviderId,
    RepositoryAccess,
    required_execution_capabilities,
)
from .config import (
    AgentSettings,
    AppConfig,
    ProjectSettings,
    RunnerSettings,
    VerificationCommand,
    VerificationSettings,
)

RESOLVED_RUN_POLICY_SCHEMA_VERSION = 2
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_POLICY_ASSETS = {
    "prompt.implementation": _PROJECT_ROOT / "prompts" / "implement.md",
    "prompt.correction": _PROJECT_ROOT / "prompts" / "correct.md",
    "prompt.review": _PROJECT_ROOT / "prompts" / "review.md",
    "result-contract.implementation": _PROJECT_ROOT
    / "schemas"
    / "implementation-result.schema.json",
    "result-contract.review": _PROJECT_ROOT / "schemas" / "review-result.schema.json",
}


class ResolvedRunPolicyError(ValueError):
    """Raised when immutable run policy cannot be captured or trusted."""


class ProviderPolicyCodec(Protocol):
    provider_id: ProviderId
    policy_version: str
    capabilities: frozenset[AgentCapability]

    def encode_run_policy(self, policy: object) -> object: ...
    def decode_run_policy(self, payload: object) -> object: ...
    def runtime_compatibility_problem(self, policy: object) -> str | None: ...


@dataclass(frozen=True)
class ResolvedTaskPolicy:
    task_kind: AgentTaskKind
    provider_id: ProviderId
    repository_access: RepositoryAccess
    required_capabilities: frozenset[AgentCapability]

    def __post_init__(self) -> None:
        if not isinstance(self.task_kind, AgentTaskKind):
            raise ResolvedRunPolicyError("resolved task kind must be typed.")
        if not isinstance(self.provider_id, ProviderId):
            raise ResolvedRunPolicyError("resolved task provider ID must be typed.")
        if not isinstance(self.repository_access, RepositoryAccess):
            raise ResolvedRunPolicyError(
                "resolved task repository access must be typed."
            )
        _validate_capabilities(self.required_capabilities, provider_id=self.provider_id)
        expected_access = _task_repository_access(self.task_kind)
        expected_capabilities = required_execution_capabilities(expected_access)
        if (
            self.repository_access is not expected_access
            or self.required_capabilities != expected_capabilities
        ):
            raise ResolvedRunPolicyError(
                f"resolved_policy task requirements are incompatible: {self.task_kind.value}."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "provider_id": str(self.provider_id),
            "repository_access": self.repository_access.value,
            "required_capabilities": sorted(
                c.value for c in self.required_capabilities
            ),
        }


@dataclass(frozen=True)
class ResolvedProviderPolicy:
    provider_id: ProviderId
    adapter_policy_version: str
    declared_capabilities: frozenset[AgentCapability]
    payload_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.provider_id, ProviderId):
            raise ResolvedRunPolicyError("resolved provider ID must be typed.")
        if (
            not isinstance(self.adapter_policy_version, str)
            or not self.adapter_policy_version.strip()
        ):
            raise ResolvedRunPolicyError(
                "resolved provider adapter policy version must be non-empty."
            )
        _validate_capabilities(self.declared_capabilities, provider_id=self.provider_id)
        if not isinstance(self.payload_json, str):
            raise ResolvedRunPolicyError(
                "resolved provider payload must use canonical JSON."
            )
        try:
            payload = json.loads(self.payload_json)
        except (TypeError, ValueError) as error:
            raise ResolvedRunPolicyError(
                "resolved provider payload must use valid JSON."
            ) from error
        if _canonical_json(payload) != self.payload_json:
            raise ResolvedRunPolicyError(
                "resolved provider payload must use canonical JSON."
            )

    def payload(self) -> object:
        return json.loads(self.payload_json)

    def to_dict(self) -> dict[str, object]:
        return {
            "provider_id": str(self.provider_id),
            "adapter_policy_version": self.adapter_policy_version,
            "declared_capabilities": sorted(
                c.value for c in self.declared_capabilities
            ),
            "payload": self.payload(),
        }


@dataclass(frozen=True)
class ResolvedRunPolicy:
    """Complete provider-neutral policy for one run."""

    target_repository_path: str
    protected_branches: tuple[str, ...]
    verification_commands: tuple[VerificationCommand, ...]
    max_correction_rounds: int
    task_policies: tuple[ResolvedTaskPolicy, ...]
    provider_policies: tuple[ResolvedProviderPolicy, ...]
    ticket_automation_package: str
    ticket_automation_version: str
    ticket_automation_source_kind: str
    ticket_automation_source_revision: str | None
    policy_asset_sha256: tuple[tuple[str, str], ...]
    schema_version: int = RESOLVED_RUN_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != RESOLVED_RUN_POLICY_SCHEMA_VERSION
        ):
            raise ResolvedRunPolicyError(
                "resolved_policy has an unsupported schema version."
            )
        if (
            not isinstance(self.target_repository_path, str)
            or not self.target_repository_path.strip()
            or not Path(self.target_repository_path).is_absolute()
        ):
            raise ResolvedRunPolicyError(
                "resolved_policy target_repository.path must be absolute."
            )
        if not isinstance(self.task_policies, tuple) or not isinstance(
            self.provider_policies, tuple
        ):
            raise ResolvedRunPolicyError(
                "resolved_policy task and provider records must be immutable tuples."
            )
        if not isinstance(self.protected_branches, tuple) or not isinstance(
            self.verification_commands, tuple
        ):
            raise ResolvedRunPolicyError(
                "resolved_policy repository and verification values must be immutable."
            )
        _validate_non_empty_strings(
            self.protected_branches,
            field="target_repository.protected_branches",
        )
        _validate_verification_commands(self.verification_commands)
        if (
            not isinstance(self.max_correction_rounds, int)
            or isinstance(self.max_correction_rounds, bool)
            or self.max_correction_rounds < 1
        ):
            raise ResolvedRunPolicyError(
                "resolved_policy verification.max_correction_rounds must be positive."
            )
        if self.ticket_automation_package != "ticket-automation":
            raise ResolvedRunPolicyError(
                "resolved_policy ticket_automation.package is unsupported."
            )
        if (
            not isinstance(self.ticket_automation_version, str)
            or not self.ticket_automation_version.strip()
        ):
            raise ResolvedRunPolicyError(
                "resolved_policy ticket_automation.version must be non-empty."
            )
        _validate_source_identity(
            self.ticket_automation_source_kind,
            self.ticket_automation_source_revision,
        )
        if not isinstance(self.policy_asset_sha256, tuple):
            raise ResolvedRunPolicyError(
                "resolved_policy asset versions must be an immutable tuple."
            )
        _validate_policy_assets(self.policy_asset_sha256)
        _validate_policy_relationships(self.task_policies, self.provider_policies)

    @property
    def assignments(self) -> Mapping[AgentTaskKind, ProviderId]:
        return {task.task_kind: task.provider_id for task in self.task_policies}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_repository": {
                "path": self.target_repository_path,
                "protected_branches": list(self.protected_branches),
            },
            "verification": {
                "commands": [
                    {
                        "name": c.name,
                        "argv": list(c.argv),
                        "timeout_seconds": c.timeout_seconds,
                    }
                    for c in self.verification_commands
                ],
                "max_correction_rounds": self.max_correction_rounds,
            },
            "tasks": {
                task.task_kind.value: task.to_dict() for task in self.task_policies
            },
            "providers": {
                str(provider.provider_id): provider.to_dict()
                for provider in self.provider_policies
            },
            "ticket_automation": {
                "package": self.ticket_automation_package,
                "version": self.ticket_automation_version,
                "source": {
                    "kind": self.ticket_automation_source_kind,
                    "revision": self.ticket_automation_source_revision,
                },
            },
            "policy_assets": dict(self.policy_asset_sha256),
        }

    @classmethod
    def from_dict(cls, data: Any) -> ResolvedRunPolicy:
        if not isinstance(data, dict):
            raise ResolvedRunPolicyError("resolved_policy must be an object.")
        version = data.get("schema_version")
        if (
            not isinstance(version, int)
            or isinstance(version, bool)
            or version != RESOLVED_RUN_POLICY_SCHEMA_VERSION
        ):
            raise ResolvedRunPolicyError(
                f"Unsupported resolved policy schema version: {version!r}; "
                f"expected {RESOLVED_RUN_POLICY_SCHEMA_VERSION}."
            )
        _require_exact_keys(
            data,
            {
                "schema_version",
                "target_repository",
                "verification",
                "tasks",
                "providers",
                "ticket_automation",
                "policy_assets",
            },
            context="resolved_policy",
        )
        target = _require_table(data, "target_repository")
        verification = _require_table(data, "verification")
        package = _require_table(data, "ticket_automation")
        source = _require_table(package, "source")
        assets = _require_table(data, "policy_assets")
        _require_exact_keys(
            target,
            {"path", "protected_branches"},
            context="resolved_policy target_repository",
        )
        _require_exact_keys(
            verification,
            {"commands", "max_correction_rounds"},
            context="resolved_policy verification",
        )
        _require_exact_keys(
            package,
            {"package", "version", "source"},
            context="resolved_policy ticket_automation",
        )
        _require_exact_keys(
            source,
            {"kind", "revision"},
            context="resolved_policy ticket_automation.source",
        )
        target_path = _require_string(target, "path")
        if not Path(target_path).is_absolute():
            raise ResolvedRunPolicyError(
                "resolved_policy target_repository.path must be absolute."
            )
        revision = source.get("revision")
        source_kind = _require_string(source, "kind")
        if source_kind == "git":
            if not _is_sha(revision):
                raise ResolvedRunPolicyError(
                    "resolved_policy ticket_automation.source.revision must be a SHA."
                )
        elif source_kind == "installed-package":
            if revision is not None:
                raise ResolvedRunPolicyError(
                    "resolved_policy installed-package source revision must be null."
                )
        else:
            raise ResolvedRunPolicyError(
                "resolved_policy ticket_automation.source.kind is unsupported."
            )
        parsed_assets = tuple(sorted(assets.items()))
        if set(dict(parsed_assets)) != set(_POLICY_ASSETS) or not all(
            isinstance(name, str) and _is_sha(digest) for name, digest in parsed_assets
        ):
            raise ResolvedRunPolicyError(
                "resolved_policy policy_assets is incomplete or invalid."
            )
        return cls(
            schema_version=RESOLVED_RUN_POLICY_SCHEMA_VERSION,
            target_repository_path=target_path,
            protected_branches=_require_string_tuple(target, "protected_branches"),
            verification_commands=_parse_verification_commands(verification),
            max_correction_rounds=_require_positive_int(
                verification, "max_correction_rounds"
            ),
            task_policies=_parse_task_policies(_require_table(data, "tasks")),
            provider_policies=_parse_provider_policies(
                _require_table(data, "providers")
            ),
            ticket_automation_package=_require_package_name(package),
            ticket_automation_version=_require_string(package, "version"),
            ticket_automation_source_kind=source_kind,
            ticket_automation_source_revision=revision,
            policy_asset_sha256=parsed_assets,
        )

    def restore_provider_policies(
        self, registry: Mapping[ProviderId, ProviderPolicyCodec]
    ) -> Mapping[ProviderId, object]:
        """Decode and compatibility-check each provider policy exactly once."""

        try:
            actual_assets = _policy_asset_sha256()
        except ResolvedRunPolicyError as error:
            raise ResolvedRunPolicyError(str(error)) from error
        expected_assets = dict(self.policy_asset_sha256)
        for name, actual in actual_assets.items():
            if expected_assets.get(name) != actual:
                raise ResolvedRunPolicyError(
                    "Persisted policy asset is incompatible with this "
                    f"TicketAutomation installation: {name}."
                )
        if self.ticket_automation_version != __version__:
            raise ResolvedRunPolicyError(
                "Persisted TicketAutomation version is incompatible with this installation: "
                f"expected {self.ticket_automation_version!r}, got {__version__!r}."
            )
        if (
            self.ticket_automation_source_kind == "git"
            and _ticket_automation_git_sha() != self.ticket_automation_source_revision
        ):
            raise ResolvedRunPolicyError(
                "Persisted TicketAutomation source revision is incompatible with this installation."
            )
        decoded_policies: dict[ProviderId, object] = {}
        for persisted in self.provider_policies:
            registration = registry.get(persisted.provider_id)
            if registration is None:
                raise ResolvedRunPolicyError(
                    f"Persisted provider is not registered: {persisted.provider_id}."
                )
            if registration.provider_id != persisted.provider_id:
                raise ResolvedRunPolicyError(
                    f"Provider registry identity does not match {persisted.provider_id}."
                )
            if registration.policy_version != persisted.adapter_policy_version:
                raise ResolvedRunPolicyError(
                    f"Provider {persisted.provider_id} adapter policy version is incompatible: "
                    f"expected {persisted.adapter_policy_version!r}, got {registration.policy_version!r}."
                )
            try:
                _validate_capabilities(
                    registration.capabilities,
                    provider_id=persisted.provider_id,
                )
            except ResolvedRunPolicyError as error:
                raise ResolvedRunPolicyError(
                    f"Provider {persisted.provider_id} registration is invalid: {error}"
                ) from error
            if registration.capabilities != persisted.declared_capabilities:
                raise ResolvedRunPolicyError(
                    f"Provider {persisted.provider_id} declared capabilities changed."
                )
            try:
                decoded = registration.decode_run_policy(persisted.payload())
                problem = registration.runtime_compatibility_problem(decoded)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                raise ResolvedRunPolicyError(
                    f"Provider {persisted.provider_id} persisted policy is invalid: {error}"
                ) from error
            if problem is not None:
                raise ResolvedRunPolicyError(
                    f"Provider {persisted.provider_id} is incompatible: {problem}"
                )
            decoded_policies[persisted.provider_id] = decoded
        return MappingProxyType(decoded_policies)

    def runtime_compatibility_problem(
        self, registry: Mapping[ProviderId, ProviderPolicyCodec]
    ) -> str | None:
        try:
            self.restore_provider_policies(registry)
        except ResolvedRunPolicyError as error:
            return str(error)
        return None


def resolve_run_policy(
    config: AppConfig,
    *,
    target_repository_path: Path | str,
    assignments: Mapping[AgentTaskKind, ProviderId],
    provider_policies: Mapping[ProviderId, object],
    provider_registrations: Mapping[ProviderId, ProviderPolicyCodec],
) -> ResolvedRunPolicy:
    tasks = tuple(
        _resolved_task_policy(task_kind, assignments.get(task_kind))
        for task_kind in AgentTaskKind
    )
    referenced = {task.provider_id for task in tasks}
    if set(provider_policies) != referenced:
        raise ResolvedRunPolicyError(
            "Resolved provider policies must exactly match task assignments."
        )
    if set(provider_registrations) != referenced:
        raise ResolvedRunPolicyError(
            "Provider registrations must exactly match task assignments."
        )
    providers: list[ResolvedProviderPolicy] = []
    for provider_id in sorted(referenced, key=str):
        registration = provider_registrations[provider_id]
        if registration.provider_id != provider_id:
            raise ResolvedRunPolicyError(
                f"Provider registration identity does not match {provider_id}."
            )
        _validate_capabilities(registration.capabilities, provider_id=provider_id)
        version = registration.policy_version
        if not isinstance(version, str) or not version.strip():
            raise ResolvedRunPolicyError(
                f"Provider {provider_id} policy version must be a non-empty string."
            )
        try:
            payload_json = _canonical_json(
                registration.encode_run_policy(provider_policies[provider_id])
            )
            registration.decode_run_policy(json.loads(payload_json))
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ResolvedRunPolicyError(
                f"Provider {provider_id} could not encode its resolved policy: {error}"
            ) from error
        providers.append(
            ResolvedProviderPolicy(
                provider_id=provider_id,
                adapter_policy_version=version,
                declared_capabilities=registration.capabilities,
                payload_json=payload_json,
            )
        )
    git_sha = _ticket_automation_git_sha()
    return ResolvedRunPolicy(
        target_repository_path=str(Path(target_repository_path).resolve()),
        protected_branches=config.project.protected_branches,
        verification_commands=config.verification.commands,
        max_correction_rounds=config.runner.max_correction_rounds,
        task_policies=tasks,
        provider_policies=tuple(providers),
        ticket_automation_package="ticket-automation",
        ticket_automation_version=__version__,
        ticket_automation_source_kind="git"
        if git_sha is not None
        else "installed-package",
        ticket_automation_source_revision=git_sha,
        policy_asset_sha256=tuple(sorted(_policy_asset_sha256().items())),
    )


def config_from_resolved_run_policy(resolved: ResolvedRunPolicy) -> AppConfig:
    provider_ids = {task.provider_id for task in resolved.task_policies}
    return AppConfig(
        project=ProjectSettings(
            name="Persisted run policy",
            repo=Path(resolved.target_repository_path),
            protected_branches=resolved.protected_branches,
        ),
        runner=RunnerSettings(max_correction_rounds=resolved.max_correction_rounds),
        agents=AgentSettings(
            assignments=resolved.assignments,
            providers={provider_id: {} for provider_id in provider_ids},
        ),
        verification=VerificationSettings(commands=resolved.verification_commands),
        source_files=(),
        configuration_directory=_PROJECT_ROOT,
    )


def _resolved_task_policy(
    task_kind: AgentTaskKind, provider_id: ProviderId | None
) -> ResolvedTaskPolicy:
    if not isinstance(provider_id, ProviderId):
        raise ResolvedRunPolicyError(
            f"Missing typed provider assignment for {task_kind.value}."
        )
    access = _task_repository_access(task_kind)
    return ResolvedTaskPolicy(
        task_kind=task_kind,
        provider_id=provider_id,
        repository_access=access,
        required_capabilities=required_execution_capabilities(access),
    )


def _parse_task_policies(data: dict[str, Any]) -> tuple[ResolvedTaskPolicy, ...]:
    if set(data) != {kind.value for kind in AgentTaskKind}:
        raise ResolvedRunPolicyError(
            "resolved_policy tasks must contain implementation, review, and correction."
        )
    parsed: list[ResolvedTaskPolicy] = []
    for task_kind in AgentTaskKind:
        task = _require_table(data, task_kind.value)
        _require_exact_keys(
            task,
            {"provider_id", "repository_access", "required_capabilities"},
            context=f"resolved_policy task {task_kind.value}",
        )
        expected = _resolved_task_policy(
            task_kind, _provider_id(_require_string(task, "provider_id"))
        )
        try:
            access = RepositoryAccess(_require_string(task, "repository_access"))
        except ValueError as error:
            raise ResolvedRunPolicyError(
                f"resolved_policy task access is unsupported: {task_kind.value}."
            ) from error
        capabilities = _parse_capabilities(task, "required_capabilities")
        if (
            access is not expected.repository_access
            or capabilities != expected.required_capabilities
        ):
            raise ResolvedRunPolicyError(
                f"resolved_policy task requirements are incompatible: {task_kind.value}."
            )
        parsed.append(expected)
    return tuple(parsed)


def _parse_provider_policies(
    data: dict[str, Any],
) -> tuple[ResolvedProviderPolicy, ...]:
    parsed: list[ResolvedProviderPolicy] = []
    for key, value in data.items():
        provider_id = _provider_id(key)
        if not isinstance(value, dict):
            raise ResolvedRunPolicyError(
                f"resolved_policy provider record must be an object: {key}."
            )
        _require_exact_keys(
            value,
            {
                "provider_id",
                "adapter_policy_version",
                "declared_capabilities",
                "payload",
            },
            context=f"resolved_policy provider {key}",
        )
        if _provider_id(_require_string(value, "provider_id")) != provider_id:
            raise ResolvedRunPolicyError(
                f"resolved_policy provider key does not match its typed ID: {key}."
            )
        parsed.append(
            ResolvedProviderPolicy(
                provider_id=provider_id,
                adapter_policy_version=_require_string(value, "adapter_policy_version"),
                declared_capabilities=_parse_capabilities(
                    value, "declared_capabilities"
                ),
                payload_json=_canonical_json(value.get("payload")),
            )
        )
    if not parsed:
        raise ResolvedRunPolicyError(
            "resolved_policy providers must contain at least one provider."
        )
    return tuple(sorted(parsed, key=lambda item: str(item.provider_id)))


def _validate_policy_relationships(
    tasks: tuple[ResolvedTaskPolicy, ...], providers: tuple[ResolvedProviderPolicy, ...]
) -> None:
    if tuple(task.task_kind for task in tasks) != tuple(AgentTaskKind):
        raise ResolvedRunPolicyError(
            "resolved_policy task records must be complete and ordered."
        )
    provider_by_id = {provider.provider_id: provider for provider in providers}
    referenced = {task.provider_id for task in tasks}
    if len(provider_by_id) != len(providers) or set(provider_by_id) != referenced:
        raise ResolvedRunPolicyError(
            "resolved_policy providers must exactly match task assignments."
        )
    for task in tasks:
        if (
            not task.required_capabilities
            <= provider_by_id[task.provider_id].declared_capabilities
        ):
            raise ResolvedRunPolicyError(
                f"resolved_policy provider {task.provider_id} lacks capabilities required for {task.task_kind.value}."
            )


def _task_repository_access(task_kind: AgentTaskKind) -> RepositoryAccess:
    return (
        RepositoryAccess.READ_ONLY
        if task_kind is AgentTaskKind.REVIEW
        else RepositoryAccess.WORKSPACE_WRITE
    )


def _validate_non_empty_strings(values: tuple[str, ...], *, field: str) -> None:
    if not values or not all(
        isinstance(value, str) and value.strip() for value in values
    ):
        raise ResolvedRunPolicyError(
            f"resolved_policy {field} must be a non-empty string tuple."
        )


def _validate_verification_commands(
    commands: tuple[VerificationCommand, ...],
) -> None:
    if not commands:
        raise ResolvedRunPolicyError(
            "resolved_policy verification.commands must be non-empty."
        )
    for index, command in enumerate(commands, start=1):
        if not isinstance(command, VerificationCommand):
            raise ResolvedRunPolicyError(
                f"resolved_policy verification.commands[{index}] has the wrong type."
            )
        if not isinstance(command.name, str) or not command.name.strip():
            raise ResolvedRunPolicyError(
                f"resolved_policy verification.commands[{index}].name is invalid."
            )
        if (
            not isinstance(command.argv, tuple)
            or not command.argv
            or not all(isinstance(item, str) and item.strip() for item in command.argv)
        ):
            raise ResolvedRunPolicyError(
                f"resolved_policy verification.commands[{index}].argv is invalid."
            )
        if (
            not isinstance(command.timeout_seconds, int)
            or isinstance(command.timeout_seconds, bool)
            or command.timeout_seconds < 1
        ):
            raise ResolvedRunPolicyError(
                f"resolved_policy verification.commands[{index}].timeout_seconds is invalid."
            )


def _validate_source_identity(kind: object, revision: object) -> None:
    if kind == "git":
        if not _is_sha(revision):
            raise ResolvedRunPolicyError(
                "resolved_policy ticket_automation.source.revision must be a SHA."
            )
        return
    if kind == "installed-package":
        if revision is not None:
            raise ResolvedRunPolicyError(
                "resolved_policy installed-package source revision must be null."
            )
        return
    raise ResolvedRunPolicyError(
        "resolved_policy ticket_automation.source.kind is unsupported."
    )


def _validate_policy_assets(assets: tuple[tuple[str, str], ...]) -> None:
    if any(
        not isinstance(item, tuple)
        or len(item) != 2
        or not isinstance(item[0], str)
        or not isinstance(item[1], str)
        for item in assets
    ):
        raise ResolvedRunPolicyError(
            "resolved_policy policy_assets is incomplete or invalid."
        )
    try:
        asset_map = dict(assets)
    except (TypeError, ValueError) as error:
        raise ResolvedRunPolicyError(
            "resolved_policy policy_assets is incomplete or invalid."
        ) from error
    if (
        len(asset_map) != len(assets)
        or set(asset_map) != set(_POLICY_ASSETS)
        or tuple(sorted(assets)) != assets
        or not all(_is_sha(digest) for _, digest in assets)
    ):
        raise ResolvedRunPolicyError(
            "resolved_policy policy_assets is incomplete or invalid."
        )


def _validate_capabilities(capabilities: object, *, provider_id: ProviderId) -> None:
    if not isinstance(capabilities, frozenset) or not all(
        isinstance(capability, AgentCapability) for capability in capabilities
    ):
        raise ResolvedRunPolicyError(
            f"Provider {provider_id} capabilities must be typed and immutable."
        )


def _parse_capabilities(data: dict[str, Any], key: str) -> frozenset[AgentCapability]:
    values = data.get(key)
    if not isinstance(values, list) or not values:
        raise ResolvedRunPolicyError(
            f"resolved_policy field must be a non-empty capability list: {key}."
        )
    try:
        parsed = frozenset(AgentCapability(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ResolvedRunPolicyError(
            f"resolved_policy field contains an unsupported capability: {key}."
        ) from error
    if len(parsed) != len(values):
        raise ResolvedRunPolicyError(
            f"resolved_policy field contains duplicate capabilities: {key}."
        )
    return parsed


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ResolvedRunPolicyError(
            "Provider resolved payload must be valid JSON."
        ) from error


def _policy_asset_sha256() -> dict[str, str]:
    try:
        return {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in _POLICY_ASSETS.items()
        }
    except OSError as error:
        raise ResolvedRunPolicyError(
            f"Could not read TicketAutomation policy asset: {error}"
        ) from error


def _ticket_automation_git_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(_PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    sha = completed.stdout.strip()
    return sha if completed.returncode == 0 and _is_sha(sha) else None


def _require_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ResolvedRunPolicyError(f"resolved_policy field must be an object: {key}.")
    return value


def _require_exact_keys(
    data: dict[str, Any], expected: set[str], *, context: str
) -> None:
    actual = set(data)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append("missing " + ", ".join(missing))
    if unexpected:
        details.append("unexpected " + ", ".join(unexpected))
    raise ResolvedRunPolicyError(f"{context} fields are invalid: {'; '.join(details)}.")


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ResolvedRunPolicyError(
            f"resolved_policy field must be a non-empty string: {key}."
        )
    return value.strip()


def _require_string_tuple(data: dict[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ResolvedRunPolicyError(
            f"resolved_policy field must be a non-empty string list: {key}."
        )
    return tuple(_require_string({key: item}, key) for item in value)


def _require_positive_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ResolvedRunPolicyError(
            f"resolved_policy field must be a positive integer: {key}."
        )
    return value


def _require_package_name(data: dict[str, Any]) -> str:
    package = _require_string(data, "package")
    if package != "ticket-automation":
        raise ResolvedRunPolicyError(
            "resolved_policy ticket_automation.package is unsupported."
        )
    return package


def _parse_verification_commands(
    verification: dict[str, Any],
) -> tuple[VerificationCommand, ...]:
    commands = verification.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ResolvedRunPolicyError(
            "resolved_policy verification.commands must be a non-empty list."
        )
    parsed: list[VerificationCommand] = []
    for index, command in enumerate(commands, start=1):
        if not isinstance(command, dict):
            raise ResolvedRunPolicyError(
                f"resolved_policy verification.commands[{index}] must be an object."
            )
        _require_exact_keys(
            command,
            {"name", "argv", "timeout_seconds"},
            context=f"resolved_policy verification.commands[{index}]",
        )
        argv = command.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(item, str) and item.strip() for item in argv)
        ):
            raise ResolvedRunPolicyError(
                f"resolved_policy verification.commands[{index}].argv is invalid."
            )
        parsed.append(
            VerificationCommand(
                name=_require_string(command, "name"),
                argv=tuple(argv),
                timeout_seconds=_require_positive_int(command, "timeout_seconds"),
            )
        )
    return tuple(parsed)


def _provider_id(value: str) -> ProviderId:
    try:
        return ProviderId(value)
    except ValueError as error:
        raise ResolvedRunPolicyError(
            f"resolved_policy contains an invalid provider ID: {value!r}."
        ) from error


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "RESOLVED_RUN_POLICY_SCHEMA_VERSION",
    "ResolvedProviderPolicy",
    "ResolvedRunPolicy",
    "ResolvedRunPolicyError",
    "ResolvedTaskPolicy",
    "config_from_resolved_run_policy",
    "resolve_run_policy",
]
