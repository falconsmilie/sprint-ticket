from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ticket_automation.application.agent_execution import (
    AgentCapability,
    AgentContractError,
    AgentExecutionStatus,
    AgentFailureCategory,
    AgentTaskKind,
    ArtifactReference,
    ArtifactRole,
    AttemptArtifactLayout,
    InvocationStart,
    ProviderId,
    RepositoryAccess,
)
from ticket_automation.execution_evidence import (
    ExecutionEvidence,
    decode_execution_evidence,
    encode_execution_evidence,
    read_execution_evidence,
    write_execution_evidence,
)
from ticket_automation.persistence import (
    CodecError,
    PersistenceError,
    atomic_write_json,
)


def _layout(tmp_path: Path) -> AttemptArtifactLayout:
    run = tmp_path / "run"
    attempt = run / "attempts" / "001-review"
    attempt.mkdir(parents=True)
    return AttemptArtifactLayout(run, attempt)


def _evidence(
    layout: AttemptArtifactLayout,
    *,
    provider: str,
    successful: bool,
    native_artifacts: bool = True,
) -> ExecutionEvidence:
    started = datetime(2026, 9, 24, 10, 0, tzinfo=UTC)
    typed_path = layout.path("provider-result.json")
    typed_path.write_text("{}\n", encoding="utf-8")
    typed = layout.reference(ArtifactRole.TYPED_RESULT, typed_path, "application/json")
    artifacts: tuple[ArtifactReference, ...] = ()
    if native_artifacts:
        events = layout.path("provider-events.jsonl")
        events.write_text("", encoding="utf-8")
        artifacts = (
            layout.reference(
                ArtifactRole.PROVIDER_EVENTS,
                events,
                "application/x-ndjson",
            ),
        )
    return ExecutionEvidence(
        execution_id=f"{layout.attempt_id}:review",
        attempt_id=layout.attempt_id,
        provider_id=ProviderId(provider),
        task_kind=AgentTaskKind.REVIEW,
        repository_access=RepositoryAccess.READ_ONLY,
        required_capabilities=frozenset(
            {AgentCapability.READ_ONLY_EXECUTION, AgentCapability.STRUCTURED_RESULT}
        ),
        started_at=started,
        ended_at=started + timedelta(seconds=2),
        duration_seconds=2,
        status=(
            AgentExecutionStatus.SUCCESS if successful else AgentExecutionStatus.FAILED
        ),
        invocation_start=InvocationStart.STARTED,
        failure_category=(
            None if successful else AgentFailureCategory.NON_SUCCESSFUL_EXECUTION
        ),
        failure_message=None if successful else "Provider failed.",
        typed_result_artifact=typed if successful else None,
        artifacts=artifacts,
        provider_metadata={"native_code": 17, "opaque": {"provider": provider}},
    )


@pytest.mark.parametrize(
    ("provider", "successful", "native_artifacts"),
    [
        ("provider-alpha", True, True),
        ("provider-beta", False, False),
    ],
)
def test_execution_evidence_round_trips_for_multiple_providers(
    tmp_path: Path,
    provider: str,
    successful: bool,
    native_artifacts: bool,
) -> None:
    layout = _layout(tmp_path)
    expected = _evidence(
        layout,
        provider=provider,
        successful=successful,
        native_artifacts=native_artifacts,
    )

    reference = write_execution_evidence(layout, expected)
    actual = read_execution_evidence(layout)

    assert reference.role is ArtifactRole.EXECUTION_EVIDENCE
    assert actual.provider_id == expected.provider_id
    assert actual.status is expected.status
    assert actual.typed_result_artifact == expected.typed_result_artifact
    assert actual.artifacts == expected.artifacts
    assert actual.provider_metadata == expected.provider_metadata


@pytest.mark.parametrize(
    "path",
    ["../escape.json", "/absolute.json", "C:/absolute.json", "nested/../../escape"],
)
def test_artifact_layout_rejects_traversal_and_absolute_paths(
    tmp_path: Path, path: str
) -> None:
    layout = _layout(tmp_path)
    with pytest.raises(AgentContractError):
        layout.path(path)


def test_artifact_layout_rejects_outside_and_missing_references(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    outside = layout.run_root / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(AgentContractError, match="owning attempt"):
        layout.reference(ArtifactRole.PROVIDER_EVENTS, outside)

    missing = ArtifactReference(
        ArtifactRole.PROVIDER_EVENTS,
        f"attempts/{layout.attempt_id}/missing.jsonl",
    )
    with pytest.raises(AgentContractError, match="does not exist"):
        layout.resolve(missing, require_exists=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 0),
        ("status", "complete"),
        ("task_kind", "analysis"),
    ],
)
def test_execution_codec_rejects_unsupported_versions_and_tokens(
    tmp_path: Path, field: str, value: object
) -> None:
    layout = _layout(tmp_path)
    payload = encode_execution_evidence(
        _evidence(layout, provider="provider-alpha", successful=True)
    )
    payload[field] = value  # type: ignore[assignment]

    with pytest.raises(CodecError):
        decode_execution_evidence(payload, layout=layout)


def test_execution_codec_rejects_wrong_typed_result_role(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    payload = encode_execution_evidence(
        _evidence(layout, provider="provider-alpha", successful=True)
    )
    typed = payload["typed_result_artifact"]
    assert isinstance(typed, dict)
    typed["role"] = ArtifactRole.STANDARD_ERROR.value

    with pytest.raises(CodecError, match="typed-result"):
        decode_execution_evidence(payload, layout=layout)


def test_execution_codec_rejects_unknown_artifact_role(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    payload = encode_execution_evidence(
        _evidence(layout, provider="provider-alpha", successful=True)
    )
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, list)
    artifact = artifacts[0]
    assert isinstance(artifact, dict)
    artifact["role"] = "provider-private-unknown"

    with pytest.raises(CodecError, match="unsupported value"):
        decode_execution_evidence(payload, layout=layout)


def test_atomic_json_write_preserves_old_record_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "record.json"
    atomic_write_json(path, {"version": 1})

    def fail_replace(source: Path | str, destination: Path | str) -> None:
        del source, destination
        raise OSError("simulated crash before replace")

    monkeypatch.setattr("ticket_automation.persistence.os.replace", fail_replace)
    with pytest.raises(PersistenceError, match="simulated crash"):
        atomic_write_json(path, {"version": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1}
    assert tuple(tmp_path.glob(".record.json.*.tmp")) == ()


def test_atomic_json_write_uses_utf8_and_one_terminal_newline(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    atomic_write_json(path, {"message": "Grüße"})

    payload = path.read_bytes()
    assert payload.endswith(b"\n")
    assert not payload.endswith(b"\n\n")
    assert json.loads(payload.decode("utf-8"))["message"] == "Grüße"
