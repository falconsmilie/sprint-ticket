from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ticket_automation.application.agent_execution import (
    AgentCapability,
    AgentContractError,
    AgentExecution,
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
from ticket_automation.attempts import attempt_result_path, start_attempt
from ticket_automation.domain.task_results import ReviewResult, ReviewVerdict
from ticket_automation.execution_evidence import (
    EXECUTION_EVIDENCE_SCHEMA_VERSION,
    ExecutionEvidence,
    decode_execution_evidence,
    encode_execution_evidence,
    read_execution_evidence,
    write_execution_evidence,
)
from ticket_automation.models import AttemptPhase, VerificationStatus
from ticket_automation.persistence import (
    CodecError,
    PersistenceError,
    atomic_write_json,
    atomic_write_text,
    exclusive_write_bytes,
    exclusive_write_text,
)
from ticket_automation.verification_evidence import (
    VERIFICATION_ROUND_FORMAT,
    VERIFICATION_SCHEMA_VERSION,
    VerificationEvidence,
    read_verification_evidence,
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
        typed_result=(
            ReviewResult(
                verdict=ReviewVerdict.PASS,
                summary="Review passed.",
                findings=(),
            )
            if successful
            else None
        ),
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
    assert actual.typed_result == expected.typed_result
    assert actual.typed_result_artifact == expected.typed_result_artifact
    assert actual.artifacts == expected.artifacts
    assert actual.provider_metadata == expected.provider_metadata
    assert actual == expected
    assert type(actual.duration_seconds) is float


@pytest.mark.parametrize(
    "schema_version",
    [True, float(EXECUTION_EVIDENCE_SCHEMA_VERSION)],
)
def test_execution_evidence_construction_requires_an_exact_integer_schema(
    tmp_path: Path,
    schema_version: object,
) -> None:
    evidence = _evidence(_layout(tmp_path), provider="provider-alpha", successful=True)

    with pytest.raises(CodecError, match="unsupported schema version"):
        replace(evidence, schema_version=schema_version)


@pytest.mark.parametrize("field", ["started_at", "ended_at"])
def test_execution_evidence_construction_rejects_naive_timestamps(
    tmp_path: Path,
    field: str,
) -> None:
    evidence = _evidence(_layout(tmp_path), provider="provider-alpha", successful=True)
    naive = evidence.started_at.replace(tzinfo=None)

    with pytest.raises(CodecError, match=rf"{field} must be a timezone-aware"):
        replace(evidence, **{field: naive})


@pytest.mark.parametrize(
    ("duration", "message"),
    [
        (True, "must be a number"),
        (float("nan"), "must be finite"),
        (float("inf"), "must be finite"),
        (-1.0, "must be finite and not negative"),
    ],
)
def test_execution_evidence_construction_rejects_invalid_durations(
    tmp_path: Path,
    duration: object,
    message: str,
) -> None:
    evidence = _evidence(_layout(tmp_path), provider="provider-alpha", successful=True)

    with pytest.raises(CodecError, match=message):
        replace(evidence, duration_seconds=duration)


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


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda payload: payload.update({"typed_result": None}), "typed result"),
        (
            lambda payload: payload.update(
                {"execution_id": f"{payload['attempt_id']}:implementation"}
            ),
            "execution_id",
        ),
        (
            lambda payload: payload.update({"typed_result_artifact": None}),
            "typed result artifact",
        ),
    ],
    ids=["missing-result", "wrong-execution-id", "missing-result-artifact"],
)
def test_successful_execution_evidence_rejects_internal_drift(
    tmp_path: Path,
    change,
    message: str,
) -> None:
    layout = _layout(tmp_path)
    payload = encode_execution_evidence(
        _evidence(layout, provider="provider-alpha", successful=True)
    )
    change(payload)

    with pytest.raises(CodecError, match=message):
        decode_execution_evidence(payload, layout=layout)


def test_execution_codec_rejects_a_result_for_the_wrong_task_kind(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    payload = encode_execution_evidence(
        _evidence(layout, provider="provider-alpha", successful=True)
    )
    payload["task_kind"] = AgentTaskKind.IMPLEMENTATION.value
    payload["execution_id"] = f"{layout.attempt_id}:implementation"

    with pytest.raises(CodecError, match="typed_result is invalid"):
        decode_execution_evidence(payload, layout=layout)


def test_execution_codec_rejects_failed_evidence_carrying_a_result(
    tmp_path: Path,
) -> None:
    layout = _layout(tmp_path)
    successful = encode_execution_evidence(
        _evidence(layout, provider="provider-alpha", successful=True)
    )
    failed = encode_execution_evidence(
        _evidence(layout, provider="provider-alpha", successful=False)
    )
    failed["typed_result"] = successful["typed_result"]

    with pytest.raises(CodecError, match="failed evidence must not contain"):
        decode_execution_evidence(failed, layout=layout)


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


@pytest.mark.parametrize("category", list(AgentFailureCategory))
@pytest.mark.parametrize("invocation_start", list(InvocationStart))
def test_runtime_and_persisted_failure_semantics_have_exact_parity(
    tmp_path: Path,
    category: AgentFailureCategory,
    invocation_start: InvocationStart,
) -> None:
    layout = _layout(tmp_path)
    payload = encode_execution_evidence(
        _evidence(layout, provider="provider-alpha", successful=False)
    )
    payload["invocation_start"] = invocation_start.value
    failure = payload["failure"]
    assert isinstance(failure, dict)
    failure["category"] = category.value
    pre_invocation = {
        AgentFailureCategory.PROVIDER_UNAVAILABLE,
        AgentFailureCategory.INVOCATION_START_FAILURE,
        AgentFailureCategory.CAPABILITY_OR_CONFIGURATION_FAILURE,
    }
    expected_valid = (
        invocation_start is InvocationStart.NOT_STARTED
        if category in pre_invocation
        else category is AgentFailureCategory.TIMEOUT
        or invocation_start is not InvocationStart.NOT_STARTED
    )

    started = datetime(2026, 9, 24, 10, 0, tzinfo=UTC)
    try:
        AgentExecution(
            provider_id=ProviderId("provider-alpha"),
            task_kind=AgentTaskKind.REVIEW,
            status=AgentExecutionStatus.FAILED,
            invocation_start=invocation_start,
            started_at=started,
            ended_at=started + timedelta(seconds=2),
            duration_seconds=2,
            failure_category=category,
            failure_message="Provider failed.",
        )
    except AgentContractError:
        runtime_valid = False
    else:
        runtime_valid = True
    try:
        decode_execution_evidence(payload, layout=layout)
    except CodecError:
        evidence_valid = False
    else:
        evidence_valid = True

    assert runtime_valid is expected_valid
    assert evidence_valid is expected_valid


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


def test_atomic_text_write_replaces_a_hardlink_without_mutating_its_target(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external.txt"
    external.write_text("external evidence", encoding="utf-8")
    destination = tmp_path / "run" / "final.patch"
    destination.parent.mkdir()
    os.link(external, destination)

    atomic_write_text(destination, "captured patch")

    assert external.read_text(encoding="utf-8") == "external evidence"
    assert destination.read_text(encoding="utf-8") == "captured patch"
    assert not os.path.samefile(external, destination)


@pytest.mark.parametrize(
    ("writer", "payload"),
    [
        (exclusive_write_text, "immutable text"),
        (exclusive_write_bytes, b"immutable bytes"),
    ],
)
def test_exclusive_artifact_creation_rejects_existing_hardlinks(
    tmp_path: Path,
    writer,
    payload,
) -> None:
    external = tmp_path / "external"
    external.write_bytes(b"external evidence")
    destination = tmp_path / "artifact"
    os.link(external, destination)

    with pytest.raises(PersistenceError, match="exclusively create"):
        writer(destination, payload)

    assert external.read_bytes() == b"external evidence"
    assert os.path.samefile(external, destination)


@pytest.mark.parametrize("status", list(VerificationStatus))
def test_verification_evidence_decodes_status_as_the_owning_enum(
    tmp_path: Path,
    status: VerificationStatus,
) -> None:
    record = start_attempt(
        tmp_path,
        phase=AttemptPhase.VERIFYING,
        before_workspace_fingerprint="before",
    )
    atomic_write_json(
        attempt_result_path(tmp_path, record),
        {
            "schema_version": VERIFICATION_SCHEMA_VERSION,
            "format": VERIFICATION_ROUND_FORMAT,
            "round_index": 0,
            "started_at": "2026-09-24T10:00:00Z",
            "ended_at": "2026-09-24T10:00:00Z",
            "duration_seconds": 0,
            "status": status.value,
            "commands": [],
        },
    )

    evidence = read_verification_evidence(tmp_path, record)

    assert evidence.status is status


def test_verification_evidence_rejects_an_unknown_status_token(tmp_path: Path) -> None:
    record = start_attempt(
        tmp_path,
        phase=AttemptPhase.VERIFYING,
        before_workspace_fingerprint="before",
    )
    atomic_write_json(
        attempt_result_path(tmp_path, record),
        {
            "schema_version": VERIFICATION_SCHEMA_VERSION,
            "format": VERIFICATION_ROUND_FORMAT,
            "round_index": 0,
            "started_at": "2026-09-24T10:00:00Z",
            "ended_at": "2026-09-24T10:00:00Z",
            "duration_seconds": 0,
            "status": "UNKNOWN",
            "commands": [],
        },
    )

    with pytest.raises(ValueError, match="unsupported status"):
        read_verification_evidence(tmp_path, record)

    with pytest.raises(TypeError, match="VerificationStatus"):
        VerificationEvidence(
            round_index=0,
            status="PASS",  # type: ignore[arg-type]
            started_at="2026-09-24T10:00:00Z",
            ended_at="2026-09-24T10:00:00Z",
            duration_seconds=0,
            command_count=0,
        )
