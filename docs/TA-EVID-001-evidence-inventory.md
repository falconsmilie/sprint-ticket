# TA-EVID-001 evidence inventory

This inventory records the persistence boundaries inspected before the evidence
codec change. It distinguishes records that can influence workflow or reporting
from adapter diagnostics that remain opaque to the application.

## Authoritative records

| Record | Owner and codec | Version/format | Mutable write policy |
| --- | --- | --- | --- |
| `run.json` | `ticket_automation.runs.RunRecord` | run schema 5 / `ticket_automation.run` | shared atomic JSON writer |
| `baseline.json` | `ticket_automation.runs.BaselineRecord` | baseline schema 2 / `ticket_automation.baseline` | shared atomic JSON writer |
| `attempts/<attempt>/attempt.json` | `ticket_automation.attempts.AttemptRecord` | attempt schema 1 / `ticket_automation.attempt` | shared atomic JSON writer |
| `attempts/<attempt>/result.json` | task-result, stage-message, or verification codec selected by the phase role | verification schema 1 / `ticket_automation.verification_round` where applicable | shared atomic JSON writer |
| `attempts/<attempt>/execution.json` | `ticket_automation.execution_evidence.ExecutionEvidence` | execution-evidence schema 1 / `ticket_automation.execution_evidence` | shared atomic JSON writer |
| `attempts/<attempt>/workspace-guard.json` | `WorkspaceGuardInspection` | workspace-guard schema 1 / `ticket_automation.workspace_environment_guard` | shared atomic JSON writer |

Repository lock metadata is operational coordination state rather than run
evidence. It now uses the same atomic JSON primitive because it is mutable and
must not expose partial JSON.

## Provider-owned diagnostics

The Codex adapter owns `prompt.md`, `events.jsonl`, `stderr.log`,
`codex-execution.json`, and `codex-result.json`. Their filenames remain local to
the adapter. The neutral envelope refers to them by typed role and validated
run-relative path. Application and presentation code do not parse these files.

`codex-result.json` is adapter input to the provider-specific result decoder.
After decoding, the lifecycle persists the provider-neutral task result in the
attempt's authoritative `result.json`.

## Consolidated boundaries

`ticket_automation.persistence` owns UTF-8 JSON encoding, one terminal newline,
temporary-file creation, file flush and `fsync`, atomic replacement, timestamp
formatting/parsing, and controlled persistence/codec errors. JSON-compatible
dictionaries remain inside record codecs, configuration parsing, hashing
canonicalization, and opaque provider metadata.

`AttemptArtifactLayout` owns conversion between trusted run/attempt roots and
artifact references. It rejects absolute paths, traversal, paths outside the
attempt, and missing files when existence is required. Provider-neutral artifact
roles and lifecycle artifact filenames are defined with the execution contract;
provider-specific filenames are defined once in the provider evidence module.

Reporting now builds `ReportViewModel`, `ControllerReportView`,
`AgentReportView`, and `AttemptReportView` instances. It reads neutral execution
envelopes and typed task/verification results and remains independent of native
provider events and stderr.
