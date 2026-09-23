# TA-WRITE-001 writable-flow comparison

This note records the line-by-line comparison performed before extracting the
guarded writable-operation service. Line numbers refer to the files as they
stood before the TA-WRITE-001 implementation.

## Duplicated writable algorithm

| Concern | `implementation.py` | `corrections.py` | Existing shared support | Extraction decision |
|---|---:|---:|---|---|
| Initial snapshot used to seed an attempt | 132-143 | 248-260 | `WorkspaceSnapshot.capture`, `start_attempt` | Remove the duplicate capture. Associate the stage-created attempt with the authoritative snapshot captured by the service. |
| Starting branch, HEAD, staging, and fingerprint checks | 171-187, 545-603 | 308-337, 949-1023 | `workspace_safety_changes` | Express the allowed starting state as immutable baseline policy and validate it in the service. |
| Workspace-write request enforcement | 192-216 | 363-387 | `writable_worker.run_writable_agent` | Keep construction provider-neutral, require executors to advertise their capabilities, and reject an incapable executor before invoking it. |
| Invocation start tracking | 207-216 | 378-387 | `WritableAttempt.mark_process_started`, executor callback | Service-owned. |
| Canonical and environment snapshots around execution | indirect through 207-216 | indirect through 378-387 | `writable_worker.py` 91-164 | Service-owned, including the `finally` post-capture path. |
| Failed invocation audit | 242-266, 620-708 | 416-440, 1026-1108 | `classify_writable_failure`, `WritableAttempt` inspection | Consolidate into the closed service outcome and one common audit value. |
| Baseline-relative tracked and untracked file audit | 360-374, 637-679 | 1056-1086 | `audit.changed_files_including_untracked` | Service-owned for success and failure. |
| Changed/uncertain workspace classification | 251-273, 683-708 | 426-448, 1072-1108 | `classify_writable_failure` | Service-owned. A started or indeterminate changed/uncertain execution is never retryable. |
| Workspace-guard evidence persistence | indirect through 207-216 | indirect through 378-387 | `writable_worker._persist_guard_evidence_if_needed` | Service-owned in the existing attempt directory and layout. |
| Failed writable message details | 711-789 | 1111-1253 | none | Consolidate the identical audit detail formatting. Stage controller messages remain stage-owned. |
| Attempt result artifact and attempt completion | 458-525 | 679-746 | `finish_phase_attempt` | Keep result encoding and stage completion in each stage because they produce distinct stage result objects; consume the service's authoritative post fingerprint instead of taking another snapshot. |

Before extraction, `writable_attempts.py` stored mutable in-memory start and
snapshot evidence and `writable_worker.py` owned only part of the execution
boundary. The stages and `failure_classification.py` then repeated workspace
classification and audit decisions. The extraction removes both helper modules;
the mutable tracker is now private to `GuardedWritableOperation`, and returned
audits contain only immutable evidence values.

Invocation-start evidence is monotonic: once the executor callback reports a
start, a contradictory returned value cannot erase it. Invocation interruptions
and provider contract mismatches are captured as uncertain typed outcomes after
post-execution evidence collection. Snapshot and environment capture remain
owned by the service; they are not replaceable through its public constructor.
Request construction validates only immutable input relationships. Mutable
attempt association is resolved by the service and association failures return
typed safety outcomes. An untrusted or completed attempt association never writes
into the requested artifact directory. Public outcome variants are final,
service-constructed values and validate the audit facts required by their names;
the public audit rejects mutable nested collections. The guard artifact is
persisted before its path is associated with the attempt ledger, so a failed
write cannot leave a dangling authoritative evidence path. Interruptions while
recording ordinary pre/post evidence become safety evidence. The invocation-start
callback still propagates an interruption so the executor can stop and clean up
the started child process before the service returns its typed uncertain outcome.
The Codex adapter uses a private start-aware process-runner path for writable
calls, which records start immediately after successful process creation and
before waiting for completion. The public process-runner signature remains
unchanged. Injected runners without the private start-aware path retain read-only
support and do not advertise workspace-write execution.

## Intentional stage differences retained

- Implementation validates the original clean baseline fingerprint and rejects
  a successful agent result when no repository files changed.
- Correction validates the source fingerprint from the latest deterministic
  verification. Existing implementation changes are expected, so correction
  does not require a clean worktree.
- Correction owns correction causes, round limits, correction ticket rendering,
  and whether the correction round advances.
- Implementation and correction keep their own safety-violation view types and
  controller messages. They map shared evidence into those views.
- Both agent tasks currently return `ImplementationResult`, but each stage
  interprets status and wording in its own domain context.
- Baseline-verification authorization is persisted evidence policy evaluated by
  the stages before an agent call. Those stage-owned rejections are passed to
  `reject_before_start` so the service still captures and persists authoritative
  pre/post evidence. Repository and workspace validation remains service-owned.
- Implementation prepares its prompt before opening the writable attempt.
  Correction preparation happens after its stage-owned authorization checks;
  preparation failures are passed through `reject_before_start` so an opened
  attempt still receives pre/post evidence without invoking the executor.
- Result artifact encoding and final stage-result construction remain in the
  stages because their public result models are intentionally distinct.

## Post-extraction duplication scan

The final scan covered `implementation.py`, `corrections.py`,
`failure_classification.py`, the removed `writable_attempts.py` and
`writable_worker.py` paths, and the new service. It searched for
`WorkspaceSnapshot`, invocation-start callbacks, snapshot/fingerprint
comparison, `classify_writable_failure`, baseline-relative change inspection,
and workspace-environment comparison. Start tracking, safety snapshots,
post-invocation change classification, failed-write audit policy, and guard
evidence persistence now occur only in
`application/guarded_writable_operation.py`. Unexpected outer controller
failures remain conservatively human-required; they do not attempt a second
writable classification without the service's in-memory evidence.

The following similar code intentionally remains in the stages:

- Each stage constructs its own `AgentExecutionRequest` because task kind,
  prompt, and result contract are domain inputs.
- Each stage maps `WritableSafetyViolation` into its public stage violation
  type and renders its own controller message.
- Each stage writes its typed `result.json` view and finishes its distinct
  stage result. It consumes the service's post fingerprint and invocation-start
  classification rather than recapturing or reclassifying them.
- Correction still calls the tracked/untracked change audit while rendering
  correction prompt context. That read is correction-ticket semantics; it does
  not classify writable success or failure.
- `workspace_guard.py` keeps its existing environment scanner and presentation
  helper surface. The service composes its immutable snapshots directly, so the
  ticket does not promote former private helpers into public API.
