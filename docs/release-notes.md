# Release notes

## Provider-neutral architecture

This release completes the provider-boundary migration. Configuration assigns
each agent task to a registered provider under `agents`, while application and
domain code depend only on provider-neutral contracts. Codex CLI settings,
transport, diagnostics, and native artifacts are owned by
`ticket_automation.providers.codex_cli`.

The prior application-facing Codex facade, settings aliases, execution bridges,
presentation compatibility exports, and architecture debt allowlist have been
removed. Runs use resolved-policy schema 3 and run-record schema 5; records with
other schema versions are unsupported and are not modified or deleted.

Prompts and result schemas ship inside the `ticket_automation` package. The
scripted provider and provider contract fixtures remain test-only support and
are not registered by the production composition root.

Run resume and status now reject traversal IDs, linked run directories, and run
records whose identity does not match their directory. Attempt-ledger writes are
physically confined to the owning run, and final patch/report publication uses
atomic replacement rather than following a pre-existing file link.
Authoritative lifecycle reads retain the controller's bound run identity as well;
a direct-directory replacement during resume, handoff, review/correction evidence
loading, verification evidence loading, or reporting is rejected before its data
can influence workflow or presentation decisions.

## Configurable agent deadlines and bounded cleanup

Implementation, review, and correction now have independent provider-neutral
deadlines under `[agents.timeouts]`. Omitted values retain the previous
3600-second default; the release example uses 7200, 5400, and 7200 seconds.
Resolved values remain in the existing task execution-policy fields, so policy
schema 3 is unchanged and resumed runs retain their original deadlines.
Deterministic verification command timeouts remain independent.

The Codex runner now drains stdout and stderr concurrently and observes complete
JSONL records against one monotonic launch deadline. A timely terminal success
gets at most five seconds for normal exit and publication of an identical
canonical result. Late completion stays diagnostic. Timeout and post-launch
failure terminate the complete Windows Job Object or POSIX process group, with
one ten-second budget shared by termination, drain, reap, and thread shutdown.
Invocation-start recording is included in the launch deadline; a blocked
recorder cannot keep the contained tree alive. Stdout/stderr use exclusive
attempt-owned staging captures that are promoted only after their workers stop
cleanly, while the canonical result remains in external provider scratch until
validation. Non-UTF-8 result bytes are retained as octet-stream diagnostics.
Provider-native execution evidence is now schema 2 and records deadline,
terminal, finalisation, termination, cleanup, and structured-result decisions.
