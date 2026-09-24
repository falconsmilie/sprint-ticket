# Release notes

## Provider-neutral architecture

This release completes the provider-boundary migration. Configuration assigns
each agent task to a registered provider under `agents`, while application and
domain code depend only on provider-neutral contracts. Codex CLI settings,
transport, diagnostics, and native artifacts are owned by
`ticket_automation.providers.codex_cli`.

The prior application-facing Codex facade, settings aliases, execution bridges,
presentation compatibility exports, and architecture debt allowlist have been
removed. Runs use resolved-policy schema 2 and run-record schema 5; records with
other schema versions are unsupported and are not modified or deleted.

Prompts and result schemas ship inside the `ticket_automation` package. The
scripted provider and provider contract fixtures remain test-only support and
are not registered by the production composition root.
