# Provider contract testing

Every `AgentExecutor` adapter is registered in the reusable suite in
`tests/provider_contract.py`. The common suite owns provider-neutral assertions for
identity, capabilities, settings and preflight, task/result pairing, repository
access, failures, invocation-start certainty, evidence envelopes, artifacts,
metadata, timeouts, and interruption. Provider test modules own transport details
such as commands, native response parsing, and fake process or API behavior.

The suite derives task coverage from declared capabilities. A provider declaring
`read-only-execution` is tested for review dispatch. A provider declaring
`workspace-write-execution` is tested for implementation and correction dispatch.
Core capabilities are always required. The suite skips only task access that the
provider does not declare.

## Adapter onboarding checklist

- Add a test class that inherits `ProviderContractTests`.
- Supply one `provider_contract` fixture returning `ProviderContractFixture`.
- Register the stable `ProviderId`, registration object, and exact declared
  capability set.
- Provide valid and invalid settings plus an offline, deterministic happy-path
  preflight setup.
- Provide `make_probe(outcome, task_kind)`. Its fake transport must implement every
  neutral `ContractOutcome`, record invocation attempts, invocation starts, and the
  final prompt received by the transport, and require no credentials or network.
- Provide `make_capability_limited_probe(task_kind)` so the suite can prove that a
  missing required capability is rejected before transport invocation.
- Declare the adapter's documented prompt supplement through
  `expected_transport_prompt`; use the identity function when the adapter adds no
  supplement.
- Keep command construction, provider-native parsing, native artifact filenames,
  model settings, and transport edge cases in provider-specific tests.
- Run the inherited contract suite and provider-specific tests, then run the full
  test and compile checks.

The Codex registration is `TestCodexProviderContract` in
`tests/test_codex_provider_contract.py`. Its transport is an in-memory deterministic
runner. It does not launch Codex, use provider credentials, or access the network.
Provider-specific tests additionally exercise live JSONL framing/timing,
structured-message agreement, bounded finalisation, partial diagnostics, and
native process-tree cleanup. Native Windows Job Object and POSIX process-group
regressions use short offline helper processes and must only be skipped when the
corresponding operating system is unavailable.
