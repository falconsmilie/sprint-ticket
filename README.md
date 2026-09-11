# TicketAutomation

TicketAutomation is a standalone Python project for coordinating automation around implementation tickets. It is independent of the repositories it works on: target projects are configured through local settings and are not part of this package.

V1 is scoped to one implementation ticket at a time. The current scaffold establishes configuration, repository preflight, persistent run snapshots, a first writable implementation stage, deterministic verification gates, an independent read-only review stage, a CLI entry point, and shared models that later workflow tickets can build on.

Human control remains explicit. TicketAutomation does not commit, push, change branches, stage files, reset work, stash work, or clean a target repository. The implementation agent may edit the working tree, and TicketAutomation verifies the branch, HEAD, and staging area after that writable boundary. The review agent runs in a separate read-only Codex invocation and must not edit the repository.

## Configuration

Copy the example configuration and edit the local file for your machine:

```powershell
Copy-Item config.example.toml config.local.toml
```

Set `project.repo` in `config.local.toml` to the target repository path on your computer. `config.local.toml` is ignored by Git and overrides values from `config.example.toml`.

## CLI

Show the available commands:

```powershell
python -m ticket_automation --help
```

Load, validate, and summarize the effective configuration:

```powershell
python -m ticket_automation config
```

Inspect the configured target repository before any writable automation runs:

```powershell
python -m ticket_automation preflight
```

Create a persistent run record from a local Markdown ticket, invoke the implementation agent, run the configured verification gates, and then run an independent review if verification passes:

```powershell
python -m ticket_automation run tickets/example.md
```

For this stage, `run` performs preflight, copies the ticket verbatim into `runs/<run-id>/ticket.md`, records the starting branch and baseline HEAD SHA, invokes Codex once with a `workspace-write` sandbox, then runs the configured verification commands. If those deterministic gates pass, TicketAutomation invokes a fresh Codex reviewer with a `read-only` sandbox. Implementation artifacts are stored in `runs/<run-id>/implementation/`, verification artifacts are stored in `runs/<run-id>/verification/`, and review artifacts are stored in `runs/<run-id>/reviews/round-1/`.

List known run records:

```powershell
python -m ticket_automation status
```

The verification commands in `config.example.toml` are examples only. They are not assumed to be the final commands for any target repository. Each command uses an argument array and an explicit timeout:

```toml
[[verification.commands]]
name = "tests"
argv = ["python", "-m", "pytest"]
timeout_seconds = 1800
```

## Repository Preflight

Preflight is the safety gate for target repositories. It checks that `project.repo` exists, is a Git working tree, is on a branch, has no unstaged, untracked, or staged changes, and is not on one of the configured protected branches such as `main` or `master`.

The target repository must start clean. TicketAutomation deliberately does not stash changes, discard files, switch branches, reset history, or perform any automatic Git cleanup. If preflight fails, fix the repository yourself and run preflight again.

## Run Records

Run snapshots are stored under `runs/`. A run ID uses a timestamp plus a sanitized ticket identifier, such as `20260911-130512_QDEB-003`. Existing run directories are not overwritten; a numeric suffix is added if a timestamp collision occurs.

Each run directory contains `run.json`, `ticket.md`, and `baseline.json`. These records are enough to reconstruct the original ticket boundary for later workflow stages: the original ticket path, the copied ticket path, the target repository path, the starting branch, the baseline HEAD SHA, correction round counters, and timestamps.

## Implementation Stage

The implementation stage renders `prompts/implement.md` with the complete snapshotted ticket, then runs a fresh Codex process against the target repository using the `workspace-write` sandbox and `schemas/implementation-result.schema.json`. The schema accepts only `COMPLETED` or `BLOCKED` agent statuses plus a concise summary, tests run, assumptions, and known issues. It does not ask Codex to report changed files because Git remains the source of truth.

After Codex exits, TicketAutomation independently inspects Git. The current branch must still match the starting branch, `HEAD` must still match the baseline SHA, and the staging area must be empty. If any of those invariants are violated, the run becomes `HUMAN_REQUIRED` and TicketAutomation does not undo the mutation. A `BLOCKED` implementation result also becomes `HUMAN_REQUIRED` with the agent result preserved in `implementation/result.json`. If Codex reports `COMPLETED` without any worktree changes for an implementation ticket, the run becomes `HUMAN_REQUIRED`. Codex execution failures become `FAILED`.

When implementation completes and the safety checks pass, TicketAutomation captures `runs/<run-id>/diffs/after-implementation.patch` and `runs/<run-id>/diffs/after-implementation.stat` from Git.

## Verification

TicketAutomation has two validation levels. Implementation-agent targeted validation is whatever the writable agent chose to run while doing the work. Those claims are kept in `runs/<run-id>/implementation/result.json` as implementation feedback.

Runner deterministic acceptance gates are the configured `[[verification.commands]]` records. TicketAutomation runs those commands itself from the target repository directory, preserves stdout and stderr, and treats those results as authoritative for workflow acceptance. Commands are executed from argument arrays without a shell.

The first verification attempt writes `runs/<run-id>/verification/round-0.json` and `runs/<run-id>/verification/round-0.log`. Later correction rounds can use `round-1`, `round-2`, and so on. Passing gates move the run to `VERIFY`, which allows review to start. Failing gates move the run to `CORRECT` and produce typed `VerificationFailure` correction reasons. Environment or process execution problems, such as a missing executable or timeout, move the run to `HUMAN_REQUIRED`.

## Review

Review deliberately uses a separate Codex invocation from implementation. The reviewer receives `prompts/review.md`, the complete snapshotted ticket, baseline SHA, starting branch, current branch, deterministic verification results, and the implementation summary where available. The prompt instructs the reviewer to inspect the complete current working tree relative to the original baseline and to respect repository authority such as AGENTS.md, ADRs, contracts, validation rules, provenance rules, tests, documentation, and established implementation patterns.

The structured result must match `schemas/review-result.schema.json` with one of `PASS`, `CORRECTIONS_REQUIRED`, or `HUMAN_REVIEW_REQUIRED`. `PASS` may include advisory or follow-up findings, but it must not include `REQUIRED` findings. `CORRECTIONS_REQUIRED` must include at least one `REQUIRED` finding. A schema-valid but contradictory result, such as `PASS` with a required finding, moves the run to `HUMAN_REQUIRED` without manufacturing a corrected verdict.

Each review writes `prompt.md`, `events.jsonl`, `stderr.log`, and `result.json` under `runs/<run-id>/reviews/round-1/`. After the reviewer exits, TicketAutomation independently checks that the branch still matches the recorded starting branch, `HEAD` still equals the baseline SHA, and the staging area is empty. If any invariant changed, the run becomes `HUMAN_REQUIRED` and TicketAutomation does not reset or repair the repository.

TA-007 is validated with disposable temporary Git repositories and mocked Codex executions. Earlier TicketAutomation development tickets may already have been manually reviewed and committed, so the review mechanism does not try to reconstruct or retroactively review TA-001 through TA-006.
