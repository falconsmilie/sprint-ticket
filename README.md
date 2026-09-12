# TicketAutomation

TicketAutomation is a standalone Python project for coordinating automation around implementation tickets. It is independent of the repositories it works on: target projects are configured through local settings and are not part of this package.

V1 is scoped to one implementation ticket at a time. TicketAutomation now runs the complete mechanical loop for one ticket: preflight, snapshot, implementation, deterministic verification, independent review, bounded correction, reverification, fresh rereview, final reporting, and audit handoff until the run reaches `READY_FOR_HUMAN`, `HUMAN_REQUIRED`, or `FAILED`.

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

Create a persistent run record from a local Markdown ticket and run the complete V1 mechanical lifecycle:

```powershell
python -m ticket_automation run tickets/example.md
```

The `run` command performs preflight, copies the ticket verbatim into `runs/<run-id>/ticket.md`, records the starting branch and baseline HEAD SHA, invokes Codex with a `workspace-write` sandbox for implementation, runs all configured deterministic verification commands, and invokes a fresh Codex reviewer with a `read-only` sandbox when verification passes. A passing review transitions through `REPORT`, writes `diffs/final.patch` and `final-report.md`, then reaches `READY_FOR_HUMAN` only if final verification, review, and Git safety evidence still hold.

List known run records:

```powershell
python -m ticket_automation status
```

Resume a non-terminal run from an explicitly safe persisted checkpoint:

```powershell
python -m ticket_automation resume <run-id>
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

Each run directory contains `run.json`, `ticket.md`, and `baseline.json`. These records are enough to reconstruct the original ticket boundary for later workflow stages: the original ticket path, the copied ticket path, the target repository path, the starting branch, the baseline HEAD SHA, the current state, the last completed state, correction and review round counters, timestamps, and a terminal reason when the run reaches `HUMAN_REQUIRED` or `FAILED`.

A successful run also contains `diffs/final.patch` and `final-report.md`. The final patch is always relative to the original baseline SHA and the complete current working tree. The final report separates controller-observed facts from implementation-agent claims, so runner verification and Git evidence are not confused with agent-reported targeted tests.

## Mechanical V1 Loop

`python -m ticket_automation run tickets/example.md` advances the ticket without manual stage commands:

```text
PREFLIGHT
  -> SNAPSHOT
  -> IMPLEMENT
  -> VERIFY
  -> REVIEW
  -> REPORT
  -> READY_FOR_HUMAN
```

If deterministic verification fails, TicketAutomation skips review and generates a corrective ticket directly from typed `VerificationFailure` records. If review returns `CORRECTIONS_REQUIRED`, TicketAutomation generates a corrective ticket from the reviewer's `REQUIRED` findings. Every correction is followed by deterministic verification. If verification passes, the next review is a completely fresh review of the original ticket against the original baseline and the complete current working tree.

Verification-driven and review-driven corrections share the same `[runner].max_correction_rounds` limit. With the default value of `3`, TicketAutomation allows implementation, review 1, correction 1, review 2, correction 2, review 3, correction 3, and review 4. If review 4 still requires correction, the run becomes `HUMAN_REQUIRED`; correction round 4 is not started.

Python owns all orchestration decisions. Codex may edit source files during writable implementation or correction phases, and Codex may return structured implementation and review judgments. TicketAutomation decides transitions from typed state only: implementation status, deterministic verification results, review verdict, correction count, and Git safety checks. It does not infer acceptance or correction needs from prose, stdout, reviewer narrative, or summary wording.

## Final Reporting And Resume

`REPORT` is a read-only target-repository phase. It gathers already persisted evidence, captures the final Git state, writes `runs/<run-id>/diffs/final.patch`, writes `runs/<run-id>/final-report.md`, and prints a concise terminal handoff. It does not stage files, commit, switch branches, or alter target source code.

`READY_FOR_HUMAN` is entered only when deterministic verification currently passes, the final independent review verdict is `PASS`, the current source diff still matches the last verified writable checkpoint, Git safety invariants still hold, and both final report artifacts were persisted. `HUMAN_REQUIRED` and `FAILED` runs get a best-effort report where enough state exists.

`resume <run-id>` resumes only from explicit checkpoints recorded in `run.json` and the run artifacts. It does not infer safety from a run directory alone. If implementation or correction was interrupted after a writable phase was marked active but before completion was proven, resume stops at `HUMAN_REQUIRED` and explains that the working tree may contain partial modifications.

Read-only and deterministic checkpoints are recoverable when the repository still matches the recorded baseline and writable checkpoint. If deterministic verification finished writing a complete round artifact but `run.json` was not advanced, resume validates that artifact against the current run, round, repository state, and configured verification gates before adopting it. If a read-only review finished writing a valid `result.json` but `run.json` was not advanced, resume validates the saved prompt, controller checkpoint metadata, result schema, and repository invariants before adopting it. Partial verification or review artifacts are preserved under recovery artifact directories and the safe stage is rerun when the repository still matches the checkpoint. If repository contents changed after the evidence was written, resume does not reuse that evidence and stops for human inspection.

## Implementation Stage

The implementation stage renders `prompts/implement.md` with the complete snapshotted ticket, then runs a fresh Codex process against the target repository using the `workspace-write` sandbox and `schemas/implementation-result.schema.json`. The schema accepts only `COMPLETED` or `BLOCKED` agent statuses plus a concise summary, tests run, assumptions, and known issues. It does not ask Codex to report changed files because Git remains the source of truth.

After Codex exits, TicketAutomation independently inspects Git. The current branch must still match the starting branch, `HEAD` must still match the baseline SHA, and the staging area must be empty. If any of those invariants are violated, the run becomes `HUMAN_REQUIRED` and TicketAutomation does not undo the mutation. A `BLOCKED` implementation result also becomes `HUMAN_REQUIRED` with the agent result preserved in `implementation/result.json`. If Codex reports `COMPLETED` without any worktree changes for an implementation ticket, the run becomes `HUMAN_REQUIRED`. Codex execution failures become `FAILED`.

When implementation completes and the safety checks pass, TicketAutomation captures `runs/<run-id>/diffs/after-implementation.patch` and `runs/<run-id>/diffs/after-implementation.stat` from Git.

## Verification

TicketAutomation has two validation levels. Implementation-agent targeted validation is whatever the writable agent chose to run while doing the work. Those claims are kept in `runs/<run-id>/implementation/result.json` as implementation feedback.

Runner deterministic acceptance gates are the configured `[[verification.commands]]` records. TicketAutomation runs those commands itself from the target repository directory, preserves stdout and stderr, and treats those results as authoritative for workflow acceptance. Commands are executed from argument arrays without a shell.

The first verification attempt writes `runs/<run-id>/verification/round-0.json` and `runs/<run-id>/verification/round-0.log`. Later correction rounds use `round-1`, `round-2`, and so on. A complete verification round includes checkpoint metadata tying it to the run, round, baseline, branch, source state, and configured verification gates that were used. Passing gates move the run to review. Failing gates move the run to `CORRECT` and produce typed `VerificationFailure` correction reasons. Environment or process execution problems, such as a missing executable or timeout, move the run to `HUMAN_REQUIRED` instead of asking a correction agent to rewrite source code for a broken local environment.

## Review

Review deliberately uses a separate Codex invocation from implementation. The reviewer receives `prompts/review.md`, the complete snapshotted ticket, baseline SHA, starting branch, current branch, deterministic verification results, and the implementation summary where available. The prompt instructs the reviewer to inspect the complete current working tree relative to the original baseline and to respect repository authority such as AGENTS.md, ADRs, contracts, validation rules, provenance rules, tests, documentation, and established implementation patterns.

The structured result must match `schemas/review-result.schema.json` with one of `PASS`, `CORRECTIONS_REQUIRED`, or `HUMAN_REVIEW_REQUIRED`. `PASS` may include advisory or follow-up findings, but it must not include `REQUIRED` findings. `CORRECTIONS_REQUIRED` must include at least one `REQUIRED` finding. A schema-valid but contradictory result, such as `PASS` with a required finding, moves the run to `HUMAN_REQUIRED` without manufacturing a corrected verdict.

Each review writes `prompt.md`, `events.jsonl`, `stderr.log`, and `result.json` under `runs/<run-id>/reviews/round-<n>/`. Review round 1 is the first independent review after implementation, even if deterministic verification drove a correction before any review could run. After a review-driven correction, the next passing verification leads to the next review round. After the reviewer exits, TicketAutomation independently checks that the branch still matches the recorded starting branch, `HEAD` still equals the baseline SHA, and the staging area is empty. If any invariant changed, the run becomes `HUMAN_REQUIRED` and TicketAutomation does not reset or repair the repository.

If a process stops after a review result is written but before the run record advances, resume may adopt the saved review instead of invoking a new reviewer. Adoption requires the saved controller checkpoint and prompt to match the current review checkpoint, the result to satisfy the review schema, and the repository to still match the verified source state. Partial read-only review artifacts are archived and a fresh review is run when those safety checks still pass.

A `PASS` review may include `ADVISORY` or `FOLLOW_UP` findings and still reaches `READY_FOR_HUMAN`. `CORRECTIONS_REQUIRED` must include at least one `REQUIRED` finding and enters the bounded correction loop. `HUMAN_REVIEW_REQUIRED` stops at `HUMAN_REQUIRED`.

## Corrections

Correction tickets are generated mechanically from structured runner data. TicketAutomation does not ask another LLM to author `runs/<run-id>/corrections/<ticket-id>-CORR-R<round>.md`; it renders Markdown directly from either deterministic `VerificationFailure` records or independent `ReviewFinding` records. Verification failures keep their command, exit code, useful output excerpts, and a reference to the persisted verification log. Review findings keep the reviewer-authored severity, category, finding text, evidence, required change, and acceptance criteria.

The two correction sources remain distinct. A failed deterministic gate is not converted into a reviewer finding, and a reviewer finding is eligible only when its disposition is `REQUIRED`. `ADVISORY` and `FOLLOW_UP` findings are retained in review artifacts but are excluded from automated corrective work because they do not block acceptance for the current ticket.

Each correction round uses a fresh Codex invocation with the `workspace-write` sandbox and stores artifacts under `runs/<run-id>/correction-executions/round-<round>/`: `prompt.md`, `events.jsonl`, `stderr.log`, and `result.json`. The correction prompt includes the complete original ticket, the generated corrective ticket, and current repository context. It explicitly tells the correction agent that existing uncommitted changes are the original implementation and must be preserved unless the corrective ticket identifies a defect.

After every writable correction invocation, TicketAutomation independently verifies that the branch still matches the starting branch, `HEAD` still equals the baseline SHA, and the staging area is empty. If any invariant is violated, the run becomes `HUMAN_REQUIRED` and TicketAutomation does not reset or repair the repository. A correction result of `BLOCKED` also becomes `HUMAN_REQUIRED` with the agent explanation preserved.

Completed corrections capture `runs/<run-id>/diffs/after-correction-<round>.patch`. That patch is always the full diff from the original baseline SHA to the complete current working tree, including valid existing implementation work, rather than only the incremental correction delta. The next step is always deterministic verification before any further review.

## V1 Safety Boundary

V1 does not create branches, switch branches, stage files, commit, amend commits, reset, stash, clean, push, merge, retrieve GitHub issues, update GitHub issues, orchestrate multiple tickets, orchestrate epics, manage releases, create pull requests, or perform automatic acceptance.

After `READY_FOR_HUMAN`, the user still inspects the final diff, accepts or rejects the implementation, commits manually if accepted, and selects the next ticket. TicketAutomation is an audit-producing assistant, not the final authority.
