# TicketAutomation

TicketAutomation is a synchronous local controller for one implementation
ticket at a time. It asks Codex to implement, verify, review, and, by default,
make one corrective pass. The target repository remains under human Git
control: TicketAutomation never stages, commits, switches branches, or cleans
up target-worktree changes.

## Run model

Each run has an immutable resolved configuration, a snapshotted ticket, and a
clean Git baseline. The controller drives this explicit state machine:

```text
PREPARING -> PREPARED -> IMPLEMENTING -> VERIFYING -> REVIEWING -> REPORTING
                                                    |              |
                                                    v              v
                                           CORRECTION_PENDING   READY_FOR_HUMAN
                                                    |
                                                    v
                                               CORRECTING
```

Verification or review can request correction. Any safety failure, failed
infrastructure call, or finding outside the configured automatic-correction
scope ends in `HUMAN_REQUIRED`.

Before the final transition, the controller captures the canonical
`WorkspaceSnapshot` and checks that branch, HEAD, staging, the latest writable
attempt fingerprint, deterministic verification, and review all agree. It
captures `final.patch`, rechecks that the workspace did not change during that
capture, then transitions to `READY_FOR_HUMAN` or `HUMAN_REQUIRED`.
`report.md` only renders those records; it does not make acceptance decisions.

## Configuration

Copy `config.example.toml` to `config.local.toml` and set the target repository
and exact verification commands. Verification is run in the target repository,
so commands must explicitly select that project's intended environment and
tooling. For example:

```toml
[[verification.commands]]
name = "tests"
argv = ["C:/Projects/my-target/.venv/Scripts/python.exe", "-m", "pytest"]
timeout_seconds = 1800
```

TicketAutomation does not install or own pytest, ruff, pyright, or any other
target-project verification tool. Its own development tools are in the `dev`
dependency group.

`runner.max_correction_rounds` defaults to `1`. The resolved configuration is
persisted in `run.json`, so changing `config.local.toml` cannot change a run
that already exists. Sandboxes are fixed by phase: implementation and
correction use `workspace-write`; verification, review, and reporting are
read-only with respect to the target project.

## Artifacts and attempts

Runs are append-only evidence directories:

```text
runs/<run-id>/
  run.json
  ticket.md
  baseline.json
  attempts/
    001-preparation/
      attempt.json
      result.json
    002-implementation/
      attempt.json
      prompt.md
      events.jsonl
      stderr.log
      execution.json
      result.json
    003-verification/
      attempt.json
      result.json
  final.patch
  report.md
```

Attempt directory numbers are monotonically increasing. `attempt.json` records
the phase, status, before/after workspace fingerprints, process-start status,
timestamps, and result/execution paths. Codex prompts, raw events, stderr, and
typed results live with the attempt that produced them. Verification stdout and
stderr live once in its typed result; there is no parallel human log. The only
human-readable patch is the final `final.patch`.

Workspace-guard evidence is recorded in the writable execution metadata.
There are no per-round patch or `.stat` copies, duplicate Codex result files,
or automatic Git cleanup.

## Resume behavior

Resume never adopts a completed artifact whose controller state transition may
have been interrupted. It allocates a new attempt for safe work.

`PREPARING`, `VERIFYING`, `REVIEWING`, and `REPORTING` can run again when the
current workspace fingerprint matches the persisted expected fingerprint.
Incomplete attempts remain diagnostic history. `IMPLEMENTING` and `CORRECTING`
are conservative: an interruption always moves the run to `HUMAN_REQUIRED` so
a person can inspect the target workspace before deciding what to do.

Attempt records are trusted only when their sequence, phase, status, paths, and
directory agree. Invalid attempt evidence stops resume for human inspection;
the controller never follows an artifact path outside its own attempt.

## Commands

```text
ticket-automation preflight
ticket-automation run tickets/TA-ARCH-009.md
ticket-automation resume <run-id>
ticket-automation status
```

The controller requires a clean target worktree and an empty staging area at
run creation. Baseline verification runs before the first writable Codex call.
