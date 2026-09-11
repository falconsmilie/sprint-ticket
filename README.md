# TicketAutomation

TicketAutomation is a standalone Python project for coordinating automation around implementation tickets. It is independent of the repositories it works on: target projects are configured through local settings and are not part of this package.

V1 is scoped to one implementation ticket at a time. The current scaffold establishes configuration, repository preflight, persistent run snapshots, a CLI entry point, and shared models that later workflow tickets can build on.

Human control remains explicit. TicketAutomation does not commit, push, change branches, or modify a target repository without later workflow code and human direction.

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

Create a persistent run record from a local Markdown ticket:

```powershell
python -m ticket_automation run tickets/example.md
```

For this stage, `run` intentionally stops after preflight and snapshot creation. It copies the ticket verbatim into `runs/<run-id>/ticket.md`, records the starting branch and baseline HEAD SHA, writes `run.json` and `baseline.json`, and exits before any implementation is attempted. Codex is not invoked by this command yet.

List known run records:

```powershell
python -m ticket_automation status
```

The verification commands in `config.example.toml` are examples only. They are not assumed to be the final commands for any target repository.

## Repository Preflight

Preflight is the safety gate for target repositories. It checks that `project.repo` exists, is a Git working tree, is on a branch, has no unstaged, untracked, or staged changes, and is not on one of the configured protected branches such as `main` or `master`.

The target repository must start clean. TicketAutomation deliberately does not stash changes, discard files, switch branches, reset history, or perform any automatic Git cleanup. If preflight fails, fix the repository yourself and run preflight again.

## Run Records

Run snapshots are stored under `runs/`. A run ID uses a timestamp plus a sanitized ticket identifier, such as `20260911-130512_QDEB-003`. Existing run directories are not overwritten; a numeric suffix is added if a timestamp collision occurs.

Each run directory contains `run.json`, `ticket.md`, and `baseline.json`. These records are enough to reconstruct the original ticket boundary for later workflow stages: the original ticket path, the copied ticket path, the target repository path, the starting branch, the baseline HEAD SHA, correction round counters, and timestamps.
