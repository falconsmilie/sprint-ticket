# TicketAutomation

TicketAutomation is a standalone Python project for coordinating automation around implementation tickets. It is independent of the repositories it works on: target projects are configured through local settings and are not part of this package.

V1 is scoped to one implementation ticket at a time. The current scaffold establishes configuration, a CLI entry point, and shared models that later workflow tickets can build on.

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

The verification commands in `config.example.toml` are examples only. They are not assumed to be the final commands for any target repository.

