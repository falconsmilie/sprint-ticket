from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, format_config_summary, load_config
from .implementation import (
    ImplementationError,
    format_implementation_result,
    run_implementation_stage,
)
from .preflight import format_preflight_result, run_preflight
from .runs import (
    RUNS_DIR_NAME,
    RunError,
    RunPreflightError,
    TicketInputError,
    create_run_snapshot,
    format_status,
    list_run_records,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ticket_automation",
        description="Coordinate one ticket-automation run at a time.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing config.example.toml and optional config.local.toml.",
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    config_parser = subparsers.add_parser(
        "config",
        help="Load, validate, and summarize the effective configuration.",
    )
    config_parser.set_defaults(handler=_handle_config)

    preflight_parser = subparsers.add_parser(
        "preflight",
        help="Inspect the configured repository and reject unsafe starting states.",
    )
    preflight_parser.set_defaults(handler=_handle_preflight)

    run_parser = subparsers.add_parser(
        "run",
        help="Create a ticket snapshot, run implementation, and stop before verification.",
    )
    run_parser.add_argument(
        "ticket",
        type=Path,
        help="Local Markdown ticket file to copy into the run directory.",
    )
    run_parser.set_defaults(handler=_handle_run)

    status_parser = subparsers.add_parser(
        "status",
        help="List known ticket automation runs.",
    )
    status_parser.set_defaults(handler=_handle_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except ConfigError as error:
        parser.exit(status=2, message=f"Configuration error: {error}\n")


def _handle_config(args: argparse.Namespace) -> int:
    config = load_config(args.config_dir)
    print(format_config_summary(config))
    return 0


def _handle_preflight(args: argparse.Namespace) -> int:
    config = load_config(args.config_dir)
    result = run_preflight(config)
    print(format_preflight_result(result))
    return 0 if result.passed else 1


def _handle_run(args: argparse.Namespace) -> int:
    config = load_config(args.config_dir)
    runs_dir = args.config_dir / RUNS_DIR_NAME
    try:
        snapshot = create_run_snapshot(config, args.ticket, runs_dir=runs_dir)
        implementation = run_implementation_stage(config, snapshot.run_dir)
    except TicketInputError as error:
        print(f"Ticket input error: {error}", file=sys.stderr)
        return 2
    except RunPreflightError as error:
        print(format_preflight_result(error.result))
        return 1
    except ImplementationError as error:
        print(f"Implementation error: {error}", file=sys.stderr)
        return 1
    except RunError as error:
        print(f"Run error: {error}", file=sys.stderr)
        return 1

    print(
        "\n".join(
            [
                f"Snapshot created for run {snapshot.run_record.run_id}.",
                f"Run directory: {snapshot.run_dir}",
                format_implementation_result(implementation),
                "No verification or review has been attempted.",
            ]
        )
    )
    return 0 if implementation.successful else 1


def _handle_status(args: argparse.Namespace) -> int:
    runs_dir = args.config_dir / RUNS_DIR_NAME
    print(format_status(list_run_records(runs_dir), runs_dir=runs_dir))
    return 0
