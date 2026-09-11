from __future__ import annotations

import argparse
from pathlib import Path

from .config import ConfigError, format_config_summary, load_config


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

