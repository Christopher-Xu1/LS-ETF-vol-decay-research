"""Command-line entrypoint for leveraged decay research workflows."""

from __future__ import annotations

import argparse

from src.experiments.run_baseline import build as build_cmd
from src.experiments.run_baseline import fetch as fetch_cmd
from src.experiments.run_baseline import run as backtest_cmd
from src.experiments.run_regimes import run as regimes_cmd
from src.experiments.run_stress import run as stress_cmd
from src.experiments.run_sweeps import run as sweep_cmd


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leveraged decay research CLI")
    parser.add_argument("command", choices=["fetch", "build", "backtest", "regimes", "sweep", "stress", "report"])
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    if args.command == "fetch":
        fetch_cmd(args.config)
        return

    if args.command == "build":
        build_cmd(args.config)
        return

    if args.command == "backtest":
        backtest_cmd(args.config)
        return

    if args.command == "regimes":
        regimes_cmd(args.config)
        return

    if args.command == "sweep":
        sweep_cmd(args.config)
        return

    if args.command == "stress":
        stress_cmd(args.config)
        return

    if args.command == "report":
        backtest_cmd(args.config)
        regimes_cmd(args.config)
        stress_cmd(args.config)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
