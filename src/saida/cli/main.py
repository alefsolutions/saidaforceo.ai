"""Minimal CLI entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from saida import Saida
from saida.adapters import CSVAdapter


def build_parser() -> argparse.ArgumentParser:
    """Create the SAIDA CLI argument parser."""
    parser = argparse.ArgumentParser(description="SAIDA library CLI")
    subparsers = parser.add_subparsers(dest="command")

    version_parser = subparsers.add_parser("version", help="Show the SAIDA CLI version.")
    version_parser.set_defaults(command="version")

    profile_parser = subparsers.add_parser("profile", help="Profile a CSV dataset.")
    profile_parser.add_argument("--csv", required=True, help="Path to the CSV file.")
    profile_parser.add_argument("--context", help="Optional path to a markdown context file.")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a CSV dataset with a question.")
    analyze_parser.add_argument("--csv", required=True, help="Path to the CSV file.")
    analyze_parser.add_argument("--question", required=True, help="Question to analyze.")
    analyze_parser.add_argument("--context", help="Optional path to a markdown context file.")

    return parser


def main() -> int:
    """Run the SAIDA CLI."""
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "version":
        print("SAIDA CLI 0.1.0")
        return 0

    if args.command == "profile":
        dataset = _load_csv_dataset(args.csv, args.context)
        profile = Saida().profile(dataset)
        print(f"Dataset: {profile.dataset_name}")
        print(f"Rows: {profile.row_count}")
        print(f"Columns: {profile.column_count}")
        print(f"Measures: {', '.join(profile.measure_columns) or 'none'}")
        print(f"Dimensions: {', '.join(profile.dimension_columns) or 'none'}")
        print(f"Time columns: {', '.join(profile.time_columns) or 'none'}")
        return 0

    if args.command == "analyze":
        dataset = _load_csv_dataset(args.csv, args.context)
        result = Saida().analyze(dataset, args.question)
        print(result.summary)
        print("Tables:", ", ".join(table.name for table in result.tables))
        if result.warnings:
            print("Warnings:", "; ".join(result.warnings))
        return 0

    parser.print_help()
    return 1


def _load_csv_dataset(csv_path: str, context_path: str | None) -> object:
    """Load a CSV dataset for CLI commands."""
    csv_adapter = CSVAdapter(Path(csv_path), context_path=Path(context_path) if context_path else None)
    return csv_adapter.load()
