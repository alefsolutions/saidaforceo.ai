"""Minimal CLI entry point."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from saida import Saida
from saida.adapters import CSVAdapter
from saida.schemas import AnalysisResult, DatasetProfile


def build_parser() -> argparse.ArgumentParser:
    """Create the SAIDA CLI argument parser."""
    parser = argparse.ArgumentParser(description="SAIDA library CLI")
    subparsers = parser.add_subparsers(dest="command")

    version_parser = subparsers.add_parser("version", help="Show the SAIDA CLI version.")
    version_parser.set_defaults(command="version")

    profile_parser = subparsers.add_parser("profile", help="Profile a CSV dataset.")
    profile_parser.add_argument("--csv", required=True, help="Path to the CSV file.")
    profile_parser.add_argument("--context", help="Optional path to a markdown context file.")
    profile_parser.add_argument("--json", action="store_true", help="Print the profile as JSON.")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a CSV dataset with a question.")
    analyze_parser.add_argument("--csv", required=True, help="Path to the CSV file.")
    analyze_parser.add_argument("--question", required=True, help="Question to analyze.")
    analyze_parser.add_argument("--context", help="Optional path to a markdown context file.")
    analyze_parser.add_argument("--json", action="store_true", help="Print the analysis result as JSON.")
    analyze_parser.add_argument("--show-plan", action="store_true", help="Print plan steps after the summary.")
    analyze_parser.add_argument("--show-trace", action="store_true", help="Print execution trace events after the summary.")

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
        if args.json:
            print(json.dumps(_profile_payload(profile), indent=2))
        else:
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
        if args.json:
            print(json.dumps(_analysis_payload(result), indent=2))
        else:
            print(result.summary)
            print("Tables:", ", ".join(table.name for table in result.tables))
            if args.show_plan:
                print("Plan:")
                for step in result.plan.steps:
                    print(f"- {step.step_id}: {step.description}")
            if args.show_trace:
                print("Trace:")
                for event in result.trace:
                    print(f"- {event.stage}: {event.message}")
            if result.warnings:
                print("Warnings:", "; ".join(result.warnings))
        return 0

    parser.print_help()
    return 1


def _load_csv_dataset(csv_path: str, context_path: str | None) -> object:
    """Load a CSV dataset for CLI commands."""
    csv_adapter = CSVAdapter(Path(csv_path), context_path=Path(context_path) if context_path else None)
    return csv_adapter.load()


def _profile_payload(profile: DatasetProfile) -> dict[str, object]:
    """Convert a dataset profile into a small JSON-safe payload."""
    return {
        "dataset_name": profile.dataset_name,
        "row_count": profile.row_count,
        "column_count": profile.column_count,
        "measure_columns": list(profile.measure_columns),
        "dimension_columns": list(profile.dimension_columns),
        "time_columns": list(profile.time_columns),
        "identifier_columns": list(profile.identifier_columns),
        "warnings": list(profile.warnings),
    }


def _analysis_payload(result: AnalysisResult) -> dict[str, object]:
    """Convert an analysis result into a small JSON-safe payload."""
    return {
        "summary": result.summary,
        "warnings": list(result.warnings),
        "metrics": [asdict(metric) for metric in result.metrics],
        "tables": [
            {
                "name": table.name,
                "description": table.description,
                "rows": int(len(table.dataframe)),
                "columns": list(table.dataframe.columns),
            }
            for table in result.tables
        ],
        "plan": {
            "task_type": result.plan.task_type,
            "rationale": result.plan.rationale,
            "steps": [asdict(step) for step in result.plan.steps],
        },
        "trace": [asdict(event) for event in result.trace],
    }
