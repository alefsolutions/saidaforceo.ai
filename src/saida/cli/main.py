"""Minimal CLI entry point."""

from __future__ import annotations

import argparse


def main() -> int:
    """Run the SAIDA CLI."""
    parser = argparse.ArgumentParser(description="SAIDA library CLI")
    parser.add_argument("--version", action="store_true", help="Show the SAIDA CLI version.")
    args = parser.parse_args()
    if args.version:
        print("SAIDA CLI 0.1.0")
        return 0
    parser.print_help()
    return 0
