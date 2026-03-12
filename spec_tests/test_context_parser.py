from __future__ import annotations

from pathlib import Path

import pytest

from saida.context import SourceContextParser
from saida.exceptions import ContextError


def test_parse_file_raises_for_missing_context_file(tmp_path: Path) -> None:
    parser = SourceContextParser()

    with pytest.raises(ContextError):
        parser.parse_file(tmp_path / "missing.md")


def test_parse_supports_colon_and_equals_syntax() -> None:
    markdown = """
# Dataset: Sales

## Metrics
revenue = total revenue
profit: net profit

## Field Descriptions
posted_at: posting date
customer_id = customer identifier
""".strip()

    context = SourceContextParser().parse(markdown)

    assert context.metric_definitions["revenue"] == "total revenue"
    assert context.metric_definitions["profit"] == "net profit"
    assert context.field_descriptions["posted_at"] == "posting date"
    assert context.field_descriptions["customer_id"] == "customer identifier"


def test_parse_collects_rule_and_freshness_alias_sections() -> None:
    markdown = """
# Summary
sales data

## Rules
- cancelled orders must be excluded

## Freshness Expectations
- refresh daily
""".strip()

    context = SourceContextParser().parse(markdown)

    assert context.business_rules == ["cancelled orders must be excluded"]
    assert context.freshness_notes == ["refresh daily"]
