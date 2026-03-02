from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ResourceRecord:
    connector: str
    resource_id: str
    content: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetAsset:
    dataset_id: str
    source_connector: str
    source_resource_id: str
    hash: str
    kind: str
    parquet_path: str | None = None
    text_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryResult:
    query: str
    route: str
    sql: str | None
    analytics_rows: list[dict[str, Any]]
    retrieved_context: list[dict[str, Any]]
    explanation: str
    trace_id: str | None = None


@dataclass(slots=True)
class BenchmarkCase:
    name: str
    query: str
    expected_sql_nonempty: bool = False
    expected_rows_min: int = 0


@dataclass(slots=True)
class IntelligenceScores:
    ais: float
    ses: float
    ris: float
    sss: float
    composite: float


@dataclass(slots=True)
class BenchmarkReport:
    total: int
    passed_analytics: int
    passed_semantic: int
    passed_reasoning: int
    successful_executions: int
    scores: IntelligenceScores
    details: list[dict[str, Any]] = field(default_factory=list)
