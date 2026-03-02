from __future__ import annotations

from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from saida.models.types import DatasetAsset
from saida.storage.db import session_scope
from saida.storage.schema import (
    BenchmarkLineageRow,
    BenchmarkReportRow,
    DatasetRow,
    ExecutionLogRow,
    QueryTraceRow,
    ResourceHashRow,
)


class ControlPlaneStore:
    def __init__(self, session_factory: sessionmaker[Session]):
        self.session_factory = session_factory

    def is_unchanged(self, resource_key: str, content_hash: str) -> bool:
        with session_scope(self.session_factory) as session:
            row = session.get(ResourceHashRow, resource_key)
            return bool(row and row.content_hash == content_hash)

    def update_hash(self, resource_key: str, content_hash: str) -> None:
        with session_scope(self.session_factory) as session:
            row = session.get(ResourceHashRow, resource_key)
            if row is None:
                row = ResourceHashRow(resource_key=resource_key, content_hash=content_hash)
                session.add(row)
            else:
                row.content_hash = content_hash

    def upsert_dataset(self, asset: DatasetAsset) -> None:
        with session_scope(self.session_factory) as session:
            row = session.get(DatasetRow, asset.dataset_id)
            if row is None:
                row = DatasetRow(
                    dataset_id=asset.dataset_id,
                    source_connector=asset.source_connector,
                    source_resource_id=asset.source_resource_id,
                    content_hash=asset.hash,
                    kind=asset.kind,
                    parquet_path=asset.parquet_path,
                    text_summary=asset.text_summary,
                    metadata_json=asset.metadata,
                )
                session.add(row)
            else:
                row.source_connector = asset.source_connector
                row.source_resource_id = asset.source_resource_id
                row.content_hash = asset.hash
                row.kind = asset.kind
                row.parquet_path = asset.parquet_path
                row.text_summary = asset.text_summary
                row.metadata_json = asset.metadata

    def list_datasets(self) -> list[DatasetAsset]:
        with session_scope(self.session_factory) as session:
            rows = session.scalars(select(DatasetRow).order_by(DatasetRow.created_at.desc())).all()
            return [
                DatasetAsset(
                    dataset_id=row.dataset_id,
                    source_connector=row.source_connector,
                    source_resource_id=row.source_resource_id,
                    hash=row.content_hash,
                    kind=row.kind,
                    parquet_path=row.parquet_path,
                    text_summary=row.text_summary,
                    metadata=row.metadata_json or {},
                )
                for row in rows
            ]

    def save_benchmark_report(
        self,
        report: dict,
        suite_name: str = "default",
        suite_version: str = "v1",
        dataset_path: str | None = None,
        run_id: str | None = None,
    ) -> tuple[int, str]:
        final_run_id = run_id or uuid4().hex
        with session_scope(self.session_factory) as session:
            row = BenchmarkReportRow(
                run_id=final_run_id,
                suite_name=suite_name,
                suite_version=suite_version,
                dataset_path=dataset_path,
                report_json=report,
            )
            session.add(row)
            session.flush()
            return int(row.id), final_run_id

    def list_benchmark_reports(self) -> list[dict]:
        with session_scope(self.session_factory) as session:
            rows = session.scalars(select(BenchmarkReportRow).order_by(BenchmarkReportRow.created_at.desc())).all()
            return [row.report_json for row in rows]

    def save_benchmark_lineage(
        self,
        report_id: int,
        run_id: str,
        suite_name: str,
        suite_version: str,
        dataset_path: str | None,
        details: list[dict],
    ) -> None:
        with session_scope(self.session_factory) as session:
            for detail in details:
                if "case" not in detail:
                    continue
                session.add(
                    BenchmarkLineageRow(
                        benchmark_report_id=report_id,
                        run_id=run_id,
                        suite_name=suite_name,
                        suite_version=suite_version,
                        dataset_path=dataset_path,
                        case_name=str(detail.get("case")),
                        query_text=str(detail.get("query", "")),
                        analytics_ok=bool(detail.get("analytics_ok", False)),
                        semantic_ok=bool(detail.get("semantic_ok", False)),
                        reasoning_ok=bool(detail.get("reasoning_ok", False)),
                    )
                )

    def log_execution(
        self,
        trace_id: str,
        query_text: str,
        route: str | None,
        status: str,
        sql_text: str | None,
        analytics_row_count: int,
        retrieved_count: int,
        duration_ms: float | None,
        error_text: str | None = None,
        metadata: dict | None = None,
        traces: list[dict] | None = None,
    ) -> None:
        with session_scope(self.session_factory) as session:
            row = ExecutionLogRow(
                trace_id=trace_id,
                query_text=query_text,
                route=route,
                status=status,
                sql_text=sql_text,
                analytics_row_count=analytics_row_count,
                retrieved_count=retrieved_count,
                duration_ms=duration_ms,
                error_text=error_text,
                metadata_json=metadata or {},
            )
            session.add(row)
            session.flush()

            for tr in traces or []:
                session.add(
                    QueryTraceRow(
                        execution_log_id=int(row.id),
                        trace_id=trace_id,
                        step_name=str(tr.get("step_name", "unknown")),
                        status=str(tr.get("status", "unknown")),
                        duration_ms=float(tr["duration_ms"]) if tr.get("duration_ms") is not None else None,
                        input_json=tr.get("input_json", {}) or {},
                        output_json=tr.get("output_json", {}) or {},
                        error_text=tr.get("error_text"),
                    )
                )
