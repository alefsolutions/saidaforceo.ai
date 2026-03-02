from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

try:
    from pgvector.sqlalchemy import Vector
except Exception:  # pragma: no cover - optional dependency in some environments
    Vector = None

DEFAULT_EMBEDDING_DIMENSIONS = 1536


class Base(DeclarativeBase):
    pass


class VectorCompat(TypeDecorator):
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql" and Vector is not None:
            return dialect.type_descriptor(Vector(DEFAULT_EMBEDDING_DIMENSIONS))
        return dialect.type_descriptor(JSON())


class DatasetRow(Base):
    __tablename__ = "datasets"

    dataset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_connector: Mapped[str] = mapped_column(String(64), nullable=False)
    source_resource_id: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    parquet_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    text_summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    embedding: Mapped["SemanticEmbeddingRow"] = relationship(back_populates="dataset", uselist=False)


class ResourceHashRow(Base):
    __tablename__ = "resource_hashes"

    resource_key: Mapped[str] = mapped_column(Text, primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class BenchmarkReportRow(Base):
    __tablename__ = "benchmark_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    suite_name: Mapped[str] = mapped_column(String(128), nullable=False, default="default")
    suite_version: Mapped[str] = mapped_column(String(64), nullable=False, default="v1")
    dataset_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SemanticEmbeddingRow(Base):
    __tablename__ = "semantic_embeddings"

    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.dataset_id", ondelete="CASCADE"), primary_key=True)
    embedding_json: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    embedding_vector: Mapped[list[float] | None] = mapped_column(VectorCompat(), nullable=True)
    embedding_norm: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    dataset: Mapped[DatasetRow] = relationship(back_populates="embedding")


class ExecutionLogRow(Base):
    __tablename__ = "execution_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    route: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    sql_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    analytics_row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retrieved_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    traces: Mapped[list["QueryTraceRow"]] = relationship(back_populates="execution_log", cascade="all, delete-orphan")


class QueryTraceRow(Base):
    __tablename__ = "query_traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    execution_log_id: Mapped[int] = mapped_column(ForeignKey("execution_logs.id", ondelete="CASCADE"), nullable=False)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    step_name: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    input_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    output_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    execution_log: Mapped[ExecutionLogRow] = relationship(back_populates="traces")


class BenchmarkLineageRow(Base):
    __tablename__ = "benchmark_lineage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    benchmark_report_id: Mapped[int] = mapped_column(ForeignKey("benchmark_reports.id", ondelete="CASCADE"), nullable=False)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    suite_name: Mapped[str] = mapped_column(String(128), nullable=False)
    suite_version: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    case_name: Mapped[str] = mapped_column(String(128), nullable=False)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    analytics_ok: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    semantic_ok: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    reasoning_ok: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
