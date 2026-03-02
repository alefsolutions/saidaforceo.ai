"""add observability and audit schema

Revision ID: 0002_observability
Revises: 0001_initial
Create Date: 2026-03-02
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0002_observability"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"

    op.add_column("benchmark_reports", sa.Column("run_id", sa.String(length=64), nullable=True))
    op.add_column("benchmark_reports", sa.Column("suite_name", sa.String(length=128), nullable=True))
    op.add_column("benchmark_reports", sa.Column("suite_version", sa.String(length=64), nullable=True))
    op.add_column("benchmark_reports", sa.Column("dataset_path", sa.Text(), nullable=True))

    op.execute("UPDATE benchmark_reports SET run_id = 'legacy' WHERE run_id IS NULL")
    op.execute("UPDATE benchmark_reports SET suite_name = 'default' WHERE suite_name IS NULL")
    op.execute("UPDATE benchmark_reports SET suite_version = 'v1' WHERE suite_version IS NULL")

    if is_postgres:
        op.alter_column("benchmark_reports", "run_id", nullable=False)
        op.alter_column("benchmark_reports", "suite_name", nullable=False)
        op.alter_column("benchmark_reports", "suite_version", nullable=False)
    op.create_index("ix_benchmark_reports_run_id", "benchmark_reports", ["run_id"])

    op.create_table(
        "execution_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("route", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("sql_text", sa.Text(), nullable=True),
        sa.Column("analytics_row_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("retrieved_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_execution_logs_trace_id", "execution_logs", ["trace_id"])

    op.create_table(
        "query_traces",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("execution_log_id", sa.Integer(), sa.ForeignKey("execution_logs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("step_name", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column("input_json", sa.JSON(), nullable=False),
        sa.Column("output_json", sa.JSON(), nullable=False),
        sa.Column("error_text", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_query_traces_trace_id", "query_traces", ["trace_id"])

    op.create_table(
        "benchmark_lineage",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("benchmark_report_id", sa.Integer(), sa.ForeignKey("benchmark_reports.id", ondelete="CASCADE"), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("suite_name", sa.String(length=128), nullable=False),
        sa.Column("suite_version", sa.String(length=64), nullable=False),
        sa.Column("dataset_path", sa.Text(), nullable=True),
        sa.Column("case_name", sa.String(length=128), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("analytics_ok", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("semantic_ok", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("reasoning_ok", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_benchmark_lineage_run_id", "benchmark_lineage", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_benchmark_lineage_run_id", table_name="benchmark_lineage")
    op.drop_table("benchmark_lineage")

    op.drop_index("ix_query_traces_trace_id", table_name="query_traces")
    op.drop_table("query_traces")

    op.drop_index("ix_execution_logs_trace_id", table_name="execution_logs")
    op.drop_table("execution_logs")

    op.drop_index("ix_benchmark_reports_run_id", table_name="benchmark_reports")
    op.drop_column("benchmark_reports", "dataset_path")
    op.drop_column("benchmark_reports", "suite_version")
    op.drop_column("benchmark_reports", "suite_name")
    op.drop_column("benchmark_reports", "run_id")
