from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, text

from saida import SaidaAgent
from saida.connectors.filesystem import FileSystemConnector
from saida.utils.config import SaidaConfig


def test_query_writes_execution_logs_and_traces(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "notes.txt").write_text("Revenue trend analysis for Q3.", encoding="utf-8")

    db_path = tmp_path / "obs.db"
    dsn = f"sqlite+pysqlite:///{db_path.as_posix()}"

    agent = SaidaAgent(SaidaConfig(control_plane_dsn=dsn, llm_provider="mock", embedding_provider="mock"))
    agent.add_connector(FileSystemConnector(str(data)))
    agent.ingest_all()
    result = agent.query("show revenue analysis")

    engine = create_engine(dsn, future=True)
    with engine.connect() as conn:
        log_row = conn.execute(
            text("SELECT trace_id, status, query_text FROM execution_logs ORDER BY id DESC LIMIT 1")
        ).mappings().one()
        assert log_row["status"] == "success"
        assert log_row["query_text"] == "show revenue analysis"
        assert log_row["trace_id"] == result.trace_id

        trace_count = conn.execute(
            text("SELECT COUNT(*) FROM query_traces WHERE trace_id = :trace_id"),
            {"trace_id": result.trace_id},
        ).scalar_one()
        assert trace_count >= 3


def test_benchmark_writes_lineage(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "sales.csv").write_text("quarter,revenue\nQ1,10\nQ2,20\n", encoding="utf-8")

    db_path = tmp_path / "obs_bench.db"
    dsn = f"sqlite+pysqlite:///{db_path.as_posix()}"

    agent = SaidaAgent(SaidaConfig(control_plane_dsn=dsn, llm_provider="mock", embedding_provider="mock"))
    agent.add_connector(FileSystemConnector(str(data)))
    agent.ingest_all()
    report = agent.run_benchmarks(suite_name="spec-suite", suite_version="v9", dataset_path=str(data))
    assert report.total > 0

    engine = create_engine(dsn, future=True)
    with engine.connect() as conn:
        report_row = conn.execute(
            text("SELECT id, run_id, suite_name, suite_version FROM benchmark_reports ORDER BY id DESC LIMIT 1")
        ).mappings().one()
        assert report_row["suite_name"] == "spec-suite"
        assert report_row["suite_version"] == "v9"

        lineage_count = conn.execute(
            text("SELECT COUNT(*) FROM benchmark_lineage WHERE benchmark_report_id = :report_id"),
            {"report_id": report_row["id"]},
        ).scalar_one()
        assert lineage_count == report.total
