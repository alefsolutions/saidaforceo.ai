from __future__ import annotations

import duckdb


class DuckDBAnalyticsEngine:
    name = "duckdb"

    def execute(self, sql: str) -> list[dict]:
        con = duckdb.connect()
        try:
            cursor = con.execute(sql)
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in rows]
        finally:
            con.close()

    def query_parquet(self, parquet_path: str, sql_projection: str = "*") -> list[dict]:
        safe_projection = sql_projection if sql_projection.strip() else "*"
        sql = f"SELECT {safe_projection} FROM read_parquet(?)"
        con = duckdb.connect()
        try:
            cursor = con.execute(sql, [parquet_path])
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in rows]
        finally:
            con.close()
