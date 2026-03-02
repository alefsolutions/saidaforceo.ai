from __future__ import annotations

from pathlib import Path

import duckdb


class DatasetProfiler:
    def profile_tabular(self, path: str) -> dict:
        ext = Path(path).suffix.lower()
        con = duckdb.connect()
        try:
            escaped = path.replace("'", "''")
            if ext == ".csv":
                relation = con.sql(f"SELECT * FROM read_csv_auto('{escaped}')")
            else:
                relation = con.sql(f"SELECT * FROM read_json_auto('{escaped}')")
            columns = [c[0] for c in relation.description]
            count = relation.count("*").fetchone()[0]
            return {"columns": columns, "row_count": int(count)}
        finally:
            con.close()
