from __future__ import annotations

from pathlib import Path
import json

from saida.cli.main import main


def test_cli_analyze_runs_with_sample_csv(monkeypatch: object, tmp_path: Path, capsys: object) -> None:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "posted_at,revenue,region\n"
        "2026-01-01,100,West\n"
        "2026-02-01,90,West\n"
        "2026-03-01,80,East\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        ["saida", "analyze", "--csv", str(csv_path), "--question", "Why did revenue drop in March?"],
    )

    exit_code = main()
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Completed a diagnostic analysis for revenue on sales." in output


def test_cli_profile_supports_json_output(monkeypatch: object, tmp_path: Path, capsys: object) -> None:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "posted_at,revenue,region\n"
        "2026-01-01,100,West\n"
        "2026-02-01,90,West\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("sys.argv", ["saida", "profile", "--csv", str(csv_path), "--json"])

    exit_code = main()
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert exit_code == 0
    assert payload["dataset_name"] == "sales"
    assert payload["row_count"] == 2
    assert "revenue" in payload["measure_columns"]


def test_cli_analyze_supports_json_plan_and_trace_output(monkeypatch: object, tmp_path: Path, capsys: object) -> None:
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "posted_at,revenue,region\n"
        "2026-01-01,100,West\n"
        "2026-02-01,90,West\n"
        "2026-03-01,80,East\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        ["saida", "analyze", "--csv", str(csv_path), "--question", "Why did revenue drop in March?", "--json"],
    )

    exit_code = main()
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert exit_code == 0
    assert payload["plan"]["task_type"] == "diagnostic"
    assert payload["trace"][0]["stage"] == "adapter"
    assert any(table["name"] == "period_comparison" for table in payload["tables"])
