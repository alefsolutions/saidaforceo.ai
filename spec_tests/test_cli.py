from __future__ import annotations

from pathlib import Path

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
