from __future__ import annotations

import json
from pathlib import Path


TEXT_EXTENSIONS = {".txt", ".md", ".log", ".py", ".sql", ".json", ".csv"}
TABULAR_EXTENSIONS = {".csv", ".json"}


def detect_kind(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in TABULAR_EXTENSIONS:
        return "tabular"
    if ext in TEXT_EXTENSIONS:
        return "document"
    if ext in {".pdf", ".docx", ".xlsx"}:
        return "document"
    return "binary"


def parse_text(path: str) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".json":
        raw = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        return json.dumps(raw, ensure_ascii=True)
    return p.read_text(encoding="utf-8", errors="ignore")


def semantic_summary(text: str, max_len: int = 300) -> str:
    compact = " ".join(text.split())
    return compact[:max_len]
