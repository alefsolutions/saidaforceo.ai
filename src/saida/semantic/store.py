from __future__ import annotations

from math import sqrt

from saida.models.types import DatasetAsset


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    an = sqrt(sum(x * x for x in a[:n]))
    bn = sqrt(sum(x * x for x in b[:n]))
    if an == 0.0 or bn == 0.0:
        return 0.0
    return dot / (an * bn)


class SemanticStore:
    """Semantic layer abstraction.

    In production, implement this with PostgreSQL + pgvector.
    """

    def __init__(self):
        self._rows: list[tuple[DatasetAsset, list[float]]] = []

    def upsert_dataset(self, asset: DatasetAsset, embedding: list[float]) -> None:
        self._rows = [r for r in self._rows if r[0].dataset_id != asset.dataset_id]
        self._rows.append((asset, embedding))

    def retrieve(self, query_embedding: list[float], limit: int = 5) -> list[dict]:
        scored = []
        for asset, emb in self._rows:
            scored.append({"dataset": asset, "score": _cosine(query_embedding, emb)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]
