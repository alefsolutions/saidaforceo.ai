from __future__ import annotations

from dataclasses import asdict

from saida.models.types import DatasetAsset


class ControlPlaneStore:
    """Control-plane metadata store.

    Uses in-memory storage by default to keep the core deterministic and runnable.
    Can be swapped with PostgreSQL-backed implementation without changing call sites.
    """

    def __init__(self):
        self._datasets: dict[str, DatasetAsset] = {}
        self._resource_hashes: dict[str, str] = {}
        self._benchmark_reports: list[dict] = []

    def is_unchanged(self, resource_key: str, content_hash: str) -> bool:
        return self._resource_hashes.get(resource_key) == content_hash

    def update_hash(self, resource_key: str, content_hash: str) -> None:
        self._resource_hashes[resource_key] = content_hash

    def upsert_dataset(self, asset: DatasetAsset) -> None:
        self._datasets[asset.dataset_id] = asset

    def list_datasets(self) -> list[DatasetAsset]:
        return list(self._datasets.values())

    def save_benchmark_report(self, report: dict) -> None:
        self._benchmark_reports.append(report)

    def list_benchmark_reports(self) -> list[dict]:
        return list(self._benchmark_reports)

    def snapshot(self) -> dict:
        return {
            "datasets": [asdict(v) for v in self._datasets.values()],
            "hashes": dict(self._resource_hashes),
            "benchmarks": list(self._benchmark_reports),
        }
