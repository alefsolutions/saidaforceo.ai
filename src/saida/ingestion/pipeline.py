from __future__ import annotations

from pathlib import Path

from saida.connectors.base import BaseConnector
from saida.embeddings.base import BaseEmbeddingProvider
from saida.ingestion.parsers import detect_kind, parse_text, semantic_summary
from saida.ingestion.profiler import DatasetProfiler
from saida.models.types import DatasetAsset, ResourceRecord
from saida.semantic.store import SemanticStore
from saida.storage.control_plane import ControlPlaneStore
from saida.storage.parquet_store import ParquetStore
from saida.utils.hashing import sha256_bytes


class IngestionPipeline:
    def __init__(
        self,
        control_plane: ControlPlaneStore,
        parquet_store: ParquetStore,
        semantic_store: SemanticStore,
        embedding_provider: BaseEmbeddingProvider,
    ):
        self.control_plane = control_plane
        self.parquet_store = parquet_store
        self.semantic_store = semantic_store
        self.embedding_provider = embedding_provider
        self.profiler = DatasetProfiler()

    def ingest_connector(self, connector: BaseConnector) -> list[DatasetAsset]:
        assets: list[DatasetAsset] = []
        for resource_id in connector.discover():
            record = ResourceRecord(
                connector=connector.name,
                resource_id=resource_id,
                content=connector.load(resource_id),
                metadata=connector.get_metadata(),
            )
            asset = self.ingest_resource(record)
            if asset is not None:
                assets.append(asset)
        return assets

    def ingest_resource(self, record: ResourceRecord) -> DatasetAsset | None:
        if isinstance(record.content, bytes):
            raw_bytes = record.content
        elif isinstance(record.content, str):
            raw_bytes = record.content.encode("utf-8", errors="ignore")
        else:
            raw_bytes = str(record.content).encode("utf-8", errors="ignore")

        content_hash = sha256_bytes(raw_bytes)
        resource_key = f"{record.connector}:{record.resource_id}"
        if self.control_plane.is_unchanged(resource_key, content_hash):
            return None

        kind = detect_kind(record.resource_id)
        dataset_id = content_hash[:16]
        parquet_path: str | None = None
        profile: dict = {}

        path = Path(record.resource_id)
        if kind == "tabular" and path.exists():
            if path.suffix.lower() == ".csv":
                parquet_path = self.parquet_store.csv_to_parquet(str(path), dataset_id)
            elif path.suffix.lower() == ".json":
                parquet_path = self.parquet_store.json_to_parquet(str(path), dataset_id)
            profile = self.profiler.profile_tabular(str(path))

        text_for_summary = ""
        if path.exists() and kind in {"document", "tabular"}:
            text_for_summary = parse_text(str(path))

        summary = semantic_summary(text_for_summary)
        embeddings = self.embedding_provider.embed([summary]) if summary else [[0.0, 0.0, 0.0]]

        asset = DatasetAsset(
            dataset_id=dataset_id,
            source_connector=record.connector,
            source_resource_id=record.resource_id,
            hash=content_hash,
            kind=kind,
            parquet_path=parquet_path,
            text_summary=summary,
            metadata={**record.metadata, **profile},
        )
        self.control_plane.update_hash(resource_key, content_hash)
        self.control_plane.upsert_dataset(asset)
        self.semantic_store.upsert_dataset(asset, embeddings[0])
        return asset
