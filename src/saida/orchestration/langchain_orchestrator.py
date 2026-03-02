from __future__ import annotations

from saida.analytics.engine import DuckDBAnalyticsEngine
from saida.embeddings.base import BaseEmbeddingProvider
from saida.llm.base import BaseLLMProvider
from saida.models.types import QueryResult
from saida.orchestration.router import QueryRouter
from saida.semantic.store import SemanticStore
from saida.storage.control_plane import ControlPlaneStore


class LangChainOrchestrator:
    """Workflow layer. Uses LangChain-style prompt orchestration semantics.

    A full LangChain chain graph can be added here without changing agent API.
    """

    def __init__(
        self,
        control_plane: ControlPlaneStore,
        semantic_store: SemanticStore,
        embedding_provider: BaseEmbeddingProvider,
        llm_provider: BaseLLMProvider,
        analytics_engine: DuckDBAnalyticsEngine,
    ):
        self.control_plane = control_plane
        self.semantic_store = semantic_store
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.analytics_engine = analytics_engine
        self.router = QueryRouter()

    def run_query(self, prompt: str) -> QueryResult:
        route = self.router.classify(prompt)
        q_emb = self.embedding_provider.embed([prompt])[0]
        retrieved = self.semantic_store.retrieve(q_emb, limit=5)

        sql: str | None = None
        analytics_rows: list[dict] = []
        if route == "analytics":
            top_dataset = next((r["dataset"] for r in retrieved if r["dataset"].parquet_path), None)
            if top_dataset and top_dataset.parquet_path:
                sql = "SELECT * FROM read_parquet(?) LIMIT 25"
                analytics_rows = self.analytics_engine.execute(
                    sql.replace("?", f"'{top_dataset.parquet_path.replace("'", "''")}'")
                )

        explanation_prompt = (
            "You are SAIDA. Use deterministic analytics results and retrieved context. "
            "Do not invent numeric calculations not present in analytics_rows.\n"
            f"user_prompt={prompt}\n"
            f"route={route}\n"
            f"retrieved_count={len(retrieved)}\n"
            f"analytics_rows_count={len(analytics_rows)}"
        )
        explanation = self.llm_provider.explain(explanation_prompt)
        retrieved_context = [
            {
                "dataset_id": item["dataset"].dataset_id,
                "source": item["dataset"].source_resource_id,
                "score": item["score"],
            }
            for item in retrieved
        ]
        return QueryResult(
            query=prompt,
            route=route,
            sql=sql,
            analytics_rows=analytics_rows,
            retrieved_context=retrieved_context,
            explanation=explanation,
        )
