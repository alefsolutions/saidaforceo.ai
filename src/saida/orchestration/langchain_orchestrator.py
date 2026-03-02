from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from saida.analytics.engine import DuckDBAnalyticsEngine
from saida.analytics.sql_planner import SQLPlanner
from saida.analytics.sql_validator import SQLValidator
from saida.embeddings.base import BaseEmbeddingProvider
from saida.llm.base import BaseLLMProvider
from saida.models.types import QueryResult
from saida.orchestration.llm_guard import enforce_no_unverified_numbers
from saida.orchestration.router import QueryRouter
from saida.semantic.store import SemanticStore
from saida.storage.control_plane import ControlPlaneStore


@dataclass(slots=True)
class GraphState:
    trace_id: str
    query: str
    route: str
    retrieved: list[dict[str, Any]]
    sql: str | None
    analytics_rows: list[dict[str, Any]]
    explanation: str
    trace_events: list[dict[str, Any]]


class LangChainOrchestrator:
    """LangChain runnable graph using explicit tools.

    Graph:
      route -> semantic_retrieve_tool -> analytics_query_tool -> llm_synthesis
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
        self.sql_planner = SQLPlanner()
        self.sql_validator = SQLValidator()

        self.semantic_retrieve_tool = self._build_semantic_retrieve_tool()
        self.analytics_query_tool = self._build_analytics_query_tool()

        self.graph = (
            RunnableLambda(self._route_step)
            | RunnableLambda(self._semantic_step)
            | RunnableLambda(self._analytics_step)
            | RunnableLambda(self._synthesis_step)
        )

    def _build_semantic_retrieve_tool(self):
        @tool("semantic_retrieve")
        def semantic_retrieve(query: str) -> list[dict]:
            """Retrieve semantically relevant datasets and context for a user query."""
            q_emb = self.embedding_provider.embed([query])[0]
            rows = self.semantic_store.retrieve(q_emb, limit=5)
            out: list[dict] = []
            for item in rows:
                dataset = item["dataset"]
                out.append(
                    {
                        "dataset_id": dataset.dataset_id,
                        "source": dataset.source_resource_id,
                        "parquet_path": dataset.parquet_path,
                        "score": float(item["score"]),
                    }
                )
            return out

        return semantic_retrieve

    def _build_analytics_query_tool(self):
        @tool("analytics_query")
        def analytics_query(sql: str | None) -> dict:
            """Run deterministic DuckDB analytics from a pre-validated SQL plan."""
            if not sql:
                return {"sql": None, "rows": []}
            self.sql_validator.validate(sql)
            rows = self.analytics_engine.execute(sql)
            return {"sql": sql, "rows": rows}

        return analytics_query

    def _route_step(self, prompt: str) -> GraphState:
        started = perf_counter()
        route = self.router.classify(prompt)
        duration_ms = (perf_counter() - started) * 1000.0
        return GraphState(
            trace_id=uuid4().hex,
            query=prompt,
            route=route,
            retrieved=[],
            sql=None,
            analytics_rows=[],
            explanation="",
            trace_events=[
                {
                    "step_name": "route",
                    "status": "success",
                    "duration_ms": duration_ms,
                    "input_json": {"query": prompt},
                    "output_json": {"route": route},
                }
            ],
        )

    def _semantic_step(self, state: GraphState) -> GraphState:
        started = perf_counter()
        try:
            retrieved = self.semantic_retrieve_tool.invoke({"query": state.query})
            state.retrieved = retrieved
            state.trace_events.append(
                {
                    "step_name": "semantic_retrieve",
                    "status": "success",
                    "duration_ms": (perf_counter() - started) * 1000.0,
                    "input_json": {"query": state.query},
                    "output_json": {"retrieved_count": len(retrieved)},
                }
            )
        except Exception as exc:
            state.trace_events.append(
                {
                    "step_name": "semantic_retrieve",
                    "status": "error",
                    "duration_ms": (perf_counter() - started) * 1000.0,
                    "input_json": {"query": state.query},
                    "output_json": {},
                    "error_text": str(exc),
                }
            )
            raise
        return state

    def _analytics_step(self, state: GraphState) -> GraphState:
        if state.route != "analytics":
            state.trace_events.append(
                {
                    "step_name": "analytics",
                    "status": "skipped",
                    "duration_ms": 0.0,
                    "input_json": {"route": state.route},
                    "output_json": {},
                }
            )
            return state

        started = perf_counter()
        plan = self.sql_planner.plan(state.query, state.route, state.retrieved)
        if not plan.sql:
            state.sql = None
            state.analytics_rows = []
            state.trace_events.append(
                {
                    "step_name": "analytics_plan",
                    "status": "skipped",
                    "duration_ms": (perf_counter() - started) * 1000.0,
                    "input_json": {"query": state.query},
                    "output_json": {"reason": plan.reason},
                }
            )
            return state

        try:
            result = self.analytics_query_tool.invoke({"sql": plan.sql})
            state.sql = result.get("sql")
            state.analytics_rows = result.get("rows", [])
            state.trace_events.append(
                {
                    "step_name": "analytics_execute",
                    "status": "success",
                    "duration_ms": (perf_counter() - started) * 1000.0,
                    "input_json": {"sql": plan.sql},
                    "output_json": {"row_count": len(state.analytics_rows)},
                }
            )
        except Exception as exc:
            state.trace_events.append(
                {
                    "step_name": "analytics_execute",
                    "status": "error",
                    "duration_ms": (perf_counter() - started) * 1000.0,
                    "input_json": {"sql": plan.sql},
                    "output_json": {},
                    "error_text": str(exc),
                }
            )
            raise
        return state

    def _synthesis_step(self, state: GraphState) -> QueryResult:
        started = perf_counter()
        prompt = (
            "You are SAIDA. Use tool outputs only. "
            "Do not fabricate numeric values not present in analytics rows.\n"
            f"query={state.query}\n"
            f"route={state.route}\n"
            f"retrieved_context={[{'dataset_id': r.get('dataset_id'), 'source': r.get('source')} for r in state.retrieved]}\n"
            f"analytics_rows={state.analytics_rows}"
        )
        try:
            explanation = self.llm_provider.explain(prompt)
            if state.route == "analytics":
                enforce_no_unverified_numbers(state.query, state.analytics_rows, explanation)
            state.trace_events.append(
                {
                    "step_name": "synthesis",
                    "status": "success",
                    "duration_ms": (perf_counter() - started) * 1000.0,
                    "input_json": {"route": state.route},
                    "output_json": {"explanation_length": len(explanation)},
                }
            )
        except Exception as exc:
            state.trace_events.append(
                {
                    "step_name": "synthesis",
                    "status": "error",
                    "duration_ms": (perf_counter() - started) * 1000.0,
                    "input_json": {"route": state.route},
                    "output_json": {},
                    "error_text": str(exc),
                }
            )
            raise
        return QueryResult(
            query=state.query,
            route=state.route,
            sql=state.sql,
            analytics_rows=state.analytics_rows,
            retrieved_context=[
                {
                    "dataset_id": item.get("dataset_id"),
                    "source": item.get("source"),
                    "score": item.get("score"),
                }
                for item in state.retrieved
            ],
            explanation=explanation,
            trace_id=state.trace_id,
        )

    def run_query(self, prompt: str) -> QueryResult:
        started = perf_counter()
        trace_id = uuid4().hex
        try:
            state = self._route_step(prompt)
            state.trace_id = trace_id
            state = self._semantic_step(state)
            state = self._analytics_step(state)
            result = self._synthesis_step(state)
            result.trace_id = trace_id

            self.control_plane.log_execution(
                trace_id=trace_id,
                query_text=prompt,
                route=result.route,
                status="success",
                sql_text=result.sql,
                analytics_row_count=len(result.analytics_rows),
                retrieved_count=len(result.retrieved_context),
                duration_ms=(perf_counter() - started) * 1000.0,
                metadata={"orchestrator": "langchain_graph"},
                traces=state.trace_events,
            )
            return result
        except Exception as exc:
            self.control_plane.log_execution(
                trace_id=trace_id,
                query_text=prompt,
                route=None,
                status="error",
                sql_text=None,
                analytics_row_count=0,
                retrieved_count=0,
                duration_ms=(perf_counter() - started) * 1000.0,
                error_text=str(exc),
                metadata={"orchestrator": "langchain_graph"},
                traces=[],
            )
            raise
