from __future__ import annotations

from saida.benchmarking.scoring import compute_scores
from saida.models.types import BenchmarkCase, BenchmarkReport
from saida.orchestration.langchain_orchestrator import LangChainOrchestrator


class BenchmarkRunner:
    def __init__(self, orchestrator: LangChainOrchestrator):
        self.orchestrator = orchestrator

    def run(self, cases: list[BenchmarkCase]) -> BenchmarkReport:
        details: list[dict] = []
        passed_analytics = 0
        passed_semantic = 0
        passed_reasoning = 0
        successful_executions = 0

        for case in cases:
            ok_exec = True
            try:
                result = self.orchestrator.run_query(case.query)
            except Exception as exc:  # pragma: no cover
                ok_exec = False
                result = None
                details.append({"case": case.name, "error": str(exc)})

            if ok_exec and result is not None:
                successful_executions += 1
                analytics_ok = (len(result.analytics_rows) >= case.expected_rows_min)
                if case.expected_sql_nonempty:
                    analytics_ok = analytics_ok and bool(result.sql)
                semantic_ok = len(result.retrieved_context) > 0
                reasoning_ok = len(result.explanation.strip()) > 0

                passed_analytics += int(analytics_ok)
                passed_semantic += int(semantic_ok)
                passed_reasoning += int(reasoning_ok)

                details.append(
                    {
                        "case": case.name,
                        "analytics_ok": analytics_ok,
                        "semantic_ok": semantic_ok,
                        "reasoning_ok": reasoning_ok,
                    }
                )

        total = len(cases)
        scores = compute_scores(
            total=total,
            passed_analytics=passed_analytics,
            passed_semantic=passed_semantic,
            passed_reasoning=passed_reasoning,
            successful_executions=successful_executions,
        )
        return BenchmarkReport(
            total=total,
            passed_analytics=passed_analytics,
            passed_semantic=passed_semantic,
            passed_reasoning=passed_reasoning,
            successful_executions=successful_executions,
            scores=scores,
            details=details,
        )
