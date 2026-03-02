from __future__ import annotations

from saida.models.types import IntelligenceScores


def compute_scores(
    total: int,
    passed_analytics: int,
    passed_semantic: int,
    passed_reasoning: int,
    successful_executions: int,
    w_ais: float = 0.40,
    w_ses: float = 0.30,
    w_ris: float = 0.20,
    w_sss: float = 0.10,
) -> IntelligenceScores:
    if total <= 0:
        return IntelligenceScores(0.0, 0.0, 0.0, 0.0, 0.0)

    ais = (passed_analytics / total) * 100.0
    ses = (passed_semantic / total) * 100.0
    ris = (passed_reasoning / total) * 100.0
    sss = (successful_executions / total) * 100.0
    composite = (w_ais * ais) + (w_ses * ses) + (w_ris * ris) + (w_sss * sss)
    return IntelligenceScores(ais=ais, ses=ses, ris=ris, sss=sss, composite=composite)
