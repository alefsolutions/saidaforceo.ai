from __future__ import annotations


class QueryRouter:
    def classify(self, prompt: str) -> str:
        p = prompt.lower()
        if any(w in p for w in ["sum", "count", "avg", "revenue", "margin", "group by", "quarter", "q1", "q2", "q3", "q4"]):
            return "analytics"
        return "semantic_reasoning"
