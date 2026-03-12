from pathlib import Path
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from _env import load_project_env
from saida import Saida
from saida.adapters import CSVAdapter
from saida.config import LlmConfig, SaidaConfig


def main() -> None:
    load_project_env(PROJECT_ROOT)

    dataset = CSVAdapter(
        PROJECT_ROOT / "examples" / "sales.csv",
        context_path=PROJECT_ROOT / "examples" / "sales_context.md",
    ).load()

    config = SaidaConfig(
        llm=LlmConfig(
            enabled=True,
            provider="ollama",
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            use_for_prompting=True,
            use_for_reasoning=True,
        )
    )

    engine = Saida(config=config)
    result = engine.analyze(dataset, "Why did revenue drop in March by region?")

    print(result.summary)
    print("Tables:", ", ".join(table.name for table in result.tables))
    if result.warnings:
        print("Warnings:", "; ".join(result.warnings))


if __name__ == "__main__":
    main()
