<p align="center">
  <img src="assets/banner.png" alt="SAIDA Banner" />
</p>

# SAIDA

SAIDA is a deterministic, measurable AI intelligence framework for business data.

It combines deterministic analytics, semantic retrieval, and controlled LLM reasoning into one library that developers can embed in Python applications.

## Quick Links

- Full architecture and contract: [ARCHITECTURE.md](./ARCHITECTURE.md)
- Release history: [CHANGELOG.md](./CHANGELOG.md)
- License: [LICENSE](./LICENSE)

## Technical Baseline

SAIDA is a Python library/framework (not a hosted service).

### Runtime and Toolchain

- Python: `>=3.11` (recommended: `3.11` or `3.12`)
- Packaging: `pyproject.toml` (setuptools)
- Migrations: Alembic
- Type checking: MyPy
- Linting/formatting: Ruff
- Pre-commit hooks: pre-commit

### Core Runtime Dependencies (Pinned)

- `openai==2.21.0`
- `duckdb==1.4.4`
- `langchain-core==1.2.13`
- `sqlalchemy==2.0.46`
- `alembic==1.18.4`
- `psycopg[binary]==3.3.2`
- `pgvector==0.4.2`
- `pymysql==1.1.2`
- `google-api-python-client==2.190.0`
- `google-auth==2.48.0`
- `pypdf==6.7.0`
- `python-docx==1.2.0`
- `openpyxl==3.1.5`

See lock files:
- Runtime lock: [`requirements.lock.txt`](./requirements.lock.txt)
- Dev/tooling lock: [`requirements-dev.lock.txt`](./requirements-dev.lock.txt)

## Install and Run

### 1. Install dependencies

```bash
pip install -r requirements-dev.lock.txt
pip install -e . --no-deps
```

### 2. (Optional) Setup pre-commit

```bash
pre-commit install
pre-commit run --all-files
```

### 3. Run migrations

```bash
python -m alembic -c alembic.ini upgrade head
```

### 4. Try SAIDA from CLI

```bash
python -m saida.cli.main query --path ./benchmarks/datasets --prompt "show revenue by quarter"
```

### 5. Run quality gates

```bash
python -m ruff check src/saida spec_tests
python -m mypy --config-file pyproject.toml src/saida
python -m pytest -q
python -m saida.benchmarking.ci_gate --suite benchmarks/suites/core_v1.json --datasets benchmarks/datasets
```

## Python Usage Examples

### Example 1: Query your local datasets

```python
from saida import SaidaAgent
from saida.connectors.filesystem import FileSystemConnector
from saida.utils.config import SaidaConfig

config = SaidaConfig(
    control_plane_dsn="sqlite+pysqlite:///./saida_demo.db",
    llm_provider="mock",
    embedding_provider="mock",
    parquet_root="./.saida/parquet",
)

agent = SaidaAgent(config)
agent.add_connector(FileSystemConnector("./benchmarks/datasets"))
agent.ingest_all()

result = agent.query("show revenue by quarter")
print(result.route)
print(result.sql)
print(result.analytics_rows[:3])
print(result.explanation)
```

### Example 2: Run benchmark suite in Python

```python
from saida import SaidaAgent
from saida.connectors.filesystem import FileSystemConnector
from saida.utils.config import SaidaConfig
from saida.benchmarking.suite import load_benchmark_suite

suite = load_benchmark_suite("benchmarks/suites/core_v1.json")

agent = SaidaAgent(
    SaidaConfig(
        control_plane_dsn="sqlite+pysqlite:///./saida_bench.db",
        llm_provider="mock",
        embedding_provider="mock",
        parquet_root="./.saida/bench-parquet",
    )
)
agent.add_connector(FileSystemConnector("benchmarks/datasets"))
agent.ingest_all()

report = agent.run_benchmarks(
    cases=suite.cases,
    suite_name=suite.name,
    suite_version="v1",
    dataset_path="benchmarks/datasets",
)

print(report.scores)
```

## SAIDA Use Cases (Developer-Focused)

- Executive KPI copilots for quarterly reviews
- Financial variance analysis over CSV/XLSX/DB data
- Revenue trend analysis from mixed files + databases
- Cost center diagnostics with deterministic SQL outputs
- Board memo generation grounded in verified analytics
- Business operations intelligence assistants
- Internal BI copilots for analytics teams
- Natural-language querying over Parquet datasets
- Cross-dataset semantic search over internal docs
- Data room summarization for leadership updates
- Post-merger reporting and dataset consolidation workflows
- Compliance-oriented explanation generation with audit trails
- Postgres + pgvector semantic retrieval applications
- MySQL-backed analysis assistants for ops teams
- Google Drive document intelligence pipelines
- File-system analytics workflows for local/private datasets
- Benchmark-driven model and prompt evaluation loops
- Quality-gated AI analytics in CI/CD
- Risk monitoring pipelines with reproducible scoring
- Metric reporting bots with traceable query lineage
- Product analytics copilots with safe SQL planning
- Sales analytics assistants with deterministic rollups
- Operations note summarization tied to source evidence
- Investor update assistants grounded in source data
- Internal knowledge copilots for strategy teams

## Support

For support, enterprise collaboration, or deployment help:

- `hello@alefpng.com`

---

Sponsored and supported by Alef Digital Solutions.
