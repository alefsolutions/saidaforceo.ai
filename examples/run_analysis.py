from saida import Saida
from saida.adapters import CSVAdapter


dataset = CSVAdapter("examples/sales.csv", context_path="examples/sales_context.md").load()
result = Saida().analyze(dataset, "Why did revenue drop in March by region?")

print(result.summary)
print("Tables:", ", ".join(table.name for table in result.tables))
