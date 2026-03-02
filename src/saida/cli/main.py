from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from saida import SaidaAgent
from saida.connectors.filesystem import FileSystemConnector


def main() -> None:
    parser = argparse.ArgumentParser(description="SAIDA CLI")
    parser.add_argument("command", choices=["ingest", "query", "bench"])
    parser.add_argument("--path", default="./data")
    parser.add_argument("--prompt", default="What data do we have?")
    args = parser.parse_args()

    agent = SaidaAgent()
    agent.add_connector(FileSystemConnector(args.path))

    if args.command == "ingest":
        assets = agent.ingest_all()
        print(json.dumps([a.dataset_id for a in assets]))
    elif args.command == "query":
        agent.ingest_all()
        result = agent.query(args.prompt)
        print(json.dumps(asdict(result), default=str))
    else:
        agent.ingest_all()
        report = agent.run_benchmarks()
        print(json.dumps(asdict(report), default=str))


if __name__ == "__main__":
    main()
