# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

Quick links: [Main README](./README.md) | [Architecture Contract](./ARCHITECTURE.md) | [License](./LICENSE)

## [Unreleased]
### Added
- Release hygiene tooling with pre-commit, Ruff linting, MyPy type checks, and pinned lock files.
- CI benchmark threshold gates, DB integration tests, and observability/audit persistence.

## [0.1.0] - 2026-03-02
### Added
- Initial SAIDA core architecture: agent, connectors, ingestion, storage, analytics, semantic, orchestration, benchmarking.
- PostgreSQL control-plane + semantic persistence with Alembic migrations and pgvector-ready schema.
- Document parsing support for TXT/CSV/JSON/PDF/DOCX/XLSX.
- LangChain runnable orchestration graph and analytics safety guards.

### Changed
- Migrated from legacy `saida_core` scaffold to spec-aligned `saida` package structure.

---

Sponsored and supported by Alef Digital Solutions.
