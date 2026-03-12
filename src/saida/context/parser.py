"""Markdown semantic context parsing."""

from __future__ import annotations

from pathlib import Path

from saida.exceptions import ContextError
from saida.schemas import SourceContext


class SourceContextParser:
    """Parse semantic markdown into a SourceContext."""

    def parse_file(self, path: str | Path) -> SourceContext:
        """Load and parse a markdown context file."""
        markdown_path = Path(path)
        if not markdown_path.exists():
            raise ContextError(f"Context file not found: {markdown_path}")

        try:
            markdown = markdown_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ContextError(f"Failed to read context file: {markdown_path}") from exc

        return self.parse(markdown)

    def parse(self, markdown: str) -> SourceContext:
        """Parse markdown text into a tolerant semantic context object."""
        sections = self._split_sections(markdown)
        return SourceContext(
            raw_markdown=markdown,
            source_summary=self._join_section(sections.get("dataset")) or self._join_section(sections.get("summary")),
            table_descriptions=self._parse_mapping(sections.get("table descriptions", [])),
            field_descriptions=self._parse_mapping(sections.get("field descriptions", [])),
            metric_definitions=self._parse_mapping(sections.get("metrics", [])),
            business_rules=self._parse_bullets(sections.get("important rules", []))
            or self._parse_bullets(sections.get("business rules", [])),
            caveats=self._parse_bullets(sections.get("caveats", [])),
            trusted_date_fields=self._parse_bullets(sections.get("trusted date field", []))
            or self._parse_bullets(sections.get("trusted date fields", [])),
            preferred_identifiers=self._parse_bullets(sections.get("preferred identifiers", [])),
            freshness_notes=self._parse_bullets(sections.get("freshness", [])),
        )

    def _split_sections(self, markdown: str) -> dict[str, list[str]]:
        sections: dict[str, list[str]] = {"dataset": []}
        current_section = "dataset"
        for raw_line in markdown.splitlines():
            line = raw_line.strip()
            if line.startswith("#"):
                current_section = line.lstrip("#").strip().lower()
                sections.setdefault(current_section, [])
                continue
            if line:
                sections.setdefault(current_section, []).append(line)
        return sections

    def _parse_mapping(self, lines: list[str]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for line in lines:
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            mapping[key.strip()] = value.strip()
        return mapping

    def _parse_bullets(self, lines: list[str]) -> list[str]:
        values: list[str] = []
        for line in lines:
            cleaned = line.lstrip("-* ").strip()
            if cleaned:
                values.append(cleaned)
        return values

    def _join_section(self, lines: list[str] | None) -> str | None:
        if not lines:
            return None
        return " ".join(line.strip() for line in lines if line.strip()) or None
