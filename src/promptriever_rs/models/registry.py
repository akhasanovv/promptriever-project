from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from promptriever_rs.config import load_yaml


@dataclass(slots=True)
class ModelSpec:
    name: str
    hf_id: str
    family: str
    query_prefix: str = ""
    document_prefix: str = ""
    query_template: str = "{query}"
    document_template: str = "{passage}"
    normalize_embeddings: bool = True

    def format_query(self, query: str, instruction: str | None = None) -> str:
        payload = self.query_template.format(
            query=query.strip(),
            instruction=(instruction or "").strip(),
        ).strip()
        return f"{self.query_prefix}{payload}".strip()

    def format_document(self, passage: str) -> str:
        payload = self.document_template.format(passage=passage.strip()).strip()
        return f"{self.document_prefix}{payload}".strip()


def load_model_spec(config_path: str | Path) -> ModelSpec:
    data = load_yaml(config_path)
    return ModelSpec(**data)
