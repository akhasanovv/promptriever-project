from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from datasets import load_dataset

from promptriever_rs.config import ensure_dir, load_yaml
from promptriever_rs.utils.io import write_jsonl


def iter_sberquad_records(config: dict) -> Iterable[dict]:
    dataset_name = config["dataset_name"]
    split_mapping = config["dataset_splits"]
    deduplicate_contexts = bool(config.get("deduplicate_contexts", False))
    min_context_chars = int(config.get("min_context_chars", 0))
    max_samples_per_split = config.get("max_samples_per_split")

    seen_contexts: set[tuple[str, str]] = set()

    for split_alias, hf_split in split_mapping.items():
        dataset = load_dataset(dataset_name, split=hf_split)
        processed = 0

        for index, row in enumerate(dataset):
            context = str(row["context"]).strip()
            query = str(row["question"]).strip()
            answer = str(row["answers"]["text"][0]).strip() if row.get("answers", {}).get("text") else ""

            if len(context) < min_context_chars:
                continue

            context_id = f"{split_alias}-ctx-{index:06d}"
            dedup_key = (split_alias, context) if deduplicate_contexts else (context_id, "")
            if deduplicate_contexts and dedup_key in seen_contexts:
                continue
            seen_contexts.add(dedup_key)

            yield {
                "sample_id": f"{split_alias}-{processed:06d}",
                "split": split_alias,
                "query": query,
                "answer": answer,
                "positive_passage": context,
                "metadata": {
                    "source_dataset": dataset_name,
                    "context_id": context_id,
                    "original_id": row.get("id"),
                    "title": row.get("title"),
                },
            }
            processed += 1

            if max_samples_per_split and processed >= int(max_samples_per_split):
                break


def build_sberquad_records(config_path: str | Path) -> Path:
    config = load_yaml(config_path)
    output_dir = ensure_dir(config["output_dir"])
    output_path = output_dir / "sberquad_records.jsonl"
    write_jsonl(output_path, iter_sberquad_records(config))
    return output_path
