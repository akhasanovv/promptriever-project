from __future__ import annotations

from pathlib import Path

from tqdm.auto import tqdm

from promptriever_rs.config import load_yaml
from promptriever_rs.utils.io import read_jsonl, write_jsonl


def assemble_instruction_dataset(config_path: str | Path) -> Path:
    config = load_yaml(config_path)
    base_records = read_jsonl(config["base_records_path"])
    negative_rows = read_jsonl(config["negative_instructions_path"])
    negative_by_id = {row["sample_id"]: row for row in negative_rows}
    positive_by_id: dict[str, dict] = {}
    if config.get("positive_instructions_path"):
        positive_rows = read_jsonl(config["positive_instructions_path"])
        positive_by_id = {row["sample_id"]: row for row in positive_rows}

    positive_instruction_template = config.get("positive_instruction_template", "").strip()
    allowed_splits = set(config.get("splits", ["train", "validation", "test"]))
    include_answer = bool(config.get("include_answer_in_metadata", True))

    assembled: list[dict] = []
    for row in tqdm(base_records, desc="Assembling instruction dataset", total=len(base_records)):
        if row["split"] not in allowed_splits:
            continue

        negative = negative_by_id.get(row["sample_id"])
        if not negative:
            continue

        positive = positive_by_id.get(row["sample_id"])
        positive_instruction = (
            positive["positive_instruction"].strip()
            if positive and positive.get("positive_instruction")
            else positive_instruction_template
        )
        if not positive_instruction:
            continue

        metadata = dict(row.get("metadata", {}))
        if include_answer:
            metadata["answer"] = row.get("answer", "")
        metadata["violation_reason"] = negative.get("violation_reason", "")
        metadata["relevance_reason"] = positive.get("relevance_reason", "") if positive else ""

        assembled.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "query": row["query"],
                "answer": row.get("answer", ""),
                "positive_passage": row["positive_passage"],
                "positive_instruction": positive_instruction,
                "negative_instruction": negative["negative_instruction"],
                "negative_passages": row.get("negative_passages", []),
                "metadata": metadata,
            }
        )

    output_path = Path(config["output_path"])
    write_jsonl(output_path, assembled)
    return output_path
