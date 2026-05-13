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

        assembled.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "query": row["query"],
                "positive_passage": row["positive_passage"],
                "positive_instruction": positive_instruction,
                "negative_instruction": negative["negative_instruction"],
            }
        )

    output_path = Path(config["output_path"])
    write_jsonl(output_path, assembled)
    return output_path


def assemble_promptriever_dataset(config_path: str | Path) -> Path:
    config = load_yaml(config_path)
    base_records = read_jsonl(config["base_records_path"])
    passage_rows = read_jsonl(config["generated_passages_path"])
    passages_by_id = {row["sample_id"]: row for row in passage_rows}
    hard_negative_rows = read_jsonl(config["hard_negatives_path"]) if config.get("hard_negatives_path") else []
    hard_negatives_by_id = {row["sample_id"]: row for row in hard_negative_rows}
    positive_validation_rows = (
        read_jsonl(config["positive_instruction_validation_path"])
        if config.get("positive_instruction_validation_path")
        else []
    )
    positive_validation_by_id = {row["sample_id"]: row for row in positive_validation_rows}

    allowed_splits = set(config.get("splits", ["train", "validation", "test"]))
    use_generated_positive = bool(config.get("use_generated_positive_passage", True))
    require_positive_validation = bool(config.get("require_positive_instruction_validation", False))
    require_generated_positive = bool(config.get("require_generated_positive_validation", True))
    min_hard_negatives = int(config.get("min_hard_negatives", 0))

    assembled: list[dict] = []
    stats = {
        "total": 0,
        "wrong_split": 0,
        "missing_generated": 0,
        "invalid_positive_instruction": 0,
        "missing_instruction": 0,
        "invalid_generated_positive": 0,
        "not_enough_instruction_negatives": 0,
        "not_enough_hard_negatives": 0,
        "assembled": 0,
    }
    for row in tqdm(base_records, desc="Assembling Promptriever dataset", total=len(base_records)):
        stats["total"] += 1
        if row["split"] not in allowed_splits:
            stats["wrong_split"] += 1
            continue

        generated = passages_by_id.get(row["sample_id"])
        if not generated:
            stats["missing_generated"] += 1
            continue
        if require_positive_validation:
            positive_validation = positive_validation_by_id.get(row["sample_id"])
            if not positive_validation or not positive_validation.get("is_valid", False):
                stats["invalid_positive_instruction"] += 1
                continue

        instruction = str(generated.get("instruction", "")).strip()
        if not instruction:
            stats["missing_instruction"] += 1
            continue
        if require_generated_positive and not generated.get("generated_positive_is_valid", False):
            stats["invalid_generated_positive"] += 1
            continue

        instruction_negatives = [
            str(text).strip()
            for text in generated.get("instruction_negative_passages", [])
            if str(text).strip()
        ]
        if not instruction_negatives:
            stats["not_enough_instruction_negatives"] += 1
            continue
        hard_negative_row = hard_negatives_by_id.get(row["sample_id"], {})
        hard_negative_passages = [
            str(text).strip()
            for text in hard_negative_row.get("hard_negative_passages", [])
            if str(text).strip()
        ]
        if len(hard_negative_passages) < min_hard_negatives:
            stats["not_enough_hard_negatives"] += 1
            continue

        positive_passage = row["positive_passage"]
        if use_generated_positive and str(generated.get("generated_positive_passage", "")).strip():
            positive_passage = str(generated["generated_positive_passage"]).strip()

        assembled.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "query": row["query"],
                "instruction": instruction,
                "positive_passage": positive_passage,
                "instruction_negative_passages": instruction_negatives,
                "hard_negative_passages": hard_negative_passages,
            }
        )
        stats["assembled"] += 1

    output_path = Path(config["output_path"])
    write_jsonl(output_path, assembled)
    print(
        "Promptriever assembly summary: "
        f"total={stats['total']}, wrong_split={stats['wrong_split']}, "
        f"missing_generated={stats['missing_generated']}, "
        f"invalid_positive_instruction={stats['invalid_positive_instruction']}, "
        f"missing_instruction={stats['missing_instruction']}, "
        f"invalid_generated_positive={stats['invalid_generated_positive']}, "
        f"not_enough_instruction_negatives={stats['not_enough_instruction_negatives']}, "
        f"not_enough_hard_negatives={stats['not_enough_hard_negatives']}, "
        f"assembled={stats['assembled']}"
    )
    return output_path
