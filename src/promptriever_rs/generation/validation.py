from __future__ import annotations

from pathlib import Path

from tqdm.auto import tqdm

from promptriever_rs.config import load_yaml
from promptriever_rs.utils.io import read_jsonl, write_jsonl
from promptriever_rs.validation.judges import RerankerJudge


def _print_positive_summary(rows: list[dict]) -> None:
    num_valid = sum(1 for row in rows if row.get("is_valid", False))
    print(
        "Positive instruction validation summary: "
        f"validated={len(rows)}, is_valid={num_valid}, invalid={len(rows) - num_valid}"
    )


def _print_passage_summary(rows: list[dict]) -> None:
    num_positive_valid = sum(1 for row in rows if row.get("generated_positive_is_valid", False))
    num_negative_valid = sum(len(row.get("instruction_negative_passages", [])) for row in rows)
    print(
        "Generated passage validation summary: "
        f"validated_rows={len(rows)}, positive_is_valid={num_positive_valid}, "
        f"instruction_negative_is_valid={num_negative_valid}"
    )


def validate_positive_instructions(config_path: str | Path) -> Path:
    config = load_yaml(config_path)
    base_records = read_jsonl(config["base_records_path"])
    positive_rows = read_jsonl(config["positive_instructions_path"])
    positives_by_id = {row["sample_id"]: row for row in positive_rows}

    judge = RerankerJudge(
        model_name=config["judge_model"],
        device=str(config.get("device", "auto")),
        max_length=int(config.get("max_length", 512)),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        use_fp16=bool(config.get("use_fp16", False)),
    )
    threshold = float(config.get("positive_threshold", 0.0))

    output_rows: list[dict] = []
    for row in tqdm(base_records, desc="Validating positive instructions", total=len(base_records)):
        positive = positives_by_id.get(row["sample_id"])
        if not positive or not positive.get("positive_instruction"):
            continue

        combined_query = f"{positive['positive_instruction'].strip()}\n{row['query'].strip()}".strip()
        score = judge.score(
            [(combined_query, row["positive_passage"])],
            batch_size=1,
        )[0]
        is_valid = bool(score >= threshold)

        output_rows.append(
            {
                "sample_id": row["sample_id"],
                "positive_instruction": positive["positive_instruction"].strip(),
                "relevance_reason": positive.get("relevance_reason", "").strip(),
                "validation_score": score,
                "is_valid": is_valid,
            }
        )

    output_path = Path(config["output_path"])
    write_jsonl(output_path, output_rows)
    _print_positive_summary(output_rows)
    return output_path


def validate_promptriever_passages(config_path: str | Path) -> Path:
    config = load_yaml(config_path)
    base_records = read_jsonl(config["base_records_path"])
    base_by_id = {row["sample_id"]: row for row in base_records}
    passage_rows = read_jsonl(config["generated_passages_path"])

    judge = RerankerJudge(
        model_name=config["judge_model"],
        device=str(config.get("device", "auto")),
        max_length=int(config.get("max_length", 512)),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        use_fp16=bool(config.get("use_fp16", False)),
    )
    query_positive_threshold = float(config.get("query_positive_threshold", 0.0))
    instruction_positive_threshold = float(config.get("instruction_positive_threshold", 0.0))
    instruction_negative_threshold = float(config.get("instruction_negative_threshold", 0.0))

    output_rows: list[dict] = []
    for row in tqdm(passage_rows, desc="Validating generated passages", total=len(passage_rows)):
        base = base_by_id.get(row["sample_id"])
        if not base:
            continue

        query = base["query"].strip()
        instruction = str(row.get("instruction", "")).strip()
        if not instruction:
            continue
        query_plus_instruction = f"{instruction}\n{query}".strip()

        generated_positive = str(row.get("generated_positive_passage", "")).strip()
        positive_score = judge.score(
            [(query_plus_instruction, generated_positive)],
            batch_size=1,
        )[0] if generated_positive else float("-inf")

        validated_negatives: list[str] = []
        negative_scores: list[dict] = []
        candidate_negatives = [
            str(text).strip() for text in row.get("instruction_negative_passages", []) if str(text).strip()
        ]
        if candidate_negatives:
            query_scores = judge.score(
                [(query, candidate) for candidate in candidate_negatives],
                batch_size=int(config.get("batch_size", 8)),
            )
            instruction_scores = judge.score(
                [(query_plus_instruction, candidate) for candidate in candidate_negatives],
                batch_size=int(config.get("batch_size", 8)),
            )
            for candidate, q_score, qi_score in zip(
                candidate_negatives,
                query_scores,
                instruction_scores,
                strict=False,
            ):
                if q_score >= query_positive_threshold and qi_score <= instruction_negative_threshold:
                    validated_negatives.append(candidate)
                negative_scores.append(
                    {
                        "passage": candidate,
                        "query_score": q_score,
                        "query_instruction_score": qi_score,
                    }
                )

        output_rows.append(
            {
                "sample_id": row["sample_id"],
                "instruction": instruction,
                "generated_positive_passage": generated_positive,
                "generated_positive_score": positive_score,
                "generated_positive_is_valid": bool(positive_score >= instruction_positive_threshold),
                "instruction_negative_passages": validated_negatives,
                "instruction_negative_scores": negative_scores,
                "positive_rationale": row.get("positive_rationale", ""),
                "negative_rationales": row.get("negative_rationales", []),
            }
        )

    output_path = Path(config["output_path"])
    write_jsonl(output_path, output_rows)
    _print_passage_summary(output_rows)
    return output_path


def apply_positive_thresholds(config_path: str | Path) -> Path:
    config = load_yaml(config_path)
    rows = read_jsonl(config["input_path"])
    threshold = float(config.get("positive_threshold", 0.0))

    output_rows: list[dict] = []
    for row in rows:
        updated_row = dict(row)
        updated_row["is_valid"] = bool(float(row["validation_score"]) >= threshold)
        output_rows.append(updated_row)

    output_path = Path(config.get("output_path", config["input_path"]))
    write_jsonl(output_path, output_rows)
    _print_positive_summary(output_rows)
    return output_path


def apply_passage_thresholds(config_path: str | Path) -> Path:
    config = load_yaml(config_path)
    rows = read_jsonl(config["input_path"])
    instruction_positive_threshold = float(config.get("instruction_positive_threshold", 0.0))
    query_positive_threshold = float(config.get("query_positive_threshold", 0.0))
    instruction_negative_threshold = float(config.get("instruction_negative_threshold", 0.0))

    output_rows: list[dict] = []
    for row in rows:
        updated_row = dict(row)
        updated_row["generated_positive_is_valid"] = bool(
            float(row.get("generated_positive_score", float("-inf"))) >= instruction_positive_threshold
        )

        validated_negatives: list[str] = []
        for score_row in row.get("instruction_negative_scores", []):
            query_score = float(score_row["query_score"])
            query_instruction_score = float(score_row["query_instruction_score"])
            if query_score >= query_positive_threshold and query_instruction_score <= instruction_negative_threshold:
                validated_negatives.append(str(score_row["passage"]).strip())

        updated_row["instruction_negative_passages"] = validated_negatives
        output_rows.append(updated_row)

    output_path = Path(config.get("output_path", config["input_path"]))
    write_jsonl(output_path, output_rows)
    _print_passage_summary(output_rows)
    return output_path
