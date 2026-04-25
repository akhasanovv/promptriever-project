from __future__ import annotations

import json
import os
import time
from pathlib import Path

from tqdm.auto import tqdm

from promptriever_rs.config import load_yaml
from promptriever_rs.utils.io import append_jsonl, read_jsonl


def _build_negative_user_prompt(record: dict) -> str:
    return (
        "Сгенерируй одну негативную инструкцию для retrieval-модели.\n\n"
        f"Вопрос пользователя:\n{record['query']}\n\n"
        f"Позитивный контекст:\n{record['positive_passage']}\n\n"
        f"Короткий ответ:\n{record.get('answer', '')}\n\n"
        "Нужна инструкция, которая тематически близка к вопросу, "
        "но делает этот контекст неподходящим."
    )


def _build_positive_user_prompt(record: dict) -> str:
    return (
        "Сгенерируй одну позитивную инструкцию для retrieval-модели.\n\n"
        f"Вопрос пользователя:\n{record['query']}\n\n"
        f"Позитивный контекст:\n{record['positive_passage']}\n\n"
        f"Короткий ответ:\n{record.get('answer', '')}\n\n"
        "Нужна инструкция, которая тематически соответствует вопросу "
        "и делает этот контекст подходящим и релевантным."
    )


def _call_groq(
    *,
    api_key: str,
    api_base: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    try:
        from groq import Groq
    except ImportError as exc:
        raise ImportError(
            "groq is required for instruction generation. Install it in your environment first."
        ) from exc

    client = Groq(api_key=api_key)

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        model=model,
    )

    content = chat_completion.choices[0].message.content
    parsed = json.loads(content)
    
    if "negative_instruction" not in parsed:
        raise ValueError("Groq response does not contain 'negative_instruction'.")
    return parsed


def _prepare_generation_run(config_path: str | Path) -> tuple[dict, list[dict], set[str], int]:
    config = load_yaml(config_path)
    input_path = Path(config["input_path"])
    records = read_jsonl(input_path)
    start_index = int(config.get("start_index", 0))

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")

    if start_index < 0:
        raise ValueError("start_index must be non-negative.")
    if start_index > len(records):
        raise ValueError(
            f"start_index={start_index} is larger than the dataset size ({len(records)})."
        )

    existing_ids: set[str] = set()
    output_path = Path(config["output_path"])
    if config.get("resume", True) and output_path.exists():
        existing_ids = {row["sample_id"] for row in read_jsonl(output_path)}

    candidate_records = records[start_index:]
    remaining_records = [
        record for record in candidate_records if record["sample_id"] not in existing_ids
    ]
    return config, remaining_records, existing_ids, len(records)


def generate_negative_instructions(config_path: str | Path) -> Path:
    config, remaining_records, existing_ids, total_records = _prepare_generation_run(config_path)
    output_path = Path(config["output_path"])
    delay = 60.0 / max(int(config.get("requests_per_minute", 25)), 1)
    start_index = int(config.get("start_index", 0))

    print(
        "Negative generation summary: "
        f"dataset_size={total_records}, start_index={start_index}, "
        f"already_generated={len(existing_ids)}, to_generate={len(remaining_records)}"
    )

    for offset, record in enumerate(
        tqdm(remaining_records, desc="Generating negatives", total=len(remaining_records)),
        start=1,
    ):
        if record["sample_id"] in existing_ids:
            continue

        parsed = _call_groq(
            api_key=api_key,
            api_base=config["api_base"],
            model=config["model"],
            system_prompt=config["system_prompt"],
            user_prompt=_build_negative_user_prompt(record),
            temperature=float(config.get("temperature", 0.2)),
            max_tokens=int(config.get("max_tokens", 220)),
        )
        append_jsonl(
            output_path,
            {
                "sample_id": record["sample_id"],
                "negative_instruction": parsed["negative_instruction"].strip(),
                "violation_reason": parsed.get("violation_reason", "").strip(),
            },
        )
        if offset == 1 or offset % 50 == 0:
            original_index = start_index + offset - 1
            print(
                f"Saved negative instruction {offset}/{len(remaining_records)} "
                f"(approx original index >= {original_index}, sample_id={record['sample_id']})"
            )
        time.sleep(delay)

    return output_path


def generate_positive_instructions(config_path: str | Path) -> Path:
    config, remaining_records, existing_ids, total_records = _prepare_generation_run(config_path)
    output_path = Path(config["output_path"])
    delay = 60.0 / max(int(config.get("requests_per_minute", 25)), 1)
    start_index = int(config.get("start_index", 0))

    print(
        "Positive generation summary: "
        f"dataset_size={total_records}, start_index={start_index}, "
        f"already_generated={len(existing_ids)}, to_generate={len(remaining_records)}"
    )

    for offset, record in enumerate(
        tqdm(remaining_records, desc="Generating positives", total=len(remaining_records)),
        start=1,
    ):
        parsed = _call_groq(
            api_key=os.getenv("GROQ_API_KEY", ""),
            api_base=config["api_base"],
            model=config["model"],
            system_prompt=config["system_prompt"],
            user_prompt=_build_positive_user_prompt(record),
            temperature=float(config.get("temperature", 0.2)),
            max_tokens=int(config.get("max_tokens", 220)),
        )
        if "positive_instruction" not in parsed:
            raise ValueError("Groq response does not contain 'positive_instruction'.")

        append_jsonl(
            output_path,
            {
                "sample_id": record["sample_id"],
                "positive_instruction": parsed["positive_instruction"].strip(),
                "relevance_reason": parsed.get("relevance_reason", "").strip(),
            },
        )
        if offset == 1 or offset % 50 == 0:
            original_index = start_index + offset - 1
            print(
                f"Saved positive instruction {offset}/{len(remaining_records)} "
                f"(approx original index >= {original_index}, sample_id={record['sample_id']})"
            )
        time.sleep(delay)

    return output_path
