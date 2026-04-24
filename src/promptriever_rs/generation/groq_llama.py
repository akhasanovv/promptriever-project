from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
from tqdm import tqdm

from groq import Groq

from promptriever_rs.config import load_yaml
from promptriever_rs.utils.io import append_jsonl, read_jsonl


def _build_user_prompt(record: dict) -> str:
    return (
        "Сгенерируй одну негативную инструкцию для retrieval-модели.\n\n"
        f"Вопрос пользователя:\n{record['query']}\n\n"
        f"Позитивный контекст:\n{record['positive_passage']}\n\n"
        f"Короткий ответ:\n{record.get('answer', '')}\n\n"
        "Нужна инструкция, которая тематически близка к вопросу, "
        "но делает этот контекст неподходящим."
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
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        model="llama-3.1-8b-instant",
    )

    content = chat_completion.choices[0].message.content
    parsed = json.loads(content)
    
    if "negative_instruction" not in parsed:
        raise ValueError("Groq response does not contain 'negative_instruction'.")
    return parsed


def generate_negative_instructions(config_path: str | Path) -> Path:
    config = load_yaml(config_path)
    input_path = Path(config["input_path"])
    output_path = Path(config["output_path"])
    records = read_jsonl(input_path)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")

    existing_ids: set[str] = set()
    if config.get("resume", True) and output_path.exists():
        existing_ids = {row["sample_id"] for row in read_jsonl(output_path)}

    delay = 60.0 / max(int(config.get("requests_per_minute", 25)), 1)

    for record in tqdm(records, desc="Generating negatives"):
        if record["sample_id"] in existing_ids:
            continue

        parsed = _call_groq(
            api_key=api_key,
            api_base=config["api_base"],
            model=config["model"],
            system_prompt=config["system_prompt"],
            user_prompt=_build_user_prompt(record),
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
        time.sleep(delay)

    return output_path
