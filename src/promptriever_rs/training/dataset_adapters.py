from __future__ import annotations

from typing import Any

from tqdm.auto import tqdm


def require_sentence_transformers() -> Any:
    try:
        from sentence_transformers import InputExample
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for training. Install with `pip install -e .[train]`."
        ) from exc
    return InputExample


def build_binary_instruction_pairs(records: list[dict], model_spec) -> list[Any]:
    InputExample = require_sentence_transformers()
    samples: list[Any] = []
    for row in tqdm(records, desc="Building binary train pairs", total=len(records)):
        positive_query = model_spec.format_query(row["query"], row["positive_instruction"])
        negative_query = model_spec.format_query(row["query"], row["negative_instruction"])
        passage = model_spec.format_document(row["positive_passage"])

        samples.append(InputExample(texts=[positive_query, passage], label=1.0))
        samples.append(InputExample(texts=[negative_query, passage], label=0.0))
    return samples


def build_mnrl_pairs(records: list[dict], model_spec) -> list[Any]:
    InputExample = require_sentence_transformers()
    samples: list[Any] = []
    for row in tqdm(records, desc="Building MNRL pairs", total=len(records)):
        positive_query = model_spec.format_query(row["query"], row["positive_instruction"])
        passage = model_spec.format_document(row["positive_passage"])
        samples.append(InputExample(texts=[positive_query, passage]))
    return samples


def build_promptriever_examples(
    records: list[dict],
    model_spec,
    negatives_per_sample: int = 3,
    include_hard_negatives: bool = True,
) -> list[Any]:
    InputExample = require_sentence_transformers()
    samples: list[Any] = []
    target_negatives = max(1, int(negatives_per_sample))

    for row in tqdm(records, desc="Building Promptriever examples", total=len(records)):
        anchor = model_spec.format_query(row["query"], row.get("instruction"))
        positive = model_spec.format_document(row["positive_passage"])

        negatives: list[str] = []
        negatives.extend(row.get("instruction_negative_passages", []))
        if include_hard_negatives:
            negatives.extend(row.get("hard_negative_passages", []))
        cleaned_negatives = [model_spec.format_document(text) for text in negatives if str(text).strip()]
        if not cleaned_negatives:
            continue

        selected_negatives = cleaned_negatives[:target_negatives]
        while len(selected_negatives) < target_negatives:
            selected_negatives.append(selected_negatives[-1])

        samples.append(InputExample(texts=[anchor, positive, *selected_negatives]))
    return samples
