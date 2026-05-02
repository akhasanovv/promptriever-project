from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from promptriever_rs.config import load_yaml
from promptriever_rs.models.registry import load_model_spec
from promptriever_rs.utils.io import write_jsonl, read_jsonl
from promptriever_rs.utils.device import resolve_device
from promptriever_rs.validation.judges import RerankerJudge


def _require_sentence_transformers():
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for hard-negative mining."
        ) from exc
    return SentenceTransformer, torch


def mine_hard_negatives(config_path: str | Path) -> Path:
    config = load_yaml(config_path)
    records = read_jsonl(config["base_records_path"])
    model_spec = load_model_spec(config["retrieval_model_config"])
    SentenceTransformer, torch = _require_sentence_transformers()

    corpus_texts: list[str] = []
    corpus_index_by_text: dict[str, int] = {}
    for row in records:
        passage = str(row["positive_passage"]).strip()
        if passage and passage not in corpus_index_by_text:
            corpus_index_by_text[passage] = len(corpus_texts)
            corpus_texts.append(passage)

    device = resolve_device(torch, str(config.get("device", "auto")))
    model = SentenceTransformer(model_spec.hf_id, device=device)
    encoded_corpus = model.encode(
        [model_spec.format_document(text) for text in corpus_texts],
        batch_size=int(config.get("embedding_batch_size", 64)),
        show_progress_bar=True,
        normalize_embeddings=bool(model_spec.normalize_embeddings),
        convert_to_numpy=True,
    )
    encoded_queries = model.encode(
        [model_spec.format_query(row["query"], None) for row in records],
        batch_size=int(config.get("embedding_batch_size", 64)),
        show_progress_bar=True,
        normalize_embeddings=bool(model_spec.normalize_embeddings),
        convert_to_numpy=True,
    )

    top_k = max(int(config.get("retrieval_top_k", 50)), int(config.get("num_hard_negatives", 2)) + 1)
    num_hard_negatives = int(config.get("num_hard_negatives", 2))
    judge = RerankerJudge(
        model_name=config["judge_model"],
        device=str(config.get("judge_device", config.get("device", "auto"))),
        max_length=int(config.get("judge_max_length", 512)),
        trust_remote_code=bool(config.get("judge_trust_remote_code", False)),
        use_fp16=bool(config.get("judge_use_fp16", False)),
    )
    query_negative_threshold = float(config.get("query_negative_threshold", 0.0))

    output_rows: list[dict] = []
    for row, query_embedding in tqdm(
        list(zip(records, encoded_queries, strict=False)),
        desc="Mining hard negatives",
        total=len(records),
    ):
        similarities = np.matmul(encoded_corpus, query_embedding)
        ranked_indices = np.argsort(-similarities)[:top_k]

        candidate_passages: list[str] = []
        positive_passage = str(row["positive_passage"]).strip()
        for corpus_idx in ranked_indices:
            candidate = corpus_texts[int(corpus_idx)]
            if candidate == positive_passage:
                continue
            candidate_passages.append(candidate)

        if not candidate_passages:
            continue

        judge_scores = judge.score(
            [(row["query"], candidate) for candidate in candidate_passages],
            batch_size=int(config.get("judge_batch_size", 8)),
        )

        kept_passages: list[str] = []
        kept_scores: list[float] = []
        for candidate, score in zip(candidate_passages, judge_scores, strict=False):
            if score <= query_negative_threshold:
                kept_passages.append(candidate)
                kept_scores.append(score)
            if len(kept_passages) >= num_hard_negatives:
                break

        output_rows.append(
            {
                "sample_id": row["sample_id"],
                "hard_negative_passages": kept_passages,
                "hard_negative_scores": kept_scores,
            }
        )

    output_path = Path(config["output_path"])
    write_jsonl(output_path, output_rows)
    return output_path
