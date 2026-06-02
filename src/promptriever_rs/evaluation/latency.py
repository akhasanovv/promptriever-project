from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from promptriever_rs.config import ensure_dir, load_yaml
from promptriever_rs.evaluation.mteb_eval import _load_sentence_transformer
from promptriever_rs.models.registry import load_model_spec
from promptriever_rs.utils.device import resolve_device


def _require_latency_stack():
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Latency benchmark dependencies are missing. Install with `pip install -e .[eval]`."
        ) from exc
    return torch, SentenceTransformer


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile for an empty list.")
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sync_device(torch_module: Any, device: str) -> None:
    if device == "cuda":
        torch_module.cuda.synchronize()
    elif device == "mps" and hasattr(torch_module, "mps"):
        torch_module.mps.synchronize()


def benchmark_latency(config_path: str | Path) -> Path:
    config = load_yaml(config_path)
    model_spec = load_model_spec(config["model_config"])
    output_dir = ensure_dir(config["output_dir"])

    torch, SentenceTransformer = _require_latency_stack()
    device = resolve_device(torch, config.get("device", "auto"))
    model_path = str(config.get("model_path") or model_spec.hf_id)
    if config.get("model_path"):
        model, load_info = _load_sentence_transformer(
            sentence_transformer=SentenceTransformer,
            model_path=model_path,
            device=device,
            prompts={
                "query": model_spec.query_prefix,
                "document": model_spec.document_prefix,
            },
            base_model_id=model_spec.hf_id,
        )
        print(
            "Latency model load: "
            f"mode={load_info['mode']}, "
            f"adapter={load_info['lora_adapter_dir']}, "
            f"base={load_info['base_model']}"
        )
    else:
        model = SentenceTransformer(model_path, device=device)

    text_type = str(config.get("text_type", "query"))
    if text_type == "query":
        text = model_spec.format_query(
            str(config.get("query", "Кто написал роман Война и мир?")),
            instruction=config.get("instruction"),
        )
    elif text_type == "document":
        text = model_spec.format_document(
            str(config.get("document", "Лев Толстой написал роман Война и мир."))
        )
    else:
        raise ValueError("text_type must be either 'query' or 'document'.")

    batch_size = int(config.get("batch_size", 1))
    warmup_runs = int(config.get("warmup_runs", 20))
    runs = int(config.get("runs", 200))
    normalize = bool(config.get("normalize_embeddings", model_spec.normalize_embeddings))
    show_progress = bool(config.get("show_progress_bar", False))

    for _ in range(warmup_runs):
        model.encode(
            text,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
        )

    latencies_ms: list[float] = []
    for _ in range(runs):
        _sync_device(torch, device)
        started_at = time.perf_counter()
        model.encode(
            text,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
        )
        _sync_device(torch, device)
        latencies_ms.append((time.perf_counter() - started_at) * 1000.0)

    result = {
        "model_name": model_spec.name,
        "model_path": model_path,
        "device": device,
        "text_type": text_type,
        "batch_size": batch_size,
        "warmup_runs": warmup_runs,
        "runs": runs,
        "mean_ms": statistics.fmean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "p95_ms": _percentile(latencies_ms, 95),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }

    output_path = output_dir / "latency.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    print(
        "Latency: "
        f"mean={result['mean_ms']:.2f} ms, "
        f"median={result['median_ms']:.2f} ms, "
        f"p95={result['p95_ms']:.2f} ms"
    )
    return output_path
