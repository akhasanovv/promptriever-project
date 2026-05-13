from __future__ import annotations

import json
from pathlib import Path

from tqdm.auto import tqdm

from promptriever_rs.config import ensure_dir, load_yaml
from promptriever_rs.models.registry import load_model_spec
from promptriever_rs.utils.device import resolve_device


def _require_eval_stack():
    try:
        import torch
        import mteb
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Evaluation dependencies are missing. Install with `pip install -e .[eval]`."
        ) from exc
    return torch, mteb, SentenceTransformer


def _build_mteb_model(
    *,
    mteb_module,
    sentence_transformer,
    model_path: str,
    device: str,
    query_prefix: str,
    document_prefix: str,
    model_name_override: str | None = None,
):
    model = sentence_transformer(
        model_path,
        device=device,
        prompts={
            "query": query_prefix.strip(),
            "document": document_prefix.strip(),
        },
    )

    wrapper = mteb_module.SentenceTransformerEncoderWrapper(model)
    model_name = model_name_override or str(Path(model_path).resolve())
    wrapper.mteb_model_meta.name = model_name
    wrapper.mteb_model_meta.revision = "local"
    return wrapper


def evaluate_mteb(config_path: str | Path) -> Path:
    config = load_yaml(config_path)
    model_spec = load_model_spec(config["model_config"])
    torch, mteb, SentenceTransformer = _require_eval_stack()
    device = resolve_device(torch, config.get("device", "auto"))

    model = _build_mteb_model(
        mteb_module=mteb,
        sentence_transformer=SentenceTransformer,
        model_path=config["model_path"],
        device=device,
        query_prefix=model_spec.query_prefix,
        document_prefix=model_spec.document_prefix,
        model_name_override=config.get("model_name"),
    )

    tasks = list(mteb.get_tasks(tasks=config["tasks"], languages=config.get("languages")))
    task_results: list[dict] = []
    use_mteb_cache = bool(config.get("use_mteb_cache", False))

    for task in tqdm(tasks, desc="Evaluating tasks", total=len(tasks)):
        task_name = getattr(task.metadata, "name", None) or getattr(task, "name", str(task))
        print(f"Running evaluation for task: {task_name}")
        result = mteb.evaluate(
            model,
            [task],
            encode_kwargs={"batch_size": int(config.get("batch_size", 64))},
            cache=mteb.ResultCache() if use_mteb_cache else None,
            overwrite_strategy="always",
            show_progress_bar=True,
        )
        if hasattr(result, "to_dict"):
            task_results.append(result.to_dict())
        elif hasattr(result, "to_dataframe"):
            task_results.append(
                {
                    "task_name": task_name,
                    "rows": result.to_dataframe().to_dict(orient="records"),
                }
            )
        else:
            task_results.append({"task_name": task_name, "results": str(result)})

    output_path = Path(config["output_path"])
    ensure_dir(output_path.parent)
    payload = {
        "model_path": config["model_path"],
        "model_name": config.get("model_name", str(Path(config["model_path"]).resolve())),
        "device": device,
        "tasks": task_results,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return output_path
