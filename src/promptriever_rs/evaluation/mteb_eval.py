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


def _find_lora_adapter_dir(model_path: str | Path) -> Path | None:
    path = Path(model_path)
    candidates = [path, path / "0"]
    for candidate in candidates:
        if (candidate / "adapter_config.json").is_file():
            return candidate
    if path.is_dir():
        matches = sorted(path.rglob("adapter_config.json"))
        if matches:
            return matches[0].parent
    return None


def _local_weight_files_size_bytes(model_path: str | Path) -> int:
    path = Path(model_path)
    if not path.is_dir():
        return 0
    patterns = ("*.safetensors", "*.bin")
    total = 0
    for pattern in patterns:
        total += sum(file.stat().st_size for file in path.rglob(pattern) if file.is_file())
    return total


def _format_bytes(size: int) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GiB"
    if size >= 1024**2:
        return f"{size / 1024**2:.2f} MiB"
    if size >= 1024:
        return f"{size / 1024:.2f} KiB"
    return f"{size} B"


def _validate_local_sentence_transformer_artifact(model_path: str | Path) -> dict:
    path = Path(model_path)
    total_weight_size = _local_weight_files_size_bytes(path)
    info = {
        "local_weight_size_bytes": total_weight_size,
        "local_weight_size": _format_bytes(total_weight_size),
    }
    if path.is_dir() and total_weight_size and total_weight_size < 100 * 1024**2:
        raise ValueError(
            f"Model path '{model_path}' does not contain a LoRA adapter_config.json, "
            f"but its local weight files are only {_format_bytes(total_weight_size)}. "
            "This looks like a partial PEFT/LoRA checkpoint saved as a full SentenceTransformer model. "
            "Evaluate the actual adapter directory containing adapter_config.json, or retrain/resave with "
            "the fixed training code so LoRA is merged into the base model before saving."
        )
    return info


def _load_sentence_transformer(
    *,
    sentence_transformer,
    model_path: str,
    device: str,
    prompts: dict[str, str],
    base_model_id: str,
):
    adapter_dir = _find_lora_adapter_dir(model_path)
    if adapter_dir is None:
        artifact_info = _validate_local_sentence_transformer_artifact(model_path)
        return sentence_transformer(model_path, device=device, prompts=prompts), {
            "mode": "sentence_transformer",
            "model_path": str(model_path),
            "lora_adapter_dir": None,
            "base_model": None,
            "lora_merged": False,
            **artifact_info,
        }

    try:
        from peft import PeftConfig, PeftModel
    except ImportError as exc:
        raise ImportError(
            "The evaluation model path contains a LoRA adapter, but PEFT is not installed. "
            "Install with `pip install -e .[eval]` or `pip install peft`."
        ) from exc

    peft_config = PeftConfig.from_pretrained(str(adapter_dir))
    base_name = base_model_id or peft_config.base_model_name_or_path
    if not base_name:
        raise ValueError(
            f"LoRA adapter at {adapter_dir} does not declare a base model; "
            "set `hf_id` in the model config."
        )

    print(f"Loading base model '{base_name}' and applying LoRA adapter from {adapter_dir}.")
    model = sentence_transformer(base_name, device=device, prompts=prompts)
    if len(model) == 0 or not hasattr(model[0], "auto_model"):
        raise ValueError("SentenceTransformer backbone does not expose `auto_model` for LoRA loading.")

    peft_model = PeftModel.from_pretrained(model[0].auto_model, str(adapter_dir), is_trainable=False)
    lora_merged = False
    if hasattr(peft_model, "merge_and_unload"):
        model[0].auto_model = peft_model.merge_and_unload()
        lora_merged = True
        print("Merged LoRA adapter into the base model for evaluation.")
    else:
        model[0].auto_model = peft_model
        print("Using LoRA adapter wrapper for evaluation.")
    adapter_weight_size = _local_weight_files_size_bytes(adapter_dir)
    return model, {
        "mode": "lora_adapter",
        "model_path": str(model_path),
        "lora_adapter_dir": str(adapter_dir),
        "base_model": str(base_name),
        "lora_merged": lora_merged,
        "local_weight_size_bytes": adapter_weight_size,
        "local_weight_size": _format_bytes(adapter_weight_size),
    }


def _build_mteb_model(
    *,
    mteb_module,
    sentence_transformer,
    model_path: str,
    device: str,
    query_prefix: str,
    document_prefix: str,
    base_model_id: str,
    normalize_embeddings: bool,
    model_name_override: str | None = None,
):
    prompts = {
        "query": query_prefix,
        "document": document_prefix,
    }
    model, load_info = _load_sentence_transformer(
        sentence_transformer=sentence_transformer,
        model_path=model_path,
        device=device,
        prompts=prompts,
        base_model_id=base_model_id,
    )

    try:
        native_wrapper = mteb_module.SentenceTransformerEncoderWrapper(model, model_prompts=prompts)
    except TypeError:
        native_wrapper = mteb_module.SentenceTransformerEncoderWrapper(model)
    model_name = model_name_override or str(Path(model_path).resolve())
    native_wrapper.mteb_model_meta.name = model_name
    native_wrapper.mteb_model_meta.revision = "local"
    native_wrapper.load_info = load_info
    native_wrapper.normalize_embeddings = normalize_embeddings
    return native_wrapper


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
        base_model_id=model_spec.hf_id,
        normalize_embeddings=bool(model_spec.normalize_embeddings),
        model_name_override=config.get("model_name"),
    )
    print(
        "Evaluation model load: "
        f"mode={model.load_info['mode']}, "
        f"adapter={model.load_info['lora_adapter_dir']}, "
        f"base={model.load_info['base_model']}, "
        f"weights={model.load_info['local_weight_size']}, "
        f"normalize_embeddings={bool(model_spec.normalize_embeddings)}"
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
            encode_kwargs={
                "batch_size": int(config.get("batch_size", 64)),
                "normalize_embeddings": bool(model_spec.normalize_embeddings),
            },
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
        "model_load": model.load_info,
        "normalize_embeddings": bool(model_spec.normalize_embeddings),
        "tasks": task_results,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return output_path
