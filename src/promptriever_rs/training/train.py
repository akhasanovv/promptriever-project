from __future__ import annotations

import inspect
import json
import random
from pathlib import Path

from tqdm.auto import tqdm

from promptriever_rs.config import ensure_dir, load_yaml
from promptriever_rs.models.registry import load_model_spec
from promptriever_rs.training.dataset_adapters import (
    build_binary_instruction_pairs,
    build_mnrl_pairs,
)
from promptriever_rs.utils.device import resolve_device
from promptriever_rs.utils.io import read_jsonl


def _require_training_stack():
    try:
        import torch
        from accelerate import Accelerator
        from sentence_transformers import SentenceTransformer, losses
        from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError(
            "Training dependencies are missing. Install with `pip install -e .[train]`."
        ) from exc
    return torch, SentenceTransformer, losses, DataLoader, Accelerator, EmbeddingSimilarityEvaluator


def _patch_accelerate_unwrap_model_if_needed(Accelerator) -> None:
    signature = inspect.signature(Accelerator.unwrap_model)
    if "keep_torch_compile" in signature.parameters:
        return

    original_method = Accelerator.unwrap_model

    def _compat_unwrap_model(self, model, keep_fp32_wrapper: bool = True, keep_torch_compile: bool = False):
        return original_method(self, model, keep_fp32_wrapper=keep_fp32_wrapper)

    Accelerator.unwrap_model = _compat_unwrap_model


def _split_records(records: list[dict], train_splits: set[str], eval_splits: set[str]) -> tuple[list[dict], list[dict]]:
    train_rows = [row for row in records if row["split"] in train_splits]
    eval_rows = [row for row in records if row["split"] in eval_splits]
    return train_rows, eval_rows


def _build_binary_evaluator(eval_rows: list[dict], model_spec, EmbeddingSimilarityEvaluator, batch_size: int):
    if not eval_rows:
        return None

    sentences1: list[str] = []
    sentences2: list[str] = []
    scores: list[float] = []

    for row in tqdm(eval_rows, desc="Preparing validation pairs", total=len(eval_rows)):
        passage = model_spec.format_document(row["positive_passage"])
        sentences1.append(model_spec.format_query(row["query"], row["positive_instruction"]))
        sentences2.append(passage)
        scores.append(1.0)
        sentences1.append(model_spec.format_query(row["query"], row["negative_instruction"]))
        sentences2.append(passage)
        scores.append(0.0)

    return EmbeddingSimilarityEvaluator(
        sentences1=sentences1,
        sentences2=sentences2,
        scores=scores,
        batch_size=batch_size,
        name="validation",
        show_progress_bar=True,
        write_csv=True,
    )


def fit(config_path: str | Path) -> Path:
    config = load_yaml(config_path)
    model_spec = load_model_spec(config["model_config"])
    output_dir = ensure_dir(config["output_dir"])

    seed = int(config.get("seed", 42))
    random.seed(seed)

    records = read_jsonl(config["dataset_path"])
    train_rows, eval_rows = _split_records(
        records,
        set(config.get("train_splits", ["train"])),
        set(config.get("eval_splits", ["validation"])),
    )

    if config.get("max_train_samples"):
        train_rows = train_rows[: int(config["max_train_samples"])]
    if config.get("max_eval_samples"):
        eval_rows = eval_rows[: int(config["max_eval_samples"])]

    torch, SentenceTransformer, losses, DataLoader, Accelerator, EmbeddingSimilarityEvaluator = _require_training_stack()
    _patch_accelerate_unwrap_model_if_needed(Accelerator)
    torch.manual_seed(seed)

    device = resolve_device(torch, config.get("device", "auto"))
    model = SentenceTransformer(model_spec.hf_id, device=device)

    mode = config.get("training_mode", "binary_instruction_pairs")
    if mode == "binary_instruction_pairs":
        train_dataset = build_binary_instruction_pairs(train_rows, model_spec)
        eval_dataset = build_binary_instruction_pairs(eval_rows, model_spec)
        loss = losses.CosineSimilarityLoss(model)
        evaluator = _build_binary_evaluator(
            eval_rows,
            model_spec,
            EmbeddingSimilarityEvaluator,
            batch_size=int(config.get("eval_batch_size", config.get("batch_size", 32))),
        )
    elif mode == "mnrl":
        train_dataset = build_mnrl_pairs(train_rows, model_spec)
        eval_dataset = build_mnrl_pairs(eval_rows, model_spec)
        loss = losses.MultipleNegativesRankingLoss(model)
        evaluator = None
    else:
        raise ValueError(f"Unsupported training_mode: {mode}")

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=int(config.get("batch_size", 32)),
    )
    warmup_steps = max(
        1,
        int(len(train_dataloader) * float(config.get("num_epochs", 1)) * float(config.get("warmup_ratio", 0.1))),
    )
    evaluation_steps = int(config.get("evaluation_steps", 0)) if evaluator is not None else 0

    use_fp16 = bool(config.get("use_fp16", False))
    if device != "cuda" and use_fp16:
        print(f"Disabling fp16 because device='{device}' does not support the CUDA fp16 training path.")
        use_fp16 = False

    print(
        "Training summary: "
        f"device={device}, train_records={len(train_rows)}, train_examples={len(train_dataset)}, "
        f"val_records={len(eval_rows)}, val_examples={len(eval_dataset)}, "
        f"epochs={int(config.get('num_epochs', 1))}, batch_size={int(config.get('batch_size', 32))}"
    )
    if evaluator is not None:
        if evaluation_steps > 0:
            print(f"Validation will run every {evaluation_steps} training steps with a visible progress bar.")
        else:
            print("Validation will run at the end of each epoch with a visible progress bar.")
        print("Validation uses the current in-memory model state from the training loop.")
    else:
        print("Validation evaluator is disabled for this training mode.")

    model.fit(
        train_objectives=[(train_dataloader, loss)],
        evaluator=evaluator,
        epochs=int(config.get("num_epochs", 1)),
        warmup_steps=warmup_steps,
        evaluation_steps=evaluation_steps,
        output_path=str(output_dir / "model"),
        optimizer_params={"lr": float(config.get("learning_rate", 1e-4))},
        save_best_model=False,
        use_amp=use_fp16,
        checkpoint_path=str(output_dir / "checkpoints"),
        checkpoint_save_steps=int(config.get("save_every_steps", 500)),
        show_progress_bar=True,
    )
    model.save(str(output_dir / "model"))

    manifest = {
        "run_name": config["run_name"],
        "model_name": model_spec.name,
        "hf_id": model_spec.hf_id,
        "device": device,
        "training_mode": mode,
        "evaluation_steps": evaluation_steps,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "output_dir": str(output_dir),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return output_dir / "model"
