from __future__ import annotations

from pathlib import Path

from promptriever_rs.evaluation.mteb_eval import evaluate_mteb


def evaluate_mfollowir(config_path: str | Path) -> Path:
    return evaluate_mteb(config_path)
