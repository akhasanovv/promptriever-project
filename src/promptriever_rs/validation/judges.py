from __future__ import annotations

from dataclasses import dataclass

from promptriever_rs.utils.device import resolve_device


@dataclass(slots=True)
class PairScore:
    text_a: str
    text_b: str
    score: float


class RerankerJudge:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_length: int = 512,
        trust_remote_code: bool = False,
        use_fp16: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for validation. Install the project dependencies first."
            ) from exc

        self._torch = torch
        self.device = resolve_device(torch, device)
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        self.model.to(self.device)

        self._autocast_enabled = bool(use_fp16 and self.device == "cuda")

    def score(self, pairs: list[tuple[str, str]], batch_size: int = 8) -> list[float]:
        scores: list[float] = []

        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            features = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )
            features = {key: value.to(self.device) for key, value in features.items()}

            with self._torch.no_grad():
                if self._autocast_enabled:
                    with self._torch.autocast(device_type="cuda", dtype=self._torch.float16):
                        logits = self.model(**features, return_dict=True).logits.view(-1).float()
                else:
                    logits = self.model(**features, return_dict=True).logits.view(-1).float()
            scores.extend(float(value) for value in logits.cpu())

        return scores
