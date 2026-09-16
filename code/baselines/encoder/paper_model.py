"""Table 22 building blocks; new implementation, not historical checkpoints."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

try:
    import torch
except ImportError:
    torch = None


@dataclass(frozen=True)
class LogTargetScaler:
    mean: float
    scale: float

    @classmethod
    def fit(cls, train_eur):
        values = np.asarray(train_eur, dtype=float)
        if values.size == 0 or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("training targets must be nonempty finite nonnegative EUR")
        z = np.log1p(values)
        return cls(float(z.mean()), float(z.std()) or 1.0)

    def transform(self, eur):
        values = np.asarray(eur, dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("targets must be finite nonnegative EUR")
        return (np.log1p(values) - self.mean) / self.scale

    def inverse(self, standardized):
        values = np.asarray(standardized, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("predictions must be finite")
        with np.errstate(over="raise"):
            return np.maximum(np.expm1(values * self.scale + self.mean), 0.0)


def encode_chunks(text: str, tokenizer: Any, architecture: str):
    """Special tokens count toward each 4096/8192-token budget.

    Legal-Longformer uses the first two nonoverlapping chunks. Empty trailing
    chunks are masked out of CLS pooling; this padding convention is a newly
    documented choice because it is not specified in the paper.
    """
    if architecture not in {"modernbert", "legal_longformer"}:
        raise ValueError("unknown architecture")
    length, count = (8192, 1) if architecture == "modernbert" else (4096, 2)
    capacity = length - tokenizer.num_special_tokens_to_add(pair=False)
    if capacity <= 0:
        raise ValueError("invalid tokenizer special-token budget")
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=capacity * count)
    result = {"input_ids": [], "attention_mask": [], "chunk_mask": []}
    if architecture == "legal_longformer":
        result["global_attention_mask"] = []
    for index in range(count):
        payload = tokens[index * capacity:(index + 1) * capacity]
        prepared = tokenizer.prepare_for_model(payload, add_special_tokens=True, padding="max_length", max_length=length,
                                               truncation=True, return_attention_mask=True)
        result["input_ids"].append(prepared["input_ids"])
        result["attention_mask"].append(prepared["attention_mask"])
        present = bool(payload) or index == 0
        result["chunk_mask"].append(float(present))
        if architecture == "legal_longformer":
            # Longformer needs a globally attended CLS representation.
            result["global_attention_mask"].append([1 if present else 0] + [0] * (length - 1))
    return result


if torch is not None:
    class PaperEncoderRegressor(torch.nn.Module):
        def __init__(self, backbone, *, structured_dim=0, activation=None):
            super().__init__()
            self.backbone = backbone
            self.config = backbone.config
            hidden = int(backbone.config.hidden_size)
            if structured_dim:
                if activation not in {"relu", "gelu"}:
                    raise ValueError("choose an explicit MLP activation; historical activation is unknown")
                nonlinearity = torch.nn.ReLU() if activation == "relu" else torch.nn.GELU()
                self.structured = torch.nn.Sequential(torch.nn.Linear(structured_dim, 256), nonlinearity,
                                                      torch.nn.Linear(256, 768))
            else:
                self.structured = None
            self.regression = torch.nn.Linear(hidden + (768 if structured_dim else 0), 1)

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        def forward(self, input_ids, attention_mask, chunk_mask, labels=None, structured_features=None,
                    global_attention_mask=None):
            batch, chunks, length = input_ids.shape
            kwargs = {"input_ids": input_ids.reshape(batch * chunks, length),
                      "attention_mask": attention_mask.reshape(batch * chunks, length)}
            if global_attention_mask is not None:
                kwargs["global_attention_mask"] = global_attention_mask.reshape(batch * chunks, length)
            hidden = self.backbone(**kwargs).last_hidden_state[:, 0].reshape(batch, chunks, -1)
            mask = chunk_mask.to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            if self.structured is not None:
                if structured_features is None:
                    raise ValueError("late fusion requires structured features")
                pooled = torch.cat((pooled, self.structured(structured_features.to(pooled.dtype))), dim=-1)
            elif structured_features is not None:
                raise ValueError("text-only model cannot silently ignore supplied structured features")
            logits = self.regression(pooled).squeeze(-1)
            output = {"logits": logits}
            if labels is not None:
                output["loss"] = torch.nn.functional.smooth_l1_loss(logits.float(), labels.float(), beta=1.0)
            return output
else:
    class PaperEncoderRegressor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for the paper encoder model")
