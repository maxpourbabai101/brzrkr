"""FinBERT‑based sentiment encoder.

Wraps a pre‑trained FinBERT model (``ProsusAI/finbert`` by default) and
exposes a single helper, :func:`score_news`, that maps a string (or list
of strings) to a sentiment score in [-1, 1] plus an "event salience"
flag indicating whether at least one input crossed an absolute‑score
threshold.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, List, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("FINBERT_MODEL_NAME", "ProsusAI/finbert")
SALIENCE_THRESHOLD = 0.65  # |score| above this is treated as an event


@dataclass
class SentimentResult:
    score: float          # in [-1, 1]
    salience: bool        # event flag
    label_probs: dict     # raw per‑label probabilities


class SentimentEncoder:
    """Thin wrapper around a FinBERT classifier head."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        # Imported lazily so the rest of the codebase remains usable without
        # the heavy ``transformers`` dependency installed.
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading FinBERT model %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        # FinBERT labels: 0=positive, 1=negative, 2=neutral.
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def _batch_logits(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        return self.model(**inputs).logits

    def score_news(
        self,
        text: Union[str, Iterable[str]],
    ) -> SentimentResult:
        """Score one or many headlines and return an aggregate result.

        The aggregate score is the mean of per‑item ``P(pos) - P(neg)``
        values. ``salience`` is True iff any individual item's absolute
        score exceeds :data:`SALIENCE_THRESHOLD`.
        """
        texts = [text] if isinstance(text, str) else list(text)
        if not texts:
            return SentimentResult(score=0.0, salience=False, label_probs={})

        logits = self._batch_logits(texts)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        pos_idx = next(i for i, l in self.id2label.items() if "pos" in l.lower())
        neg_idx = next(i for i, l in self.id2label.items() if "neg" in l.lower())

        per_item = probs[:, pos_idx] - probs[:, neg_idx]
        agg = float(np.mean(per_item))
        salience = bool(np.max(np.abs(per_item)) >= SALIENCE_THRESHOLD)

        # Average label probabilities for diagnostics.
        avg_probs = {
            self.id2label[i]: float(np.mean(probs[:, i]))
            for i in range(probs.shape[1])
        }
        return SentimentResult(score=agg, salience=salience, label_probs=avg_probs)


# ---------------------------------------------------------------------------
# Module‑level convenience
# ---------------------------------------------------------------------------
_cached_encoder: SentimentEncoder | None = None


def score_news(text: Union[str, Iterable[str]]) -> SentimentResult:
    """Convenience wrapper that lazily instantiates a cached encoder."""
    global _cached_encoder
    if _cached_encoder is None:
        _cached_encoder = SentimentEncoder()
    return _cached_encoder.score_news(text)
