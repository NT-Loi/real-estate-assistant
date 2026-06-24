"""
Vietnamese reranker wrapper for retrieval candidates.

Uses AITeamVN/Vietnamese_Reranker as a cross-encoder style model. The wrapper
is lazy and fail-open: if the model is unavailable, callers keep their original
retrieval ordering.
"""
from __future__ import annotations

import logging
from typing import Iterable

from db.config import RERANKER_MAX_LENGTH, RERANKER_MODEL

log = logging.getLogger("bds_reranker")


class VietnameseReranker:
    """Lazy wrapper around a sequence-classification reranker."""

    def __init__(self, model_name: str = RERANKER_MODEL):
        self._model_name = model_name
        self._tokenizer = None
        self._model = None
        self._device = None
        self._disabled = False

    def _load(self) -> bool:
        if self._disabled:
            return False
        if self._model is not None and self._tokenizer is not None:
            return True

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("Loading reranker model: %s", self._model_name)
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._model_name,
                    local_files_only=True,
                )
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self._model_name,
                    local_files_only=True,
                )
            except Exception:
                log.info("Reranker not cached yet; attempting HuggingFace download")
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)

            self._model.to(self._device)
            self._model.eval()
            return True
        except Exception as exc:
            log.warning("Reranker disabled because it could not load: %s", exc)
            self._disabled = True
            return False

    def score(self, query: str, passages: Iterable[str], batch_size: int = 8) -> list[float]:
        """Return reranker scores for query/passage pairs."""
        passages = list(passages)
        if not passages or not self._load():
            return []

        import torch

        scores: list[float] = []
        pairs = [[query, passage] for passage in passages]
        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=RERANKER_MAX_LENGTH,
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                logits = self._model(**inputs, return_dict=True).logits.view(-1).float()
                scores.extend(float(x) for x in logits.detach().cpu())
        return scores

    def rerank(self, query: str, docs: list, top_k: int) -> list:
        """Return docs sorted by reranker score, preserving docs on failure."""
        if not docs:
            return docs
        scores = self.score(query, [getattr(doc, "text", "") for doc in docs])
        if len(scores) != len(docs):
            return docs[:top_k]

        for doc, score in zip(docs, scores):
            doc.metadata = {**(doc.metadata or {}), "reranker_score": score}
            doc.score = score

        return sorted(docs, key=lambda d: d.score, reverse=True)[:top_k]
