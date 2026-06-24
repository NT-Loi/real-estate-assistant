"""
Embedder — thin wrapper around sentence-transformers for generating
text embeddings.

Uses the model specified in db.config.EMBEDDING_MODEL.
"""
from __future__ import annotations

import logging
from typing import Union

from sentence_transformers import SentenceTransformer

from db.config import EMBEDDING_MODEL, EMBEDDING_NORMALIZE

log = logging.getLogger("bds_embedder")


class Embedder:
    """Lazy-loading embedding model wrapper."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            log.info(f"Loading embedding model: {self._model_name}")
            try:
                # Try loading from local cache first — no network calls
                self._model = SentenceTransformer(
                    self._model_name,
                    local_files_only=True,
                )
                log.info("Loaded embedding model from local cache (offline)")
            except Exception:
                # Cache miss — download from HuggingFace (first run only)
                log.info("Model not cached yet — downloading from HuggingFace…")
                self._model = SentenceTransformer(self._model_name)
                log.info("Embedding model downloaded and cached")

            dim = self._model.get_embedding_dimension()
            log.info(f"Embedding model ready. Dimension: {dim}")
        return self._model

    def embed(self, texts: Union[str, list[str]]) -> list[list[float]]:
        """
        Embed one or more texts.

        Args:
            texts: a single string or list of strings

        Returns:
            List of embedding vectors (list of floats).
        """
        model = self._load()
        if isinstance(texts, str):
            texts = [texts]
        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=EMBEDDING_NORMALIZE,
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        model = self._load()
        return model.get_embedding_dimension()
