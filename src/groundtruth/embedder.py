"""Text -> vectors.

Wraps sentence-transformers so that the asymmetry of bge-style models is impossible
to get wrong: queries receive an instruction prefix, documents do not (invariant #5).
Calling the underlying model directly anywhere else is a bug.
"""

from __future__ import annotations

import numpy as np

from groundtruth.config import EmbeddingConfig


class Embedder:
    """Lazily-loaded CPU embedding model.

    Loading is deferred to first use because several CLI verbs (`ingest`) need the
    tokenizer but never the weights, and model load is the slowest part of startup.
    """

    def __init__(self, cfg: EmbeddingConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            print(f"  loading embedding model {self.cfg.model} (cpu)...")
            self._model = SentenceTransformer(self.cfg.model, device="cpu")
        return self._model

    @property
    def tokenizer(self):
        """Tokenizer only - used by the chunker without paying for model weights."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            from transformers.utils import logging as hf_logging

            # The chunker deliberately measures spans longer than the model window in
            # order to decide where to split them. transformers warns about that as if
            # it were an impending error, which it is not - the span never reaches the
            # model unsplit. Suppress it so real warnings stay visible.
            hf_logging.set_verbosity_error()
            self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model)
        return self._tokenizer

    @property
    def dim(self) -> int:
        # Renamed in sentence-transformers 6.x; the old name still works on 3.x/5.x.
        if hasattr(self.model, "get_embedding_dimension"):
            return int(self.model.get_embedding_dimension())
        return int(self.model.get_sentence_embedding_dimension())

    def encode_documents(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """Embed passages. No prefix - bge documents are embedded raw."""
        return self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    def encode_query(self, query: str) -> np.ndarray:
        """Embed a search query, with the retrieval instruction prefix applied."""
        return self.model.encode(
            self.cfg.query_prefix + query,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
