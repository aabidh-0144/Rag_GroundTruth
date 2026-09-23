"""Lexical retrieval.

Not one of the three headline experiments, but it is the sparse half of the hybrid
strategy and a useful baseline row in the results table - dense retrieval is
routinely beaten by BM25 on exact-term queries (an author name, "BLEU", "GELU").
"""

from __future__ import annotations

import numpy as np

from groundtruth.index import BM25Index, load_chunks, tokenize_bm25
from groundtruth.retrievers.base import rank
from groundtruth.schemas import Chunk, RetrievedChunk
from groundtruth.utils import Timer


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self._index: BM25Index | None = None
        self._chunks = chunks
        self._by_id: dict[str, Chunk] | None = None

    @property
    def index(self) -> BM25Index:
        if self._index is None:
            self._index = BM25Index.load()
        return self._index

    @property
    def by_id(self) -> dict[str, Chunk]:
        if self._by_id is None:
            if self._chunks is None:
                self._chunks = load_chunks()
            self._by_id = {c.chunk_id: c for c in self._chunks}
        return self._by_id

    def warmup(self) -> None:
        """Unpickle the index and build the id map before any timed query."""
        _ = self.index
        _ = self.by_id

    def retrieve(self, query: str, k: int) -> tuple[list[RetrievedChunk], dict[str, float]]:
        timer = Timer()

        with timer("search"):
            tokens = tokenize_bm25(query)
            scores = np.asarray(self.index.bm25.get_scores(tokens))
            # argpartition is O(n) vs a full sort; at 10k chunks it is not the
            # bottleneck, but it keeps BM25 honest as the "fast" retriever.
            top_n = min(k, len(scores))
            candidates = np.argpartition(-scores, top_n - 1)[:top_n] if top_n > 0 else []

        results: list[RetrievedChunk] = []
        for idx in candidates:
            chunk_id = self.index.chunk_ids[int(idx)]
            chunk = self.by_id.get(chunk_id)
            if chunk is None:
                # Index and chunk file disagree - a stale store, not a soft error.
                raise RuntimeError(
                    f"BM25 index references unknown chunk {chunk_id!r}. "
                    "Re-run `index` to rebuild against the current chunks.jsonl."
                )
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=chunk.text,
                    score=float(scores[idx]),
                    rank=0,
                    metadata=chunk.metadata(),
                )
            )

        return rank(results), timer.total()
