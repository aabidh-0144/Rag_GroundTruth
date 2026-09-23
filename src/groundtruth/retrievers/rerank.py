"""Experiment C: cross-encoder reranking on top of any base retriever.

A bi-encoder embeds query and passage independently, so it can only ever measure
similarity in a shared space. A cross-encoder reads the pair jointly and is markedly
better at judging relevance - at the cost of one forward pass per candidate, which
on CPU is the dominant latency in the whole pipeline. Surfacing exactly that
quality-versus-latency tradeoff is the point of this experiment.
"""

from __future__ import annotations

from groundtruth.config import RerankerConfig
from groundtruth.retrievers.base import Retriever, rank
from groundtruth.schemas import RetrievedChunk
from groundtruth.utils import Timer


class RerankRetriever:
    """Wraps a base retriever and re-scores its top-N with a cross-encoder."""

    name = "rerank"

    def __init__(self, base: Retriever, cfg: RerankerConfig) -> None:
        self.base = base
        self.cfg = cfg
        self._model = None
        self.name = f"{base.name}_rerank"

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            print(f"  loading reranker {self.cfg.model} (cpu)...")
            self._model = CrossEncoder(self.cfg.model, device="cpu", max_length=self.cfg.max_length)
        return self._model

    def warmup(self) -> None:
        """Force model load and one forward pass before any timed query.

        Loading the cross-encoder takes tens of seconds. Left lazy, that cost lands
        inside the first query's `rerank_ms` and corrupts both the mean and the p95
        that this framework exists to report.
        """
        self.model.predict([("warmup", "warmup passage")], show_progress_bar=False)
        base_warmup = getattr(self.base, "warmup", None)
        if callable(base_warmup):
            base_warmup()

    def retrieve(self, query: str, k: int) -> tuple[list[RetrievedChunk], dict[str, float]]:
        timer = Timer()

        # Deliberately over-fetch: the reranker can only promote what the base
        # retriever surfaced, so top_n is the real recall ceiling of this strategy.
        candidates, base_timings = self.base.retrieve(query, self.cfg.top_n)
        timer.merge(base_timings)

        if not candidates:
            return [], timer.total()

        with timer("rerank"):
            pairs = [(query, c.text) for c in candidates]
            scores = self.model.predict(pairs, show_progress_bar=False)

        reranked = [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                text=c.text,
                score=float(score),
                rank=0,
                metadata=c.metadata,
            )
            for c, score in zip(candidates, scores)
        ]

        return rank(reranked)[:k], timer.total()
