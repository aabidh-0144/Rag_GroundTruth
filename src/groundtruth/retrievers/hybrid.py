"""Experiment B: hybrid retrieval via Reciprocal Rank Fusion.

RRF rather than weighted score normalization because cosine similarity and BM25
scores live on incomparable scales, and any normalization scheme needs a weight that
has to be tuned per corpus. RRF consumes only *ranks*, so it needs no tuning and
cannot be destabilised by one retriever's score distribution shifting.

    score(d) = sum over retrievers of  1 / (k + rank_r(d))

k=60 damps the influence of top ranks enough that a document must do well in at
least one list, but a near-miss in both lists can still surface.
"""

from __future__ import annotations

from groundtruth.config import CandidatesConfig, FusionConfig
from groundtruth.retrievers.base import Retriever, rank
from groundtruth.schemas import RetrievedChunk
from groundtruth.utils import Timer


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]], k: int = 60
) -> list[RetrievedChunk]:
    """Fuse several ranked lists into one.

    Chunks are matched by `chunk_id`; the text and metadata of the first occurrence
    win, which is safe because both retrievers read from the same chunk set.
    """
    scores: dict[str, float] = {}
    seen: dict[str, RetrievedChunk] = {}

    for ranking in rankings:
        for chunk in ranking:
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + chunk.rank)
            seen.setdefault(chunk.chunk_id, chunk)

    fused = [
        RetrievedChunk(
            chunk_id=chunk_id,
            text=seen[chunk_id].text,
            score=score,
            rank=0,
            metadata=seen[chunk_id].metadata,
        )
        for chunk_id, score in scores.items()
    ]
    return rank(fused)


class HybridRetriever:
    """Dense + BM25, fused by RRF."""

    name = "hybrid"

    def __init__(
        self,
        dense: Retriever,
        sparse: Retriever,
        candidates: CandidatesConfig,
        fusion: FusionConfig,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.candidates = candidates
        self.fusion = fusion

        if fusion.method != "rrf":
            raise ValueError(f"Unsupported fusion method {fusion.method!r}; only 'rrf' is implemented")

    def warmup(self) -> None:
        for retriever in (self.dense, self.sparse):
            warmup = getattr(retriever, "warmup", None)
            if callable(warmup):
                warmup()

    def retrieve(self, query: str, k: int) -> tuple[list[RetrievedChunk], dict[str, float]]:
        timer = Timer()

        dense_hits, dense_timings = self.dense.retrieve(query, self.candidates.dense)
        sparse_hits, sparse_timings = self.sparse.retrieve(query, self.candidates.sparse)
        timer.merge(dense_timings)
        timer.merge(sparse_timings)

        with timer("fuse"):
            fused = reciprocal_rank_fusion([dense_hits, sparse_hits], k=self.fusion.k)

        return fused[:k], timer.total()
