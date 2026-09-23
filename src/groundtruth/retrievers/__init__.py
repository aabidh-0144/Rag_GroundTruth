"""Retriever implementations and the factory that maps a config to one.

The factory is the only place that knows the strategy names, so adding a fourth
experiment means adding one implementation and one branch here - never editing an
existing retriever.
"""

from __future__ import annotations

from groundtruth.config import Config
from groundtruth.embedder import Embedder
from groundtruth.retrievers.base import Retriever, rank
from groundtruth.retrievers.bm25 import BM25Retriever
from groundtruth.retrievers.dense import DenseRetriever
from groundtruth.retrievers.hybrid import HybridRetriever, reciprocal_rank_fusion
from groundtruth.retrievers.rerank import RerankRetriever

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "RerankRetriever",
    "Retriever",
    "build_retriever",
    "rank",
    "reciprocal_rank_fusion",
]


def build_retriever(cfg: Config, embedder: Embedder) -> Retriever:
    """Construct the retriever named by `cfg.retrieval.strategy`."""
    strategy = cfg.retrieval.strategy

    if strategy == "dense":
        return DenseRetriever(embedder)

    if strategy == "bm25":
        return BM25Retriever()

    if strategy in ("hybrid", "hybrid_rerank"):
        hybrid = HybridRetriever(
            dense=DenseRetriever(embedder),
            sparse=BM25Retriever(),
            candidates=cfg.retrieval.candidates,
            fusion=cfg.retrieval.fusion,
        )
        if strategy == "hybrid":
            return hybrid
        return RerankRetriever(base=hybrid, cfg=cfg.retrieval.reranker)

    raise ValueError(f"Unknown retrieval strategy: {strategy!r}")
