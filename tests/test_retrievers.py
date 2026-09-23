"""Retrieval tests.

RRF is tested against hand-computed values rather than a golden snapshot: fusion is
the one piece of retrieval maths written from scratch here, so it is worth pinning
exactly rather than approximately.
"""

from __future__ import annotations

import pytest

from groundtruth.config import (
    CandidatesConfig,
    Config,
    FusionConfig,
    RetrievalConfig,
)
from groundtruth.retrievers.base import rank
from groundtruth.retrievers.hybrid import HybridRetriever, reciprocal_rank_fusion
from groundtruth.schemas import RetrievedChunk
from groundtruth.utils import Timer


def make_hits(chunk_ids: list[str]) -> list[RetrievedChunk]:
    """A ranked list, best first, with descending placeholder scores."""
    return [
        RetrievedChunk(chunk_id=cid, text=f"text of {cid}", score=1.0 - i * 0.1, rank=i + 1)
        for i, cid in enumerate(chunk_ids)
    ]


class StubRetriever:
    """Returns a fixed ranking. Lets fusion be tested without any index on disk."""

    def __init__(self, name: str, chunk_ids: list[str]) -> None:
        self.name = name
        self.chunk_ids = chunk_ids
        self.last_k: int | None = None

    def retrieve(self, query: str, k: int):
        self.last_k = k
        timer = Timer()
        with timer(self.name):
            hits = make_hits(self.chunk_ids[:k])
        return hits, timer.total()


# --- rank ------------------------------------------------------------------


def test_rank_orders_by_score_descending():
    chunks = [
        RetrievedChunk(chunk_id="a", text="", score=0.2, rank=0),
        RetrievedChunk(chunk_id="b", text="", score=0.9, rank=0),
        RetrievedChunk(chunk_id="c", text="", score=0.5, rank=0),
    ]
    assert [c.chunk_id for c in rank(chunks)] == ["b", "c", "a"]


def test_rank_assigns_contiguous_one_based_ranks():
    ranked = rank(make_hits(["x", "y", "z"]))
    assert [c.rank for c in ranked] == [1, 2, 3]


def test_rank_breaks_ties_deterministically():
    """Determinism across runs depends on tie-breaking by chunk_id."""
    chunks = [
        RetrievedChunk(chunk_id="zebra", text="", score=0.5, rank=0),
        RetrievedChunk(chunk_id="apple", text="", score=0.5, rank=0),
    ]
    assert [c.chunk_id for c in rank(chunks)] == ["apple", "zebra"]


# --- RRF -------------------------------------------------------------------


def test_rrf_matches_hand_computed_scores():
    dense = make_hits(["a", "b"])  # a@1, b@2
    sparse = make_hits(["b", "a"])  # b@1, a@2

    fused = {c.chunk_id: c.score for c in reciprocal_rank_fusion([dense, sparse], k=60)}

    expected = 1 / 61 + 1 / 62  # both appear at rank 1 in one list and 2 in the other
    assert fused["a"] == pytest.approx(expected)
    assert fused["b"] == pytest.approx(expected)


def test_rrf_rewards_agreement_between_retrievers():
    """A chunk both retrievers like must beat one that only a single retriever ranks first."""
    dense = make_hits(["shared", "dense_only"])
    sparse = make_hits(["shared", "sparse_only"])

    fused = reciprocal_rank_fusion([dense, sparse], k=60)

    assert fused[0].chunk_id == "shared"
    assert fused[0].score == pytest.approx(2 / 61)


def test_rrf_unions_both_result_sets():
    fused = reciprocal_rank_fusion([make_hits(["a", "b"]), make_hits(["c", "d"])], k=60)
    assert {c.chunk_id for c in fused} == {"a", "b", "c", "d"}


def test_rrf_ignores_raw_scores():
    """Only ranks matter - that is the whole reason RRF is used over score blending."""
    inflated = make_hits(["a", "b"])
    for chunk in inflated:
        chunk.score *= 1000

    assert [c.chunk_id for c in reciprocal_rank_fusion([inflated], k=60)] == [
        c.chunk_id for c in reciprocal_rank_fusion([make_hits(["a", "b"])], k=60)
    ]


def test_rrf_k_controls_top_rank_dominance():
    """Small k sharpens the advantage of a rank-1 hit; large k flattens it."""
    dense = make_hits(["a", "b", "c"])
    sparse = make_hits(["c", "b", "a"])

    sharp = {c.chunk_id: c.score for c in reciprocal_rank_fusion([dense, sparse], k=1)}
    flat = {c.chunk_id: c.score for c in reciprocal_rank_fusion([dense, sparse], k=1000)}

    assert sharp["a"] - sharp["b"] > flat["a"] - flat["b"]


def test_rrf_preserves_text_and_metadata():
    hits = make_hits(["a"])
    hits[0].metadata = {"doc_id": "paper_1"}

    fused = reciprocal_rank_fusion([hits], k=60)

    assert fused[0].text == "text of a"
    assert fused[0].metadata == {"doc_id": "paper_1"}


def test_rrf_handles_empty_input():
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []


# --- HybridRetriever -------------------------------------------------------


def test_hybrid_overfetches_candidates_then_truncates():
    """Fusion needs deep candidate lists; the caller only ever sees k."""
    dense = StubRetriever("dense", [f"d{i}" for i in range(50)])
    sparse = StubRetriever("sparse", [f"s{i}" for i in range(50)])

    hybrid = HybridRetriever(dense, sparse, CandidatesConfig(dense=40, sparse=30), FusionConfig())
    results, _ = hybrid.retrieve("query", k=5)

    assert dense.last_k == 40
    assert sparse.last_k == 30
    assert len(results) == 5
    assert [c.rank for c in results] == [1, 2, 3, 4, 5]


def test_hybrid_merges_timings_from_both_retrievers():
    hybrid = HybridRetriever(
        StubRetriever("dense", ["a"]),
        StubRetriever("sparse", ["b"]),
        CandidatesConfig(),
        FusionConfig(),
    )
    _, timings = hybrid.retrieve("query", k=2)

    assert {"dense_ms", "sparse_ms", "fuse_ms"} <= set(timings)


def test_hybrid_rejects_unknown_fusion_method():
    with pytest.raises(ValueError, match="fusion"):
        HybridRetriever(
            StubRetriever("dense", []),
            StubRetriever("sparse", []),
            CandidatesConfig(),
            FusionConfig(method="linear"),
        )


# --- config / experiment integrity ----------------------------------------


def test_experiments_share_one_index_fingerprint():
    """The three experiments must be comparable - invariant #3.

    They may differ in retrieval and generation, but any difference in chunking or
    embedding would mean they were reading from different indexes.
    """
    from groundtruth.config import load_config

    fingerprints = {
        name: load_config(f"configs/{name}.yaml").index_fingerprint()
        for name in ("dense", "hybrid", "hybrid_rerank")
    }
    assert len(set(fingerprints.values())) == 1, fingerprints


def test_config_fingerprint_distinguishes_experiments():
    from groundtruth.config import load_config

    fingerprints = {
        load_config(f"configs/{name}.yaml").fingerprint()
        for name in ("dense", "hybrid", "hybrid_rerank")
    }
    assert len(fingerprints) == 3


def test_unknown_strategy_is_rejected():
    cfg = Config(retrieval=RetrievalConfig(strategy="magic"))
    from groundtruth.embedder import Embedder
    from groundtruth.retrievers import build_retriever

    with pytest.raises(ValueError, match="strategy"):
        build_retriever(cfg, Embedder(cfg.embedding))
