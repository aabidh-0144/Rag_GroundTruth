"""The retrieval contract.

Every strategy returns the same shape - a ranked list plus per-stage timings - so the
evaluation layer can treat Dense, Hybrid and Hybrid+Reranker identically (invariant #4).
New strategies are new implementations of this protocol, never a branch inside an
existing one; that is what keeps the experiment matrix clean.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from groundtruth.schemas import RetrievedChunk


@runtime_checkable
class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, k: int) -> tuple[list[RetrievedChunk], dict[str, float]]:
        """Return the top-k chunks for `query`, plus a `{stage}_ms` timing dict."""
        ...


def rank(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Sort by score descending and (re)assign contiguous 1-based ranks.

    Ties are broken by chunk_id so that repeated runs are byte-identical - the
    determinism check in the verification plan depends on this.
    """
    ordered = sorted(chunks, key=lambda c: (-c.score, c.chunk_id))
    for i, chunk in enumerate(ordered, start=1):
        chunk.rank = i
    return ordered
