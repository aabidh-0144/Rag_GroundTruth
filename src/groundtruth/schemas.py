"""Core data structures.

These types are the contract between pipeline stages, and — more importantly — the
contract the evaluation layer will later depend on. Changing a field name here means
changing the on-disk format of `chunks.jsonl` and `results.jsonl`.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


def make_chunk_id(doc_id: str, start_char: int, end_char: int) -> str:
    """Stable, content-independent chunk identifier.

    Derived from the character span rather than the text so that it is reproducible
    across machines and re-derivable from a golden-dataset label alone. See invariant
    #1 in .claude/CLAUDE.md.
    """
    return hashlib.sha1(f"{doc_id}:{start_char}:{end_char}".encode()).hexdigest()[:16]


@dataclass(slots=True)
class PageSpan:
    """Where a source PDF page lives in the document's character coordinate system."""

    page: int
    start_char: int
    end_char: int


@dataclass(slots=True)
class Document:
    """A normalized source document.

    `text` is the single coordinate system for the whole project: chunks and golden
    dataset labels both address into it by character offset.
    """

    doc_id: str
    title: str
    source_path: str
    n_pages: int
    text: str
    pages: list[PageSpan] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Document:
        return cls(
            doc_id=d["doc_id"],
            title=d["title"],
            source_path=d["source_path"],
            n_pages=d["n_pages"],
            text=d["text"],
            pages=[PageSpan(**p) for p in d.get("pages", [])],
        )

    def page_range(self, start_char: int, end_char: int) -> tuple[int, int]:
        """First and last PDF page touched by a character span."""
        touched = [
            p.page for p in self.pages if p.start_char < end_char and p.end_char > start_char
        ]
        if not touched:
            return (0, 0)
        return (min(touched), max(touched))


@dataclass(slots=True)
class Chunk:
    """An indexed unit of retrieval."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    start_char: int
    end_char: int
    page_start: int
    page_end: int
    section_path: str
    n_tokens: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        return cls(**d)

    def metadata(self) -> dict[str, Any]:
        """Chroma-safe metadata (scalars only, text excluded — it is the document body)."""
        d = self.to_dict()
        d.pop("text")
        return d


@dataclass(slots=True)
class RetrievedChunk:
    """One ranked retrieval result.

    `score` is comparable only within a single retriever — dense cosine, BM25, RRF and
    cross-encoder scores are on different scales by nature. Ranking metrics (MRR, nDCG)
    use `rank`; only diagnostics should read `score` across strategies.
    """

    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QueryResult:
    """One row of `results.jsonl` — the seam the evaluation layer consumes."""

    query_id: str
    question: str
    retrieved: list[RetrievedChunk]
    answer: str
    timings: dict[str, float]
    config_name: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "retrieved": [r.to_dict() for r in self.retrieved],
            "answer": self.answer,
            "timings": self.timings,
            "config_name": self.config_name,
            "error": self.error,
        }
