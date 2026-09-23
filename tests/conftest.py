"""Shared fixtures.

A fake whitespace tokenizer keeps the chunking tests fast and hermetic - they are
about span arithmetic and boundary logic, not about any particular model's
vocabulary. The real tokenizer is exercised by scripts/smoke.py instead.
"""

from __future__ import annotations

import pytest

from groundtruth.schemas import Document, PageSpan


class FakeTokenizer:
    """Approximates a wordpiece tokenizer as `len(text.split())`."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [0] * len(text.split())


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


def make_document(text: str, doc_id: str = "test_doc") -> Document:
    return Document(
        doc_id=doc_id,
        title="Test Document",
        source_path="data/raw/test.pdf",
        n_pages=1,
        text=text,
        pages=[PageSpan(page=1, start_char=0, end_char=len(text))],
    )


@pytest.fixture
def paper_text() -> str:
    """A miniature paper: preamble, nested headers, short and long sections."""
    return (
        "A Study of Retrieval Systems\n\n"
        "This paper examines retrieval. We present results. The approach is simple.\n\n"
        "# 1 Introduction\n\n"
        "Retrieval systems find documents. They rank them by relevance. "
        "Users issue queries to the system. The system returns ranked results.\n\n"
        "## 1.1 Motivation\n\n"
        "Manual evaluation is slow. Automated evaluation is repeatable.\n\n"
        "# 2 Method\n\n"
        "We use BM25 as a baseline. We compare it against dense retrieval. "
        "Hybrid fusion combines both. Reranking improves precision further.\n"
    )
