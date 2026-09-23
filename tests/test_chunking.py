"""Chunking tests.

The char-span round-trip test is the most important test in the repo: the golden
dataset will anchor its labels to character spans, so if a span does not slice back
to the chunk's own text, every label built on it is silently wrong.
"""

from __future__ import annotations

from conftest import make_document

from groundtruth.chunking import Chunker, split_sections
from groundtruth.config import ChunkingConfig
from groundtruth.schemas import make_chunk_id


def test_chunk_id_is_deterministic():
    a = make_chunk_id("paper_x", 100, 500)
    b = make_chunk_id("paper_x", 100, 500)
    assert a == b
    assert len(a) == 16


def test_chunk_id_depends_only_on_span():
    """IDs must be derivable from a (doc_id, start, end) label alone - invariant #1."""
    assert make_chunk_id("paper_x", 100, 500) != make_chunk_id("paper_x", 100, 501)
    assert make_chunk_id("paper_x", 100, 500) != make_chunk_id("paper_y", 100, 500)


def test_char_spans_round_trip(paper_text, tokenizer):
    """doc.text[start:end] == chunk.text for every chunk. The core invariant."""
    doc = make_document(paper_text)
    chunks = Chunker(ChunkingConfig(size=30, overlap=5, min_tokens=3), tokenizer).chunk_document(doc)

    assert chunks
    for chunk in chunks:
        assert doc.text[chunk.start_char : chunk.end_char] == chunk.text


def test_chunking_is_reproducible(paper_text, tokenizer):
    doc = make_document(paper_text)
    cfg = ChunkingConfig(size=30, overlap=5, min_tokens=3)

    first = Chunker(cfg, tokenizer).chunk_document(doc)
    second = Chunker(cfg, tokenizer).chunk_document(doc)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_chunk_ids_are_unique(paper_text, tokenizer):
    """Duplicate IDs would silently shrink the Chroma collection."""
    doc = make_document(paper_text)
    chunks = Chunker(ChunkingConfig(size=30, overlap=5, min_tokens=3), tokenizer).chunk_document(doc)

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_index_is_contiguous(paper_text, tokenizer):
    doc = make_document(paper_text)
    chunks = Chunker(ChunkingConfig(size=30, overlap=5, min_tokens=3), tokenizer).chunk_document(doc)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunks_respect_size_limit(paper_text, tokenizer):
    doc = make_document(paper_text)
    cfg = ChunkingConfig(size=25, overlap=5, min_tokens=3)
    chunks = Chunker(cfg, tokenizer).chunk_document(doc)

    # A merge may push a chunk to exactly `size`, but never past it - anything more
    # would be truncated by the embedding model without warning.
    for chunk in chunks:
        assert chunk.n_tokens <= cfg.size


def test_oversized_sentence_is_hard_split(tokenizer):
    """A table row or long equation must not overflow the embedder's context."""
    doc = make_document("word " * 300)
    cfg = ChunkingConfig(size=20, overlap=4, min_tokens=2)
    chunks = Chunker(cfg, tokenizer).chunk_document(doc)

    assert len(chunks) > 1
    assert "".join(c.text for c in chunks) == doc.text


def test_sections_are_detected(paper_text):
    sections = split_sections(paper_text)
    paths = [s.path for s in sections]

    assert "" in paths  # preamble before the first header
    assert "1 Introduction" in paths
    assert "1 Introduction > 1.1 Motivation" in paths
    assert "2 Method" in paths


def test_sections_tile_the_document(paper_text):
    """Sections must cover the text exactly once, or chunking would lose or duplicate content."""
    sections = split_sections(paper_text)

    assert sections[0].start_char == 0
    assert sections[-1].end_char == len(paper_text)
    for previous, current in zip(sections, sections[1:]):
        assert previous.end_char == current.start_char


def test_document_without_headers_yields_one_section():
    text = "Just some prose with no markdown headers at all."
    sections = split_sections(text)

    assert len(sections) == 1
    assert sections[0].start_char == 0
    assert sections[0].end_char == len(text)


def test_section_path_is_carried_onto_chunks(paper_text, tokenizer):
    doc = make_document(paper_text)
    chunks = Chunker(ChunkingConfig(size=30, overlap=5, min_tokens=3), tokenizer).chunk_document(doc)

    assert any("Introduction" in c.section_path for c in chunks)


def test_overlap_repeats_content_between_neighbours(tokenizer):
    """Overlap exists so a fact spanning a boundary survives intact in one chunk."""
    sentences = " ".join(f"Sentence number {i} has content." for i in range(40))
    doc = make_document(sentences)
    cfg = ChunkingConfig(size=20, overlap=8, min_tokens=2)
    chunks = Chunker(cfg, tokenizer).chunk_document(doc)

    assert len(chunks) > 1
    # Consecutive chunks from one section should share a character range.
    assert any(b.start_char < a.end_char for a, b in zip(chunks, chunks[1:]))


def test_tiny_chunks_are_merged_away(tokenizer):
    doc = make_document("# H1\n\nShort.\n\n# H2\n\n" + "word " * 60)
    cfg = ChunkingConfig(size=40, overlap=5, min_tokens=10)
    chunks = Chunker(cfg, tokenizer).chunk_document(doc)

    # Only a chunk with no contiguous predecessor may remain under min_tokens.
    for chunk in chunks[1:]:
        assert chunk.n_tokens >= cfg.min_tokens or chunk.chunk_index == 0


def test_content_budget_leaves_room_for_special_tokens(tokenizer):
    """A chunk of exactly model_max_length content tokens would be truncated.

    Truncation means the stored embedding does not represent the chunk's own text -
    a silent way to corrupt every retrieval metric built on top of it.
    """
    tokenizer.model_max_length = 512
    chunker = Chunker(ChunkingConfig(size=512), tokenizer)
    assert chunker.max_tokens == 510


def test_content_budget_respects_a_smaller_configured_size(tokenizer):
    tokenizer.model_max_length = 512
    assert Chunker(ChunkingConfig(size=256), tokenizer).max_tokens == 256


def test_content_budget_ignores_sentinel_model_max_length(tokenizer):
    """Tokenizers with no real limit report a huge sentinel value."""
    tokenizer.model_max_length = 1_000_000_000_000
    assert Chunker(ChunkingConfig(size=512), tokenizer).max_tokens == 512


def test_hard_split_respects_the_content_budget(tokenizer):
    tokenizer.model_max_length = 30
    doc = make_document("word " * 200)
    chunks = Chunker(ChunkingConfig(size=100, overlap=4, min_tokens=2), tokenizer).chunk_document(doc)

    for chunk in chunks:
        assert chunk.n_tokens <= 28


def test_section_paths_strip_markdown_emphasis():
    """pymupdf4llm emits bold headings; raw asterisks in a section path are noise."""
    sections = split_sections("# **3.2 Attention**\n\nBody text here.\n")
    assert "3.2 Attention" in [s.path for s in sections]
