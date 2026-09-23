"""Document -> Chunks.

Two-level strategy, which is the right default for research papers:

1. Split on markdown headers into sections, carrying a `section_path` for context.
2. Within each section, split into token windows using the *embedding model's own
   tokenizer*, preferring sentence boundaries.

The invariant this module owns, and which the whole evaluation rests on:
`document.text[chunk.start_char:chunk.end_char] == chunk.text`. Golden dataset labels
anchor to those character spans, so they survive a change of chunk size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from groundtruth.config import ChunkingConfig
from groundtruth.schemas import Chunk, Document, make_chunk_id

# ATX markdown headers, as emitted by pymupdf4llm.
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

# pymupdf4llm marks bold headings, so titles arrive as "**3.2 Attention**".
_EMPHASIS_RE = re.compile(r"[*_`]+")


def clean_heading(title: str) -> str:
    """Strip markdown emphasis from a heading so section paths stay readable."""
    return _EMPHASIS_RE.sub("", title).strip()


# Sentence boundary: terminator + whitespace, or a blank line. Kept simple on
# purpose - paper text has enough abbreviations ("et al.", "Fig.", "e.g.") that a
# heavier splitter buys little here and costs a dependency.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])|\n\n+")


@dataclass(slots=True)
class Section:
    """A header-delimited region of the document, in document coordinates."""

    path: str
    start_char: int
    end_char: int


def split_sections(text: str) -> list[Section]:
    """Split on markdown headers, tracking the nesting path.

    A document with no headers yields a single section spanning the whole text, so
    callers never need a special case.
    """
    headers = list(_HEADER_RE.finditer(text))
    if not headers:
        return [Section(path="", start_char=0, end_char=len(text))]

    sections: list[Section] = []

    # Any preamble before the first header (title block, abstract) is still content.
    if headers[0].start() > 0:
        sections.append(Section(path="", start_char=0, end_char=headers[0].start()))

    stack: list[tuple[int, str]] = []  # (level, title)
    for i, match in enumerate(headers):
        level = len(match.group(1))
        title = clean_heading(match.group(2))

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

        start = match.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        sections.append(
            Section(path=" > ".join(t for _, t in stack), start_char=start, end_char=end)
        )

    return sections


def _sentence_spans(text: str, offset: int) -> list[tuple[int, int]]:
    """Absolute (start, end) spans of sentence-ish units within `text`."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_RE.finditer(text):
        end = match.start()
        if end > cursor:
            spans.append((offset + cursor, offset + end))
        cursor = match.end()
    if cursor < len(text):
        spans.append((offset + cursor, offset + len(text)))
    return spans


class Chunker:
    """Token-window chunker bound to the embedding model's tokenizer.

    Counting in the embedder's own tokens (rather than characters or tiktoken) is what
    guarantees a chunk actually fits the model's 512-token context without truncation.
    """

    def __init__(self, cfg: ChunkingConfig, tokenizer) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.max_tokens = self._content_budget(cfg, tokenizer)

    @staticmethod
    def _content_budget(cfg: ChunkingConfig, tokenizer) -> int:
        """Largest chunk, in content tokens, the embedder can consume whole.

        `model_max_length` counts [CLS] and [SEP], so a chunk of exactly that many
        content tokens would be silently truncated - and a truncated chunk is a
        chunk whose embedding does not represent its own text, which is a subtle
        way to corrupt every retrieval metric downstream.
        """
        limit = getattr(tokenizer, "model_max_length", None)
        # Tokenizers with no real limit use a huge sentinel value.
        if not isinstance(limit, int) or limit > 1_000_000:
            return cfg.size
        return min(cfg.size, max(1, limit - 2))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def chunk_document(self, doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in split_sections(doc.text):
            chunks.extend(self._chunk_section(doc, section))

        chunks = self._merge_tiny(doc, chunks)

        # chunk_index is assigned last so it is contiguous per document even after
        # merges - useful for "show me the neighbouring chunk" debugging.
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
        return chunks

    def _chunk_section(self, doc: Document, section: Section) -> list[Chunk]:
        body = doc.text[section.start_char : section.end_char]
        if not body.strip():
            return []

        spans = _sentence_spans(body, section.start_char)
        if not spans:
            return []

        out: list[Chunk] = []
        window: list[tuple[int, int]] = []
        window_tokens = 0

        for span in spans:
            span_text = doc.text[span[0] : span[1]]
            span_tokens = self.count_tokens(span_text)

            # A single sentence longer than the window (tables, long equations) is
            # hard-split rather than allowed to overflow the embedder's context.
            if span_tokens > self.max_tokens:
                if window:
                    out.append(self._emit(doc, section, window))
                    window, window_tokens = [], 0
                out.extend(self._hard_split(doc, section, span))
                continue

            if window_tokens + span_tokens > self.max_tokens and window:
                out.append(self._emit(doc, section, window))
                window, window_tokens = self._carry_overlap(doc, window)

            window.append(span)
            window_tokens += span_tokens

        if window:
            out.append(self._emit(doc, section, window))
        return out

    def _carry_overlap(
        self, doc: Document, window: list[tuple[int, int]]
    ) -> tuple[list[tuple[int, int]], int]:
        """Seed the next window with trailing sentences worth ~`overlap` tokens.

        Overlap exists so a fact straddling a boundary is retrievable from at least
        one chunk in full.
        """
        carried: list[tuple[int, int]] = []
        total = 0
        for span in reversed(window):
            span_tokens = self.count_tokens(doc.text[span[0] : span[1]])
            if total + span_tokens > self.cfg.overlap and carried:
                break
            carried.insert(0, span)
            total += span_tokens
        return carried, total

    def _hard_split(
        self, doc: Document, section: Section, span: tuple[int, int]
    ) -> list[Chunk]:
        """Token-exact fallback for a single oversized unit (a table, a long equation).

        Splitting on a chars-per-token estimate is not good enough here: dense
        technical text (numbers, symbols, citation markers) tokenizes far below the
        ~4 chars/token of ordinary prose, so an estimate overshoots and produces
        chunks the embedder silently truncates. The tokenizer's offset mapping gives
        exact character boundaries instead.

        Pieces are contiguous and non-overlapping so the span arithmetic stays
        trivially correct; this is a degenerate path, not a place for overlap.
        """
        start, end = span
        text = doc.text[start:end]

        offsets = self._token_offsets(text)
        if not offsets:
            return [self._make_chunk(doc, section, start, end)]

        out: list[Chunk] = []
        cursor = start
        for i in range(0, len(offsets), self.max_tokens):
            window = offsets[i : i + self.max_tokens]
            is_last = i + self.max_tokens >= len(offsets)
            # The final piece runs to the true span end so trailing whitespace or
            # punctuation outside the last token is not dropped.
            piece_end = end if is_last else start + window[-1][1]
            if piece_end <= cursor:
                continue
            out.append(self._make_chunk(doc, section, cursor, piece_end))
            cursor = piece_end
        return out

    def _token_offsets(self, text: str) -> list[tuple[int, int]]:
        """Per-token (start, end) character offsets, or [] if unsupported.

        Only fast tokenizers provide offset mapping; the fallback keeps the test
        fakes and any slow tokenizer working, at the cost of a rough split.
        """
        try:
            encoding = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            return [(int(a), int(b)) for a, b in encoding["offset_mapping"]]
        except Exception:
            # Fall back to one pseudo-token per character. Coarse, but it keeps the
            # windowing logic in _hard_split correct rather than adding a branch.
            return [(i, i + 1) for i in range(len(text))]

    def _emit(self, doc: Document, section: Section, window: list[tuple[int, int]]) -> Chunk:
        return self._make_chunk(doc, section, window[0][0], window[-1][1])

    def _make_chunk(self, doc: Document, section: Section, start: int, end: int) -> Chunk:
        text = doc.text[start:end]
        page_start, page_end = doc.page_range(start, end)
        return Chunk(
            chunk_id=make_chunk_id(doc.doc_id, start, end),
            doc_id=doc.doc_id,
            chunk_index=-1,  # assigned after merging
            start_char=start,
            end_char=end,
            page_start=page_start,
            page_end=page_end,
            section_path=section.path,
            n_tokens=self.count_tokens(text),
            text=text,
        )

    def _can_join(self, doc: Document, first: Chunk, second: Chunk) -> bool:
        """Whether two chunks can be fused without violating the span invariant.

        They must be from the same document and separated by nothing but whitespace,
        so that the merged span still slices back to the merged text. Section
        boundaries emit a blank line between spans, so requiring *strict* adjacency
        would block almost every useful merge.
        """
        if first.doc_id != second.doc_id or first.end_char > second.start_char:
            return False
        if doc.text[first.end_char : second.start_char].strip():
            return False
        # Measure the joined text rather than summing the parts: subword tokenizers
        # do not distribute over concatenation, and the sum can understate the total.
        return self.count_tokens(doc.text[first.start_char : second.end_char]) <= self.max_tokens

    def _join(self, doc: Document, first: Chunk, second: Chunk, section_path: str) -> Chunk:
        section = Section(section_path, first.start_char, second.end_char)
        return self._make_chunk(doc, section, first.start_char, second.end_char)

    def _merge_tiny(self, doc: Document, chunks: list[Chunk]) -> list[Chunk]:
        """Fold sub-`min_tokens` chunks into an adjacent one.

        Stray fragments ("3.1 Encoder", a caption line) are noise in the index: they
        match on almost nothing yet occasionally outrank real content on short
        queries, because a 3-token chunk is trivially "about" its 3 tokens.

        Two passes, because direction matters. A bare section header belongs with the
        text *beneath* it, so tiny chunks that could not be absorbed by a predecessor
        are given a second chance to merge forwards.
        """
        if not chunks:
            return chunks

        backward: list[Chunk] = []
        for chunk in chunks:
            if (
                chunk.n_tokens < self.cfg.min_tokens
                and backward
                and self._can_join(doc, backward[-1], chunk)
            ):
                prev = backward[-1]
                backward[-1] = self._join(doc, prev, chunk, prev.section_path)
            else:
                backward.append(chunk)

        merged: list[Chunk] = []
        i = 0
        while i < len(backward):
            chunk = backward[i]
            nxt = backward[i + 1] if i + 1 < len(backward) else None
            if (
                chunk.n_tokens < self.cfg.min_tokens
                and nxt is not None
                and self._can_join(doc, chunk, nxt)
            ):
                # The successor's section path wins: a header fragment takes the
                # identity of the body it introduces.
                merged.append(self._join(doc, chunk, nxt, nxt.section_path))
                i += 2
                continue
            merged.append(chunk)
            i += 1

        # A tiny chunk with no whitespace-adjacent neighbour on either side survives.
        # Dropping it would lose text, which is worse than one noisy fragment.
        return [c for c in merged if c.text.strip()]


def chunk_documents(docs: list[Document], cfg: ChunkingConfig, tokenizer) -> list[Chunk]:
    chunker = Chunker(cfg, tokenizer)
    all_chunks: list[Chunk] = []
    for doc in docs:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc.doc_id}: {len(chunks)} chunks")
    return all_chunks
