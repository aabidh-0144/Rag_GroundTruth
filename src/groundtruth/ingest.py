"""PDF -> normalized Document JSON.

Uses pymupdf4llm's markdown output rather than raw text extraction because the `#`
headers it recovers are what make section-aware chunking possible downstream.

The invariant this module owns: the emitted `text` is the document's single character
coordinate system, and every recorded page span must slice back to the right text.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from groundtruth.config import DOCS_DIR, RAW_DIR
from groundtruth.schemas import Document, PageSpan
from groundtruth.utils import read_json, slugify, write_json

# A references heading, on its own line, optionally numbered. Everything after the
# last such heading is bibliography — high term overlap with real content but never
# a useful answer, so it is dropped rather than left to pollute retrieval.
_REFERENCES_RE = re.compile(
    r"^#{0,6}\s*(?:\d+\.?\s*)?(references|bibliography)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# "attrac-\ntive" -> "attractive". PDF line-wrapping hyphenation otherwise fragments
# terms that both the tokenizer and BM25 need intact.
_DEHYPHENATE_RE = re.compile(r"(\w)-\s*\n\s*(\w)")

_MULTI_BLANKLINE_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)


def _find_repeated_lines(pages: list[str], min_pages: int = 3) -> set[str]:
    """Identify running headers/footers: short lines appearing on most pages.

    Threshold is 60% of pages, which is lenient enough to catch headers that start
    after the title page but strict enough not to eat genuine repeated content.
    """
    if len(pages) < min_pages:
        return set()

    counts: Counter[str] = Counter()
    for page in pages:
        lines = {ln.strip() for ln in page.splitlines() if ln.strip()}
        # Only short lines are plausible headers/footers.
        counts.update(ln for ln in lines if len(ln) < 80)

    threshold = max(min_pages, int(len(pages) * 0.6))
    return {line for line, count in counts.items() if count >= threshold}


def _clean_page(text: str, boilerplate: set[str]) -> str:
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in boilerplate:
            continue
        # Bare page numbers.
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        kept.append(line)
    text = "\n".join(kept)
    text = _DEHYPHENATE_RE.sub(r"\1\2", text)
    text = _TRAILING_WS_RE.sub("", text)
    return text


def _strip_references(text: str) -> str:
    matches = list(_REFERENCES_RE.finditer(text))
    if not matches:
        return text
    cut = matches[-1].start()
    # Only trust the cut if it leaves the bulk of the paper intact; a paper that
    # merely *mentions* "References" early on should not be truncated to nothing.
    if cut < len(text) * 0.4:
        return text
    return text[:cut].rstrip()


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if len(stripped) > 10:
            return stripped[:200]
    return fallback


def ingest_pdf(pdf_path: Path) -> Document:
    """Parse one PDF into a Document with page-level character offsets."""
    import pymupdf4llm  # imported lazily: heavy, and not needed by every CLI verb

    doc_id = slugify(pdf_path.stem)
    page_dicts = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, show_progress=False)
    raw_pages = [p.get("text", "") for p in page_dicts]

    boilerplate = _find_repeated_lines(raw_pages)

    parts: list[str] = []
    pages: list[PageSpan] = []
    cursor = 0
    for i, raw in enumerate(raw_pages, start=1):
        cleaned = _clean_page(raw, boilerplate)
        # Page separator keeps offsets honest and stops the last line of one page
        # from being glued to the first line of the next.
        piece = cleaned + "\n\n"
        parts.append(piece)
        pages.append(PageSpan(page=i, start_char=cursor, end_char=cursor + len(piece)))
        cursor += len(piece)

    text = "".join(parts)

    # References are stripped *after* offsets are assigned, so spans stay valid; the
    # trailing page spans are then clamped to the shortened text.
    text = _strip_references(text)
    text = _MULTI_BLANKLINE_RE.sub("\n\n", text)
    # Collapsing blank lines shifts offsets, so pages are re-anchored by re-walking.
    pages = _reanchor_pages(text, pages, cursor)

    return Document(
        doc_id=doc_id,
        title=_extract_title(text, pdf_path.stem),
        source_path=str(pdf_path.relative_to(pdf_path.parents[2]) if len(pdf_path.parents) > 2 else pdf_path),
        n_pages=len(raw_pages),
        text=text,
        pages=pages,
    )


def _reanchor_pages(text: str, pages: list[PageSpan], original_len: int) -> list[PageSpan]:
    """Rescale page spans after text-length-changing cleanup.

    An approximation — page boundaries are only used to report "this chunk came from
    pages 5-6" in the UI, never for correctness — but a proportional rescale keeps
    them close and, critically, keeps them inside the text.
    """
    new_len = len(text)
    if original_len == 0 or new_len == original_len:
        return [PageSpan(p.page, min(p.start_char, new_len), min(p.end_char, new_len)) for p in pages]

    scale = new_len / original_len
    out: list[PageSpan] = []
    for p in pages:
        start = min(int(p.start_char * scale), new_len)
        end = min(int(p.end_char * scale), new_len)
        out.append(PageSpan(page=p.page, start_char=start, end_char=max(end, start)))
    return out


def ingest_all(force: bool = False) -> list[Document]:
    """Ingest every PDF in data/raw/, skipping outputs that are already up to date."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {RAW_DIR}. Drop your research papers there and re-run."
        )

    docs: list[Document] = []
    for pdf in pdfs:
        out_path = DOCS_DIR / f"{slugify(pdf.stem)}.json"
        if not force and out_path.exists() and out_path.stat().st_mtime >= pdf.stat().st_mtime:
            docs.append(Document.from_dict(read_json(out_path)))
            print(f"  skip  {pdf.name} (up to date)")
            continue

        doc = ingest_pdf(pdf)
        write_json(out_path, doc.to_dict())
        docs.append(doc)
        print(f"  ok    {pdf.name} -> {doc.doc_id} ({doc.n_pages}p, {len(doc.text):,} chars)")

    return docs


def load_documents() -> list[Document]:
    paths = sorted(DOCS_DIR.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No documents in {DOCS_DIR}. Run `ingest` first.")
    return [Document.from_dict(read_json(p)) for p in paths]
