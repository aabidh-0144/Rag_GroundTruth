"""Ingestion tests.

These cover the cleanup heuristics without touching a real PDF - pymupdf4llm's
output is exercised by scripts/smoke.py instead, since mocking it would only test
the mock.
"""

from __future__ import annotations

from conftest import make_document

from groundtruth.ingest import (
    _clean_page,
    _extract_title,
    _find_repeated_lines,
    _reanchor_pages,
    _strip_references,
)
from groundtruth.schemas import PageSpan
from groundtruth.utils import slugify


# --- boilerplate removal ---------------------------------------------------


def test_running_header_is_detected():
    pages = [f"Journal of Testing, Vol 3\n\nBody text for page {i}." for i in range(5)]
    assert "Journal of Testing, Vol 3" in _find_repeated_lines(pages)


def test_body_text_is_not_treated_as_boilerplate():
    pages = [f"Unique body content number {i}." for i in range(5)]
    assert _find_repeated_lines(pages) == set()


def test_short_documents_skip_boilerplate_detection():
    """With two pages, 'repeated' is indistinguishable from coincidence."""
    assert _find_repeated_lines(["Same line", "Same line"]) == set()


def test_long_repeated_lines_are_kept():
    """A repeated long paragraph is content (a disclaimer, an abstract), not a header."""
    long_line = "This sentence is far too long to plausibly be a running header. " * 2
    pages = [f"{long_line}\nPage {i}" for i in range(5)]
    assert long_line not in _find_repeated_lines(pages)


def test_clean_page_removes_boilerplate_and_page_numbers():
    cleaned = _clean_page("Running Header\nReal content here.\n7", {"Running Header"})
    assert cleaned.strip() == "Real content here."


def test_clean_page_rejoins_hyphenated_line_breaks():
    """PDF line-wrap hyphenation fragments terms BM25 and the tokenizer both need whole."""
    assert "attractive" in _clean_page("an attrac-\ntive option", set())


def test_clean_page_keeps_numbers_that_are_content():
    assert "12345678" in _clean_page("12345678", set())  # too long to be a page number


# --- references stripping --------------------------------------------------


def test_references_section_is_stripped():
    body = "Body content. " * 100
    text = body + "\n# References\n\n[1] Some Author. A paper. 2020."

    stripped = _strip_references(text)

    assert "Some Author" not in stripped
    assert "Body content." in stripped


def test_early_references_mention_does_not_truncate_the_paper():
    """A paper that merely discusses 'references' early must not be gutted."""
    text = "# References\n\n" + "Real body content that follows. " * 100
    assert len(_strip_references(text)) == len(text)


def test_document_without_references_is_untouched():
    text = "A paper with no bibliography heading at all. " * 20
    assert _strip_references(text) == text


# --- page offsets ----------------------------------------------------------


def test_reanchored_pages_stay_within_the_text():
    text = "shortened text"
    pages = [PageSpan(page=1, start_char=0, end_char=100), PageSpan(page=2, start_char=100, end_char=200)]

    reanchored = _reanchor_pages(text, pages, original_len=200)

    for page in reanchored:
        assert 0 <= page.start_char <= page.end_char <= len(text)


def test_reanchoring_is_a_noop_when_length_is_unchanged():
    text = "x" * 50
    pages = [PageSpan(page=1, start_char=0, end_char=50)]

    assert _reanchor_pages(text, pages, original_len=50) == pages


def test_page_range_maps_a_char_span_to_pages():
    doc = make_document("x" * 300)
    doc.pages = [
        PageSpan(page=1, start_char=0, end_char=100),
        PageSpan(page=2, start_char=100, end_char=200),
        PageSpan(page=3, start_char=200, end_char=300),
    ]

    assert doc.page_range(150, 250) == (2, 3)
    assert doc.page_range(10, 20) == (1, 1)


# --- misc ------------------------------------------------------------------


def test_title_is_the_first_substantial_line():
    assert _extract_title("# Attention Is All You Need\n\nbody", "fallback") == "Attention Is All You Need"


def test_title_falls_back_when_no_line_qualifies():
    assert _extract_title("hi\nok\n", "my_paper") == "my_paper"


def test_slugify_produces_stable_doc_ids():
    assert slugify("Attention Is All You Need (2017).pdf") == "attention_is_all_you_need_2017pdf"
    assert slugify("BERT: Pre-training") == slugify("BERT:  Pre-training")


def test_slugify_never_returns_empty():
    assert slugify("!!!") == "untitled"
