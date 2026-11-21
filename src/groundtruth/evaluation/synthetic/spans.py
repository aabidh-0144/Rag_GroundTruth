"""Anchor a generated sample's reference contexts to character spans.

Project invariant #2: ground truth anchors to (doc_id, start_char, end_char), never
to chunk_id, so a dataset survives a re-chunking experiment. Ragas returns
reference contexts as plain text; this maps each one back to where it lives.

Lookup order for each reference context:
1. "chunk"  - it is exactly one of the chunks we fed Ragas (the normal case: the
              generator is given our own chunks, so the text is verbatim)
2. "exact"  - it occurs verbatim inside a document's text
3. "none"   - not found (e.g. Ragas paraphrased or merged text); kept, but
              flagged so a reviewer knows this sample can't drive span-based
              retrieval metrics
"""

from __future__ import annotations

from typing import Any

from groundtruth.schemas import Chunk, Document


class SpanIndex:
    def __init__(self, chunks: list[Chunk], documents: list[Document]) -> None:
        self._by_text: dict[str, Chunk] = {}
        for chunk in chunks:
            self._by_text.setdefault(chunk.text.strip(), chunk)
        self._documents = documents

    def anchor(self, context: str) -> dict[str, Any]:
        text = context.strip()
        chunk = self._by_text.get(text)
        if chunk is not None:
            return {
                "doc_id": chunk.doc_id,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "match": "chunk",
            }
        if text:
            for doc in self._documents:
                start = doc.text.find(text)
                if start != -1:
                    return {
                        "doc_id": doc.doc_id,
                        "start_char": start,
                        "end_char": start + len(text),
                        "match": "exact",
                    }
        return {"doc_id": None, "start_char": None, "end_char": None, "match": "none"}

    def anchor_all(self, contexts: list[str]) -> list[dict[str, Any]]:
        return [self.anchor(c) for c in contexts]
