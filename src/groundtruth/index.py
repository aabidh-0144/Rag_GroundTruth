"""Chunks -> searchable indexes.

Two indexes are built from one canonical chunk set:

* **Dense** - ChromaDB persistent collection, cosine space.
* **Sparse** - rank_bm25 BM25Okapi, pickled. Chroma has no lexical search, so the
  hybrid experiment needs its own sparse index.

Both are built once and shared by every retrieval strategy (invariant #3), which is
what makes Dense / Hybrid / Hybrid+Reranker an apples-to-apples comparison.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from groundtruth.config import (
    BM25_PATH,
    CHROMA_DIR,
    CHUNKS_PATH,
    COLLECTION_NAME,
    Config,
)
from groundtruth.embedder import Embedder
from groundtruth.schemas import Chunk
from groundtruth.utils import read_jsonl, sha256_file

# Split on anything that is not a word character or a hyphen. Hyphens are kept
# because technical terms depend on them ("self-attention", "bi-encoder"), and no
# stemming is applied for the same reason - "transformers" and "transformer" being
# distinct costs little, while stemming "BLEU" or "GELU" into nonsense costs a lot.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize_bm25(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(slots=True)
class BM25Index:
    """Pickled sparse index plus the chunk-ID ordering its scores align to."""

    bm25: object
    chunk_ids: list[str]

    def save(self, path: Path = BM25_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"bm25": self.bm25, "chunk_ids": self.chunk_ids}, fh)

    @classmethod
    def load(cls, path: Path = BM25_PATH) -> BM25Index:
        if not path.exists():
            raise FileNotFoundError(f"BM25 index missing at {path}. Run `index` first.")
        with path.open("rb") as fh:
            data = pickle.load(fh)
        return cls(bm25=data["bm25"], chunk_ids=data["chunk_ids"])


def load_chunks(path: Path = CHUNKS_PATH) -> list[Chunk]:
    if not path.exists():
        raise FileNotFoundError(f"No chunks at {path}. Run `chunk` first.")
    return [Chunk.from_dict(row) for row in read_jsonl(path)]


def get_collection(create: bool = False):
    """Open the Chroma collection.

    `hnsw:search_ef` is raised well above the default so that, at this corpus size,
    approximate search returns effectively exact results. Otherwise ANN recall error
    would leak into the Recall@10 numbers and we would be measuring the index rather
    than the retriever.
    """
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    metadata = {
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 400,
        "hnsw:search_ef": 400,
        "hnsw:M": 32,
    }
    if create:
        return client.get_or_create_collection(name=COLLECTION_NAME, metadata=metadata)
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:  # chroma raises several types across versions
        raise FileNotFoundError(
            f"Chroma collection {COLLECTION_NAME!r} not found in {CHROMA_DIR}. Run `index` first."
        ) from exc


def build_dense_index(chunks: list[Chunk], embedder: Embedder, batch: int = 512) -> None:
    """Embed every chunk and (re)populate the Chroma collection."""
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Rebuild from scratch: a partially-updated index is worse than no index, because
    # it fails silently rather than loudly.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = get_collection(create=True)

    print(f"  embedding {len(chunks):,} chunks...")
    vectors = embedder.encode_documents([c.text for c in chunks])

    print(f"  writing to chroma ({CHROMA_DIR})...")
    for start in range(0, len(chunks), batch):
        window = chunks[start : start + batch]
        collection.add(
            ids=[c.chunk_id for c in window],
            documents=[c.text for c in window],
            embeddings=vectors[start : start + batch].tolist(),
            metadatas=[c.metadata() for c in window],
        )

    count = collection.count()
    if count != len(chunks):
        raise RuntimeError(
            f"Chroma holds {count} vectors but {len(chunks)} chunks were indexed. "
            "Duplicate chunk_ids would cause this - the index is not trustworthy."
        )
    print(f"  chroma: {count:,} vectors, dim={embedder.dim}")


def build_bm25_index(chunks: list[Chunk]) -> None:
    from rank_bm25 import BM25Okapi

    print(f"  building bm25 over {len(chunks):,} chunks...")
    corpus = [tokenize_bm25(c.text) for c in chunks]
    index = BM25Index(bm25=BM25Okapi(corpus), chunk_ids=[c.chunk_id for c in chunks])
    index.save()
    print(f"  bm25: saved to {BM25_PATH}")


def build_all(cfg: Config, chunks: list[Chunk], embedder: Embedder) -> dict[str, str]:
    """Build both indexes and return the fingerprint manifest."""
    build_dense_index(chunks, embedder)
    build_bm25_index(chunks)

    manifest = {
        "n_chunks": str(len(chunks)),
        "chunks_sha256": sha256_file(CHUNKS_PATH),
        "embedding_model": cfg.embedding.model,
        "index_fingerprint": cfg.index_fingerprint(),
    }
    from groundtruth.utils import write_json

    write_json(CHROMA_DIR.parent / "index_manifest.json", manifest)
    return manifest


def load_index_manifest() -> dict[str, str]:
    from groundtruth.utils import read_json

    path = CHROMA_DIR.parent / "index_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Index manifest missing at {path}. Run `index` first.")
    return read_json(path)
