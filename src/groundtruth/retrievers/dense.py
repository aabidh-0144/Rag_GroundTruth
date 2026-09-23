"""Experiment A: pure vector retrieval."""

from __future__ import annotations

from groundtruth.embedder import Embedder
from groundtruth.index import get_collection
from groundtruth.retrievers.base import rank
from groundtruth.schemas import RetrievedChunk
from groundtruth.utils import Timer


class DenseRetriever:
    """Cosine nearest-neighbour search over the Chroma collection."""

    name = "dense"

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            self._collection = get_collection()
        return self._collection

    def warmup(self) -> None:
        """Load the model and open the collection before any timed query."""
        self.embedder.encode_query("warmup")
        self.collection.count()

    def retrieve(self, query: str, k: int) -> tuple[list[RetrievedChunk], dict[str, float]]:
        timer = Timer()

        with timer("embed"):
            vector = self.embedder.encode_query(query)

        with timer("search"):
            response = self.collection.query(
                query_embeddings=[vector.tolist()],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )

        results: list[RetrievedChunk] = []
        ids = response["ids"][0]
        documents = response["documents"][0]
        metadatas = response["metadatas"][0]
        distances = response["distances"][0]

        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            # Chroma reports cosine *distance*; convert so higher is better and the
            # score reads as a similarity like every other retriever here.
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    score=1.0 - float(distance),
                    rank=0,
                    metadata=dict(metadata),
                )
            )

        return rank(results), timer.total()
