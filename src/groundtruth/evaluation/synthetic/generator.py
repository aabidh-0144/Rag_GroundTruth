"""Synthetic dataset generation with Ragas' TestsetGenerator.

Workflow (Ragas 0.4.4 APIs, verified against the installed package):

1. Feed Ragas OUR chunks (data/processed/chunks.jsonl) as pre-chunked nodes, so
   it never re-chunks the corpus. Its reference contexts then come back as our
   chunk texts verbatim, which makes span anchoring exact (spans.py).
2. Build the knowledge graph with Ragas' own `default_transforms_for_prechunked`
   (summaries, entities, themes, relationships) via `apply_transforms`.
3. Build the query distribution from the configured weights:
       factual    -> SingleHopSpecificQuerySynthesizer
       reasoning  -> MultiHopAbstractQuerySynthesizer
       multihop   -> MultiHopSpecificQuerySynthesizer
   and drop any synthesizer the graph cannot support (multi-hop needs related
   chunks - a small corpus may have none), the same check Ragas' default
   distribution uses. Dropped types are recorded in the manifest, not hidden.
4. `TestsetGenerator.generate()` writes questions, reference answers and
   reference contexts.
5. Save every sample as origin="synthetic", verification_status="unverified".

Expensive: the transforms and synthesizers each make many LLM calls. On a local
CPU model keep `size` small and `concurrency` at 1.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Callable

from groundtruth.config import CHUNKS_PATH, GenerationConfig
from groundtruth.evaluation.config import EvaluationConfig
from groundtruth.evaluation.ragas_mode import adapters
from groundtruth.evaluation.synthetic.datasets import Dataset, DatasetSample, DatasetStore
from groundtruth.evaluation.synthetic.spans import SpanIndex
from groundtruth.schemas import Chunk, Document
from groundtruth.utils import sha256_file

QUESTION_TYPES = {
    "factual": "SingleHopSpecificQuerySynthesizer",
    "reasoning": "MultiHopAbstractQuerySynthesizer",
    "multihop": "MultiHopSpecificQuerySynthesizer",
}
# Ragas' synthesizer `name` -> our question type, for labelling generated samples.
_SYNTHESIZER_TYPE = {
    "single_hop_specific_query_synthesizer": "factual",
    "multi_hop_abstract_query_synthesizer": "reasoning",
    "multi_hop_specific_query_synthesizer": "multihop",
}


def validate_distribution(distribution: dict[str, float]) -> dict[str, float]:
    unknown = [k for k in distribution if k not in QUESTION_TYPES]
    if unknown:
        raise ValueError(f"unknown question type(s) {unknown}; available: {sorted(QUESTION_TYPES)}")
    if any(float(v) < 0 for v in distribution.values()):
        raise ValueError("question distribution weights must not be negative")
    weights = {k: float(v) for k, v in distribution.items() if float(v) > 0}
    if not weights:
        raise ValueError("question distribution needs at least one positive weight")
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def parse_distribution(text: str) -> dict[str, float]:
    """'factual=0.5,reasoning=0.25,multihop=0.25' -> dict."""
    out: dict[str, float] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        key, sep, value = part.partition("=")
        if not sep:
            raise ValueError(f"bad distribution entry {part!r}; expected type=weight")
        out[key.strip()] = float(value)
    return validate_distribution(out)


def _build_knowledge_graph(chunks: list[Chunk], llm: Any, embeddings: Any, run_config: Any) -> Any:
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.transforms import apply_transforms, default_transforms_for_prechunked

    nodes = [
        Node(
            type=NodeType.CHUNK,
            properties={
                "page_content": c.text,
                "document_metadata": {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "section_path": c.section_path,
                },
            },
        )
        for c in chunks
        if c.text.strip()
    ]
    kg = KnowledgeGraph(nodes=nodes)
    apply_transforms(
        kg, default_transforms_for_prechunked(llm=llm, embedding_model=embeddings), run_config=run_config
    )
    return kg


def _build_distribution(
    weights: dict[str, float], llm: Any, kg: Any, progress: Callable[[str], None]
) -> tuple[list[tuple[Any, float]], dict[str, str]]:
    from ragas.testset.synthesizers import multi_hop
    from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer

    classes = {
        "SingleHopSpecificQuerySynthesizer": SingleHopSpecificQuerySynthesizer,
        "MultiHopAbstractQuerySynthesizer": multi_hop.MultiHopAbstractQuerySynthesizer,
        "MultiHopSpecificQuerySynthesizer": multi_hop.MultiHopSpecificQuerySynthesizer,
    }
    available: list[tuple[Any, float]] = []
    dropped: dict[str, str] = {}
    for qtype, weight in weights.items():
        synthesizer = classes[QUESTION_TYPES[qtype]](llm=llm)
        try:
            supported = bool(synthesizer.get_node_clusters(kg))
        except Exception as exc:
            supported, why = False, f"{type(exc).__name__}: {exc}"
        else:
            why = "the knowledge graph has no node clusters this question type needs"
        if supported:
            available.append((synthesizer, weight))
        else:
            dropped[qtype] = why
            progress(f"  dropping question type {qtype!r}: {why}")
    if not available:
        raise ValueError(
            f"no requested question type is possible on this corpus ({dropped}); "
            "try 'factual' only, or add more documents"
        )
    total = sum(w for _, w in available)
    return [(s, w / total) for s, w in available], dropped


def generate_dataset(
    cfg: EvaluationConfig,
    judge: GenerationConfig,
    size: int | None = None,
    distribution: dict[str, float] | None = None,
    chunks: list[Chunk] | None = None,
    documents: list[Document] | None = None,
    store: DatasetStore | None = None,
    progress: Callable[[str], None] = print,
) -> Dataset:
    """Generate, anchor and save a synthetic dataset. Returns it (all unverified)."""
    from groundtruth.index import load_chunks
    from groundtruth.ingest import load_documents

    size = size or cfg.synthetic.size
    if size < 1:
        raise ValueError("dataset size must be at least 1")
    weights = validate_distribution(distribution or cfg.synthetic.distribution)
    chunks = chunks if chunks is not None else load_chunks()
    documents = documents if documents is not None else load_documents()
    if not chunks:
        raise ValueError("no chunks to generate from - run ingest, chunk and index first")

    adapters.import_ragas()
    from ragas.run_config import RunConfig
    from ragas.testset import TestsetGenerator

    llm = adapters.build_ragas_llm(judge)
    embeddings = adapters.build_embeddings(cfg.ragas.embedding_model)
    # Ragas defaults to 16 parallel workers - far too many for one local model.
    run_config = RunConfig(
        timeout=judge.timeout_s, max_retries=judge.max_retries, max_workers=cfg.concurrency
    )

    progress(f"building knowledge graph from {len(chunks)} chunks (many LLM calls)...")
    kg = _build_knowledge_graph(chunks, llm, embeddings, run_config)
    query_distribution, dropped = _build_distribution(weights, llm, kg, progress)

    progress(f"generating {size} sample(s)...")
    generator = TestsetGenerator(llm=llm, embedding_model=embeddings, knowledge_graph=kg)
    testset = generator.generate(
        testset_size=size, query_distribution=query_distribution, run_config=run_config
    )

    spans = SpanIndex(chunks, documents)
    store = store or DatasetStore()
    dataset_id = store.new_id()
    created = dt.datetime.now().isoformat(timespec="seconds")
    samples = []
    for i, item in enumerate(testset.samples, start=1):
        sample = item.eval_sample
        contexts = list(getattr(sample, "reference_contexts", None) or [])
        samples.append(
            DatasetSample(
                sample_id=f"{dataset_id}_{i:03d}",
                question=sample.user_input or "",
                ground_truth=getattr(sample, "reference", None),
                reference_contexts=contexts,
                reference_spans=spans.anchor_all(contexts),
                question_type=_SYNTHESIZER_TYPE.get(item.synthesizer_name, item.synthesizer_name),
                generator_model=f"{judge.provider}:{judge.model}",
                created_at=created,
            )
        )

    dataset = Dataset(
        dataset_id=dataset_id,
        manifest={
            "created_at": created,
            "kind": "synthetic",
            "generator": {
                "framework": f"ragas {adapters.import_ragas().__version__}",
                "provider": judge.provider,
                "model": judge.model,
                "embedding_model": cfg.ragas.embedding_model,
            },
            "requested_size": size,
            "distribution_requested": weights,
            "question_types_dropped": dropped,
            "corpus": {
                "n_chunks": len(chunks),
                "doc_ids": sorted({c.doc_id for c in chunks}),
                "chunks_sha256": sha256_file(CHUNKS_PATH) if CHUNKS_PATH.exists() else None,
            },
            "note": "All samples are unverified synthetic data until a human reviews them.",
        },
        samples=samples,
    )
    store.save(dataset)
    return dataset
