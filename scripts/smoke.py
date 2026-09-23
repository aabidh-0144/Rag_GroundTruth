"""End-to-end smoke test against the real corpus, models and indexes.

The unit tests use fakes so they stay fast; this script is the counterpart that
exercises the parts fakes cannot cover - pymupdf's actual output, the real
tokenizer, Chroma persistence, and the cross-encoder.

    python scripts/smoke.py [--skip-index] [--generate]

Assumes `ingest`, `chunk` and `index` have already run unless --full is passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundtruth.config import CHUNKS_PATH, Config, load_config  # noqa: E402
from groundtruth.embedder import Embedder  # noqa: E402
from groundtruth.index import load_chunks  # noqa: E402
from groundtruth.ingest import load_documents  # noqa: E402
from groundtruth.retrievers import build_retriever  # noqa: E402

QUERY = "What is the attention mechanism?"

PASS = "  PASS "
FAIL = "  FAIL "

failures: list[str] = []


def check(condition: bool, label: str) -> None:
    print(f"{PASS if condition else FAIL}{label}")
    if not condition:
        failures.append(label)


def check_documents() -> None:
    print("\n[1] documents")
    docs = load_documents()
    check(len(docs) > 0, f"{len(docs)} document(s) loaded")

    for doc in docs:
        check(bool(doc.text.strip()), f"{doc.doc_id}: text is non-empty")
        check(
            all(0 <= p.start_char <= p.end_char <= len(doc.text) for p in doc.pages),
            f"{doc.doc_id}: page offsets lie inside the text",
        )


def check_chunks() -> None:
    print("\n[2] chunks")
    docs = {d.doc_id: d for d in load_documents()}
    chunks = load_chunks()
    check(len(chunks) > 0, f"{len(chunks):,} chunk(s) loaded")

    ids = [c.chunk_id for c in chunks]
    check(len(ids) == len(set(ids)), "chunk ids are unique")

    # The invariant the golden dataset will depend on, verified against real text.
    bad = [
        c.chunk_id
        for c in chunks
        if c.doc_id in docs and docs[c.doc_id].text[c.start_char : c.end_char] != c.text
    ]
    check(not bad, f"char spans round-trip ({len(bad)} mismatches)")

    oversized = [c.chunk_id for c in chunks if c.n_tokens > 512]
    check(not oversized, f"no chunk exceeds the embedder's 512-token window ({len(oversized)} over)")


def check_retrievers(cfg: Config) -> None:
    print("\n[3] retrievers")
    embedder = Embedder(cfg.embedding)

    for strategy in ("dense", "bm25", "hybrid", "hybrid_rerank"):
        cfg.retrieval.strategy = strategy
        retriever = build_retriever(cfg, embedder)
        results, timings = retriever.retrieve(QUERY, k=5)

        check(len(results) > 0, f"{strategy}: returned {len(results)} result(s)")
        check(
            all(a.score >= b.score for a, b in zip(results, results[1:])),
            f"{strategy}: scores are non-increasing",
        )
        check(
            [c.rank for c in results] == list(range(1, len(results) + 1)),
            f"{strategy}: ranks are contiguous and 1-based",
        )
        check(bool(timings), f"{strategy}: timings recorded {sorted(timings)}")
        total = sum(timings.values())
        print(f"         top hit: {results[0].chunk_id} score={results[0].score:.4f} ({total:.0f}ms)")


def check_determinism(cfg: Config) -> None:
    print("\n[4] determinism")
    embedder = Embedder(cfg.embedding)
    cfg.retrieval.strategy = "hybrid"
    retriever = build_retriever(cfg, embedder)

    first, _ = retriever.retrieve(QUERY, k=10)
    second, _ = retriever.retrieve(QUERY, k=10)

    check(
        [c.chunk_id for c in first] == [c.chunk_id for c in second],
        "repeated retrieval returns an identical ranking",
    )


def check_generation(cfg: Config) -> None:
    print("\n[5] generation")
    from groundtruth.generator import REFUSAL, build_generator

    generator = build_generator(cfg.generation)
    healthy, message = generator.health_check()
    if not healthy:
        print(f"  SKIP  {message}")
        return
    check(True, f"ollama reachable, model {cfg.generation.model} present")

    embedder = Embedder(cfg.embedding)
    cfg.retrieval.strategy = "hybrid"
    results, _ = build_retriever(cfg, embedder).retrieve(QUERY, k=5)

    answer = generator.generate(QUERY, results)
    check(bool(answer.strip()), "generated a non-empty answer")
    print(f"         {answer[:200]}")

    # The refusal path matters: it is what separates a retrieval failure from a
    # hallucination once the evaluation layer exists.
    refusal = generator.generate("What is the capital city of Bolivia?", results)
    check(REFUSAL in refusal, f"refuses out-of-context questions (got: {refusal[:80]!r})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hybrid_rerank.yaml")
    parser.add_argument("--generate", action="store_true", help="also test Ollama generation")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if not CHUNKS_PATH.exists():
        print("No chunks found. Run: ingest -> chunk -> index first.", file=sys.stderr)
        return 1

    check_documents()
    check_chunks()
    check_retrievers(cfg)
    check_determinism(cfg)
    if args.generate:
        check_generation(cfg)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
