"""Command-line entry point.

    python -m groundtruth.cli ingest
    python -m groundtruth.cli chunk
    python -m groundtruth.cli index
    python -m groundtruth.cli ask "What is multi-head attention?" --config configs/hybrid.yaml
    python -m groundtruth.cli run --config configs/dense.yaml --queries queries.jsonl

argparse rather than a CLI framework - five verbs do not justify a dependency.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from groundtruth.config import CHUNKS_PATH, ROOT, Config, load_config


def _load(config_path: str | None) -> Config:
    return load_config(config_path) if config_path else Config()


def cmd_ingest(args: argparse.Namespace) -> int:
    from groundtruth.ingest import ingest_all

    print("Ingesting PDFs from data/raw/ ...")
    docs = ingest_all(force=args.force)
    total_chars = sum(len(d.text) for d in docs)
    print(f"\n{len(docs)} documents, {total_chars:,} characters total.")
    return 0


def cmd_chunk(args: argparse.Namespace) -> int:
    from groundtruth.chunking import chunk_documents
    from groundtruth.embedder import Embedder
    from groundtruth.ingest import load_documents
    from groundtruth.utils import write_jsonl

    cfg = _load(args.config)
    docs = load_documents()
    embedder = Embedder(cfg.embedding)

    print(f"Chunking {len(docs)} documents (size={cfg.chunking.size}, overlap={cfg.chunking.overlap}) ...")
    chunks = chunk_documents(docs, cfg.chunking, embedder.tokenizer)

    n = write_jsonl(CHUNKS_PATH, [c.to_dict() for c in chunks])
    token_counts = sorted(c.n_tokens for c in chunks)
    print(f"\n{n:,} chunks -> {CHUNKS_PATH.relative_to(ROOT)}")
    if token_counts:
        median = token_counts[len(token_counts) // 2]
        print(f"tokens per chunk: min={token_counts[0]} median={median} max={token_counts[-1]}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    from groundtruth.embedder import Embedder
    from groundtruth.index import build_all, load_chunks

    cfg = _load(args.config)
    chunks = load_chunks()
    embedder = Embedder(cfg.embedding)

    print(f"Indexing {len(chunks):,} chunks ...")
    manifest = build_all(cfg, chunks, embedder)
    print(f"\nDone. index_fingerprint={manifest['index_fingerprint']}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from groundtruth.pipeline import RagPipeline

    cfg = _load(args.config)
    pipeline = RagPipeline(cfg, generate=not args.no_generate)

    if pipeline.generator is not None:
        healthy, message = pipeline.generator.health_check()
        if not healthy:
            print(f"warning: generation disabled - {message}\n", file=sys.stderr)
            pipeline.generator = None

    if args.warmup:
        print("warming up models (timings will reflect steady state)...")
        pipeline.warmup()

    result = pipeline.answer(args.question)

    print(f"\n=== retrieved ({cfg.retrieval.strategy}, k={cfg.retrieval.k}) ===")
    for chunk in result.retrieved:
        source = chunk.metadata.get("doc_id", "?")
        section = chunk.metadata.get("section_path") or "-"
        pages = f"p{chunk.metadata.get('page_start', '?')}-{chunk.metadata.get('page_end', '?')}"
        preview = " ".join(chunk.text.split())[:160]
        print(f"\n[{chunk.rank}] score={chunk.score:.4f}  {source} | {section} | {pages}")
        print(f"    {chunk.chunk_id}  {preview}...")

    if result.answer:
        print("\n=== answer ===")
        print(result.answer)
    if result.error:
        print(f"\n=== error ===\n{result.error}", file=sys.stderr)

    print("\n=== timings (ms) ===")
    for stage, value in sorted(result.timings.items()):
        print(f"  {stage:<14} {value:>9.2f}")
    if not args.warmup:
        print("  note: includes one-off model loading. Pass --warmup for steady-state timings.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from groundtruth.pipeline import run_batch
    from groundtruth.utils import read_json, read_jsonl

    cfg = _load(args.config)

    path = Path(args.queries)
    queries = read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)
    if args.limit:
        queries = queries[: args.limit]

    print(f"Running {len(queries)} queries under config '{cfg.name}' ...")
    run_dir = run_batch(cfg, queries, generate=not args.no_generate)

    from groundtruth.utils import read_json as _read_json

    manifest = _read_json(run_dir / "manifest.json")
    print(f"\nrun -> {run_dir.relative_to(ROOT)}")
    print(f"  queries={manifest['n_queries']} errors={manifest['n_errors']}")
    print(
        f"  latency mean={manifest['latency_ms']['mean']}ms "
        f"p50={manifest['latency_ms']['p50']}ms p95={manifest['latency_ms']['p95']}ms"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="groundtruth", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="path to an experiment YAML (defaults to built-in defaults)")

    p_ingest = sub.add_parser("ingest", help="parse PDFs in data/raw/ into documents")
    p_ingest.add_argument("--force", action="store_true", help="re-ingest even if up to date")
    p_ingest.set_defaults(func=cmd_ingest)

    p_chunk = sub.add_parser("chunk", parents=[common], help="split documents into chunks")
    p_chunk.set_defaults(func=cmd_chunk)

    p_index = sub.add_parser("index", parents=[common], help="build the chroma + bm25 indexes")
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", parents=[common], help="run one question interactively")
    p_ask.add_argument("question")
    p_ask.add_argument("--no-generate", action="store_true", help="retrieval only")
    p_ask.add_argument(
        "--warmup", action="store_true", help="preload models so timings exclude model loading"
    )
    p_ask.set_defaults(func=cmd_ask)

    p_run = sub.add_parser("run", parents=[common], help="batch-run a query file into runs/")
    p_run.add_argument("--queries", required=True, help="jsonl/json with {query_id, question}")
    p_run.add_argument("--limit", type=int, help="only run the first N queries")
    p_run.add_argument("--no-generate", action="store_true", help="retrieval only")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)

    try:
        return args.func(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
