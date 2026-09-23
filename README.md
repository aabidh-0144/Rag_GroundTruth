# Groundtruth

An automated evaluation and regression-testing framework for RAG systems.

RAG systems fail in ways that are hard to see: the retriever misses the right chunk, or ranks
it too low, or the LLM hallucinates despite being given the right context. Worse, a change to
chunking, embeddings, retrieval or prompts can silently degrade quality. Groundtruth exists to
answer two questions automatically, and repeatably:

1. **Did the retriever fetch the right information?**
2. **Did the LLM produce a correct and faithful answer from it?**

## Documentation

| Document | Read it for |
|---|---|
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | **Start here.** What was built and why: folder-by-folder guide, the full RAG pipeline (chunking method, embedding model, retrieval strategies), data formats, design invariants, measured performance |
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | Step-by-step setup and run instructions, from creating a venv to producing experiment results |
| [docs/OLLAMA_SETUP.md](docs/OLLAMA_SETUP.md) | Installing Ollama and enabling answer generation (optional — retrieval works without it) |

## Status

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | The RAG system under test | **done** |
| 2 | Golden dataset (~100 Q/A pairs) | not started |
| 3 | Evaluation metrics (Recall@K, MRR, nDCG, faithfulness, relevance, p95) | not started |
| 4 | Experiment comparison report | not started |

Phase 1 builds the system that phases 2-4 will measure. It is deliberately transparent - no
LangChain, no LlamaIndex - because an evaluation framework whose retrieval internals you cannot
inspect is not worth much.

## Architecture

```
data/raw/*.pdf
      |
      v  [1] INGEST      pymupdf4llm -> markdown, headers preserved
data/processed/documents/<doc_id>.json
      |
      v  [2] CHUNK       section-aware + token windows
data/processed/chunks.jsonl
      |
      +--> [3a] EMBED    bge-small-en-v1.5 -> ChromaDB      (store/chroma/)
      +--> [3b] BM25     rank_bm25                          (store/bm25.pkl)
                  |
                  v  [4] RETRIEVE   dense | bm25 | hybrid (RRF) | hybrid+rerank
                  v  [5] GENERATE   Ollama, grounded prompt with citations
                  v  [6] ARTIFACTS  runs/<run_id>/{config.yaml, results.jsonl, manifest.json}
```

`results.jsonl` is the seam the evaluation layer will consume: one record per query holding the
question, the full ranked list of chunk IDs with scores, the answer, and per-stage latency.

## Setup

Requires Python 3.11+ (developed on 3.13) and ~2 GB of disk for models.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-torch.txt   # CPU-only torch, first
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -e . --no-deps
```

For answer generation, install [Ollama](https://ollama.com/download) and pull a model:

```bash
ollama pull qwen3:4b
```

Retrieval works without Ollama - pass `--no-generate` to skip generation.

## Usage

Drop research-paper PDFs into `data/raw/`, then:

```bash
.venv/Scripts/python.exe -m groundtruth.cli ingest
.venv/Scripts/python.exe -m groundtruth.cli chunk --config configs/hybrid.yaml
.venv/Scripts/python.exe -m groundtruth.cli index --config configs/hybrid.yaml
```

Ask a single question (the main debugging tool - prints ranked chunks, scores, and timings):

```bash
.venv/Scripts/python.exe -m groundtruth.cli ask "What is multi-head attention?" \
    --config configs/hybrid_rerank.yaml --warmup
```

Batch-run a query file into a reproducible run directory:

```bash
.venv/Scripts/python.exe -m groundtruth.cli run \
    --config configs/hybrid.yaml --queries data/sample_queries.jsonl
```

## Experiments

Four configs, differing **only** in their `retrieval` block so the comparison isolates one
variable at a time. They share one index; the pipeline refuses to run if a config's chunking or
embedding settings disagree with the index that was built.

| Config | Strategy | Warm latency/query* |
|---|---|---|
| `configs/dense.yaml` | vector only | ~50 ms |
| `configs/hybrid.yaml` | BM25 + vector, fused with RRF | ~50 ms |
| `configs/hybrid_rerank.yaml` | hybrid + MiniLM-L6 cross-encoder | ~2.8 s |
| `configs/hybrid_rerank_large.yaml` | hybrid + bge-reranker-base | ~29 s |

\* measured on an i7-1185G7 (4c/8t), CPU only, 50 rerank candidates, retrieval only.

Reranking dominates latency on CPU. That is not a flaw to hide - quantifying the
quality-versus-latency tradeoff is one of the things this framework is for.

## Design invariants

These are not style preferences; breaking them invalidates the evaluation.

1. **Chunk IDs are `sha1(f"{doc_id}:{start_char}:{end_char}")[:16]`** - deterministic and
   reproducible across machines.
2. **Ground truth anchors to character spans, never to chunk IDs.** This is what lets a golden
   dataset survive a change of chunk size. `document.text[start:end] == chunk.text` is enforced
   by tests and by `scripts/smoke.py`.
3. **All experiments share one index.** Retrieval strategies are read-side only.
4. **Every retriever returns `(list[RetrievedChunk], timings)`.** The eval layer needs ranked
   chunk IDs with scores, not just an answer string.
5. **bge queries always get the instruction prefix**; documents never do. Handled inside
   `Embedder` so it cannot be forgotten.
6. **Latency is measured per stage**, and models are warmed up before timed runs so model
   loading never lands inside a p95.
7. **Runs record a corpus fingerprint.** A config/index mismatch fails loudly rather than
   producing plausible, meaningless numbers.

## Testing

```bash
.venv/Scripts/python.exe -m pytest              # 52 unit tests, fakes only, <1s
.venv/Scripts/python.exe scripts/smoke.py       # end-to-end against real models and indexes
.venv/Scripts/python.exe scripts/smoke.py --generate   # also exercises Ollama
```

The unit tests use a fake tokenizer and stub retrievers so they stay fast and hermetic.
`scripts/smoke.py` is the counterpart that exercises what fakes cannot: real PDF parsing, the
real tokenizer, Chroma persistence, and the cross-encoder.

## Layout

```
configs/             experiment YAML - all tunables live here, none in code
src/groundtruth/
  ingest.py          PDF -> Document (page offsets, boilerplate/reference stripping)
  chunking.py        Document -> Chunk (section-aware, token-exact)
  embedder.py        bge wrapper enforcing the query/document asymmetry
  index.py           Chroma + BM25 construction, fingerprinting
  retrievers/        dense, bm25, hybrid (RRF), rerank - one file each
  generator.py       Ollama client + grounded prompt
  pipeline.py        end-to-end orchestration and run artifacts
  cli.py             ingest | chunk | index | ask | run
data/, store/, runs/ gitignored; all regenerable from data/raw/ + configs
```
