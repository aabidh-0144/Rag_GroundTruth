# Groundtruth — Project Context

## What this is

**Groundtruth** is an automated evaluation and regression-testing framework for RAG systems.
It answers two questions without manual checking:

1. Did the retriever fetch the correct information?
2. Did the LLM generate a correct and faithful answer from what was retrieved?

This is a **personal learning project**, not a production system. Do not over-engineer.
Prefer simple, reproducible, minimal-infrastructure solutions.

## Project phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | **The RAG system under test** — ingest, chunk, index, retrieve, generate | complete |
| 2 | **Golden dataset** — ~100 Q/A pairs over ~25 research papers | tooling complete (Ragas synthetic generation + human review); dataset itself not yet generated |
| 3 | **Evaluation layer** | LLM-judge (Mode 1) + Ragas (Mode 2) complete — see docs/EVALUATION.md. **Not built yet:** rank metrics Recall@K, MRR, nDCG against span-labelled datasets |
| 4 | **Experiment comparison** — Dense vs Hybrid vs Hybrid+Reranker report | side-by-side compare in the Streamlit app; report not started |

Likely next work: Recall@K / MRR / nDCG using `reference_spans` from reviewed datasets.

## Hardware constraints (non-negotiable)

- Windows 11, Intel i7-1185G7 (4 cores / 8 threads), 31 GB RAM, ~39 GB free disk
- **No GPU** — never propose CUDA-dependent tooling; always install the CPU torch wheel
- **No Docker** — never propose containerised services (no Qdrant/Weaviate/Elasticsearch servers)
- Everything must run embedded/in-process

## Locked stack

| Layer | Choice |
|-------|--------|
| Vector store | ChromaDB, `PersistentClient(path="store/chroma")`, cosine |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim) via sentence-transformers |
| Sparse | `rank_bm25.BM25Okapi`, pickled to `store/bm25.pkl` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` (default); `BAAI/bge-reranker-base` as Experiment D |
| Generator | Ollama at `localhost:11434`, model `qwen3:4b` (fallback `llama3.2:3b`) |
| PDF parsing | `pymupdf4llm.to_markdown(page_chunks=True)` |
| Python | `.venv` at repo root, Python 3.13 |
| Judge (eval) | Same provider/model as generation by default (`configs/evaluation.yaml`, `GROUNDTRUTH_JUDGE_*`). **User wants local Ollama only — no Groq, no paid LLMs.** |
| Eval framework | Ragas 0.4.x (`ragas.metrics.collections` API only — the `ragas.metrics` imports are deprecated) |
| Eval UI | Streamlit (`groundtruth.cli ui`) |

Do not swap these without being asked. Config-level changes (chunk size, k, fusion params)
are fine and expected; stack-level changes are not.

## Invariants — breaking these invalidates the evaluation

1. **Chunk IDs are `sha1(f"{doc_id}:{start_char}:{end_char}")[:16]`.** Deterministic and
   content-independent, so they are stable across runs and machines.
2. **Ground truth anchors to character spans `(doc_id, start_char, end_char)`, never to
   `chunk_id`.** This is what lets the golden dataset survive a re-chunking experiment.
   If a chunking change would break labels, the labels are wrong, not the chunker.
3. **All experiments share one index.** Dense / Hybrid / Hybrid+Reranker are read-side
   strategies over the same Chroma collection and BM25 index. Chunking and embeddings are
   held constant so the comparison isolates one variable.
4. **Every retriever returns `(list[RetrievedChunk], timings: dict)`.** The eval layer
   depends on ranked chunk IDs with scores, not just the answer string.
5. **bge queries always get the instruction prefix**
   `"Represent this sentence for searching relevant passages: "`. Documents do not.
   This is handled inside `Embedder`; never call the raw model directly.
6. **Latency is measured per stage** (`embed_ms`, `search_ms`, `rerank_ms`, `generate_ms`,
   `total_ms`) from day one, because p95 is a phase-3 deliverable.
7. **Runs record a corpus fingerprint.** A run whose fingerprint disagrees with the index
   must fail loudly rather than produce an incomparable result.
8. **A missing score is never 0.** Every metric result is `success | failed | skipped`;
   failed/skipped are excluded from means and counted. `overall_score` is null when a
   requested metric applied but never succeeded. All rules live in `evaluation/aggregate.py`.
9. **Synthetic data is never ground truth until a human verifies it.** Generated samples
   start `origin: synthetic`, `verification_status: unverified`; rejected samples are never
   evaluated; edits keep the original and reset to unverified. Never fabricate a ground truth —
   reference-based metrics without one are `skipped`.

## Environment facts learned while building (do not rediscover these)

- **Corporate TLS inspection breaks Python HTTPS.** certifi lacks the proxy's root CA, so
  huggingface.co downloads fail with CERTIFICATE_VERIFY_FAILED even though curl works.
  Fixed by `truststore`, injected in `src/groundtruth/__init__.py`. Do not remove it.
- **GitHub release assets are blocked (HTTP 403).** This means Ollama cannot be installed on
  this machine via winget or a direct download - both redirect to
  `release-assets.githubusercontent.com`. PyPI, Hugging Face and arXiv are all reachable.
  The user runs Ollama (a qwen 14b model) on a separate personal laptop instead.
- **This machine's `.env` sets `GROUNDTRUTH_GENERATION_PROVIDER=groq`** (from an earlier
  session). The judge inherits generation settings, so evaluations run *here* use Groq unless
  that line is commented out or `GROUNDTRUTH_JUDGE_PROVIDER=ollama` is set. The user does not
  want Groq used for real work — never send their corpus to it without asking.
- **Ragas 0.4.4 gotchas (verified against the installed package):** the legacy
  `ragas.embeddings.base.HuggingfaceEmbeddings` cannot be instantiated (abstract async
  methods) — use the modern `ragas.embeddings.HuggingFaceEmbeddings` for metrics AND testset
  transforms; testset generation needs `rapidfuzz`, which Ragas does not declare;
  `RunConfig` defaults to 16 workers (we pass `max_workers=concurrency`); Ragas posts usage
  analytics unless `RAGAS_DO_NOT_TRACK=true` (set by `ragas_mode/adapters.import_ragas()`).
  `tests/test_eval_ragas.py` checks the metric registry against the installed Ragas API.
- **Measured CPU latency** (i7-1185G7, warm, 50 rerank candidates, retrieval only):
  dense ~22ms, hybrid ~24ms, hybrid+MiniLM-L6 ~4.4s, hybrid+bge-reranker-base ~29s.
  The reranker dominates; this is real, and worth reporting rather than hiding.
- **Embedding throughput** is ~6 chunks/sec on CPU. 2 papers -> 87 chunks -> ~14s.
  Expect ~1,100 chunks and ~3 minutes of indexing for a 25-paper corpus.
- **Model loading costs 40-130s** and must never land inside a timed query. Every retriever
  exposes `warmup()`; `run_batch` calls it before the loop. Keep it that way or p95 is garbage.
- Two arXiv PDFs (Attention Is All You Need, BERT) sit in `data/raw/` as test fixtures.
  They are gitignored and can be deleted once the real corpus lands.

## Layout

```
data/raw/*.pdf                      user drops PDFs here (gitignored)
data/processed/documents/*.json     normalized text + page char offsets
data/processed/chunks.jsonl         canonical chunks
store/chroma/ , store/bm25.pkl      indexes
runs/<run_id>/                      config.yaml, results.jsonl, manifest.json
evals/<evaluation_id>/              config.yaml, results.jsonl (append-only), summary.json
datasets/<dataset_id>/              samples.jsonl, manifest.json (tracked - commit reviewed ones)
configs/{dense,hybrid,hybrid_rerank,hybrid_rerank_large}.yaml , configs/evaluation.yaml
src/groundtruth/                    package; llm.py = provider transport shared by generator + judges
src/groundtruth/evaluation/         Phase 3 - __init__.py is a map of the package
src/groundtruth/ui/                 Streamlit app (app.py + views/ one file per page)
```

`data/`, `store/`, `runs/`, `evals/` are gitignored — all regenerable from `data/raw/` + configs.

## Commands

```bash
.venv/Scripts/python.exe -m groundtruth.cli ingest                       # PDFs   -> documents
.venv/Scripts/python.exe -m groundtruth.cli chunk                        # docs   -> chunks.jsonl
.venv/Scripts/python.exe -m groundtruth.cli index                        # chunks -> chroma + bm25
.venv/Scripts/python.exe -m groundtruth.cli ask "question" --config configs/hybrid.yaml
.venv/Scripts/python.exe -m groundtruth.cli run --config configs/dense.yaml --queries q.jsonl
.venv/Scripts/python.exe -m groundtruth.cli eval reference-free --run <run_id>     # Mode 1
.venv/Scripts/python.exe -m groundtruth.cli eval ragas --run <run_id>              # Mode 2
.venv/Scripts/python.exe -m groundtruth.cli eval generate-dataset --size 5
.venv/Scripts/python.exe -m groundtruth.cli eval review-dataset <dataset_id>
.venv/Scripts/python.exe -m groundtruth.cli eval history | results <id>
.venv/Scripts/python.exe -m groundtruth.cli ui                                     # Streamlit
.venv/Scripts/python.exe -m pytest                     # ~30s, no network
GROUNDTRUTH_LIVE_TESTS=1 .venv/Scripts/python.exe -m pytest -m live   # real judge / models
```

## Conventions

- **Configs drive behaviour.** No magic numbers in code; every tunable lives in `configs/*.yaml`.
- **New retrieval strategies are new `Retriever` implementations**, never a branch inside an
  existing one. That is how the experiment matrix stays clean.
- Prefer stdlib and explicit code over frameworks. There is deliberately no LangChain /
  LlamaIndex here — the whole point is that retrieval internals stay inspectable.
  **Exception:** Ragas (which depends on LangChain). Import it only inside
  `evaluation/ragas_mode/` and `evaluation/synthetic/`, via `adapters.import_ragas()`.
- **Structure for readability (user's explicit request):** one responsibility per file; one
  Mode 1 metric per file; prompts only in `evaluation/prompts.py`; scoring rules only in
  `evaluation/aggregate.py`; new eval metrics are a new class + registry entry, like retrievers.
- The Streamlit app never scores anything itself; it launches `groundtruth.cli` subprocesses
  and reads `evals/` and `datasets/`.
