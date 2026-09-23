# Getting Started — Running Groundtruth from Scratch

Step-by-step, from an empty machine to a working RAG system with experiment results.

**Time required:** ~20 minutes, most of it waiting on downloads.
**Prerequisites:** Python 3.11+ and ~3 GB free disk.

> For *what* the system does and *why*, read [`ARCHITECTURE.md`](ARCHITECTURE.md).
> For answer generation, read [`OLLAMA_SETUP.md`](OLLAMA_SETUP.md) — **not required** for
> retrieval, which is the bulk of the system.

---

## Quick reference

If you've done this before and just want the commands:

```powershell
cd C:\Users\301755\Documents\Personal\Rag_GroundTruth

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-torch.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps

# put PDFs in data\raw\ then:
.\.venv\Scripts\python.exe -m groundtruth.cli ingest
.\.venv\Scripts\python.exe -m groundtruth.cli chunk --config configs\hybrid.yaml
.\.venv\Scripts\python.exe -m groundtruth.cli index --config configs\hybrid.yaml
.\.venv\Scripts\python.exe -m groundtruth.cli ask "your question" --config configs\hybrid.yaml --no-generate
```

Everything below explains those steps in detail.

---

## Step 1 — Check your Python

```powershell
python --version
```

Needs **3.11 or newer**. Developed and tested on 3.13.14.

If Python isn't installed, get it from [python.org](https://www.python.org/downloads/) and tick
**"Add Python to PATH"** during setup.

---

## Step 2 — Open a terminal in the project folder

```powershell
cd to \Rag_GroundTruth
```

Confirm you're in the right place — you should see `src`, `configs`, `tests`:

```powershell
ls
```

---

## Step 3 — Create the virtual environment

A venv keeps this project's packages isolated from your system Python.

```powershell
python -m venv .venv
```

Creates a `.venv\` folder. Takes a few seconds.

### Optional: activate it

You can activate the venv so you can type `python` instead of the full path:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Your prompt gains a `(.venv)` prefix. The `Set-ExecutionPolicy` line is needed once per terminal
because PowerShell blocks scripts by default; `-Scope Process` means it only affects this window.

> **The rest of this guide uses the full path** (`.\.venv\Scripts\python.exe`) so the commands
> work whether or not you activated. If you did activate, plain `python` works too.

---

## Step 4 — Install PyTorch (CPU-only) ⚠️ do this FIRST

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-torch.txt
```

**Why separately, and why first?** `requirements-torch.txt` points at PyTorch's CPU-only wheel
index. If you skip it, the next step pulls the default PyTorch build — which includes **~2.5 GB
of CUDA libraries** that are useless on a machine with no NVIDIA GPU.

Downloads ~200 MB. Takes 2–5 minutes.

**Verify:**

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__)"
```

Expect something like `2.14.0+cpu`. **The `+cpu` suffix is what you're checking for.**

---

## Step 5 — Install everything else

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Installs ChromaDB, sentence-transformers, pymupdf4llm, rank-bm25, truststore, pytest and their
dependencies. Downloads ~300 MB. Takes 3–7 minutes.

**Verify:**

```powershell
.\.venv\Scripts\python.exe -c "import chromadb, sentence_transformers, pymupdf4llm; print('ok')"
```

---

## Step 6 — Install the project itself

```powershell
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

Registers `src\groundtruth` as an importable package so `python -m groundtruth.cli` works.

- `-e` = editable: your code edits take effect immediately, no reinstall needed.
- `--no-deps` = don't re-resolve dependencies, you just installed them.

**Skipping this causes:** `No module named 'groundtruth'`

---

## Step 7 — Run the tests

Confirms the install before you touch any data.

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Expect:

```
...................................................................  [100%]
71 passed in 0.19s
```

These use fake tokenizers and stub retrievers — no models downloaded, no network needed.

---

## Step 8 — Add your PDFs

Copy research papers into `data\raw\`:

```powershell
copy "C:\path\to\your\papers\*.pdf" data\raw\
ls data\raw\
```

**Notes:**

- Filenames become document IDs (`Attention Is All You Need.pdf` → `attention_is_all_you_need`).
  Use descriptive names; avoid spaces if you can.
- Target ~20–30 papers for the full project. Start with 2–3 to check the pipeline works.
- Must be **text-based** PDFs, not scans. A scanned image PDF extracts no text and needs OCR
  first (not supported here).
- `data\raw\` is gitignored — your PDFs won't be committed.

> ℹ️ Two arXiv papers (Attention Is All You Need, BERT) may already be there as test fixtures.
> Delete them once you add your own corpus.

---

## Step 9 — Ingest: PDFs → text

```powershell
.\.venv\Scripts\python.exe -m groundtruth.cli ingest
```

```
Ingesting PDFs from data/raw/ ...
  ok    attention_is_all_you_need.pdf -> attention_is_all_you_need (15p, 41,415 chars)
  ok    bert.pdf -> bert (16p, 66,047 chars)

2 documents, 107,462 characters total.
```

**What happened:** each PDF was parsed to Markdown (preserving section headings), had running
headers/footers and its bibliography stripped, and was written to
`data\processed\documents\<doc_id>.json`.

**Fast** (~1 second per paper) and **idempotent** — re-running skips unchanged PDFs. Use
`--force` to re-parse everything.

**Sanity check:** if a paper reports far fewer characters than you'd expect, it may be a scanned
PDF with no extractable text.

---

## Step 10 — Chunk: text → retrievable pieces

```powershell
.\.venv\Scripts\python.exe -m groundtruth.cli chunk --config configs\hybrid.yaml
```

```
Chunking 2 documents (size=512, overlap=64) ...
  attention_is_all_you_need: 35 chunks
  bert: 52 chunks

87 chunks -> data\processed\chunks.jsonl
tokens per chunk: min=80 median=380 max=510
```

**First run downloads the bge-small tokenizer** (~130 MB) from Hugging Face. Subsequent runs are
instant.

**Read the output:** `max` should be **≤ 510**. That's the 512-token model window minus the
`[CLS]`/`[SEP]` special tokens. Anything higher means chunks are being silently truncated by the
embedder.

> ⚠️ **Any config works here** as long as its `chunking` and `embedding` blocks match the others
> — all four configs share those blocks deliberately. `hybrid.yaml` is just convention.

---

## Step 11 — Index: build the search structures

```powershell
.\.venv\Scripts\python.exe -m groundtruth.cli index --config configs\hybrid.yaml
```

```
Indexing 87 chunks ...
  embedding 87 chunks...
  loading embedding model BAAI/bge-small-en-v1.5 (cpu)...
  writing to chroma (...\store\chroma)...
  chroma: 87 vectors, dim=384
  building bm25 over 87 chunks...
  bm25: saved to ...\store\bm25.pkl

Done. index_fingerprint=8145406428fb8256
```

**First run downloads the embedding model** (~130 MB) and takes ~45 s extra.

**Timing:** ~6 chunks/second. 87 chunks ≈ 15 s. A 25-paper corpus (~1,100 chunks) ≈ 3 minutes.

**Builds two indexes:** the ChromaDB vector store and the BM25 lexical index. Both are needed —
the hybrid strategy uses both.

**Note the `index_fingerprint`.** Every experiment config must produce this same value; that's
what proves they're comparing against the same index.

> 🔄 **Re-run `chunk` AND `index`** whenever you add PDFs or change chunking/embedding settings.
> The pipeline refuses to run on a mismatch, so you'll get a clear error rather than bad data.

---

## Step 12 — Ask a question

This is your main debugging tool.

```powershell
.\.venv\Scripts\python.exe -m groundtruth.cli ask "What is multi-head attention?" --config configs\hybrid.yaml --no-generate
```

```
=== retrieved (hybrid, k=10) ===

[1] score=0.0325  attention_is_all_you_need | Attention Is All You Need > 3 Model Architecture > 3.2 Attention | p3-4
    b93fbb22ebbe7951  ### 3.2 Attention An attention function can be described as mapping a query...

[2] score=0.0298  attention_is_all_you_need | ... > 3.2.2 Multi-Head Attention | p4-5
    79e04a6063b67b0c  ### 3.2.2 Multi-Head Attention Instead of performing a single attention...

=== timings (ms) ===
  embed_ms           21.70
  fuse_ms             0.08
  search_ms           5.36
  total_ms           27.14
  note: includes one-off model loading. Pass --warmup for steady-state timings.
```

### Flags

| Flag | Purpose |
|---|---|
| `--no-generate` | Retrieval only. **Use this until Ollama is set up.** |
| `--warmup` | Preload models first, so timings reflect steady state rather than including model loading |
| `--config` | Which strategy to use |

### Try the different strategies

```powershell
# Fast vector search
... ask "What is multi-head attention?" --config configs\dense.yaml --no-generate

# BM25 + vector fused — better on exact terms
... ask "What is multi-head attention?" --config configs\hybrid.yaml --no-generate

# + cross-encoder reranking (~4s, notably better ordering)
... ask "What is multi-head attention?" --config configs\hybrid_rerank.yaml --no-generate
```

**A good test of hybrid:** ask about a rare exact term — an author name, a dataset name, a metric.
Dense retrieval often misses these entirely; hybrid should catch them.

---

## Step 13 — Batch run: produce experiment results

```powershell
.\.venv\Scripts\python.exe -m groundtruth.cli run --config configs\hybrid.yaml --queries data\sample_queries.jsonl --no-generate
```

```
Running 10 queries under config 'hybrid' ...
  warming up models (excluded from timings)...
  [1/10] q0001: What is multi-head attention and why is it used instead of...
  ...

run -> runs\20260923_122141_hybrid
  queries=10 errors=0
  latency mean=24.3ms p50=24.4ms p95=27.3ms
```

Creates `runs\<timestamp>_<config>\`:

| File | Contents |
|---|---|
| `config.yaml` | Exact settings used — snapshot for reproducibility |
| `results.jsonl` | One record per query: question, ranked chunks + scores, answer, timings |
| `manifest.json` | Fingerprints, error count, latency summary (mean/p50/p95/max) |

`results.jsonl` is what the Phase 3 evaluation layer will consume.

### Your own query file

JSONL, one object per line:

```jsonl
{"query_id": "q0001", "question": "What is multi-head attention?"}
{"query_id": "q0002", "question": "How is BERT pre-trained?"}
```

Then `--queries path\to\your_queries.jsonl`.

### Run all experiments for comparison

```powershell
foreach ($c in "dense","hybrid","hybrid_rerank") {
    .\.venv\Scripts\python.exe -m groundtruth.cli run --config "configs\$c.yaml" --queries data\sample_queries.jsonl --no-generate
}
```

> ⏱️ Add `hybrid_rerank_large` only when you have time — it's ~29 s/query (~48 min per 100
> questions).

**Before comparing two runs, check their `index_fingerprint` matches** in `manifest.json`.
Different fingerprints = different indexes = invalid comparison.

---

## Step 14 — Run the smoke test

End-to-end verification against real models and indexes (unlike the unit tests, which use fakes).

```powershell
.\.venv\Scripts\python.exe scripts\smoke.py --config configs\hybrid_rerank.yaml
```

26 checks across documents, chunks, all four retrievers, and determinism. Ends with:

```
============================================================
all checks passed
```

Run this after any change to chunking, indexing or retrieval.

---

## Command reference

| Command | Purpose | Speed |
|---|---|---|
| `ingest` | PDFs → documents | ~1 s/paper |
| `chunk --config X` | Documents → chunks | seconds |
| `index --config X` | Chunks → Chroma + BM25 | ~6 chunks/s |
| `ask "Q" --config X` | One question, interactive | ms to seconds |
| `run --config X --queries F` | Batch → `runs/` | depends on strategy |

Useful flags: `--force` (ingest), `--no-generate`, `--warmup` (ask), `--limit N` (run).

---

## Troubleshooting

### `No module named 'groundtruth'`
You skipped Step 6, or you're using system Python. Run
`.\.venv\Scripts\python.exe -m pip install -e . --no-deps` and use the full `.venv` path.

### `CERTIFICATE_VERIFY_FAILED` when downloading models
Your network does TLS inspection and Python's `certifi` bundle lacks the corporate root CA. The
`truststore` package handles this automatically — confirm it's installed:

```powershell
.\.venv\Scripts\python.exe -c "import truststore; print('ok')"
```

If missing: `.\.venv\Scripts\python.exe -m pip install truststore`

### `No PDFs found in .../data/raw`
Put PDFs in `data\raw\`. Check they're `.pdf` (lowercase extension matters for the glob).

### `No chunks at .../chunks.jsonl. Run 'chunk' first.`
Run steps 9–11 in order: `ingest` → `chunk` → `index`.

### `Index/config mismatch`
```
Index/config mismatch.
  index was built with: ... (fingerprint 8145406428fb8256)
  this config expects:  ... (fingerprint e9f185b6e0a29e2c)
```
**Working as designed** — it caught a stale index. You changed `chunking` or `embedding`.
Re-run `chunk` then `index`.

### PowerShell: "running scripts is disabled on this system"
Only affects `Activate.ps1`. Either run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first, or skip activation and
use the full `.\.venv\Scripts\python.exe` path.

### First command is very slow
Expected. Models download once (~260 MB total) and load in 40–130 s. Subsequent runs are fast.
Use `--warmup` when you care about accurate timings.

### `warning: generation disabled - Ollama not reachable`
Expected if you haven't set up Ollama. Retrieval works fine — use `--no-generate` to silence it.
See [`OLLAMA_SETUP.md`](OLLAMA_SETUP.md).

### A paper produced suspiciously few chunks
Likely a scanned/image PDF with no extractable text. Check:

```powershell
.\.venv\Scripts\python.exe -c "import json; d=json.load(open('data/processed/documents/YOUR_DOC.json', encoding='utf-8')); print(len(d['text'])); print(d['text'][:300])"
```

Near-empty text means you need OCR before this pipeline can use it.

---

## What to do next

1. **Add your real corpus** — 20–30 papers in `data\raw\`, then re-run `ingest` → `chunk` →
   `index`.
2. **Explore retrieval quality** with `ask`. Try questions you know the answers to, and compare
   `dense` against `hybrid` on rare exact terms.
3. **Optionally set up Ollama** for answer generation — [`OLLAMA_SETUP.md`](OLLAMA_SETUP.md).
4. **Move to Phase 2** — building the golden dataset, which is what turns this from a RAG system
   into an evaluation framework.
