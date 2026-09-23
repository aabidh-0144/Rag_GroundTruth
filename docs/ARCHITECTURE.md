# Groundtruth — Architecture & Developer Guide

> **Read this first.** It explains what the system is, how every folder and file fits together,
> and — most importantly — *why* each RAG decision was made the way it was.
>
> Companion documents: [`GETTING_STARTED.md`](GETTING_STARTED.md) (how to run it) and
> [`OLLAMA_SETUP.md`](OLLAMA_SETUP.md) (how to enable answer generation).

---

## Table of contents

1. [What this project is](#1-what-this-project-is)
2. [The big picture](#2-the-big-picture)
3. [Folder-by-folder guide](#3-folder-by-folder-guide)
4. [The RAG pipeline, stage by stage](#4-the-rag-pipeline-stage-by-stage)
   - [4.1 Ingestion](#41-ingestion--pdf--document)
   - [4.2 Chunking](#42-chunking--document--chunks)
   - [4.3 Embedding](#43-embedding--text--vectors)
   - [4.4 Storage](#44-storage--where-the-vectors-live)
   - [4.5 Retrieval](#45-retrieval--the-four-strategies)
   - [4.6 Generation](#46-generation--context--answer)
5. [The seven invariants](#5-the-seven-invariants)
6. [Data formats](#6-data-formats)
7. [Configuration system](#7-configuration-system)
8. [Measured performance](#8-measured-performance)
9. [Testing strategy](#9-testing-strategy)
10. [How to extend it](#10-how-to-extend-it)

---

## 1. What this project is

**Groundtruth** is an automated evaluation and regression-testing framework for RAG systems.

A RAG system can give you a wrong answer for several distinct reasons, and from the outside they
all look identical:

| Failure | Where it happens |
|---|---|
| The right chunk was never retrieved | Retriever |
| The right chunk was retrieved but ranked 47th | Ranking |
| The right chunk was in context, but the LLM hallucinated anyway | Generator |
| A change to chunking/embeddings/prompts silently made things worse | Anywhere |

You cannot fix what you cannot distinguish. Groundtruth exists to separate these automatically
and repeatably, instead of you reading answers one by one and forming an impression.

### Project phases

| Phase | Scope | Status |
|-------|-------|--------|
| **1** | **The RAG system under test** | ✅ **Complete — this document** |
| 2 | Golden dataset (~100 Q/A pairs over ~25 papers) | Not started |
| 3 | Metrics: Recall@K, MRR, nDCG, Faithfulness, Answer Relevance, p95 | Not started |
| 4 | Experiment comparison report | Not started |

**Phase 1 builds the thing that phases 2–4 will measure.** Everything in this document is about
making that measurable later — which is why some choices look stricter than a normal RAG demo.

### A deliberate non-choice: no framework

There is no LangChain and no LlamaIndex here. That is intentional, not laziness.

An evaluation framework whose retrieval internals you cannot inspect is close to useless — when
Recall@10 drops, you need to see the actual ranked list and the actual scores, not a chain
abstraction. The whole system is ~3,000 lines of plain Python, and every retrieval decision is
readable in one file.

---

## 2. The big picture

```
                         data/raw/*.pdf
                               │
                               │  [1] INGEST
                               │      pymupdf4llm → markdown (headers preserved)
                               │      strip headers/footers, references, hyphenation
                               ▼
              data/processed/documents/<doc_id>.json
                    "one long text + page offsets"
                               │
                               │  [2] CHUNK
                               │      split on markdown headers → sections
                               │      then token windows (512 / 64 overlap)
                               ▼
                  data/processed/chunks.jsonl
                    "87 chunks, each with a char span"
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
        [3a] EMBED                     [3b] BM25
     bge-small-en-v1.5                rank_bm25 (Okapi)
        384 dims                       lexical index
                 │                           │
                 ▼                           ▼
          store/chroma/                store/bm25.pkl
        (ChromaDB, cosine)
                 │                           │
                 └─────────────┬─────────────┘
                               │
                               │  [4] RETRIEVE  ← the experiment variable
                               │      dense │ bm25 │ hybrid(RRF) │ hybrid+rerank
                               ▼
                    ranked list of chunks + scores
                               │
                               │  [5] GENERATE
                               │      Ollama, grounded prompt, inline citations
                               ▼
                      runs/<run_id>/
              ├── config.yaml      ← exact settings used
              ├── results.jsonl    ← one record per query  ◄── the eval layer reads THIS
              └── manifest.json    ← fingerprints + latency summary
```

### The one thing to understand

Steps **1–3 happen once**. Steps **4–5 happen per query**.

All four retrieval strategies read from **the same index**. That is what makes the experiment
table meaningful: when you compare Dense vs Hybrid, the *only* thing that differs is the
retrieval strategy. Chunking and embeddings are held constant, so any difference in Recall@10 is
attributable to retrieval and nothing else.

---

## 3. Folder-by-folder guide

```
Rag_GroundTruth/
│
├── .claude/
│   └── CLAUDE.md               Project memory. Context for AI assistants: locked stack,
│                               invariants, hardware limits, environment gotchas.
│                               Read by Claude Code automatically each session.
│
├── configs/                    ⭐ EXPERIMENTS LIVE HERE
│   ├── _base.md                Note explaining the shared-index rule
│   ├── dense.yaml              Experiment A — vector search only
│   ├── hybrid.yaml             Experiment B — BM25 + vector, fused with RRF
│   ├── hybrid_rerank.yaml      Experiment C — hybrid + MiniLM-L6 cross-encoder
│   └── hybrid_rerank_large.yaml Experiment D — hybrid + bge-reranker-base (slow, stronger)
│
├── data/
│   ├── raw/                    ⭐ YOU PUT PDFs HERE            (gitignored)
│   ├── processed/                                              (gitignored)
│   │   ├── documents/*.json    Parsed text + page char offsets
│   │   └── chunks.jsonl        The canonical chunk set
│   └── sample_queries.jsonl    10 example questions for batch runs (tracked)
│
├── store/                      Built indexes                   (gitignored)
│   ├── chroma/                 ChromaDB persistent collection
│   ├── bm25.pkl                Pickled BM25 index
│   └── index_manifest.json     Fingerprint of what was indexed
│
├── runs/                       Experiment outputs              (gitignored)
│   └── 20260923_122236_hybrid_rerank/
│       ├── config.yaml
│       ├── results.jsonl
│       └── manifest.json
│
├── src/groundtruth/            ⭐ THE CODE
│   ├── __init__.py       (25)  Package init + truststore TLS fix
│   ├── schemas.py       (142)  Core data types + chunk ID function
│   ├── config.py        (151)  Config dataclasses, YAML loading, fingerprints
│   ├── ingest.py        (183)  PDF → Document
│   ├── chunking.py      (339)  Document → Chunks  ← most complex file
│   ├── embedder.py       (75)  Text → vectors (bge wrapper)
│   ├── index.py         (167)  Chunks → Chroma + BM25
│   ├── generator.py     (138)  Context → answer (Ollama client + prompt)
│   ├── pipeline.py      (180)  End-to-end orchestration, run artifacts
│   ├── cli.py           (179)  Command-line interface
│   │
│   ├── retrievers/             ⭐ ONE FILE PER STRATEGY
│   │   ├── __init__.py   (51)  Factory: config → retriever
│   │   ├── base.py       (34)  The Retriever protocol + rank()
│   │   ├── dense.py      (64)  Vector search
│   │   ├── bm25.py       (76)  Lexical search
│   │   ├── hybrid.py     (88)  RRF fusion
│   │   └── rerank.py     (76)  Cross-encoder wrapper
│   │
│   └── utils/
│       ├── io.py         (64)  JSON/JSONL, slugify, sha256
│       └── timing.py     (44)  Per-stage latency timer
│
├── tests/                      71 unit tests, all fakes, runs in <1s
│   ├── conftest.py       (52)  Fake tokenizer + sample paper fixture
│   ├── test_chunking.py (179)  ⭐ Includes the char-span invariant test
│   ├── test_retrievers.py(214) RRF maths, ranking, config integrity
│   ├── test_ingest.py   (135)  Boilerplate/reference stripping
│   └── test_generator.py(218)  Prompt building, refusal, retries (Ollama stubbed)
│
├── scripts/
│   └── smoke.py         (167)  End-to-end check against REAL models and indexes
│
├── docs/                       You are here
├── requirements-torch.txt      CPU-only torch (install first!)
├── requirements.txt            Everything else
└── pyproject.toml              Package metadata
```

### Which files matter most?

If you read only three files, read these:

1. **`src/groundtruth/chunking.py`** — the hardest logic, and it owns the invariant everything
   else depends on.
2. **`src/groundtruth/retrievers/hybrid.py`** — the RRF fusion that makes Experiment B work.
3. **`src/groundtruth/pipeline.py`** — how it all comes together and what gets written to disk.

---

## 4. The RAG pipeline, stage by stage

### 4.1 Ingestion — PDF → Document

**File:** `src/groundtruth/ingest.py` · **Library:** `pymupdf4llm`

**What it does:** turns a PDF into one long clean text string, remembering which character
positions belong to which page.

**Why `pymupdf4llm` and not plain PyMuPDF?** Plain text extraction gives you a flat wall of text.
`pymupdf4llm` emits **Markdown**, so section headings survive as `## 3.2 Attention`. Those
headings are what make section-aware chunking possible in the next step. Without them you are
chunking blind.

**Cleanup applied (in order):**

| Step | Why it matters |
|---|---|
| Detect running headers/footers | A line appearing on 60%+ of pages is journal boilerplate. Left in, it pollutes every chunk and inflates BM25 term frequencies. |
| Strip bare page numbers | Same reason, and they add nothing retrievable. |
| De-hyphenate line breaks | `attrac-\ntive` → `attractive`. PDF line wrapping otherwise splits terms that both the tokenizer and BM25 need whole. |
| Truncate at References | Bibliographies have huge term overlap with real content but never contain answers. Guarded: only cuts if it leaves ≥40% of the paper, so a paper *mentioning* "References" early isn't gutted. |
| Collapse blank lines | Cosmetic, keeps chunk text tidy. |

**Output** — `data/processed/documents/<doc_id>.json`:

```json
{
  "doc_id": "attention_is_all_you_need",
  "title": "Attention Is All You Need",
  "source_path": "data/raw/attention_is_all_you_need.pdf",
  "n_pages": 15,
  "text": "<41,415 characters of markdown>",
  "pages": [{"page": 1, "start_char": 0, "end_char": 3021}, ...]
}
```

> 🔑 **`text` is the coordinate system for the entire project.** Chunks address into it by
> character offset. So will your golden dataset labels. Everything downstream is a range of
> characters in this string.

---

### 4.2 Chunking — Document → Chunks

**File:** `src/groundtruth/chunking.py` · **The most important file to understand**

#### The method: two-level, section-aware token windowing

**Level 1 — split on Markdown headers.** Each region becomes a *section* carrying a readable
path:

```
"Attention Is All You Need > 3 Model Architecture > 3.2 Attention"
```

Why: chunk boundaries that respect document structure are better than boundaries that fall
mid-argument. A chunk that starts halfway through §3.1 and ends halfway through §3.2 is about
neither topic. Section paths also make debugging vastly easier — you immediately see *where* a
retrieved chunk came from.

**Level 2 — token windows within each section.**

| Parameter | Value | Reasoning |
|---|---|---|
| `size` | 512 tokens | Matches bge-small's context window exactly. |
| `overlap` | 64 tokens | A fact straddling a boundary survives whole in at least one chunk. |
| `min_tokens` | 80 | Below this, chunks get merged (see below). |

Three details that are easy to get wrong:

**(a) Counted in the embedding model's own tokens.** Not characters, not `tiktoken`. We load
bge-small's tokenizer and count with it. Only this guarantees a chunk actually *fits* the model
without silent truncation.

**(b) Split on sentence boundaries where possible.** The windower accumulates whole sentences
until adding one more would exceed the budget. Chunks end at a full stop, not mid-word.

**(c) A real 512-token budget is 510.** `model_max_length` counts `[CLS]` and `[SEP]`. A chunk of
exactly 512 content tokens gets truncated by two tokens — and a truncated chunk is one whose
stored embedding *does not represent its own text*. That is a silent corruption of every
retrieval metric built on top of it. `Chunker._content_budget()` subtracts the special tokens
automatically.

#### Edge cases handled

**Oversized single units** (a table row, a long equation) can't be sentence-split. They're
hard-split using the tokenizer's **offset mapping**, which gives exact character boundaries per
token.

> ⚠️ This was a real bug caught during the build. The original code estimated 4 characters per
> token. Dense technical text (numbers, symbols, citation markers) tokenizes far below that, so
> the estimate overshot and produced **657-token chunks** — silently truncated by the embedder.
> Offset mapping is exact; the estimate was not.

**Tiny fragments** (a bare heading like `### 3.1 Encoder`) are noise: a 3-token chunk is
trivially "about" its 3 tokens and can outrank real content on short queries. They're merged
in two passes:

1. **Backward** — absorb into the preceding chunk.
2. **Forward** — anything left (a heading has no useful predecessor) merges into the chunk
   *below* it, which is semantically correct: a heading belongs with the text it introduces.

Merging only occurs across **whitespace-only gaps**, so the character-span invariant survives.

#### Output — `data/processed/chunks.jsonl`

```json
{
  "chunk_id": "b93fbb22ebbe7951",
  "doc_id": "attention_is_all_you_need",
  "chunk_index": 12,
  "start_char": 18233,
  "end_char": 20551,
  "page_start": 3,
  "page_end": 4,
  "section_path": "Attention Is All You Need > 3 Model Architecture > 3.2 Attention",
  "n_tokens": 498,
  "text": "### 3.2 Attention\nAn attention function can be described as..."
}
```

#### 🔑 The chunk ID design — the single most important decision

```python
chunk_id = sha1(f"{doc_id}:{start_char}:{end_char}").hexdigest()[:16]
```

The ID is derived from **the character span**, not the text, not a counter.

**Why this matters enormously for Phase 2.** Your golden dataset will label which chunk answers
each question. The naive approach stores `correct_chunk_id`. Then, six weeks later, you want to
run the experiment "does chunk size 256 beat 512?" — you re-chunk, every chunk ID changes, and
**all 100 of your hand-checked labels are now garbage.**

By anchoring labels to `(doc_id, start_char, end_char)` instead, a re-chunking changes which
chunks exist but *not* where the answer lives in the document. You can always recompute which
new chunk overlaps the labelled span.

This costs nothing now and saves the entire dataset later.

---

### 4.3 Embedding — text → vectors

**File:** `src/groundtruth/embedder.py`

#### The model: `BAAI/bge-small-en-v1.5`

| Property | Value |
|---|---|
| Dimensions | 384 |
| Parameters | ~33M |
| Context | 512 tokens |
| Size on disk | ~130 MB |
| Speed here | ~6 chunks/sec on CPU |

**Why this model:**

- **CPU-friendly.** No GPU on this machine. 384 dims and 33M params keep indexing to minutes,
  not hours. (`bge-base` is ~4× slower for a modest quality gain.)
- **Strong on technical/academic English**, which is the corpus.
- **Free and fully offline** after the first download — so experiments stay reproducible with no
  API in the loop.
- **Swappable.** The embedding model is a config value; making it a 4th experiment dimension
  later is trivial.

#### ⚠️ The asymmetry trap

**bge models are asymmetric.** Queries must be prefixed; documents must not:

```python
# Query — prefixed
"Represent this sentence for searching relevant passages: What is attention?"

# Document — raw
"An attention function can be described as mapping a query and..."
```

Forgetting the prefix is a *silent* accuracy loss — everything still runs, results are just
quietly worse. This is such an easy mistake that it is structurally prevented: `Embedder` exposes
only `encode_query()` and `encode_documents()`, and the prefix lives inside `encode_query()`.
**Never call the underlying SentenceTransformer directly.**

Vectors are L2-normalised, so cosine similarity reduces to a dot product.

---

### 4.4 Storage — where the vectors live

**File:** `src/groundtruth/index.py`

Two indexes, built from the same chunk list.

#### Dense index — ChromaDB

```python
chromadb.PersistentClient(path="store/chroma")
collection metadata = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 400,
    "hnsw:search_ef": 400,
    "hnsw:M": 32,
}
```

**Why ChromaDB:** embedded (no server, no Docker), persists to disk, simple API, and it stores
chunk text and metadata alongside the vectors so retrieval returns everything in one call.

**Why `search_ef` is cranked to 400** (default is much lower): HNSW is an *approximate* nearest
neighbour index. At default settings it occasionally misses a true nearest neighbour. For a
normal app, who cares. For **this** app that error would land inside your Recall@10 number — and
you would be measuring the index's approximation error, not your retriever's quality. At ~1,000
chunks, a high `search_ef` makes search effectively exact at negligible cost.

We pass **precomputed embeddings** rather than giving Chroma an embedding function, so the
embedder stays explicit and swappable.

#### Sparse index — rank_bm25

ChromaDB has no lexical search, so BM25 gets its own index: `BM25Okapi` over tokenized chunks,
pickled to `store/bm25.pkl` with the chunk-ID ordering its score array aligns to.

**Tokenisation** — `[a-z0-9]+(?:-[a-z0-9]+)*`, lowercased:

- **Hyphens preserved** — `self-attention` and `bi-encoder` are single terms.
- **No stemming.** Treating "transformer"/"transformers" separately costs little; stemming
  `BLEU` or `GELU` into nonsense costs a lot.

#### Fingerprinting

`store/index_manifest.json` records chunk count, a SHA-256 of `chunks.jsonl`, the embedding model,
and an `index_fingerprint` (a hash of **only** chunking + embedding settings).

At startup, `RagPipeline` compares that fingerprint against the config and **refuses to run** on
a mismatch:

```
Index/config mismatch.
  index was built with: BAAI/bge-small-en-v1.5 (fingerprint 8145406428fb8256)
  this config expects:  BAAI/bge-small-en-v1.5 (fingerprint e9f185b6e0a29e2c)
Chunking or embedding settings changed. Re-run `chunk` and `index`.
```

Silently retrieving from a stale index is the worst possible failure for an evaluation framework:
you get plausible-looking numbers that mean nothing.

---

### 4.5 Retrieval — the four strategies

**Folder:** `src/groundtruth/retrievers/`

Every strategy implements one interface:

```python
class Retriever(Protocol):
    name: str
    def retrieve(self, query: str, k: int) -> tuple[list[RetrievedChunk], dict[str, float]]:
        ...   # (ranked results, {stage}_ms timings)
```

Returning timings alongside results is not incidental — p95 latency is a Phase 3 deliverable, so
every stage is measured from day one.

#### A. `DenseRetriever` — vector search

Encode query (with prefix) → Chroma cosine search → convert distance to similarity
(`score = 1 - distance`) so higher is always better.

**Good at:** paraphrase, conceptual questions, synonyms.
**Bad at:** rare exact terms, proper nouns, identifiers.

#### B. `BM25Retriever` — lexical search

Classic Okapi BM25 term-frequency scoring. Not a headline experiment, but it is the sparse half
of hybrid and a useful baseline row.

**Good at:** exact terms, names, acronyms, numbers.
**Bad at:** paraphrase — it cannot match "avoid recurrence" to "dispensing with recurrence".

#### C. `HybridRetriever` — Reciprocal Rank Fusion

Fetches top-50 from each, then fuses:

```
score(d) = Σ  1 / (k + rank_r(d))          k = 60
         r∈retrievers
```

**Why RRF rather than weighted score blending?** Cosine similarity lives in roughly `[0, 1]`;
BM25 scores are unbounded (we saw 13.0). Blending them requires normalisation *plus* a weight
that must be tuned per corpus — and that weight becomes another confounding variable in your
experiments.

RRF consumes **only ranks**, never raw scores. No tuning, no normalisation, and immune to one
retriever's score distribution shifting. `k=60` is from the original paper; it damps the
dominance of rank-1 enough that agreement between retrievers can outweigh a single strong hit.

**This works — verified on the real corpus.** Query `newstest2013 beam search` (a term appearing
*only* in the Attention paper):

| Strategy | Top 3 results |
|---|---|
| dense | BERT, BERT, BERT ❌ *completely wrong paper* |
| bm25 | Attention §6, Attention §6, Attention §6 ✅ |
| hybrid | Attention §6, Attention §6, BERT ✅ |

And in reverse, on `how does the model avoid recurrence?`:

| Strategy | Top result |
|---|---|
| dense | Attention §1 Introduction ✅ |
| bm25 | BERT ❌ |
| hybrid | Attention §3 Model Architecture ✅ |

Hybrid recovers the correct answer in **both** directions. That is precisely the point.

#### D. `RerankRetriever` — cross-encoder

Wraps any base retriever: take its top-50, re-score every `(query, chunk)` pair with a
cross-encoder, return the new top-k.

**Bi-encoder vs cross-encoder:**

```
Bi-encoder (bge-small)          Cross-encoder (reranker)
  query  → [vector]               [query + chunk] → one model → score
  chunk  → [vector]
  similarity = dot product

  Fast: chunks pre-computed       Slow: one forward pass PER PAIR
  Weaker: no interaction          Stronger: reads both jointly
```

The cross-encoder sees query and passage **together**, so it can judge relevance far better. The
price is one model forward pass per candidate, which on CPU dominates everything else.

> ⚠️ `top_n` (50) is the **recall ceiling**. A reranker can only reorder what the base retriever
> already surfaced. If the right chunk was at rank 60, no reranker will save you.

**Two reranker configs:**

| Config | Model | Params | Latency/query |
|---|---|---|---|
| `hybrid_rerank.yaml` | `cross-encoder/ms-marco-MiniLM-L6-v2` | 22M | **~4.4 s** |
| `hybrid_rerank_large.yaml` | `BAAI/bge-reranker-base` | 278M | **~29 s** |

MiniLM-L6 is the default so the experiment loop stays interactive. The large model is kept as a
separate experiment — "small vs large reranker" is a genuinely interesting extra row rather than
a bottleneck you have to sit through.

#### Determinism

`rank()` sorts by score descending, **breaking ties by `chunk_id`**. Without that, equal-scoring
chunks could swap order between runs and your regression tests would flap. Verified: the same
query twice returns a byte-identical ranking.

---

### 4.6 Generation — context → answer

**File:** `src/groundtruth/generator.py` · **Status:** written and unit-tested, not yet run
against a live model (see [`OLLAMA_SETUP.md`](OLLAMA_SETUP.md))

**Client:** Ollama HTTP `POST /api/chat` at `localhost:11434`.

**Determinism settings** — `temperature=0`, fixed `seed`, explicit `num_ctx=4096`. Reproducibility
is the entire point of a regression framework: same config + same corpus must give the same
answer. `num_ctx` is set explicitly so context overflow becomes a visible config error rather
than a silent truncation that quietly corrupts faithfulness scores.

#### The prompt, and why it is shaped this way

```
Context:
[1] (source: attention_is_all_you_need | section: 3.2 Attention)
An attention function can be described as...

[2] (source: bert | section: 3 BERT)
We introduce BERT and its detailed implementation...

Question: What is multi-head attention?

Answer:
```

With a system prompt enforcing four rules:

1. Use only the context — no prior knowledge.
2. **Cite block numbers inline**, like `[1]` or `[2][3]`.
3. If the answer isn't there, reply exactly `INSUFFICIENT_CONTEXT`.
4. Be concise and factual.

**Why citations?** They give the Phase 3 faithfulness metric something concrete to verify
against, instead of forcing it to re-derive attribution from scratch.

**Why the refusal token?** This is the key design point. It separates two failure modes that
otherwise look identical:

| Model output | Diagnosis |
|---|---|
| `INSUFFICIENT_CONTEXT` | **Retrieval failed** — the right chunk never arrived |
| A confident wrong answer | **Generation failed** — hallucination despite good context |

Without an explicit refusal path, both show up as "wrong answer" and the two questions this
project exists to answer become indistinguishable.

**Reasoning traces are stripped.** Models like qwen3 emit `<think>...</think>` scratchpads. That
is not the answer, and leaving it in would let a faithfulness judge grade the model's rough
working.

Generation is decoupled from retrieval: `context_chunks` (how many chunks enter the prompt,
default 5) is independent of retrieval `k` (default 10).

---

## 5. The seven invariants

These are not style preferences. **Breaking any of them invalidates the evaluation.** They are
also listed in `.claude/CLAUDE.md` so AI assistants respect them.

| # | Invariant | What breaks if violated |
|---|---|---|
| 1 | Chunk IDs are `sha1(doc_id:start:end)[:16]` | IDs stop being reproducible across machines |
| 2 | **Ground truth anchors to char spans, never chunk IDs** | Your 100 labels die on the first re-chunk |
| 3 | All experiments share one index | Dense-vs-Hybrid comparison becomes meaningless |
| 4 | Retrievers return `(list[RetrievedChunk], timings)` | Eval layer can't compute ranking metrics |
| 5 | bge queries get the prefix; documents don't | Silent accuracy loss, no error |
| 6 | Latency measured per stage, models warmed first | p95 becomes model-load time |
| 7 | Runs record a corpus fingerprint | Stale-index runs produce plausible nonsense |

### On invariant 6 — a real bug worth knowing about

Model loading takes **40–130 seconds** on this machine. Originally models loaded lazily, so that
cost landed inside the *first query's* `rerank_ms`. The first measurement was **167,000 ms**
against a true steady state of ~2,800 ms.

Over 100 queries that single outlier would dominate the mean and could single-handedly set the
p95. Every retriever now exposes `warmup()`, and `run_batch()` calls it before the timing loop
starts. Keep it that way.

---

## 6. Data formats

### `chunks.jsonl` — one JSON object per line

Covered in [§4.2](#42-chunking--document--chunks).

### `results.jsonl` — the evaluation seam

**This is the contract with Phase 3.** One record per query:

```json
{
  "query_id": "q0001",
  "question": "What is multi-head attention?",
  "retrieved": [
    {
      "chunk_id": "79e04a6063b67b0c",
      "text": "### 3.2.2 Multi-Head Attention...",
      "score": 6.7713,
      "rank": 1,
      "metadata": {"doc_id": "...", "section_path": "...", "start_char": 18233, ...}
    }
  ],
  "answer": "Multi-head attention runs h=8 parallel attention layers [1][2].",
  "timings": {"embed_ms": 15.85, "search_ms": 5.29, "fuse_ms": 0.09,
              "rerank_ms": 2974.39, "total_ms": 2995.62},
  "config_name": "hybrid_rerank",
  "error": null
}
```

Everything Phase 3 needs is here:

- **Recall@K / MRR / nDCG** ← `retrieved[].chunk_id` + `rank`
- **Faithfulness / Answer Relevance** ← `answer` + `retrieved[].text`
- **Latency / p95** ← `timings`

> ⚠️ `score` is comparable only *within* one retriever. Cosine, BM25, RRF and cross-encoder
> scores are on completely different scales — note the negative cross-encoder scores in real
> output. **Ranking metrics must use `rank`, never `score`.**

### `manifest.json` — run provenance

```json
{
  "run_id": "20260923_122236_hybrid_rerank",
  "config_name": "hybrid_rerank",
  "config_fingerprint": "dc4fb4770d351046",
  "index_fingerprint": "8145406428fb8256",
  "n_queries": 10,
  "n_errors": 0,
  "latency_ms": {"mean": 4391.38, "p50": 4854.2, "p95": 5115.95, "max": 5115.95}
}
```

**Before putting two runs side by side, check `index_fingerprint` matches.** Different
fingerprints mean different indexes and the comparison is invalid. Verified across the three
real runs: all shared `8145406428fb8256` while `config_fingerprint` differed — exactly right.

---

## 7. Configuration system

**Principle: configs drive behaviour; there are no magic numbers in the code.**

An experiment *is* a YAML file, and it gets snapshotted into every run directory — so a run is
self-describing and reproducible months later.

```yaml
name: hybrid_rerank

chunking:                       # ─┐ affects the INDEX
  size: 512                     #  │ must be identical across all
  overlap: 64                   #  │ experiment configs
  min_tokens: 80                #  │
embedding:                      #  │
  model: BAAI/bge-small-en-v1.5 #  │
  batch_size: 32                # ─┘

retrieval:                      # ─┐ the EXPERIMENT VARIABLE
  strategy: hybrid_rerank       #  │ dense | bm25 | hybrid | hybrid_rerank
  k: 10                         #  │
  candidates: {dense: 50, sparse: 50}
  fusion: {method: rrf, k: 60}  #  │
  reranker:                     #  │
    model: cross-encoder/ms-marco-MiniLM-L6-v2
    top_n: 50                   #  │
    max_length: 512             # ─┘

generation:
  provider: ollama
  model: qwen3:4b
  context_chunks: 5
  temperature: 0.0
  seed: 42
  num_ctx: 4096
```

**Two fingerprints, two purposes:**

- `index_fingerprint` — hashes **only** `chunking` + `embedding`. All four configs must produce
  the same value, proving they share one index. (There's a test asserting this.)
- `config_fingerprint` — hashes everything. All four must **differ**, proving they're distinct
  experiments. (Also tested.)

**Unknown keys are errors, not warnings.** A typo like `overlapp: 64` would silently do nothing
and you'd spend an afternoon wondering why your experiment didn't change. `load_config()` raises.

---

## 8. Measured performance

All measured on this machine: **Intel i7-1185G7 (4 cores / 8 threads), 31 GB RAM, no GPU**.
Warm (models preloaded), retrieval only, 50 rerank candidates.

### Latency per query

| Strategy | Typical | p95 (10 queries) | Per 100 queries |
|---|---:|---:|---:|
| dense | ~22 ms | 24.8 ms | ~2 s |
| bm25 | ~3 ms | — | <1 s |
| hybrid (RRF) | ~24 ms | 27.3 ms | ~3 s |
| hybrid + MiniLM-L6 | ~4.4 s | 5,116 ms | **~7 min** |
| hybrid + bge-reranker-base | ~29 s | — | **~48 min** |

**The reranker dominates — by three orders of magnitude.** That is not a flaw to hide; quantifying
this quality-vs-latency tradeoff is one of the things this framework exists to do.

### One-off costs

| Operation | Cost |
|---|---|
| Model load (embedder) | ~43 s first time (then OS-cached) |
| Model load (reranker) | 11–133 s depending on model |
| Embedding throughput | ~6 chunks/sec |
| Ingest + chunk + index, 2 papers (87 chunks) | ~30 s |
| **Projected for 25 papers (~1,100 chunks)** | **~3–4 min** |

> 📌 The original estimate assumed ~10,000 chunks. Reality: 2 papers → 87 chunks, so ~43
> chunks/paper. **25 papers ≈ 1,100 chunks**, an order of magnitude smaller. Indexing is minutes,
> not half an hour.

---

## 9. Testing strategy

Two complementary layers — deliberately split.

### Unit tests — `pytest` · 71 tests · <1 second

Fast and hermetic. A **fake whitespace tokenizer** and **stub retrievers** mean no model
downloads and no index on disk. These test *logic*: span arithmetic, boundary conditions, fusion
maths, prompt construction.

| File | Covers |
|---|---|
| `test_chunking.py` | Char-span round-trip, ID determinism, size limits, merging, sections |
| `test_retrievers.py` | RRF against hand-computed values, ranking, tie-breaks, config integrity |
| `test_ingest.py` | Header detection, reference stripping, offset integrity, slugify |
| `test_generator.py` | Prompt structure, refusal path, retries, `<think>` stripping (Ollama stubbed) |

**The most important single test:**

```python
def test_char_spans_round_trip(paper_text, tokenizer):
    """doc.text[start:end] == chunk.text for every chunk. The core invariant."""
```

If this ever fails, every golden-dataset label built on character spans is silently wrong.

### Smoke test — `scripts/smoke.py` · ~2 minutes

The counterpart that exercises what fakes cannot: real PDF parsing, the real tokenizer, Chroma
persistence, the actual cross-encoder. 26 checks across documents, chunks, all four retrievers,
and determinism. Run it after any change to chunking, indexing or retrieval.

```bash
.venv/Scripts/python.exe scripts/smoke.py --config configs/hybrid_rerank.yaml
.venv/Scripts/python.exe scripts/smoke.py --generate     # also tests Ollama
```

---

## 10. How to extend it

### Add a retrieval strategy

1. Create `src/groundtruth/retrievers/my_strategy.py` implementing the `Retriever` protocol.
2. Add a branch in `build_retriever()` in `retrievers/__init__.py`.
3. Add `configs/my_strategy.yaml` — copying the `chunking` and `embedding` blocks **verbatim**
   from an existing config so it shares the index.

**Never** add a strategy as a branch inside an existing retriever. One file per strategy is what
keeps the experiment matrix clean.

### Change the embedding model

Edit `embedding.model` in **all** configs, then re-run `chunk` (the tokenizer changes, so chunk
boundaries change) and `index`. The fingerprint guard will stop you if you forget.

### Vary chunk size as an experiment

This is the case invariant #2 was designed for. Change `chunking.size` in all configs, re-run
`chunk` and `index`. Chunk IDs will all change — and your golden dataset will still be valid,
because labels point at character spans. Map each label to whichever new chunk overlaps its span.

### Add a generation provider

`build_generator()` in `generator.py` currently accepts only `provider: ollama`. Add a branch
and a client class exposing `generate()` and `health_check()`.

---

## Appendix: complete dependency list

| Package | Role |
|---|---|
| `pymupdf4llm`, `pymupdf` | PDF → markdown |
| `sentence-transformers`, `transformers` | Embeddings + cross-encoder |
| `torch` (CPU wheel) | Backend — **install from `requirements-torch.txt` first** |
| `chromadb` | Vector store |
| `rank-bm25` | Lexical index |
| `truststore` | OS certificate store — required on TLS-inspecting corporate networks |
| `numpy`, `pyyaml`, `requests` | Core |
| `pytest` | Tests |

### Environment gotchas discovered during the build

**1. Corporate TLS inspection breaks Python HTTPS.** Hugging Face downloads failed with
`CERTIFICATE_VERIFY_FAILED` even though `curl` worked fine — `certifi`'s CA bundle lacks the
proxy's root certificate, while curl uses the Windows certificate store. Fixed by `truststore`,
injected at package import in `src/groundtruth/__init__.py`. **Do not remove it.**

**2. GitHub release assets are blocked (HTTP 403).** This is why Ollama could not be installed —
both winget and the `ollama.com` download redirect to `release-assets.githubusercontent.com`.
PyPI, Hugging Face and arXiv are all reachable. See [`OLLAMA_SETUP.md`](OLLAMA_SETUP.md) for
workarounds.
