# Evaluation (Phase 3)

How GroundTruth measures a RAG system: what each metric means, how scores are
combined, and how to run everything from the CLI or the Streamlit app.

> New here? Read **[1. The big picture](#1-the-big-picture)** and
> **[4. Quick start](#4-quick-start)**, then come back to the rest as needed.
> For the RAG system itself, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Contents

1. [The big picture](#1-the-big-picture)
2. [Where the code lives](#2-where-the-code-lives)
3. [Choosing the judge model](#3-choosing-the-judge-model)
4. [Quick start](#4-quick-start)
5. [Mode 1 — Reference-free evaluation](#5-mode-1--reference-free-evaluation)
6. [Mode 2 — Ragas evaluation](#6-mode-2--ragas-evaluation)
7. [Synthetic datasets and human review](#7-synthetic-datasets-and-human-review)
8. [How scores are combined](#8-how-scores-are-combined)
9. [The Streamlit app](#9-the-streamlit-app)
10. [CLI reference](#10-cli-reference)
11. [Input and output formats](#11-input-and-output-formats)
12. [Configuration](#12-configuration)
13. [Testing](#13-testing)
14. [Known limitations](#14-known-limitations)
15. [Adding a metric](#15-adding-a-metric)

---

## 1. The big picture

GroundTruth answers two questions about every RAG answer:

1. **Did the retriever fetch the right information?** → *retrieval* metrics
2. **Is the answer faithful to it and on-topic?** → *generation* metrics

There are two ways to measure that:

| | **Mode 1 — Reference-free** | **Mode 2 — Ragas** |
|---|---|---|
| Needs | question + contexts + answer | the same, plus a **ground truth** for some metrics |
| Judge | our own LLM-judge prompts (readable in `prompts.py`) | the Ragas framework's metrics |
| Good for | quick checks on any run, no dataset needed | standardized metrics, regression tests against a dataset |
| Metrics | faithfulness, answer relevance, context relevance | 3 reference-free + 4 reference-based (see §6) |

Mode 2 also **generates evaluation datasets**: Ragas reads your corpus and writes
questions with reference answers, so you never have to hand-write a golden dataset.
You review them before trusting them (§7).

### How an evaluation runs

```
  samples                     a JSON/JSONL file, a RAG run (runs/<id>), or a dataset
     │                        joined with a run's answers
     ▼
  inputs.py                   validate + normalize into EvalSample objects
     │
     ▼
  orchestrator.py ──► health-check the judge (fails fast, nothing written)
     │
     ├──► Mode 1: reference_free/evaluator.py ─┐
     └──► Mode 2: ragas_mode/evaluator.py ─────┤   one MetricResult per (sample, metric):
                                               │   success | failed | skipped
                                               ▼
                     evals/<id>/results.jsonl  ◄── appended after every sample
                                               │
                                               ▼
                     aggregate.py ──► summary.json (scores, counts, status)
```

Every sample is saved the moment it finishes, so an interrupted run keeps its
work and can be resumed with `--resume <evaluation_id>`.

---

## 2. Where the code lives

Each file has one job. Start with `evaluation/__init__.py`, which is a map.

```
src/groundtruth/
├── llm.py                     LLM transport shared by the generator AND the judges
│                              (Ollama/Groq HTTP, retries, JSON mode, <think> stripping)
├── evaluation/
│   ├── __init__.py            map of the package — start here
│   ├── config.py              loads configs/evaluation.yaml, resolves the judge
│   ├── schemas.py             EvalSample, MetricResult, SampleResult, EvaluationResult
│   ├── inputs.py              JSON / JSONL / run-directory loading + validation
│   ├── prompts.py             every Mode 1 judge prompt, and nothing else
│   ├── judge.py               "ask the judge, get validated JSON back" (+1 repair)
│   ├── aggregate.py           the scoring rules (§8)
│   ├── store.py               evals/<id>/ persistence and history
│   ├── orchestrator.py        runs an evaluation end to end, either mode
│   ├── commands.py            the `groundtruth.cli eval ...` commands
│   ├── reference_free/        MODE 1
│   │   ├── base.py                Metric contract + run_metric (failures → "failed")
│   │   ├── faithfulness.py
│   │   ├── answer_relevance.py
│   │   ├── context_relevance.py
│   │   └── evaluator.py
│   ├── ragas_mode/            MODE 2
│   │   ├── metrics.py             our metric names → Ragas 0.4.4 classes
│   │   ├── adapters.py            Ollama judge + local embeddings in Ragas's format
│   │   └── evaluator.py
│   └── synthetic/             DATASETS
│       ├── generator.py           Ragas TestsetGenerator over our own chunks
│       ├── spans.py               anchor reference contexts to character spans
│       └── datasets.py            datasets/<id>/ storage + verify/reject/edit
└── ui/                        STREAMLIT APP
    ├── app.py                     entry point + navigation
    ├── common.py                  launching CLI jobs, live logs, formatting
    └── views/                     evaluate.py · results.py · datasets.py

configs/evaluation.yaml        all evaluation settings
evals/<evaluation_id>/         results (gitignored, like runs/)
datasets/<dataset_id>/         datasets (NOT gitignored — commit reviewed ones)
```

**Ragas is isolated.** It (and the LangChain it depends on) is imported only inside
`ragas_mode/` and `synthetic/`. Mode 1 and the RAG system never load it.

---

## 3. Choosing the judge model

The judge is the LLM that scores answers. **By default it is the same model you
generate with**, read from `configs/hybrid.yaml`, including your `.env` overrides.
With Ollama, that keeps everything local and free.

The resolution order, last one wins:

1. the generation model of `judge.generation_config` (default `configs/hybrid.yaml`)
2. `judge.provider` / `judge.model` in `configs/evaluation.yaml`
3. `GROUNDTRUTH_JUDGE_PROVIDER` / `GROUNDTRUTH_JUDGE_MODEL` in the environment or `.env`

> ⚠️ **Self-grading bias.** A model judging its own answers tends to rate them
> generously, faithfulness especially. Better: judge with a *different* local model
> family. For example, pull one and set it in `.env`:
>
> ```powershell
> ollama pull llama3.1:8b
> ```
> ```dotenv
> GROUNDTRUTH_JUDGE_MODEL=llama3.1:8b
> ```
>
> Every summary records which judge produced it, so results stay comparable.

> ⚠️ **Inheritance follows `.env`.** If your `.env` switches generation to Groq, an
> inheriting judge uses Groq too. Set `GROUNDTRUTH_JUDGE_PROVIDER=ollama` and a
> judge model to keep judging local regardless.

**Speed on a CPU:** each judge call takes tens of seconds with a 14B model. Mode 1
makes about 3 calls per sample, and Ragas metrics make several. Start with
`--metrics` limited to what you need, and a small sample count.

---

## 4. Quick start

Prerequisites: the RAG pipeline is set up (`ingest` → `chunk` → `index`), Ollama is
running with your model, and you have at least one run:

```powershell
.\.venv\Scripts\python.exe -m groundtruth.cli run --config configs\hybrid.yaml --queries data\sample_queries.jsonl
```

**Evaluate that run (Mode 1):**

```powershell
.\.venv\Scripts\python.exe -m groundtruth.cli eval reference-free --run <run_id>
```

**Or with Ragas (Mode 2):**

```powershell
.\.venv\Scripts\python.exe -m groundtruth.cli eval ragas --run <run_id>
```

**Or open the app** and do it all with clicks:

```powershell
.\.venv\Scripts\python.exe -m groundtruth.cli ui
```

`<run_id>` is a folder name under `runs\`, e.g. `20260923_122236_hybrid_rerank`.

---

## 5. Mode 1 — Reference-free evaluation

Needs only the question, the retrieved contexts and the answer. Every metric
returns a score in **[0, 1]**, a written **reason**, and **details** you can inspect.

### Faithfulness — is every claim backed by the context?

The judge splits the answer into atomic claims and labels each one
`supported`, `unsupported` or `contradicted` against the numbered context blocks.

    score = supported claims / total claims

`details.unsupported_claims` lists the hallucination candidates. A claim that is
true in the real world but absent from the context counts as *unsupported*: the
metric checks grounding, not general truth.

### Answer relevance — does the answer address the question?

The judge rates the answer on a 1–5 rubric and lists missing or irrelevant parts.

    score = (rating − 1) / 4        1 → 0.0 · 3 → 0.5 · 5 → 1.0

This metric checks relevance, *not* correctness, so read it together with faithfulness.

### Context relevance — did the retriever fetch the right chunks?

The judge gives each retrieved block one verdict:
`relevant` (1.0) · `partial` (0.5) · `irrelevant` (0.0).

    score = mean verdict over all blocks

`details.blocks` maps each verdict to its `chunk_id`, so you can see exactly which
retrieved chunks were noise. The judge must rate every block exactly once, and a
reply that skips one is rejected and repaired (an incomplete verdict would average
over the wrong denominator).

### When a metric is skipped instead of scored

A skipped metric means "not applicable". It is **never counted as 0**:

| Situation | faithfulness | answer relevance | context relevance |
|---|---|---|---|
| Answer is `INSUFFICIENT_CONTEXT` (refusal) | skipped | skipped | scored |
| Empty answer | skipped | skipped | scored |
| No contexts retrieved | skipped | scored | skipped |
| Judge finds no factual claims | skipped | — | — |

Why skip a refusal? When retrieval failed, refusing is the *correct* generator
behavior, and scoring it as a bad answer would blame the generator for the
retriever's miss. Context relevance still scores that sample, which is where the
miss shows up.

### When the judge misbehaves

Local models often wrap JSON in prose or code fences, drop fields, or invent
out-of-range values. `judge.py` extracts the JSON, validates it, and on failure
sends **one** repair request that repeats the original task. If that also fails,
the metric is `failed` with the validation error: no default value, no guess.

---

## 6. Mode 2 — Ragas evaluation

Uses the [Ragas](https://docs.ragas.io) framework (pinned to **0.4.x**; code verified
against 0.4.4, the `ragas.metrics.collections` API).

| Our name | Ragas class | Needs ground truth | Measures |
|---|---|:-:|---|
| `faithfulness` | `Faithfulness` | — | share of the answer's claims supported by the contexts |
| `response_relevancy` | `AnswerRelevancy` | — | how well the answer addresses the question |
| `context_precision` | `ContextPrecisionWithoutReference` | — | useful contexts ranked above useless ones (judged vs the answer) |
| `context_precision_with_reference` | `ContextPrecisionWithReference` | ✅ | same, judged vs the ground truth |
| `context_recall` | `ContextRecall` | ✅ | share of the ground truth that the contexts support |
| `answer_correctness` | `AnswerCorrectness` | ✅ | factual + semantic agreement with the ground truth |
| `answer_similarity` | `SemanticSimilarity` | ✅ | embedding similarity to the ground truth (no LLM) |

Our names stay stable even where Ragas renamed classes (`ResponseRelevancy` →
`AnswerRelevancy`, `AnswerSimilarity` → `SemanticSimilarity`). A test checks every
row of this table against the installed Ragas, so an upgrade that changes the API
fails in `pytest` instead of mid-evaluation.

**Choose metrics** with `--metrics`, e.g.
`--metrics faithfulness,context_recall,answer_correctness`.

**Ground truth is never invented.** A reference-based metric on a sample without a
ground truth is `skipped` with the reason `no ground_truth`. To get ground truths,
use a dataset (§7).

**Refusals:** metrics that judge the answer itself (faithfulness, response
relevancy, context precision) skip refusals, as in Mode 1. Reference-based metrics
**do** score them: if a ground truth exists, the question was answerable, so a
refusal is a genuine miss.

**Ragas failures are visible:** Ragas signals internal failures with `NaN`, which
is recorded as `failed`, never as a score. Each (sample, metric) has its own
timeout, and one failing metric never voids the others.

**Behind the scenes:** Ragas talks to Ollama through its OpenAI-compatible endpoint
(`http://localhost:11434/v1`). Embeddings use the same local `BAAI/bge-small-en-v1.5`
model as the index, so nothing new is downloaded. Ragas's usage analytics (on by
default in Ragas) are turned off here; set `RAGAS_DO_NOT_TRACK=false` in `.env` to
opt back in.

---

## 7. Synthetic datasets and human review

### Generating

```powershell
.\.venv\Scripts\python.exe -m groundtruth.cli eval generate-dataset --size 5
```

What happens:

1. Ragas receives **your own chunks** from `data/processed/chunks.jsonl`, so it never
   re-chunks the corpus.
2. It builds a knowledge graph (summaries, entities, themes, relationships between
   chunks). **This step makes many LLM calls and is the slow part.**
3. It writes questions of three types, weighted by `--distribution`:

   | Type | Ragas synthesizer | Example |
   |---|---|---|
   | `factual` | single-hop specific | "How many attention heads does the base Transformer use?" |
   | `reasoning` | multi-hop abstract | "Why does multi-head attention help compared to a single head?" |
   | `multihop` | multi-hop specific | a question needing facts from two related chunks |

   A type the corpus can't support (multi-hop needs related chunks, and a small
   corpus may have none) is **dropped and recorded** in the manifest, not hidden.
4. Each sample gets a reference answer (`ground_truth`) and reference contexts.
5. Each reference context is **anchored to a character span** `(doc_id, start_char,
   end_char)`. Because Ragas was given our chunks, these are almost always exact
   matches. This preserves project invariant #2: labels survive a re-chunking
   experiment.

### Reviewing — synthetic is not ground truth until a human says so

Every generated sample starts as:

```
origin: "synthetic"            verification_status: "unverified"
```

Review it in the app's **Datasets** page, or in the terminal:

```powershell
.\.venv\Scripts\python.exe -m groundtruth.cli eval review-dataset <dataset_id>
```

| Action | Effect |
|---|---|
| **verify** | status → `verified` (requires a ground truth) |
| **reject** | status → `rejected`; rejected samples are **never** evaluated |
| **edit** | change question and/or ground truth; origin → `synthetic_edited`, the original text is kept in `original`, and status resets to `unverified` until you verify again |
| **unverify** | back to `unverified` |

Nothing is ever verified automatically.

### Evaluating against a dataset

```powershell
# answer the dataset's questions with your RAG config, then score with Ragas
.\.venv\Scripts\python.exe -m groundtruth.cli eval ragas --dataset <dataset_id> --generate-answers --config configs\hybrid.yaml --metrics context_recall,answer_correctness,faithfulness

# only human-verified samples
... --verified-only
```

Every evaluated sample keeps its `verification_status` and `sample_origin` in its
`origin` field, and the summary lists which statuses were included, so results on
synthetic data and on verified data can always be told apart.

**Regression testing:** `datasets/` is not gitignored. Commit a reviewed dataset,
re-run the same evaluation after changing chunking, retrieval or prompts, and
compare in the app.

---

## 8. How scores are combined

All rules live in one file, `evaluation/aggregate.py`. The principles:

1. **A missing score is never 0.** Failed and skipped results are excluded from
   averages and counted separately (`n_failed`, `n_skipped`).
2. **Skipped ≠ failed.** *Skipped* means "doesn't apply" (no penalty). *Failed* means
   "should have been scored but couldn't", which marks the run `partial`.
3. **Only like is averaged with like.** Metrics are grouped by the question they
   answer:

   | Group | Mode 1 | Mode 2 |
   |---|---|---|
   | **retrieval** | context_relevance | context_precision, context_precision_with_reference, context_recall |
   | **generation** | faithfulness, answer_relevance | faithfulness, response_relevancy, answer_correctness, answer_similarity |

| Number | Rule |
|---|---|
| metric mean | mean of that metric's **successful** scores; `null` if none |
| sample group score | mean of the sample's successful metrics in the group; `null` if any metric in the group **failed** on that sample |
| run group score | mean of the group's metric means |
| **overall_score** | mean of the two run group scores; **`null`** if any requested metric applied but never succeeded |
| **status** | `failed`: nothing succeeded · `partial`: some results failed or samples are missing · `completed`: otherwise |

Example: faithfulness 0.9, answer relevance 0.8, context relevance 0.5 →
generation 0.85, retrieval 0.5, overall 0.675. That 0.5 retrieval score tells you
where to look, while a single blind average would have hidden it.

---

## 9. The Streamlit app

```powershell
.\.venv\Scripts\python.exe -m groundtruth.cli ui
```

Opens in your browser at `http://localhost:8501`.

| Page | What you can do |
|---|---|
| **Evaluate** | pick Mode 1 or 2, a source (a run, a dataset, or pasted JSON) and metrics, then start. The job runs in the background with a live log, and you can keep using the app. |
| **Results** | history table; inspect one evaluation (group scores, per-metric counts, every sample with color-coded success/failed/skipped and the judge's reasoning); compare up to 4 evaluations side by side, with a warning if they came from different indexes |
| **Datasets** | generate a dataset (size + question-type sliders); review samples one by one with verify / reject / reset / edit |

The app never scores anything itself. It launches `groundtruth.cli` as a background
process and reads what that writes, so closing the browser tab doesn't kill a
running evaluation, and the CLI and the app always produce identical results.

---

## 10. CLI reference

All commands: `.\.venv\Scripts\python.exe -m groundtruth.cli eval <command>`

| Command | Purpose |
|---|---|
| `reference-free` | Mode 1 |
| `ragas` | Mode 2 |
| `generate-dataset` | synthetic dataset from the indexed corpus |
| `review-dataset <id>` | interactive review; or one sample: `--sample ID --action verify\|reject\|unverify\|edit [--question ...] [--ground-truth ...] [--note ...]` |
| `datasets` | list datasets with review counts |
| `results <id> [--samples]` | one evaluation's summary, optionally every sample |
| `history [--limit N] [--mode reference_free\|ragas]` | previous evaluations, newest first |

Plus `groundtruth.cli ui` for the app.

**Options for `reference-free` and `ragas`** (choose exactly one source):

| Option | Meaning |
|---|---|
| `--input FILE` | `.json` (object or list) or `.jsonl` samples |
| `--run DIR` | a run directory, or just its id |
| `--dataset ID` | a dataset's ground truth, plus answers from `--run DIR` or `--generate-answers --config CFG` |
| `--verified-only` | with `--dataset`: only human-verified samples |
| `--metrics a,b` | which metrics (default: `configs/evaluation.yaml`) |
| `--resume ID` | continue an interrupted evaluation |
| `--eval-config FILE` | another evaluation config |

### Example output

```
=== evaluation 20261121_141502_reference_free (reference_free) - completed ===
samples evaluated: 10/10   judge: ollama:qwen3:14b   time: 412.6s

  metric                                mean   ok  fail  skip
  faithfulness                         0.920    9     0     1
  answer_relevance                     0.860    9     0     1
  context_relevance                    0.640   10     0     0

  retrieval score:  0.640
  generation score: 0.890
  overall score:    0.765

saved to evals\20261121_141502_reference_free
```

(Illustrative numbers. The skip of 1 is the deliberately unanswerable "capital of
Bolivia" sample in `data/sample_queries.jsonl`.)

---

## 11. Input and output formats

### Input sample (`--input`)

```json
{
  "question": "What is Retrieval-Augmented Generation?",
  "contexts": ["RAG combines information retrieval with language generation."],
  "answer": "RAG retrieves relevant information and uses an LLM to generate answers.",
  "ground_truth": "RAG combines information retrieval with language generation."
}
```

`sample_id`, `context_ids` and `ground_truth` are optional (`reference` is accepted
instead of `ground_truth`). Invalid input is rejected before anything runs, with
the sample number and field, e.g. `invalid sample #3: question: Field required`.

### `evals/<id>/summary.json`

```json
{
  "evaluation_id": "20261121_141502_reference_free",
  "mode": "reference_free",
  "status": "completed",
  "metrics_requested": ["faithfulness", "answer_relevance", "context_relevance"],
  "aggregates": {
    "faithfulness": {"mean": 0.92, "n_success": 9, "n_failed": 0, "n_skipped": 1}
  },
  "retrieval_score": 0.64,
  "generation_score": 0.89,
  "overall_score": 0.765,
  "overall_score_method": "mean of the retrieval and generation group scores; ...",
  "n_samples": 10,
  "n_samples_evaluated": 10,
  "source": {"type": "run", "run_id": "20260923_122236_hybrid_rerank", "index_fingerprint": "8145406428fb8256"},
  "judge": {"provider": "ollama", "model": "qwen3:14b"},
  "execution_time_s": 412.6,
  "errors": []
}
```

### One line of `evals/<id>/results.jsonl`

```json
{
  "sample_id": "q0001",
  "question": "What is multi-head attention?",
  "metrics": {
    "faithfulness": {
      "name": "faithfulness", "status": "success", "score": 0.75,
      "reason": "3/4 claims supported by the context.",
      "details": {"claims": ["..."], "unsupported_claims": [{"claim": "...", "verdict": "unsupported"}]},
      "error": null, "elapsed_ms": 18342.1
    }
  },
  "retrieval_score": 1.0, "generation_score": 0.875, "refusal": false,
  "origin": {"run": "20260923_122236_hybrid_rerank"}
}
```

### `datasets/<id>/samples.jsonl`

```json
{
  "sample_id": "20261121_150000_synthetic_001",
  "question": "How many attention heads does the base Transformer use?",
  "ground_truth": "It uses h = 8 parallel attention heads.",
  "reference_contexts": ["In this work we employ h = 8 parallel attention layers, or heads. ..."],
  "reference_spans": [{"doc_id": "attention_is_all_you_need", "start_char": 18233, "end_char": 20551, "match": "chunk"}],
  "question_type": "factual",
  "origin": "synthetic",
  "verification_status": "unverified",
  "generator_model": "ollama:qwen3:14b"
}
```

---

## 12. Configuration

`configs/evaluation.yaml` — unknown keys are errors, as with experiment configs.

| Key | Default | Meaning |
|---|---|---|
| `judge.inherit_from_generation` | `true` | judge with the generation model (§3) |
| `judge.generation_config` | `configs/hybrid.yaml` | where that model is read from |
| `judge.provider` / `judge.model` | `null` | set to judge with a different model |
| `judge.num_ctx` / `num_predict` | `8192` / `2048` | room for contexts + a JSON verdict |
| `judge.timeout_s` | `300` | per call; local CPU judging is slow |
| `concurrency` | `1` | samples in parallel — keep 1 for one local model |
| `reference_free.metrics` | all three | Mode 1 default metrics |
| `ragas.metrics` | the three reference-free | Mode 2 default metrics |
| `ragas.embedding_model` | `BAAI/bge-small-en-v1.5` | local embeddings for Ragas |
| `synthetic.size` | `5` | default dataset size |
| `synthetic.distribution` | 0.5 / 0.25 / 0.25 | factual / reasoning / multihop |

Environment: `GROUNDTRUTH_JUDGE_PROVIDER`, `GROUNDTRUTH_JUDGE_MODEL`,
`RAGAS_DO_NOT_TRACK` (defaults to `true` here).

---

## 13. Testing

```powershell
.\.venv\Scripts\python.exe -m pytest            # fast: no network, no LLM (~30s)
```

Every judge call is faked (`tests/eval_fakes.py`). Coverage: each Mode 1 metric
and its skip rules, malformed JSON and the repair turn, LLM failures, partial
failures, the Ragas registry against the installed Ragas, ground-truth skipping,
NaN and timeouts, span anchoring, the review workflow, dataset generation (Ragas
mocked at the boundary), persistence, history, resume, the CLI end to end, and
every Streamlit page.

**Live tests** use real local models and a real judge, and are opt-in:

```powershell
$env:GROUNDTRUTH_LIVE_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest -m live -v
```

They check that Mode 1 rates a grounded answer above a planted hallucination,
that Ragas faithfulness returns a real score, and that a real Ragas metric runs on
local embeddings. They skip on their own if the judge is unreachable.

---

## 14. Known limitations

- **Self-grading bias** when the judge and the generator are the same model (§3).
- **Slow on CPU.** Ragas dataset generation can take a long time even for 5
  samples: the knowledge-graph step makes many calls before the first question.
- **Small corpora limit multi-hop questions**: those types may be dropped (and
  recorded as dropped).
- **Determinism:** temperature is 0, but LLM judges are still not perfectly
  repeatable, so compare small score differences with care.
- **Ragas + reasoning models:** Mode 1 disables qwen3's `<think>` mode explicitly.
  Ragas calls go through Ollama's OpenAI-compatible endpoint, where that switch
  isn't sent. If a reasoning model makes Ragas metrics fail, judge Mode 2 with a
  non-reasoning model (`GROUNDTRUTH_JUDGE_MODEL`).
- `rapidfuzz` is required for dataset generation but not declared by Ragas itself.
  It's in `requirements.txt` for that reason.
- No HTTP API. The CLI and the app share one service layer, and an HTTP layer could
  sit on top of `orchestrator.run_evaluation` later.

---

## 15. Adding a metric

**Mode 1:** add `reference_free/my_metric.py` with a class that has `name` and
`evaluate(sample, judge) -> MetricResult`, put its prompt in `prompts.py`, register
it in `reference_free/__init__.py`, and add its name to a group in `aggregate.py`.
Return `MetricResult.skipped(...)` when it doesn't apply, and raise on failure
(`run_metric` records it as `failed`).

**Mode 2:** add one `RagasMetricSpec` to `ragas_mode/metrics.py` (class name, `ascore`
arguments, whether it needs a reference) and add its name to a group in
`aggregate.py`. The registry test then checks it against the installed Ragas.
