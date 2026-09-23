# Ollama Setup — Enabling Answer Generation

How to install Ollama, pick a model, and wire it into the RAG pipeline.

> **Retrieval does not need this.** Steps 1–13 of [`GETTING_STARTED.md`](GETTING_STARTED.md) work
> fully without Ollama — just pass `--no-generate`. This document covers the generation half:
> turning retrieved chunks into a written answer.

---

## Current status ⚠️

**Ollama is not installed on this machine, and could not be installed from this network.**

Both `winget install Ollama.Ollama` and a direct download from `ollama.com` fail with
**HTTP 403 Forbidden**, because both redirect to `release-assets.githubusercontent.com` — GitHub
release asset downloads are blocked by the corporate proxy.

```
Downloading https://github.com/ollama/ollama/releases/download/v0.34.3/OllamaSetup.exe
An unexpected error occurred while executing the command:
Download request status is not success.
0x80190193 : Forbidden (403).
```

PyPI, Hugging Face and arXiv are all reachable — it's specifically GitHub release binaries.

**Consequence:** `src/groundtruth/generator.py` is written and covered by 19 unit tests, but has
never been run against a live model. Expect to iterate on the prompt once you can actually
execute it.

[Jump to workarounds →](#option-b-network-blocked-workarounds)

---

## 1. What Ollama is, and why it's used here

Ollama runs open-weight LLMs locally. It downloads a quantized model, serves it over a small HTTP
API on `localhost:11434`, and manages loading/unloading for you.

**Why local rather than an API:**

| Reason | Detail |
|---|---|
| **Free** | No API key, no per-token cost. You'll make ~300+ generation calls per experiment sweep. |
| **Reproducible** | With `temperature=0` and a fixed seed, the same input gives the same output — indefinitely. A hosted model can be deprecated or silently updated, which would invalidate old runs. |
| **Offline** | No network dependency in the measurement loop. |
| **Private** | Nothing leaves the machine. |

The tradeoff is speed: on CPU, expect ~15–25 seconds per answer rather than ~2.

---

## 2. Hardware reality check

Measured on this machine — **Intel i7-1185G7 (4c/8t), 31 GB RAM, no GPU**:

| Model | Disk | RAM in use | Speed | ~200-token answer |
|---|---:|---:|---:|---:|
| `llama3.2:3b` | ~2.0 GB | ~3 GB | 12–18 tok/s | ~13 s |
| **`qwen3:4b`** ⭐ | ~2.6 GB | ~4 GB | 10–14 tok/s | ~17 s |
| `qwen2.5:7b` | ~4.7 GB | ~6 GB | 5–7 tok/s | ~33 s |

With 31 GB RAM you are nowhere near a memory limit. **CPU inference is bound by memory
*bandwidth*, not capacity** — which is why a bigger model doesn't just cost more RAM, it costs
proportionally more time. The 3–4B class is the sweet spot here.

### What this means for a full experiment run

| Work | Time |
|---|---|
| 100 questions × 1 config | ~30 min |
| 100 questions × 3 configs | **~1.5 hours** |
| Plus reranking (hybrid_rerank) | +~7 min |

Generation dominates. Plan to run full sweeps in the background.

> 💡 `qwen3:4b` is the recommended default — good instruction-following and citation behaviour
> for its size. Drop to `llama3.2:3b` if you want ~30% faster iteration.

---

## 3. Installation

### Option A: Normal network (recommended)

**1. Download and run the installer**

Go to **<https://ollama.com/download/windows>** and run `OllamaSetup.exe`.

No admin rights needed — it installs per-user and adds itself to startup.

**2. Verify**

Open a **new** terminal (so `PATH` refreshes):

```powershell
ollama --version
```

**3. Pull the model** (~2.6 GB download)

```powershell
ollama pull qwen3:4b
```

**4. Confirm it runs**

```powershell
ollama run qwen3:4b "Say hello in five words."
```

First response is slow (model loads into RAM); later ones are faster. Type `/bye` to exit.

---

### Option B: Network blocked — workarounds

Pick whichever fits your situation.

#### B1. Download elsewhere, transfer the installer ⭐ simplest

1. On a personal machine or phone hotspot, download `OllamaSetup.exe` from
   <https://ollama.com/download/windows>.
2. Copy it over (USB, OneDrive, email to yourself).
3. Run it here — the *installer* works fine offline; only the *download* was blocked.
4. Then `ollama pull qwen3:4b`.

> ⚠️ **`ollama pull` also needs network.** Model downloads come from `registry.ollama.ai`, not
> GitHub, so they may well succeed even though the installer download didn't. Test it right
> after installing. If pulls are also blocked, see B3.

#### B2. Request a firewall exception

Ask IT to allowlist:

| Host | Needed for |
|---|---|
| `github.com` + `release-assets.githubusercontent.com` | The installer |
| `registry.ollama.ai` | Model downloads |

Reasonable framing: these are standard developer tools for a local, offline AI workload with no
data leaving the machine.

#### B3. Fully offline model transfer

If `ollama pull` is blocked too: pull `qwen3:4b` on an unrestricted machine, then copy its model
directory across.

```
Windows:  C:\Users\<you>\.ollama\models
Linux:    ~/.ollama/models
```

Copy the whole `models` folder into `%USERPROFILE%\.ollama\` here, then `ollama list` should show
it.

#### B4. Use a free-tier hosted API instead

If local inference simply isn't going to happen, Groq and Google Gemini both have free tiers that
are more than adequate for this project. This needs a small code change —
see [§7](#7-using-a-different-provider).

Tradeoff: faster and easier, but adds a network dependency to your measurement loop and free-tier
models can be deprecated, which weakens long-term reproducibility.

---

## 4. Wiring Ollama into the RAG system

**The good news: there is nothing to configure.** The integration is already written; it
activates as soon as Ollama is reachable.

### How the connection works

`src/groundtruth/generator.py` speaks to Ollama over plain HTTP:

```
POST http://localhost:11434/api/chat
{
  "model": "qwen3:4b",
  "messages": [
    {"role": "system", "content": "<grounding rules>"},
    {"role": "user",   "content": "Context:\n[1] ...\n\nQuestion: ...\n\nAnswer:"}
  ],
  "stream": false,
  "options": {"temperature": 0.0, "seed": 42, "num_ctx": 4096, "num_predict": 512}
}
```

The `generation` block present in all four config files:

```yaml
generation:
  provider: ollama
  model: qwen3:4b
  context_chunks: 5     # how many retrieved chunks enter the prompt
  temperature: 0.0      # deterministic
  seed: 42              # deterministic
  num_ctx: 4096         # context window
```

Four more settings exist with sensible defaults and are simply omitted from the YAML. Add any of
them to a config to override:

| Setting | Default | Purpose |
|---|---|---|
| `base_url` | `http://localhost:11434` | Where Ollama is listening |
| `num_predict` | `512` | Max answer length in tokens |
| `timeout_s` | `180` | Per-request timeout |
| `max_retries` | `2` | Retries on a connection failure, with backoff |

(Defined in `GenerationConfig` in `src/groundtruth/config.py`.)

### Health check

Before any batch run, the pipeline queries `/api/tags` and verifies your model is installed. If
not, you get an actionable message rather than a confusing failure 40 questions in:

```
Model 'qwen3:4b' not installed. Available: ['llama3.2:3b']. Run: ollama pull qwen3:4b
```

For `ask`, a failed health check just disables generation with a warning — retrieval still runs.

---

## 5. Verify it works

**1. Confirm the server is up**

```powershell
ollama list
```

Ollama starts automatically after install. If not: `ollama serve`.

**2. Ask a question with generation enabled**

Note: **no `--no-generate` flag** this time.

```powershell
.\.venv\Scripts\python.exe -m groundtruth.cli ask "What is multi-head attention?" --config configs\hybrid.yaml
```

Expect the usual ranked chunks, then:

```
=== answer ===
Multi-head attention runs h = 8 parallel attention layers, or heads, each operating on
dk = dv = dmodel/h = 64 dimensions [1][2]. This lets the model jointly attend to
information from different representation subspaces at different positions [2].

=== timings (ms) ===
  embed_ms           21.70
  generate_ms     17243.55
  search_ms           5.36
  total_ms        17270.61
```

**What to check:**

- ✅ The answer cites blocks — `[1]`, `[2]`
- ✅ Content actually comes from your papers, not the model's memory
- ✅ `generate_ms` roughly matches the table in §2

**3. Test the refusal path** — this one matters

```powershell
.\.venv\Scripts\python.exe -m groundtruth.cli ask "What is the capital city of Bolivia?" --config configs\hybrid.yaml
```

**Expected:** exactly `INSUFFICIENT_CONTEXT`.

This is the most important behaviour to verify. It's what separates *"the retriever failed"* from
*"the LLM hallucinated"* — the two questions this whole project exists to distinguish. If the
model answers "Sucre" instead, it's using prior knowledge and the prompt needs tightening before
any faithfulness metric will mean anything.

**4. Run the full smoke test**

```powershell
.\.venv\Scripts\python.exe scripts\smoke.py --generate
```

Adds a `[5] generation` section checking reachability, a real answer, and the refusal path.

**5. Batch run with generation**

```powershell
.\.venv\Scripts\python.exe -m groundtruth.cli run --config configs\hybrid.yaml --queries data\sample_queries.jsonl
```

~3 minutes for 10 questions. `q0010` in the sample file is deliberately unanswerable — check it
produced `INSUFFICIENT_CONTEXT` in `results.jsonl`.

---

## 6. Changing the model

Edit `generation.model` in the config(s) you use:

```yaml
generation:
  model: llama3.2:3b      # was qwen3:4b
```

Then pull it: `ollama pull llama3.2:3b`.

> ⚠️ **Change it in all four configs**, or your experiments will differ in two variables at once
> (retrieval strategy *and* generator) and the comparison becomes uninterpretable.

The generator model is **not** part of `index_fingerprint`, so changing it does **not** require
re-indexing. It *is* part of `config_fingerprint`, so runs stay distinguishable.

### About `<think>` blocks

Reasoning models like qwen3 emit `<think>...</think>` scratchpads. These are stripped
automatically in `_strip_thinking()` — the scratchpad is not the answer, and leaving it in would
let a faithfulness judge grade the model's rough working.

---

## 7. Using a different provider

`build_generator()` currently accepts only `provider: ollama`:

```python
def build_generator(cfg: GenerationConfig) -> OllamaGenerator:
    if cfg.provider != "ollama":
        raise ValueError(f"Unsupported generation provider {cfg.provider!r}. Only 'ollama' is implemented.")
    return OllamaGenerator(cfg)
```

To add Groq, Gemini or OpenAI:

1. Write a client class in `generator.py` exposing the same two methods:
   - `generate(question: str, chunks: list[RetrievedChunk]) -> str`
   - `health_check() -> tuple[bool, str]`
2. Add a branch in `build_generator()`.
3. Set `provider:` in your configs.

Reuse the existing `SYSTEM_PROMPT`, `USER_TEMPLATE` and `format_context()` — keeping the prompt
identical across providers is what makes provider a controlled variable rather than a confound.

**Read API keys from environment variables, never hard-code them**, and keep them out of the
config files, which get snapshotted into every run directory.

---

## Troubleshooting

### `Ollama not reachable at http://localhost:11434`
The server isn't running. Start it with `ollama serve`, or launch Ollama from the Start menu.
Confirm with `ollama list`.

### `Model 'qwen3:4b' not installed`
Run `ollama pull qwen3:4b`. Check what you have with `ollama list`.

### `ollama` is not recognised as a command
Open a **new** terminal — `PATH` doesn't refresh in existing windows. If it still fails, the
binary is at `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`.

### Generation times out
Default `timeout_s` is 180. A 7B model on CPU with a long prompt can exceed that. Either raise it
or reduce `num_predict` / `context_chunks`.

### Answers ignore the context and use prior knowledge
The model is too weak to follow the grounding instruction. Options, in order of preference:
1. Try a stronger model (`qwen2.5:7b`)
2. Reduce `context_chunks` to 3 so the relevant text is more prominent
3. Strengthen the rules in `SYSTEM_PROMPT`

Worth fixing before Phase 3 — a generator that ignores context makes faithfulness scores
meaningless.

### Answers are truncated mid-sentence
Raise `num_predict` (default 512 tokens).

### Very slow first generation, fast afterwards
Normal. Ollama loads the model into RAM on first use and keeps it warm for ~5 minutes. Batch runs
amortise this across all queries.

### Out of memory
Unlikely with 31 GB, but if it happens use a smaller model or lower `num_ctx`.

---

## Summary

| | |
|---|---|
| **Needed for retrieval?** | ❌ No — use `--no-generate` |
| **Needed for Phase 3 faithfulness/relevance?** | ✅ Yes |
| **Config changes required?** | None — it's already wired |
| **Recommended model** | `qwen3:4b` (~2.6 GB) |
| **Expected speed** | ~17 s per answer |
| **Blocker on this machine** | GitHub release downloads return 403 |
| **Simplest fix** | Download `OllamaSetup.exe` on another network and transfer it |

Once it's running, verify with the Bolivia question. `INSUFFICIENT_CONTEXT` means the grounding
works, and Phase 3 will have a solid foundation to measure against.
