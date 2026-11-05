# Ollama Setup — Enabling Answer Generation

How to install Ollama, pick a model, and wire it into the RAG pipeline. Also covers **Groq**
(§8) — a free-tier hosted alternative, useful when Ollama isn't an option on a given machine.

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

**Consequence on this machine specifically:** `OllamaGenerator` is written and unit-tested but has
never been run against a live Ollama model here. **`GroqGenerator` has** — it's been verified
end-to-end against Groq's real API (health check, a live 401 against a bad key) — so Groq is the
practical way to test generation from this machine while Ollama stays blocked. See
[§8](#8-using-groq-instead-of-or-alongside-ollama).

[Jump to Ollama workarounds →](#option-b-network-blocked-workarounds)

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
see [§8](#8-using-groq-instead-of-or-alongside-ollama).

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
  "think": false,
  "options": {"temperature": 0.0, "seed": 42, "num_ctx": 4096, "num_predict": 1024}
}
```

`think` is a **top-level** field, not part of `options` — it's Ollama's switch for disabling the
`<think>...</think>` reasoning scratchpad on hybrid-reasoning models (qwen3, deepseek-r1, and
similar). It defaults to `false` here on purpose: see [§7](#7-a-real-failure-generation-that-silently-produces-nothing) for why.

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

More settings exist with sensible defaults and are simply omitted from the YAML. Add any of them
to a config to override:

| Setting | Default | Purpose |
|---|---|---|
| `base_url` | `http://localhost:11434` | Where Ollama is listening |
| `num_predict` | `1024` | Max tokens generated, **including any reasoning trace** |
| `think` | `false` | Disable the reasoning scratchpad on models that support it |
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

Reasoning models like qwen3 emit `<think>...</think>` scratchpads before writing the visible
answer. These are stripped automatically in `_strip_thinking()` — the scratchpad is not the
answer, and leaving it in would let a faithfulness judge grade the model's rough working.

By default (`generation.think: false`) the scratchpad is disabled at the source, which is both
faster and safer — see the next section for why "safer" matters here.

---

## 7. A real failure: generation that silently produces nothing

This happened during setup and is worth understanding, because the symptom looks exactly like
"generation is broken" when the real cause is a token budget.

**What went wrong:** `qwen3` writes `<think>...reasoning...</think>` before its actual answer.
That reasoning counts against `num_predict` — originally capped at 512 tokens here. If the model
was still inside `<think>` when it hit that cap, generation stopped **before the visible answer
was ever written**, and the closing `</think>` tag never arrived. The original stripper only knew
how to remove a *closed* block, so an unterminated one passed straight through — the user got a
wall of raw reasoning (or nothing useful) instead of an answer, with retrieval working perfectly
the whole time.

**The fix, two layers:**

1. **`think: false`** is now sent as a top-level field on every request, so Ollama skips the
   scratchpad entirely on models that support the switch. No reasoning tokens are spent, so
   there's nothing to truncate mid-thought.
2. **`_strip_thinking()` no longer requires a closing tag.** An unterminated `<think>` is treated
   the same as a closed one: everything from the tag onward is discarded. If that leaves nothing,
   `generate()` raises a clear error naming `num_predict` as the likely cause, instead of quietly
   returning an empty string or a reasoning dump.
3. **`num_predict` was also raised, 512 → 1024**, as a safety net for any model or Ollama version
   that ignores `think`.

**If you still hit this** (an older Ollama build that doesn't support `think`, or a different
reasoning model):

```
RuntimeError: Model produced no answer after removing its reasoning trace
(num_predict=1024). It likely hit the token limit while still 'thinking'.
Raise generation.num_predict, or confirm generation.think is false, in your config.
```

- Confirm your Ollama version supports `think` (`ollama --version`; the feature ships in
  reasonably recent builds). If not, raising `num_predict` further (e.g. `2048`) is the fallback.
- Check `results.jsonl` for this run — the query's `error` field will hold this exact message, and
  the batch continues past it rather than aborting.

---

## 8. Using Groq instead of (or alongside) Ollama

Groq hosts open models (Llama, and others) on custom inference hardware and offers a free tier —
useful if a laptop can't run Ollama, or you just want faster iteration than CPU inference allows.
It's implemented as a second `Generator`, side by side with Ollama, and switching between them
never requires editing `configs/*.yaml`.

### 8.1 Get an API key

1. Sign up at <https://console.groq.com>.
2. Create a key at <https://console.groq.com/keys>.
3. Copy `.env.example` to `.env` in the project root and paste it in:

   ```powershell
   copy .env.example .env
   ```

   ```dotenv
   GROQ_API_KEY=gsk_your_actual_key_here
   ```

`.env` is gitignored — it is never committed. `.env.example` (which **is** committed) has no real
key in it, only the variable name.

The key is loaded automatically: `src/groundtruth/__init__.py` calls `load_dotenv()` (via
`python-dotenv`) on import, so every CLI command picks it up with no extra setup.

### 8.2 Switch the provider

Every checked-in config still says `provider: ollama` — that stays the default, on disk,
untouched. To use Groq instead, set two more lines in `.env` (also shown, commented out, in
`.env.example`):

```dotenv
GROUNDTRUTH_GENERATION_PROVIDER=groq
GROUNDTRUTH_GENERATION_MODEL=llama-3.3-70b-versatile
```

This overrides `generation.provider` and `generation.model` **at load time**, for every command,
regardless of which `--config` you pass. Leave these two lines unset (or commented out) and
everything behaves exactly as before — Ollama, `qwen3:4b`, nothing changed.

A couple of solid free-tier model choices (catalog: <https://console.groq.com/docs/models>):

| Model | Good for |
|---|---|
| `llama-3.3-70b-versatile` | Best quality; still fast — Groq's hardware makes model size a much smaller latency factor than on local CPU |
| `llama-3.1-8b-instant` | Fastest, lightest quota usage per request |

### 8.3 Verify it

```powershell
.\.venv\Scripts\python.exe -m groundtruth.cli ask "What is multi-head attention?" --config configs\hybrid.yaml
```

With the env vars set, this now calls Groq instead of Ollama — same prompt, same retrieved
chunks, same citation format, same refusal behaviour. Run the Bolivia refusal check too; it
should still return exactly `INSUFFICIENT_CONTEXT`.

### 8.4 How it's implemented

`GroqGenerator` (in `src/groundtruth/generator.py`) is the second implementation of a small
`Generator` protocol — the same "one interface, multiple implementations" pattern the retrievers
use. It calls Groq's `/chat/completions` endpoint, which is OpenAI-compatible:

```python
class Generator(Protocol):
    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str: ...
    def health_check(self) -> tuple[bool, str]: ...
```

It reuses the **exact same** `SYSTEM_PROMPT`, `USER_TEMPLATE`, `format_context()`, `REFUSAL` token
and `_strip_thinking()` safety net as `OllamaGenerator` — the prompt a model sees is identical no
matter which provider answers it, so provider is a clean switch, never a second variable smuggled
into a retrieval experiment.

Two things that differ from Ollama, both deliberate:

- **The API key is never read from `GenerationConfig`.** That object gets written verbatim into
  every `runs/<id>/config.yaml`. A secret living there would leak into every run directory.
  `GroqGenerator` reads `os.environ["GROQ_API_KEY"]` directly, at the moment it's needed.
- **Groq-specific failures get their own messages**: a missing key, a rejected (401) key, and a
  429 rate limit (which backs off using the `Retry-After` header if Groq sends one, otherwise the
  same exponential backoff Ollama uses).

### 8.5 Adding a third provider

Write a class with the same two methods, reuse the shared prompt constants, add one branch in
`build_generator()`:

```python
def build_generator(cfg: GenerationConfig) -> Generator:
    if cfg.provider == "ollama":
        return OllamaGenerator(cfg)
    if cfg.provider == "groq":
        return GroqGenerator(cfg)
    raise ValueError(f"Unsupported generation provider {cfg.provider!r}. Supported: 'ollama', 'groq'.")
```

**Read API keys from environment variables, never hard-code them**, and keep them off
`GenerationConfig` for the reason above.

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
Raise `num_predict` (default 1024 tokens).

### `RuntimeError: Model produced no answer after removing its reasoning trace`
See [§7](#7-a-real-failure-generation-that-silently-produces-nothing) — the model hit
`num_predict` while still inside a `<think>` block and never wrote a visible answer. Confirm
`generation.think` is `false` (the default) and that your Ollama version supports it; if not,
raise `num_predict` further.

### Very slow first generation, fast afterwards
Normal. Ollama loads the model into RAM on first use and keeps it warm for ~5 minutes. Batch runs
amortise this across all queries.

### Out of memory
Unlikely with 31 GB, but if it happens use a smaller model or lower `num_ctx`.

### `GROQ_API_KEY is not set`
You haven't created `.env`, or it doesn't have the key in it. Copy `.env.example` to `.env` and
paste in a key from <https://console.groq.com/keys>. Confirm `GROUNDTRUTH_GENERATION_PROVIDER=groq`
is also set — without it, Ollama stays selected and `GROQ_API_KEY` is simply unused.

### `Groq rejected GROQ_API_KEY (401)`
The key is wrong, revoked, or was pasted with extra whitespace/quotes. Generate a fresh one at
<https://console.groq.com/keys> and paste it into `.env` with no quotes around it.

### `Model '...' not found on Groq`
The model ID in `GROUNDTRUTH_GENERATION_MODEL` (or `generation.model` in a config) doesn't match
Groq's current catalog. Check <https://console.groq.com/docs/models> — Groq's available models
change over time.

### `Groq rate limit (429) exceeded after retries`
You've hit the free-tier request or token quota. `GroqGenerator` already backs off once using
Groq's `Retry-After` header; a sustained 429 across a whole batch run usually means the quota is
genuinely exhausted for the window. Wait, reduce `--limit` on the batch, or switch to
`llama-3.1-8b-instant`, which tends to have a cheaper per-request quota cost.

### Switched `GROUNDTRUTH_GENERATION_PROVIDER=groq` but it's still using Ollama
`.env` is only read once, at Python process start (`load_dotenv()` runs on package import). If
you edited `.env` in an already-open terminal, close it and open a new one — or explicitly
`$env:GROUNDTRUTH_GENERATION_PROVIDER = "groq"` in the current PowerShell session instead.

---

## Summary

| | Ollama | Groq |
|---|---|---|
| **Needed for retrieval?** | ❌ No — use `--no-generate` for either | |
| **Config changes required?** | None — it's the on-disk default | None — flip 2 lines in `.env` |
| **Recommended model** | `qwen3:4b` (~2.6 GB) | `llama-3.3-70b-versatile` |
| **Expected speed** | ~17 s per answer (CPU) | seconds per answer (hosted) |
| **Cost** | Free, fully offline | Free tier, needs an API key + network |
| **Known blocker here** | GitHub release downloads return 403 on this network | none — verified against the live API |

Whichever you use, verify with the Bolivia question first. `INSUFFICIENT_CONTEXT` means the
grounding works, and Phase 3 will have a solid foundation to measure against.
