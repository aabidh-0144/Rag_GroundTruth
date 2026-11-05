"""Retrieved context -> grounded answer, via a local Ollama model or a free-tier API.

Two prompt decisions matter for the evaluation that comes later:

* **Inline citations.** The model must cite the numbered context blocks it used.
  That gives the faithfulness metric something concrete to check against rather than
  requiring it to re-derive attribution.
* **An explicit refusal token.** `INSUFFICIENT_CONTEXT` separates "the retriever
  failed" from "the generator hallucinated". Without it, both look like a wrong
  answer and the two failure modes in the problem statement become indistinguishable.

Two providers are implemented, `OllamaGenerator` and `GroqGenerator`, behind one
`Generator` protocol - the same pattern the retrievers use (see retrievers/base.py).
`SYSTEM_PROMPT`, `USER_TEMPLATE`, `format_context()`, `REFUSAL` and `_strip_thinking()`
are shared by both, so the prompt a model sees is identical regardless of which
provider is generating - provider is a config switch, never a second variable smuggled
into an experiment.

Adding a third provider means adding one more class with the same two methods
(`generate`, `health_check`) and one branch in `build_generator()` - never changing
OllamaGenerator or GroqGenerator themselves.
"""

from __future__ import annotations

import os
import re
import time
from typing import Protocol, runtime_checkable

import requests

from groundtruth.config import GenerationConfig
from groundtruth.schemas import RetrievedChunk

REFUSAL = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = """You are a precise research assistant. Answer questions using ONLY the numbered context blocks provided.

Rules:
1. Use only information stated in the context. Do not use prior knowledge.
2. Cite the block numbers you used, inline, like [1] or [2][3].
3. If the context does not contain the answer, reply with exactly: INSUFFICIENT_CONTEXT
4. Be concise and factual. Do not speculate, hedge, or add commentary.
"""

USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""


@runtime_checkable
class Generator(Protocol):
    """The generation contract. Both providers implement exactly this."""

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Return the model's answer, or REFUSAL if `chunks` is empty."""
        ...

    def health_check(self) -> tuple[bool, str]:
        """(is_usable, message) - checked once before a batch run, and by `ask`."""
        ...


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render chunks as numbered blocks, labelled with their source."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("doc_id", "unknown")
        section = chunk.metadata.get("section_path") or "-"
        blocks.append(f"[{i}] (source: {source} | section: {section})\n{chunk.text.strip()}")
    return "\n\n".join(blocks)


class OllamaGenerator:
    """Thin client over Ollama's /api/chat.

    Temperature 0 and a fixed seed because reproducibility is the whole point of a
    regression-testing framework: the same config over the same corpus must produce
    the same answer.
    """

    def __init__(self, cfg: GenerationConfig) -> None:
        self.cfg = cfg
        self.url = f"{cfg.base_url.rstrip('/')}/api/chat"

    def health_check(self) -> tuple[bool, str]:
        """Verify Ollama is up and the configured model is present."""
        try:
            response = requests.get(f"{self.cfg.base_url.rstrip('/')}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.RequestException as exc:
            return False, f"Ollama not reachable at {self.cfg.base_url}: {exc}"

        models = [m.get("name", "") for m in response.json().get("models", [])]
        # Ollama reports "qwen3:4b"; a config naming a bare "qwen3" should still match.
        if not any(m == self.cfg.model or m.startswith(self.cfg.model + ":") for m in models):
            return False, (
                f"Model {self.cfg.model!r} not installed. Available: {models or 'none'}. "
                f"Run: ollama pull {self.cfg.model}"
            )
        return True, "ok"

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return REFUSAL

        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(
                        context=format_context(chunks), question=question
                    ),
                },
            ],
            "stream": False,
            # Top-level, not under "options" - this is where Ollama's chat API expects
            # the reasoning switch for hybrid-thinking models (qwen3, deepseek-r1, ...).
            # Older Ollama builds that predate this field simply ignore it.
            "think": self.cfg.think,
            "options": {
                "temperature": self.cfg.temperature,
                "seed": self.cfg.seed,
                # Set explicitly so context overflow is a visible config problem
                # rather than a silent truncation that corrupts faithfulness scores.
                "num_ctx": self.cfg.num_ctx,
                "num_predict": self.cfg.num_predict,
            },
        }

        last_error: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                response = requests.post(self.url, json=payload, timeout=self.cfg.timeout_s)
                response.raise_for_status()
                content = response.json()["message"]["content"]
                answer = _strip_thinking(content)
                if not answer:
                    # The whole response was reasoning with no visible answer after it -
                    # almost always num_predict was hit while the model was still inside
                    # <think>, so the closing tag (and the real answer) never arrived.
                    raise RuntimeError(
                        "Model produced no answer after removing its reasoning trace "
                        f"(num_predict={self.cfg.num_predict}). It likely hit the token "
                        "limit while still 'thinking'. Raise generation.num_predict, or "
                        "confirm generation.think is false, in your config."
                    )
                return answer
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.cfg.max_retries:
                    time.sleep(2**attempt)

        raise RuntimeError(f"Ollama generation failed after retries: {last_error}")


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_API_KEY_ENV = "GROQ_API_KEY"
GROQ_KEYS_URL = "https://console.groq.com/keys"
GROQ_MODELS_URL = "https://console.groq.com/docs/models"


class GroqGenerator:
    """Thin client over Groq's OpenAI-compatible /chat/completions endpoint.

    Deliberately parallel to OllamaGenerator - same SYSTEM_PROMPT, USER_TEMPLATE,
    REFUSAL and _strip_thinking() safety net - so swapping `generation.provider`
    between "ollama" and "groq" changes nothing else about what the model is asked.

    Why the API key is never read from `GenerationConfig`: that dataclass gets
    serialised verbatim into every run's `runs/<id>/config.yaml` (see pipeline.py).
    A secret living there would leak into every run directory, and potentially into
    git if `runs/` were ever un-ignored. The key is read from the environment - via
    `.env` and python-dotenv, see `.env.example` - at the moment it's needed, and
    never stored on the config object.
    """

    def __init__(self, cfg: GenerationConfig) -> None:
        self.cfg = cfg
        self.url = f"{GROQ_BASE_URL}/chat/completions"

    @staticmethod
    def _api_key() -> str | None:
        return os.environ.get(GROQ_API_KEY_ENV)

    def health_check(self) -> tuple[bool, str]:
        """Verify a key is present, Groq is reachable, and the model exists."""
        api_key = self._api_key()
        if not api_key:
            return False, (
                f"{GROQ_API_KEY_ENV} is not set. Copy .env.example to .env and paste "
                f"your key from {GROQ_KEYS_URL}."
            )

        try:
            response = requests.get(
                f"{GROQ_BASE_URL}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
        except requests.RequestException as exc:
            return False, f"Groq API not reachable: {exc}"

        if response.status_code == 401:
            return False, f"{GROQ_API_KEY_ENV} was rejected (401). Check the key at {GROQ_KEYS_URL}."
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            return False, f"Groq API error: {exc}"

        models = [m.get("id", "") for m in response.json().get("data", [])]
        if self.cfg.model not in models:
            return False, (
                f"Model {self.cfg.model!r} not found on Groq. Available: {models or 'none'}. "
                f"See {GROQ_MODELS_URL} for current model IDs."
            )
        return True, "ok"

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return REFUSAL

        api_key = self._api_key()
        if not api_key:
            raise RuntimeError(
                f"{GROQ_API_KEY_ENV} is not set. Copy .env.example to .env and paste your key "
                f"from {GROQ_KEYS_URL}."
            )

        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(
                        context=format_context(chunks), question=question
                    ),
                },
            ],
            "temperature": self.cfg.temperature,
            "seed": self.cfg.seed,  # best-effort determinism; Groq does not guarantee it
            "max_tokens": self.cfg.num_predict,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        last_error: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                response = requests.post(
                    self.url, headers=headers, json=payload, timeout=self.cfg.timeout_s
                )
                if response.status_code == 401:
                    raise RuntimeError(
                        f"Groq rejected {GROQ_API_KEY_ENV} (401). Check the key at {GROQ_KEYS_URL}."
                    )
                if response.status_code == 429:
                    if attempt < self.cfg.max_retries:
                        wait = float(response.headers.get("Retry-After", 2**attempt))
                        time.sleep(wait)
                        continue
                    raise RuntimeError(
                        "Groq rate limit (429) exceeded after retries. Check your free-tier "
                        "quota, or slow down the batch (fewer queries / add a delay)."
                    )
                response.raise_for_status()

                content = response.json()["choices"][0]["message"]["content"]
                # Groq's own chat models don't emit <think> blocks, but if this is
                # ever pointed at a reasoning-capable model hosted on Groq (e.g. a
                # qwen3 variant), the same truncation failure mode as Ollama applies.
                answer = _strip_thinking(content)
                if not answer:
                    raise RuntimeError(
                        "Model produced no answer after removing its reasoning trace "
                        f"(max_tokens={self.cfg.num_predict}). Raise generation.num_predict."
                    )
                return answer
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.cfg.max_retries:
                    time.sleep(2**attempt)

        raise RuntimeError(f"Groq generation failed after retries: {last_error}")


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models such as qwen3.

    The reasoning trace is not the answer; leaving it in would let a faithfulness
    judge score the model's scratchpad.

    A closed block is simply cut out. An *unterminated* `<think>` - the model was
    still reasoning when `num_predict` cut generation off - is treated the same way:
    everything from `<think>` onward is discarded rather than surfaced as if it were
    the answer. That deliberately makes `generate()` see an empty string in that case
    (checked by the caller) instead of dumping a half-finished scratchpad on the user.
    """
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_OPEN_RE.split(cleaned, maxsplit=1)[0]
    return cleaned.strip()


def build_generator(cfg: GenerationConfig) -> Generator:
    """Construct the generator named by `cfg.provider`.

    Mirrors `retrievers.build_retriever()`: one factory, one branch per
    implementation, nothing else in the pipeline needs to know which provider it got.
    """
    if cfg.provider == "ollama":
        return OllamaGenerator(cfg)
    if cfg.provider == "groq":
        return GroqGenerator(cfg)
    raise ValueError(
        f"Unsupported generation provider {cfg.provider!r}. Supported: 'ollama', 'groq'."
    )
