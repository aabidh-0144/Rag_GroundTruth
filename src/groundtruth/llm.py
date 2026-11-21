"""Provider transport: "send a system + user message, get text back".

Extracted from generator.py so the same HTTP code - retries, backoff, 401/429
handling, API-key handling, the `think` switch and the reasoning-trace safety net -
serves both the RAG generator and the evaluation judges. Neither caller knows which
provider it is talking to.

`LLMClient` is the contract; `OllamaChatClient` and `GroqChatClient` implement it.
Settings come from a `GenerationConfig` (provider, model, temperature, seed,
num_ctx, num_predict, think, timeout_s, max_retries). The judge reuses that same
dataclass, so a judge is configured exactly like a generator.

API keys are read from the environment at call time and never stored on a config
object, because configs are serialised into every run and evaluation directory.
"""

from __future__ import annotations

import os
import re
import time
from typing import Protocol, runtime_checkable

import requests

from groundtruth.config import GenerationConfig

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_API_KEY_ENV = "GROQ_API_KEY"
GROQ_KEYS_URL = "https://console.groq.com/keys"
GROQ_MODELS_URL = "https://console.groq.com/docs/models"


@runtime_checkable
class LLMClient(Protocol):
    """One chat turn against a configured model."""

    cfg: GenerationConfig

    def chat(self, system: str, user: str, *, json_mode: bool = False) -> str:
        """Return the model's reply with any reasoning trace removed.

        Raises RuntimeError when the provider fails after retries, rejects the key,
        or produces nothing but a reasoning trace.
        """
        ...

    def health_check(self) -> tuple[bool, str]:
        """(is_usable, message) - checked once before a batch, not per call."""
        ...


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models such as qwen3.

    The reasoning trace is not the answer; leaving it in would let a faithfulness
    judge score the model's scratchpad, or break a judge's JSON reply.

    A closed block is simply cut out. An *unterminated* `<think>` - the model was
    still reasoning when `num_predict` cut generation off - is treated the same way:
    everything from `<think>` onward is discarded rather than surfaced as if it were
    the answer. That deliberately yields an empty string in that case, which
    `chat()` turns into a clear error instead of dumping a half-finished scratchpad.
    """
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_OPEN_RE.split(cleaned, maxsplit=1)[0]
    return cleaned.strip()


def _empty_after_thinking(cfg: GenerationConfig, label: str) -> RuntimeError:
    # The whole response was reasoning with no visible answer after it - almost
    # always num_predict was hit while the model was still inside <think>, so the
    # closing tag (and the real answer) never arrived.
    return RuntimeError(
        "Model produced no answer after removing its reasoning trace "
        f"(num_predict={cfg.num_predict}). It likely hit the token limit while still "
        f"'thinking'. Raise {label}.num_predict, or confirm {label}.think is false, "
        "in your config."
    )


class OllamaChatClient:
    """Ollama's /api/chat.

    Temperature 0 and a fixed seed by default because reproducibility is the whole
    point of a regression-testing framework.
    """

    def __init__(self, cfg: GenerationConfig, label: str = "generation") -> None:
        self.cfg = cfg
        self.label = label
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

    def chat(self, system: str, user: str, *, json_mode: bool = False) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
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
        if json_mode:
            # Constrains decoding to valid JSON - judges parse the reply.
            payload["format"] = "json"

        last_error: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                response = requests.post(self.url, json=payload, timeout=self.cfg.timeout_s)
                response.raise_for_status()
                content = response.json()["message"]["content"]
                answer = strip_thinking(content)
                if not answer:
                    raise _empty_after_thinking(self.cfg, self.label)
                return answer
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.cfg.max_retries:
                    time.sleep(2**attempt)

        raise RuntimeError(f"Ollama generation failed after retries: {last_error}")


class GroqChatClient:
    """Groq's OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, cfg: GenerationConfig, label: str = "generation") -> None:
        self.cfg = cfg
        self.label = label
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

    def chat(self, system: str, user: str, *, json_mode: bool = False) -> str:
        api_key = self._api_key()
        if not api_key:
            raise RuntimeError(
                f"{GROQ_API_KEY_ENV} is not set. Copy .env.example to .env and paste your key "
                f"from {GROQ_KEYS_URL}."
            )

        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.cfg.temperature,
            "seed": self.cfg.seed,  # best-effort determinism; Groq does not guarantee it
            "max_tokens": self.cfg.num_predict,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
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
                # Groq's own chat models don't emit <think> blocks, but a reasoning
                # model hosted there would hit the same truncation failure as Ollama.
                answer = strip_thinking(content)
                if not answer:
                    raise _empty_after_thinking(self.cfg, self.label)
                return answer
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.cfg.max_retries:
                    time.sleep(2**attempt)

        raise RuntimeError(f"Groq generation failed after retries: {last_error}")


def build_llm_client(cfg: GenerationConfig, label: str = "generation") -> LLMClient:
    """Construct the client named by `cfg.provider`.

    `label` names the config section in error messages ("generation" or "judge"),
    so a truncation error points at the setting that actually needs changing.
    """
    if cfg.provider == "ollama":
        return OllamaChatClient(cfg, label)
    if cfg.provider == "groq":
        return GroqChatClient(cfg, label)
    raise ValueError(
        f"Unsupported generation provider {cfg.provider!r}. Supported: 'ollama', 'groq'."
    )
