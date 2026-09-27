"""Retrieved context -> grounded answer, via a local Ollama model.

Two prompt decisions matter for the evaluation that comes later:

* **Inline citations.** The model must cite the numbered context blocks it used.
  That gives the faithfulness metric something concrete to check against rather than
  requiring it to re-derive attribution.
* **An explicit refusal token.** `INSUFFICIENT_CONTEXT` separates "the retriever
  failed" from "the generator hallucinated". Without it, both look like a wrong
  answer and the two failure modes in the problem statement become indistinguishable.
"""

from __future__ import annotations

import re
import time

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


def build_generator(cfg: GenerationConfig) -> OllamaGenerator:
    if cfg.provider != "ollama":
        raise ValueError(
            f"Unsupported generation provider {cfg.provider!r}. Only 'ollama' is implemented."
        )
    return OllamaGenerator(cfg)
