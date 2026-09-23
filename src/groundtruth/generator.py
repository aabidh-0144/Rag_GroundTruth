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
                content = response.json()["message"]["content"].strip()
                return _strip_thinking(content)
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.cfg.max_retries:
                    time.sleep(2**attempt)

        raise RuntimeError(f"Ollama generation failed after retries: {last_error}")


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models such as qwen3.

    The reasoning trace is not the answer; leaving it in would let a faithfulness
    judge score the model's scratchpad.
    """
    import re

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned or text.strip()


def build_generator(cfg: GenerationConfig) -> OllamaGenerator:
    if cfg.provider != "ollama":
        raise ValueError(
            f"Unsupported generation provider {cfg.provider!r}. Only 'ollama' is implemented."
        )
    return OllamaGenerator(cfg)
