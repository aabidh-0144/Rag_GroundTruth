"""Retrieved context -> grounded answer, via a local Ollama model or a free-tier API.

Two prompt decisions matter for the evaluation that comes later:

* **Inline citations.** The model must cite the numbered context blocks it used.
  That gives the faithfulness metric something concrete to check against rather than
  requiring it to re-derive attribution.
* **An explicit refusal token.** `INSUFFICIENT_CONTEXT` separates "the retriever
  failed" from "the generator hallucinated". Without it, both look like a wrong
  answer and the two failure modes in the problem statement become indistinguishable.

This module owns the *RAG prompt*. The provider transport - HTTP, retries, 401/429
handling, API keys, the `think` switch, reasoning-trace stripping - lives in llm.py,
shared with the evaluation judges. `OllamaGenerator` and `GroqGenerator` are thin
wrappers: build the prompt, hand it to the matching `LLMClient`.

`SYSTEM_PROMPT`, `USER_TEMPLATE`, `format_context()` and `REFUSAL` are shared by both
providers, so the prompt a model sees is identical regardless of which provider is
generating - provider is a config switch, never a second variable smuggled into an
experiment.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from groundtruth.config import GenerationConfig
from groundtruth.llm import (  # noqa: F401 - re-exported for existing callers
    GROQ_API_KEY_ENV,
    GROQ_BASE_URL,
    GROQ_KEYS_URL,
    GROQ_MODELS_URL,
    GroqChatClient,
    LLMClient,
    OllamaChatClient,
    strip_thinking,
)
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

# Kept under its original name for callers and tests that import it from here.
_strip_thinking = strip_thinking


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


class _PromptedGenerator:
    """Shared body of both generators: the RAG prompt over an `LLMClient`."""

    client: LLMClient

    @property
    def cfg(self) -> GenerationConfig:
        return self.client.cfg

    def health_check(self) -> tuple[bool, str]:
        return self.client.health_check()

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return REFUSAL
        user = USER_TEMPLATE.format(context=format_context(chunks), question=question)
        return self.client.chat(SYSTEM_PROMPT, user)


class OllamaGenerator(_PromptedGenerator):
    """RAG answers from a local Ollama model."""

    def __init__(self, cfg: GenerationConfig) -> None:
        self.client = OllamaChatClient(cfg)


class GroqGenerator(_PromptedGenerator):
    """RAG answers from Groq's hosted API. Key from GROQ_API_KEY, never from config."""

    def __init__(self, cfg: GenerationConfig) -> None:
        self.client = GroqChatClient(cfg)


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
