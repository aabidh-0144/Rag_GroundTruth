"""Our configured judge and embedding model, in the shapes Ragas expects.

Ragas 0.4.4 (verified against the installed package) uses:

* LLM - `ragas.llms.llm_factory(model, provider="openai", client=...)`, an
  instructor-based `InstructorBaseRagasLLM`. Both Ollama (`<base_url>/v1`) and Groq
  expose an OpenAI-compatible API, so one adapter covers both providers via the
  `openai` client with a custom base_url. `AsyncOpenAI` is used because metrics are
  scored with `await metric.ascore(...)`; Ragas detects the client type. This same
  LLM object also drives testset generation (transforms and query synthesizers).
* Embeddings - `ragas.embeddings.HuggingFaceEmbeddings` (the modern
  BaseRagasEmbedding), running `BAAI/bge-small-en-v1.5` locally through
  sentence-transformers - the model the index already uses, so nothing new is
  downloaded and nothing leaves the machine. The same object serves metrics and
  testset generation: the knowledge-graph EmbeddingExtractor accepts the modern
  interface. (The legacy `ragas.embeddings.base.HuggingfaceEmbeddings` is not used:
  in 0.4.4 it cannot even be instantiated - its async methods are unimplemented.)

Every `import ragas` in this project goes through `import_ragas()`, which first
disables Ragas's default usage analytics (it posts to t.explodinggradients.com
unless RAGAS_DO_NOT_TRACK=true). Set RAGAS_DO_NOT_TRACK=false in .env to opt in.
"""

from __future__ import annotations

import os
from typing import Any

from groundtruth.config import GenerationConfig
from groundtruth.llm import GROQ_API_KEY_ENV, GROQ_BASE_URL


def import_ragas() -> Any:
    """Import ragas with analytics off by default; clear error if it's missing."""
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    try:
        import ragas
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "Ragas is not installed. Run: .venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc
    return ragas


def openai_compatible_endpoint(cfg: GenerationConfig) -> tuple[str, str]:
    """(base_url, api_key) of the provider's OpenAI-compatible API."""
    if cfg.provider == "ollama":
        # Ollama ignores the key, but the openai client refuses to start without one.
        return f"{cfg.base_url.rstrip('/')}/v1", "ollama"
    if cfg.provider == "groq":
        key = os.environ.get(GROQ_API_KEY_ENV)
        if not key:
            raise RuntimeError(f"{GROQ_API_KEY_ENV} is not set (needed for a Groq judge)")
        return GROQ_BASE_URL, key
    raise ValueError(f"no OpenAI-compatible endpoint for provider {cfg.provider!r}")


def build_ragas_llm(cfg: GenerationConfig) -> Any:
    import_ragas()
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    base_url, api_key = openai_compatible_endpoint(cfg)
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=cfg.timeout_s,
        max_retries=cfg.max_retries,
    )
    return llm_factory(
        cfg.model,
        provider="openai",
        client=client,
        temperature=cfg.temperature,
        max_tokens=cfg.num_predict,
    )


def build_embeddings(model_name: str) -> Any:
    import_ragas()
    from ragas.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model=model_name, use_api=False, device="cpu")
