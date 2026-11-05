"""Configuration loading.

Configs drive behaviour — there are no magic numbers in the pipeline code. An
experiment is a YAML file, which is also what makes runs reproducible: the config is
snapshotted into every run directory.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repo root, resolved from this file so the CLI works from any working directory.
ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
DOCS_DIR = ROOT / "data" / "processed" / "documents"
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
CHROMA_DIR = ROOT / "store" / "chroma"
BM25_PATH = ROOT / "store" / "bm25.pkl"
RUNS_DIR = ROOT / "runs"

COLLECTION_NAME = "groundtruth"


@dataclass(slots=True)
class ChunkingConfig:
    size: int = 512
    overlap: int = 64
    min_tokens: int = 80  # below this a chunk is merged into its neighbour


@dataclass(slots=True)
class EmbeddingConfig:
    model: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 32
    # bge is an asymmetric model: queries get an instruction prefix, documents do not.
    query_prefix: str = "Represent this sentence for searching relevant passages: "


@dataclass(slots=True)
class FusionConfig:
    method: str = "rrf"
    k: int = 60  # RRF damping constant; 60 is the value from the original paper


@dataclass(slots=True)
class RerankerConfig:
    # MiniLM-L6 (22M) reranks 50 candidates in ~2.8s on this CPU; bge-reranker-base
    # (278M) takes ~29s for the same work. The small model is the default so the
    # experiment loop stays interactive; see configs/hybrid_rerank_large.yaml for
    # the quality ceiling.
    model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    top_n: int = 50  # how many candidates the cross-encoder scores; also the recall ceiling
    max_length: int = 512


@dataclass(slots=True)
class CandidatesConfig:
    dense: int = 50
    sparse: int = 50


@dataclass(slots=True)
class RetrievalConfig:
    strategy: str = "dense"  # dense | bm25 | hybrid | hybrid_rerank
    k: int = 10
    candidates: CandidatesConfig = field(default_factory=CandidatesConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)


@dataclass(slots=True)
class GenerationConfig:
    provider: str = "ollama"
    model: str = "qwen3:4b"
    base_url: str = "http://localhost:11434"
    context_chunks: int = 5  # independent of retrieval k
    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 4096
    # qwen3 and other hybrid-reasoning models spend part of this budget on a
    # <think>...</think> scratchpad before writing the visible answer. 512 was
    # tight enough that the model could hit the cap mid-thought and never emit an
    # answer at all. `think=False` below disables that scratchpad on models that
    # support the switch; num_predict is raised regardless as a safety net for
    # models that ignore it.
    num_predict: int = 1024
    think: bool = False
    timeout_s: int = 180
    max_retries: int = 2


@dataclass(slots=True)
class Config:
    name: str = "dense"
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Hash of the full config. Recorded in run manifests."""
        blob = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def index_fingerprint(self) -> str:
        """Hash of only the settings that affect the *index*.

        Dense / Hybrid / Hybrid+Reranker must share one index (invariant #3), so this
        deliberately excludes retrieval and generation settings. Two configs that
        disagree here cannot be compared.
        """
        blob = json.dumps(
            {"chunking": asdict(self.chunking), "embedding": asdict(self.embedding)},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _merge(default: Any, override: dict[str, Any] | None) -> Any:
    """Apply a YAML mapping onto a dataclass instance, recursing into nested ones."""
    if not override:
        return default
    for key, value in override.items():
        if not hasattr(default, key):
            raise ValueError(
                f"Unknown config key {key!r} for {type(default).__name__}. "
                "Typos here silently change nothing, so they are treated as errors."
            )
        current = getattr(default, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _merge(current, value)
        else:
            setattr(default, key, value)
    return default


# Free-tier model recommended for each alternate provider, used only as a fallback
# when the environment switches `provider` without also giving an explicit `model`.
# Verified 2026-09-28 against a real Groq free-tier key: fastest of the available
# chat models, correctly followed the "[1]"-style citation instruction, and produced
# clean ASCII output (a couple of the other chat models on that key's catalog used
# typographic Unicode punctuation instead). Groq's free-tier catalog changes over
# time - re-check https://console.groq.com/docs/models if this stops working.
_PROVIDER_DEFAULT_MODELS = {"groq": "qwen/qwen3.8-27b"}


def apply_generation_env_overrides(cfg: Config) -> None:
    """Let the environment (typically a local `.env`, see `.env.example`) swap the
    generator without editing any config file.

    This exists so the four experiment configs can keep `provider: ollama` as their
    on-disk, checked-in default - "the existing logic keeps working" - while a single
    `.env` line flips every run to Groq for a quick test:

        GROUNDTRUTH_GENERATION_PROVIDER=groq
        GROUNDTRUTH_GENERATION_MODEL=qwen/qwen3.8-27b

    The *resolved* config (post-override) is what gets written into
    `runs/<id>/config.yaml`, so a run is still fully self-describing - you can always
    see which provider actually produced it, even though the source YAML disagrees.
    """
    provider = os.environ.get("GROUNDTRUTH_GENERATION_PROVIDER")
    model = os.environ.get("GROUNDTRUTH_GENERATION_MODEL")

    if provider and provider != cfg.generation.provider and not model:
        # Switching provider with no explicit model would otherwise leave whatever
        # model name the YAML had (an Ollama-style tag like "qwen3:4b") pointed at a
        # provider that has never heard of it. Only kicks in when the caller didn't
        # already make an explicit choice.
        fallback = _PROVIDER_DEFAULT_MODELS.get(provider)
        if fallback:
            cfg.generation.model = fallback

    if provider:
        cfg.generation.provider = provider
    if model:
        cfg.generation.model = model


def load_config(path: str | Path) -> Config:
    """Load a YAML experiment config, falling back to defaults for absent keys."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = Config()
    _merge(cfg, data)
    apply_generation_env_overrides(cfg)

    valid = {"dense", "bm25", "hybrid", "hybrid_rerank"}
    if cfg.retrieval.strategy not in valid:
        raise ValueError(
            f"Unknown retrieval strategy {cfg.retrieval.strategy!r}; expected one of {sorted(valid)}"
        )

    valid_providers = {"ollama", "groq"}
    if cfg.generation.provider not in valid_providers:
        raise ValueError(
            f"Unknown generation provider {cfg.generation.provider!r}; "
            f"expected one of {sorted(valid_providers)}"
        )
    return cfg
