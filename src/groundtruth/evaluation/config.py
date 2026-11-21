"""What an evaluation is configured to do.

Loaded from `configs/evaluation.yaml` using the same strict merge as experiment
configs (an unknown key is an error, never silently ignored).

The judge is resolved into a plain `GenerationConfig`, the same dataclass the RAG
generator uses, so `llm.build_llm_client()` serves both. Resolution order, later
wins:

1. the generation settings of `judge.generation_config` (default
   `configs/hybrid.yaml`, including its `.env` overrides) when
   `judge.inherit_from_generation` is true - i.e. "judge with whatever model I
   generate with"
2. `judge.provider` / `judge.model` in `evaluation.yaml`, if set
3. `GROUNDTRUTH_JUDGE_PROVIDER` / `GROUNDTRUTH_JUDGE_MODEL` in the environment
4. the judge-specific limits below (`temperature`, `num_ctx`, `num_predict`, ...),
   which always apply: a judge needs room for the contexts plus a JSON verdict.

No API key is ever stored here; keys stay in the environment (see llm.py).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from groundtruth.config import ROOT, GenerationConfig, _merge, load_config

EVALUATION_CONFIG_PATH = ROOT / "configs" / "evaluation.yaml"

REFERENCE_FREE_METRICS = ("faithfulness", "answer_relevance", "context_relevance")


@dataclass(slots=True)
class JudgeConfig:
    inherit_from_generation: bool = True
    generation_config: str = "configs/hybrid.yaml"
    provider: str | None = None
    model: str | None = None
    temperature: float = 0.0
    seed: int = 42
    # A judge reads up to `context_chunks` chunks (~2.5k tokens) plus the answer and
    # writes a JSON verdict, so it needs a larger window than the generator.
    num_ctx: int = 8192
    num_predict: int = 2048
    think: bool = False
    timeout_s: int = 300
    max_retries: int = 2


@dataclass(slots=True)
class ReferenceFreeConfig:
    metrics: list[str] = field(default_factory=lambda: list(REFERENCE_FREE_METRICS))


@dataclass(slots=True)
class RagasConfig:
    metrics: list[str] = field(
        default_factory=lambda: ["faithfulness", "response_relevancy", "context_precision"]
    )
    # Local sentence-transformers model for Ragas metrics that need embeddings.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    timeout_s: int = 300


@dataclass(slots=True)
class SyntheticConfig:
    # Small by default: generation makes many judge calls, slow on a local CPU model.
    size: int = 5
    distribution: dict[str, float] = field(
        default_factory=lambda: {"factual": 0.5, "reasoning": 0.25, "multihop": 0.25}
    )


@dataclass(slots=True)
class EvaluationConfig:
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    # Samples evaluated in parallel. 1 suits a single local Ollama model.
    concurrency: int = 1
    reference_free: ReferenceFreeConfig = field(default_factory=ReferenceFreeConfig)
    ragas: RagasConfig = field(default_factory=RagasConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_evaluation_config(path: str | Path | None = None) -> EvaluationConfig:
    """Load `configs/evaluation.yaml` (or `path`), falling back to defaults."""
    cfg = EvaluationConfig()
    config_path = Path(path) if path else EVALUATION_CONFIG_PATH
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        _merge(cfg, data)
    if cfg.concurrency < 1:
        raise ValueError("evaluation concurrency must be at least 1")
    return cfg


def resolve_judge(cfg: JudgeConfig) -> GenerationConfig:
    """Turn the judge section into the GenerationConfig the LLM client consumes."""
    if cfg.inherit_from_generation:
        generation_path = Path(cfg.generation_config)
        if not generation_path.is_absolute():
            generation_path = ROOT / generation_path
        judge = replace(load_config(generation_path).generation)
    else:
        judge = GenerationConfig()

    if cfg.provider:
        judge.provider = cfg.provider
    if cfg.model:
        judge.model = cfg.model

    env_provider = os.environ.get("GROUNDTRUTH_JUDGE_PROVIDER")
    env_model = os.environ.get("GROUNDTRUTH_JUDGE_MODEL")
    if env_provider:
        judge.provider = env_provider
    if env_model:
        judge.model = env_model

    if judge.provider not in ("ollama", "groq"):
        raise ValueError(
            f"Unknown judge provider {judge.provider!r}; expected 'ollama' or 'groq'"
        )

    judge.temperature = cfg.temperature
    judge.seed = cfg.seed
    judge.num_ctx = cfg.num_ctx
    judge.num_predict = cfg.num_predict
    judge.think = cfg.think
    judge.timeout_s = cfg.timeout_s
    judge.max_retries = cfg.max_retries
    return judge
