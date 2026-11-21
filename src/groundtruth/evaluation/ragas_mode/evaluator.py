"""Mode 2 evaluator: scores samples with Ragas metrics.

Plugs into orchestrator.run_evaluation exactly like Mode 1. Each (sample, metric)
pair is scored separately with `await metric.ascore(...)` and its own timeout, so
one metric failing or returning NaN never voids the others.

Applicability is decided here, before Ragas is called:
- reference-based metric + no ground truth  -> skipped (never fabricated)
- metric needs contexts + none retrieved    -> skipped
- metric needs an answer + answer empty     -> skipped
- answer-judging metric + refusal answer    -> skipped (see metrics.py)
A NaN from Ragas (its signal for an internal failure) is recorded as failed.

All samples run inside one event loop, so the async HTTP client is created and
used on the same loop.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any

from groundtruth.config import GenerationConfig
from groundtruth.evaluation.config import RagasConfig
from groundtruth.evaluation.orchestrator import OnResult, build_sample_result
from groundtruth.evaluation.ragas_mode import adapters
from groundtruth.evaluation.ragas_mode.metrics import (
    REGISTRY,
    SAMPLE_FIELD,
    RagasMetricSpec,
    instantiate,
    validate_names,
)
from groundtruth.evaluation.reference_free.base import is_refusal
from groundtruth.evaluation.schemas import EvalSample, MetricResult
from groundtruth.llm import build_llm_client


def applicability(spec: RagasMetricSpec, sample: EvalSample) -> str | None:
    """Why a metric cannot be scored for this sample, or None if it can."""
    if spec.requires_reference and not sample.ground_truth:
        return "no ground_truth for this sample (reference-based metric)"
    if "retrieved_contexts" in spec.inputs and not sample.contexts:
        return "no contexts were retrieved"
    if "response" in spec.inputs and not sample.answer.strip():
        return "no answer to score"
    if spec.skip_on_refusal and is_refusal(sample.answer):
        return "answer is a refusal; this metric judges the answer itself"
    return None


class RagasEvaluator:
    mode = "ragas"

    def __init__(
        self,
        metrics: list[str],
        judge_cfg: GenerationConfig,
        ragas_cfg: RagasConfig,
        concurrency: int = 1,
        metric_instances: dict[str, Any] | None = None,
    ) -> None:
        self.metrics = validate_names(metrics)
        self.judge_cfg = judge_cfg
        self.ragas_cfg = ragas_cfg
        self.concurrency = max(1, concurrency)
        # Injectable so tests can supply fake metrics without loading Ragas.
        self._instances = metric_instances

    def health_check(self) -> tuple[bool, str]:
        # Same server and model Ragas will talk to, checked through our own client.
        if not any(REGISTRY[m].needs_llm for m in self.metrics):
            return True, "ok (no LLM-based metrics selected)"
        return build_llm_client(self.judge_cfg, label="judge").health_check()

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.judge_cfg.provider,
            "model": self.judge_cfg.model,
            "embedding_model": self.ragas_cfg.embedding_model,
            "framework": f"ragas {adapters.import_ragas().__version__}"
            if self._instances is None
            else "ragas",
        }

    def _build_instances(self) -> dict[str, Any]:
        specs = [REGISTRY[m] for m in self.metrics]
        llm = adapters.build_ragas_llm(self.judge_cfg) if any(s.needs_llm for s in specs) else None
        embeddings = (
            adapters.build_embeddings(self.ragas_cfg.embedding_model)
            if any(s.needs_embeddings for s in specs)
            else None
        )
        return {s.name: instantiate(s, llm, embeddings) for s in specs}

    async def _score_metric(self, name: str, instance: Any, sample: EvalSample) -> MetricResult:
        spec = REGISTRY[name]
        reason = applicability(spec, sample)
        if reason:
            return MetricResult.skipped(name, reason)

        kwargs = {arg: getattr(sample, SAMPLE_FIELD[arg]) for arg in spec.inputs}
        start = time.perf_counter()
        try:
            outcome = await asyncio.wait_for(instance.ascore(**kwargs), self.ragas_cfg.timeout_s)
            value = float(outcome.value)
            if math.isnan(value):
                result = MetricResult.failed(name, "Ragas returned NaN (internal parse/LLM failure)")
            else:
                result = MetricResult.success(
                    name,
                    min(max(value, 0.0), 1.0),
                    getattr(outcome, "reason", None) or spec.description,
                    ragas_class=spec.ragas_class,
                )
        except asyncio.TimeoutError:
            result = MetricResult.failed(name, f"timed out after {self.ragas_cfg.timeout_s}s")
        except Exception as exc:
            result = MetricResult.failed(name, f"{type(exc).__name__}: {exc}")
        result.elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        return result

    async def _run(self, samples: list[EvalSample], on_result: OnResult) -> None:
        instances = self._instances if self._instances is not None else self._build_instances()
        gate = asyncio.Semaphore(self.concurrency)

        async def one(sample: EvalSample) -> None:
            async with gate:
                metrics = {}
                for name in self.metrics:
                    metrics[name] = await self._score_metric(name, instances[name], sample)
                on_result(build_sample_result(sample, metrics))

        await asyncio.gather(*(one(s) for s in samples))

    def evaluate(self, samples: list[EvalSample], on_result: OnResult) -> None:
        asyncio.run(self._run(samples, on_result))
