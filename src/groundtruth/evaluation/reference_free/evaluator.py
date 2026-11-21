"""Mode 1 evaluator: runs the selected judge metrics over each sample.

Plugs into orchestrator.run_evaluation. Samples are scored concurrently up to
`concurrency` (default 1, right for one local Ollama model); metrics within a
sample run one after another, each through `run_metric`, so a failing metric is
recorded as "failed" without stopping the others.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from groundtruth.config import GenerationConfig
from groundtruth.evaluation.orchestrator import OnResult, build_sample_result
from groundtruth.evaluation.reference_free import METRICS, run_metric
from groundtruth.evaluation.schemas import EvalSample, SampleResult
from groundtruth.llm import LLMClient, build_llm_client


class ReferenceFreeEvaluator:
    mode = "reference_free"

    def __init__(
        self,
        metrics: list[str],
        judge_cfg: GenerationConfig,
        concurrency: int = 1,
        client: LLMClient | None = None,
    ) -> None:
        unknown = [m for m in metrics if m not in METRICS]
        if unknown:
            raise ValueError(f"unknown reference-free metric(s) {unknown}; available: {sorted(METRICS)}")
        if not metrics:
            raise ValueError("select at least one metric")
        self.metrics = list(dict.fromkeys(metrics))  # de-duplicate, keep order
        self.judge_cfg = judge_cfg
        self.concurrency = max(1, concurrency)
        # Injectable so tests can pass a fake judge instead of an HTTP client.
        self.client = client or build_llm_client(judge_cfg, label="judge")

    def health_check(self) -> tuple[bool, str]:
        return self.client.health_check()

    def describe(self) -> dict[str, Any]:
        return {"provider": self.judge_cfg.provider, "model": self.judge_cfg.model}

    def score_sample(self, sample: EvalSample) -> SampleResult:
        results = {name: run_metric(METRICS[name], sample, self.client) for name in self.metrics}
        return build_sample_result(sample, results)

    def evaluate(self, samples: list[EvalSample], on_result: OnResult) -> None:
        if self.concurrency == 1:
            for sample in samples:
                on_result(self.score_sample(sample))
            return
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            for result in pool.map(self.score_sample, samples):
                on_result(result)
