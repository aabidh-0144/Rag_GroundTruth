"""The Mode 1 metric contract and registry.

Same pattern as `retrievers/base.py`: every metric is one class with one method,
registered by name. Adding a metric = one new file + one entry in `METRICS`
(see reference_free/__init__.py). Nothing else changes.

`run_metric` is the only place a metric is invoked. It times the call and turns
any exception into a `failed` result, so one broken metric never takes the other
metrics on the same sample down with it.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

from groundtruth.evaluation.schemas import EvalSample, MetricResult
from groundtruth.generator import REFUSAL
from groundtruth.llm import LLMClient


@runtime_checkable
class Metric(Protocol):
    name: str

    def evaluate(self, sample: EvalSample, judge: LLMClient) -> MetricResult:
        """Score one sample. May return success/skipped; may raise on failure."""
        ...


def is_refusal(answer: str) -> bool:
    """The generator's explicit "the context doesn't contain this" reply."""
    return answer.strip().upper() == REFUSAL


def run_metric(metric: Metric, sample: EvalSample, judge: LLMClient) -> MetricResult:
    start = time.perf_counter()
    try:
        result = metric.evaluate(sample, judge)
    except Exception as exc:  # any failure is recorded, never propagated
        result = MetricResult.failed(metric.name, f"{type(exc).__name__}: {exc}")
    result.elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
    return result
