"""Runs one evaluation end to end, for either mode.

    samples ──► Evaluator (Mode 1 or Mode 2) ──► SampleResult per sample
                                                    │ appended to evals/<id>/results.jsonl
                                                    ▼
                                    aggregate.py ──► EvaluationResult ──► summary.json

The orchestrator knows nothing about how a metric is computed. Anything that
implements `Evaluator` plugs in; Mode 1 (reference_free/evaluator.py) and Mode 2
(ragas_mode/evaluator.py) are the two implementations. Everything mode-independent
lives here: the judge health check, resuming, incremental persistence, timing,
interruption handling and aggregation.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from typing import Any, Callable, Protocol

from groundtruth.evaluation.aggregate import (
    OVERALL_SCORE_METHOD,
    aggregate_metrics,
    sample_group_scores,
    summarize,
)
from groundtruth.evaluation.schemas import EvalSample, EvaluationResult, MetricResult, SampleResult
from groundtruth.evaluation.store import EvaluationStore

log = logging.getLogger(__name__)

OnResult = Callable[[SampleResult], None]


class Evaluator(Protocol):
    """One evaluation mode. Implementations: Mode 1 and Mode 2."""

    mode: str
    metrics: list[str]

    def health_check(self) -> tuple[bool, str]:
        """Can the judge be reached? Checked once, before any sample runs."""
        ...

    def describe(self) -> dict[str, Any]:
        """Judge/provider details recorded in the summary (never secrets)."""
        ...

    def evaluate(self, samples: list[EvalSample], on_result: OnResult) -> None:
        """Score every sample, calling `on_result` as each one finishes."""
        ...


def build_sample_result(sample: EvalSample, metrics: dict[str, MetricResult]) -> SampleResult:
    """Shared by both modes so a sample result is assembled one way only."""
    from groundtruth.evaluation.reference_free.base import is_refusal

    result = SampleResult(
        sample_id=sample.sample_id,
        question=sample.question,
        metrics=metrics,
        refusal=is_refusal(sample.answer),
        origin=sample.origin,
    )
    return sample_group_scores(result)


def run_evaluation(
    evaluator: Evaluator,
    samples: list[EvalSample],
    source: dict[str, Any],
    *,
    store: EvaluationStore | None = None,
    resume_id: str | None = None,
    extra_config: dict[str, Any] | None = None,
    progress: Callable[[str], None] = print,
) -> EvaluationResult:
    store = store or EvaluationStore()
    if not samples:
        raise ValueError("no samples to evaluate")

    healthy, message = evaluator.health_check()
    if not healthy:
        raise RuntimeError(f"judge unavailable: {message}")

    previous_time = 0.0
    if resume_id:
        evaluation_id = resume_id
        saved = store.config(evaluation_id)
        if saved.get("mode") != evaluator.mode:
            raise ValueError(
                f"cannot resume {evaluation_id!r}: it is a {saved.get('mode')} evaluation"
            )
        done = {r.sample_id for r in store.sample_results(evaluation_id)}
        try:
            previous_time = store.get(evaluation_id).execution_time_s
        except FileNotFoundError:
            pass
        started_at = saved.get("started_at") or dt.datetime.now().isoformat(timespec="seconds")
    else:
        evaluation_id = store.new_id(evaluator.mode)
        done = set()
        started_at = dt.datetime.now().isoformat(timespec="seconds")
        store.create(
            evaluation_id,
            {
                "evaluation_id": evaluation_id,
                "mode": evaluator.mode,
                "metrics": evaluator.metrics,
                "judge": evaluator.describe(),
                "source": source,
                "n_samples": len(samples),
                "started_at": started_at,
                **(extra_config or {}),
            },
        )

    pending = [s for s in samples if s.sample_id not in done]
    progress(
        f"evaluation {evaluation_id}: {len(pending)} of {len(samples)} sample(s) to score "
        f"with {', '.join(evaluator.metrics)}"
    )

    finished = [len(done)]

    def on_result(result: SampleResult) -> None:
        store.append_result(evaluation_id, result)
        finished[0] += 1
        statuses = ", ".join(f"{n}={m.status}" for n, m in result.metrics.items())
        progress(f"  [{finished[0]}/{len(samples)}] {result.sample_id}: {statuses}")

    errors: list[str] = []
    start = time.perf_counter()
    try:
        evaluator.evaluate(pending, on_result)
    except KeyboardInterrupt:
        errors.append(f"interrupted - resume with --resume {evaluation_id}")
        progress(f"\ninterrupted. Finished samples are saved; resume with --resume {evaluation_id}")
    except Exception as exc:  # record, summarize what finished, don't lose it
        log.exception("evaluation %s aborted", evaluation_id)
        errors.append(f"aborted: {type(exc).__name__}: {exc}")
    elapsed = time.perf_counter() - start

    wanted = {s.sample_id for s in samples}
    results = [r for r in store.sample_results(evaluation_id) if r.sample_id in wanted]
    aggregates = aggregate_metrics(results, evaluator.metrics)
    retrieval, generation, overall, status = summarize(aggregates, len(samples), len(results))

    summary = EvaluationResult(
        evaluation_id=evaluation_id,
        mode=evaluator.mode,  # type: ignore[arg-type]
        status=status,
        metrics_requested=evaluator.metrics,
        aggregates=aggregates,
        retrieval_score=retrieval,
        generation_score=generation,
        overall_score=overall,
        overall_score_method=OVERALL_SCORE_METHOD,
        n_samples=len(samples),
        n_samples_evaluated=len(results),
        source=source,
        judge=evaluator.describe(),
        started_at=started_at,
        execution_time_s=round(previous_time + elapsed, 2),
        errors=errors,
    )
    store.write_summary(summary)
    return summary
