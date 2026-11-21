"""How scores are combined. Every rule an aggregate number depends on lives here.

Principles:
1. A missing score is never 0. Failed and skipped metrics are excluded from
   means and counted separately (n_failed, n_skipped).
2. "skipped" means "does not apply" (refusal answer, no contexts, no ground truth),
   so it is excluded without penalty. "failed" means "should have been scored but
   couldn't", which marks the run partial.
3. Metrics are only averaged together within a group that measures the same
   thing, mirroring the project's two questions:
     retrieval  - did the retriever fetch the right information?
     generation - is the answer faithful to it and on-topic?
   Every metric is on a 0-1 "higher is better" scale.

Rules:
- metric mean        = mean of that metric's successful scores (None if none)
- sample group score = mean of the sample's successful metrics in the group;
                       None if any metric in the group FAILED on that sample
                       (a partial group average would hide the failure), or if
                       none succeeded
- run group score    = mean of the group's metric means that are not None
- overall_score      = mean of the run group scores that are not None;
                       None if any requested metric was applicable but never
                       succeeded (all its results failed) - an overall number
                       that silently drops a whole broken metric would mislead
- status             = "failed"    if no metric succeeded on any sample
                       "partial"   if any metric result failed, or samples were
                                   left unevaluated
                       "completed" otherwise
"""

from __future__ import annotations

from statistics import fmean

from groundtruth.evaluation.schemas import EvaluationStatus, MetricAggregate, SampleResult

RETRIEVAL_METRICS = frozenset(
    {"context_relevance", "context_precision", "context_precision_with_reference", "context_recall"}
)
GENERATION_METRICS = frozenset(
    {"faithfulness", "answer_relevance", "response_relevancy", "answer_correctness", "answer_similarity"}
)

OVERALL_SCORE_METHOD = (
    "mean of the retrieval and generation group scores; each group score is the mean of its "
    "metrics' means over successful samples only. Failed and skipped results are excluded, "
    "never counted as 0. Null if any requested metric applied but never succeeded."
)


def group_of(metric: str) -> str:
    if metric in RETRIEVAL_METRICS:
        return "retrieval"
    if metric in GENERATION_METRICS:
        return "generation"
    raise ValueError(f"metric {metric!r} is not assigned to a group in aggregate.py")


def _mean(values: list[float]) -> float | None:
    return round(fmean(values), 4) if values else None


def sample_group_scores(result: SampleResult) -> SampleResult:
    """Fill in a sample's retrieval_score / generation_score in place."""
    for group in ("retrieval", "generation"):
        metrics = [m for name, m in result.metrics.items() if group_of(name) == group]
        if any(m.status == "failed" for m in metrics):
            score = None
        else:
            score = _mean([m.score for m in metrics if m.status == "success" and m.score is not None])
        setattr(result, f"{group}_score", score)
    return result


def aggregate_metrics(results: list[SampleResult], metrics: list[str]) -> dict[str, MetricAggregate]:
    aggregates = {}
    for name in metrics:
        per_sample = [r.metrics[name] for r in results if name in r.metrics]
        aggregates[name] = MetricAggregate(
            mean=_mean([m.score for m in per_sample if m.status == "success" and m.score is not None]),
            n_success=sum(m.status == "success" for m in per_sample),
            n_failed=sum(m.status == "failed" for m in per_sample),
            n_skipped=sum(m.status == "skipped" for m in per_sample),
        )
    return aggregates


def summarize(
    aggregates: dict[str, MetricAggregate], n_samples: int, n_evaluated: int
) -> tuple[float | None, float | None, float | None, EvaluationStatus]:
    """(retrieval_score, generation_score, overall_score, status)."""
    groups: dict[str, float | None] = {}
    for group in ("retrieval", "generation"):
        means = [a.mean for name, a in aggregates.items() if group_of(name) == group and a.mean is not None]
        groups[group] = _mean(means)

    broken = any(a.n_success == 0 and a.n_failed > 0 for a in aggregates.values())
    present = [s for s in groups.values() if s is not None]
    overall = None if broken else _mean(present)

    any_success = any(a.n_success > 0 for a in aggregates.values())
    any_failed = any(a.n_failed > 0 for a in aggregates.values())
    if not any_success and (any_failed or n_evaluated < n_samples or n_evaluated == 0):
        status: EvaluationStatus = "failed"
    elif any_failed or n_evaluated < n_samples:
        status = "partial"
    else:
        status = "completed"
    return groups["retrieval"], groups["generation"], overall, status
