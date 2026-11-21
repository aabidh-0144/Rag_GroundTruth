"""The data shapes shared by both evaluation modes.

One result schema for Mode 1 and Mode 2 means the store, the CLI, the Streamlit app
and any later comparison code read every evaluation the same way.

Status vocabulary, used consistently:

* MetricResult.status:      "success" | "failed" | "skipped"
    - success: a score in [0, 1] was produced
    - failed:  the metric should have run but couldn't (LLM error, bad JSON, ...)
    - skipped: the metric does not apply to this sample (refusal answer, no
               contexts, no ground truth, ...) - `reason` says why
  A missing score is `None`, never 0.

* EvaluationResult.status:  "completed" | "partial" | "failed"
    - completed: every requested metric succeeded on at least one sample
    - partial:   some samples or metrics failed (or the run was interrupted)
    - failed:    nothing could be scored (e.g. the judge was unreachable)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

MetricStatus = Literal["success", "failed", "skipped"]
EvaluationStatus = Literal["completed", "partial", "failed"]
Mode = Literal["reference_free", "ragas"]


@dataclass(slots=True)
class EvalSample:
    """One thing to evaluate: a question, what was retrieved, what was answered."""

    sample_id: str
    question: str
    contexts: list[str]
    answer: str
    context_ids: list[str] = field(default_factory=list)
    ground_truth: str | None = None
    # Where the sample came from, e.g. {"run": "<run_id>"} or {"dataset": "<id>",
    # "verification_status": "verified"} - keeps synthetic and verified data apart.
    origin: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MetricResult:
    name: str
    status: MetricStatus
    score: float | None = None
    reason: str = ""
    # Metric-specific evidence: per-claim verdicts, per-chunk relevance, ...
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def success(cls, name: str, score: float, reason: str, **details: Any) -> MetricResult:
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{name} score {score} is outside [0, 1]")
        return cls(name=name, status="success", score=round(score, 4), reason=reason, details=details)

    @classmethod
    def skipped(cls, name: str, reason: str) -> MetricResult:
        return cls(name=name, status="skipped", reason=reason)

    @classmethod
    def failed(cls, name: str, error: str) -> MetricResult:
        return cls(name=name, status="failed", reason="metric could not be computed", error=error)


@dataclass(slots=True)
class SampleResult:
    sample_id: str
    question: str
    metrics: dict[str, MetricResult]
    # Group scores, None unless every metric in the group succeeded (see aggregate.py).
    retrieval_score: float | None = None
    generation_score: float | None = None
    refusal: bool = False
    origin: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "metrics": {name: m.to_dict() for name, m in self.metrics.items()},
            "retrieval_score": self.retrieval_score,
            "generation_score": self.generation_score,
            "refusal": self.refusal,
            "origin": self.origin,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SampleResult:
        return cls(
            sample_id=d["sample_id"],
            question=d["question"],
            metrics={name: MetricResult(**m) for name, m in d["metrics"].items()},
            retrieval_score=d.get("retrieval_score"),
            generation_score=d.get("generation_score"),
            refusal=d.get("refusal", False),
            origin=d.get("origin", {}),
        )


@dataclass(slots=True)
class MetricAggregate:
    """One metric across all samples. `mean` covers successes only."""

    mean: float | None
    n_success: int
    n_failed: int
    n_skipped: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvaluationResult:
    """The `summary.json` of one evaluation run."""

    evaluation_id: str
    mode: Mode
    status: EvaluationStatus
    metrics_requested: list[str]
    aggregates: dict[str, MetricAggregate]
    retrieval_score: float | None
    generation_score: float | None
    overall_score: float | None
    overall_score_method: str
    n_samples: int
    n_samples_evaluated: int
    source: dict[str, Any]
    judge: dict[str, Any]
    started_at: str
    execution_time_s: float
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["aggregates"] = {name: a.to_dict() for name, a in self.aggregates.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvaluationResult:
        d = dict(d)
        d["aggregates"] = {name: MetricAggregate(**a) for name, a in d["aggregates"].items()}
        return cls(**d)
