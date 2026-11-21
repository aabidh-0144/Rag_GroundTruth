"""Our stable metric names -> Ragas 0.4.4 metric classes.

Verified against the installed package: these all come from
`ragas.metrics.collections` (importing metrics from `ragas.metrics` is deprecated
and removed in Ragas 1.0), and each `ascore()` takes exactly the keyword arguments
listed in `inputs`. Our names stay stable even where Ragas renamed a class
(ResponseRelevancy -> AnswerRelevancy, AnswerSimilarity -> SemanticSimilarity).

`requires_reference` metrics need a ground truth. They are *skipped* - never
scored against an invented reference - when a sample has none.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from groundtruth.evaluation.ragas_mode.adapters import import_ragas

# Ragas ascore() argument -> EvalSample attribute.
SAMPLE_FIELD = {
    "user_input": "question",
    "response": "answer",
    "retrieved_contexts": "contexts",
    "reference": "ground_truth",
}


@dataclass(frozen=True, slots=True)
class RagasMetricSpec:
    name: str
    ragas_class: str
    inputs: tuple[str, ...]
    requires_reference: bool
    needs_llm: bool = True
    needs_embeddings: bool = False
    # Judges the answer's own claims/relevance, so a refusal answer is skipped
    # (there is nothing to judge). Reference-based metrics still score refusals:
    # with a ground truth the question was answerable, so refusing is a real miss.
    skip_on_refusal: bool = False
    description: str = ""


REGISTRY: dict[str, RagasMetricSpec] = {
    spec.name: spec
    for spec in (
        RagasMetricSpec(
            "faithfulness",
            "Faithfulness",
            ("user_input", "response", "retrieved_contexts"),
            requires_reference=False,
            skip_on_refusal=True,
            description="Fraction of the answer's claims supported by the contexts.",
        ),
        RagasMetricSpec(
            "response_relevancy",
            "AnswerRelevancy",
            ("user_input", "response"),
            requires_reference=False,
            needs_embeddings=True,
            skip_on_refusal=True,
            description="How well the answer addresses the question (question regeneration + similarity).",
        ),
        RagasMetricSpec(
            "context_precision",
            "ContextPrecisionWithoutReference",
            ("user_input", "response", "retrieved_contexts"),
            requires_reference=False,
            skip_on_refusal=True,
            description="Whether useful contexts are ranked above useless ones, judged against the answer.",
        ),
        RagasMetricSpec(
            "context_precision_with_reference",
            "ContextPrecisionWithReference",
            ("user_input", "reference", "retrieved_contexts"),
            requires_reference=True,
            description="Whether useful contexts are ranked first, judged against the ground truth.",
        ),
        RagasMetricSpec(
            "context_recall",
            "ContextRecall",
            ("user_input", "retrieved_contexts", "reference"),
            requires_reference=True,
            description="Fraction of the ground truth's claims that the contexts support.",
        ),
        RagasMetricSpec(
            "answer_correctness",
            "AnswerCorrectness",
            ("user_input", "response", "reference"),
            requires_reference=True,
            needs_embeddings=True,
            description="Factual + semantic agreement between the answer and the ground truth.",
        ),
        RagasMetricSpec(
            "answer_similarity",
            "SemanticSimilarity",
            ("reference", "response"),
            requires_reference=True,
            needs_llm=False,
            needs_embeddings=True,
            description="Embedding similarity between the answer and the ground truth.",
        ),
    )
}

REFERENCE_FREE = tuple(n for n, s in REGISTRY.items() if not s.requires_reference)
REFERENCE_BASED = tuple(n for n, s in REGISTRY.items() if s.requires_reference)


def validate_names(names: list[str]) -> list[str]:
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"unknown Ragas metric(s) {unknown}; available: {sorted(REGISTRY)}")
    if not names:
        raise ValueError("select at least one metric")
    return list(dict.fromkeys(names))


def instantiate(spec: RagasMetricSpec, llm: Any, embeddings: Any) -> Any:
    import_ragas()
    from ragas.metrics import collections

    cls = getattr(collections, spec.ragas_class)
    kwargs: dict[str, Any] = {}
    if spec.needs_llm:
        kwargs["llm"] = llm
    if spec.needs_embeddings:
        kwargs["embeddings"] = embeddings
    return cls(**kwargs)
