"""Mode 1 - reference-free evaluation with an LLM judge.

Needs only (question, contexts, answer). No golden dataset, no expected answers,
no labelled relevant documents.

| Metric            | Question it answers                                  | Side       |
|-------------------|------------------------------------------------------|------------|
| faithfulness      | Is every claim in the answer backed by the contexts? | generation |
| answer_relevance  | Does the answer address the question?                | generation |
| context_relevance | Are the retrieved chunks relevant to the question?   | retrieval  |

To add a metric: write a class with `name` and `evaluate(sample, judge)` in its own
file, then register it below.
"""

from __future__ import annotations

from groundtruth.evaluation.reference_free.answer_relevance import AnswerRelevance
from groundtruth.evaluation.reference_free.base import Metric, is_refusal, run_metric
from groundtruth.evaluation.reference_free.context_relevance import ContextRelevance
from groundtruth.evaluation.reference_free.faithfulness import Faithfulness

METRICS: dict[str, Metric] = {
    m.name: m for m in (Faithfulness(), AnswerRelevance(), ContextRelevance())
}

__all__ = ["METRICS", "Metric", "is_refusal", "run_metric"]
