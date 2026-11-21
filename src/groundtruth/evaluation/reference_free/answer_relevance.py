"""Answer relevance: does the answer directly and completely address the question?

Method: the judge rates the answer on a 1-5 rubric (defined in prompts.py) and
lists missing and irrelevant parts. Correctness is NOT judged here - that is
faithfulness's job - so a confidently wrong but on-topic answer can still score
high on relevance. The two metrics are meant to be read together.

    score = (rating - 1) / 4        ->  1 maps to 0.0, 5 maps to 1.0

Skipped when the answer is empty or is the INSUFFICIENT_CONTEXT refusal: a refusal
is the *correct* behaviour when retrieval failed, so scoring it as an irrelevant
answer would punish the generator for the retriever's miss.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from groundtruth.evaluation import prompts
from groundtruth.evaluation.judge import judge_json
from groundtruth.evaluation.reference_free.base import is_refusal
from groundtruth.evaluation.schemas import EvalSample, MetricResult
from groundtruth.llm import LLMClient


class RelevanceVerdict(BaseModel):
    score: int = Field(ge=1, le=5)
    missing_parts: list[str] = []
    irrelevant_parts: list[str] = []
    explanation: str = ""


class AnswerRelevance:
    name = "answer_relevance"

    def evaluate(self, sample: EvalSample, judge: LLMClient) -> MetricResult:
        if not sample.answer.strip():
            return MetricResult.skipped(self.name, "no answer to rate")
        if is_refusal(sample.answer):
            return MetricResult.skipped(
                self.name, "answer is a refusal; relevance does not apply"
            )

        verdict = judge_json(
            judge,
            prompts.ANSWER_RELEVANCE_SYSTEM,
            prompts.ANSWER_RELEVANCE_USER.format(question=sample.question, answer=sample.answer),
            RelevanceVerdict,
        )
        return MetricResult.success(
            self.name,
            (verdict.score - 1) / 4,
            f"Rated {verdict.score}/5. {verdict.explanation}".strip(),
            rating=verdict.score,
            missing_parts=verdict.missing_parts,
            irrelevant_parts=verdict.irrelevant_parts,
        )
