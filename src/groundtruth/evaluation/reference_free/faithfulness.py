"""Faithfulness: is every claim in the answer supported by the retrieved context?

Method: the judge splits the answer into atomic claims and labels each one
supported / unsupported / contradicted against the numbered context blocks.

    score = supported claims / total claims

Unsupported and contradicted claims are returned in `details` - they are the
hallucination candidates a reviewer should look at first.

Skipped (not scored) when:
- the answer is empty
- the answer is the INSUFFICIENT_CONTEXT refusal (nothing was claimed)
- there are no contexts (there is nothing to check claims against)
- the judge finds no factual claims in the answer
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from groundtruth.evaluation import prompts
from groundtruth.evaluation.judge import judge_json
from groundtruth.evaluation.reference_free.base import is_refusal
from groundtruth.evaluation.schemas import EvalSample, MetricResult
from groundtruth.llm import LLMClient


class Claim(BaseModel):
    claim: str
    verdict: Literal["supported", "unsupported", "contradicted"]
    evidence: list[int] = []
    explanation: str = ""


class FaithfulnessVerdict(BaseModel):
    claims: list[Claim]
    summary: str = ""


class Faithfulness:
    name = "faithfulness"

    def evaluate(self, sample: EvalSample, judge: LLMClient) -> MetricResult:
        if not sample.answer.strip():
            return MetricResult.skipped(self.name, "no answer to check")
        if is_refusal(sample.answer):
            return MetricResult.skipped(self.name, "answer is a refusal; it makes no claims")
        if not sample.contexts:
            return MetricResult.skipped(self.name, "no contexts to check the answer against")

        verdict = judge_json(
            judge,
            prompts.FAITHFULNESS_SYSTEM,
            prompts.FAITHFULNESS_USER.format(
                contexts=prompts.render_contexts(sample.contexts),
                question=sample.question,
                answer=sample.answer,
            ),
            FaithfulnessVerdict,
        )
        if not verdict.claims:
            return MetricResult.skipped(self.name, "the judge found no factual claims to verify")

        supported = [c for c in verdict.claims if c.verdict == "supported"]
        flagged = [c.model_dump() for c in verdict.claims if c.verdict != "supported"]
        score = len(supported) / len(verdict.claims)
        reason = (
            f"{len(supported)}/{len(verdict.claims)} claims supported by the context."
            + (f" {verdict.summary}" if verdict.summary else "")
        )
        return MetricResult.success(
            self.name,
            score,
            reason,
            claims=[c.model_dump() for c in verdict.claims],
            unsupported_claims=flagged,
        )
