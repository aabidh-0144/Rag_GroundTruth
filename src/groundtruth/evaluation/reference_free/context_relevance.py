"""Context relevance: are the retrieved chunks relevant to the question?

This is the retrieval-side metric - it answers "did the retriever fetch the right
information?" without needing labelled relevant documents.

Method: the judge gives every context block one verdict:
relevant (1.0) / partial (0.5) / irrelevant (0.0).

    score = mean verdict value over all blocks

`details.blocks` lists each block's verdict with its chunk_id (when known), so the
irrelevant chunks the retriever surfaced are visible, not just averaged away.

Skipped when there are no contexts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from groundtruth.evaluation import prompts
from groundtruth.evaluation.judge import judge_json
from groundtruth.evaluation.schemas import EvalSample, MetricResult
from groundtruth.llm import LLMClient

VERDICT_VALUE = {"relevant": 1.0, "partial": 0.5, "irrelevant": 0.0}


class BlockVerdict(BaseModel):
    block: int
    verdict: Literal["relevant", "partial", "irrelevant"]
    explanation: str = ""


class ContextRelevanceVerdict(BaseModel):
    blocks: list[BlockVerdict]
    summary: str = ""


def _require_every_block(n_blocks: int):
    """A verdict must rate blocks 1..n exactly once - otherwise the mean is a lie."""

    def check(verdict: ContextRelevanceVerdict) -> None:
        got = sorted(b.block for b in verdict.blocks)
        expected = list(range(1, n_blocks + 1))
        if got != expected:
            raise ValueError(f"expected one verdict for each block {expected}, got {got}")

    return check


class ContextRelevance:
    name = "context_relevance"

    def evaluate(self, sample: EvalSample, judge: LLMClient) -> MetricResult:
        if not sample.contexts:
            return MetricResult.skipped(self.name, "no contexts were retrieved")

        verdict = judge_json(
            judge,
            prompts.CONTEXT_RELEVANCE_SYSTEM,
            prompts.CONTEXT_RELEVANCE_USER.format(
                question=sample.question,
                contexts=prompts.render_contexts(sample.contexts),
            ),
            ContextRelevanceVerdict,
            check=_require_every_block(len(sample.contexts)),
        )

        blocks = []
        for b in sorted(verdict.blocks, key=lambda b: b.block):
            index = b.block - 1
            blocks.append(
                {
                    "block": b.block,
                    "chunk_id": sample.context_ids[index] if index < len(sample.context_ids) else None,
                    "verdict": b.verdict,
                    "explanation": b.explanation,
                }
            )
        score = sum(VERDICT_VALUE[b["verdict"]] for b in blocks) / len(blocks)
        n_relevant = sum(1 for b in blocks if b["verdict"] == "relevant")
        reason = (
            f"{n_relevant}/{len(blocks)} blocks relevant."
            + (f" {verdict.summary}" if verdict.summary else "")
        )
        return MetricResult.success(
            self.name,
            score,
            reason,
            blocks=blocks,
            irrelevant_chunk_ids=[b["chunk_id"] for b in blocks if b["verdict"] == "irrelevant"],
        )
