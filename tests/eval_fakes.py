"""Test doubles for the evaluation tests. No network, no models, no Ragas LLM calls.

FakeJudge   stands in for an LLMClient. Replies are routed by which judge prompt
            is asking (matched on a phrase from the system prompt), and can be a
            list to script successive calls (e.g. bad JSON, then a repair).
FakeRagasMetric  stands in for a Ragas collections metric: async `ascore(**kwargs)`
            returning an object with `.value` / `.reason`, recording what it got.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from groundtruth.config import GenerationConfig
from groundtruth.evaluation.schemas import EvalSample

FAITHFULNESS = "fact-checker"
ANSWER_RELEVANCE = "directly addresses"
CONTEXT_RELEVANCE = "evaluate a retrieval system"

GOOD_REPLIES = {
    FAITHFULNESS: '{"claims": [{"claim": "RAG combines retrieval and generation", '
    '"verdict": "supported", "evidence": [1]}], "summary": "all supported"}',
    ANSWER_RELEVANCE: '{"score": 5, "explanation": "direct"}',
    CONTEXT_RELEVANCE: '{"blocks": [{"block": 1, "verdict": "relevant"}], "summary": "on topic"}',
}


class FakeJudge:
    def __init__(self, replies: dict[str, Any] | None = None, healthy: bool = True) -> None:
        self.cfg = GenerationConfig(model="fake-judge")
        self.replies = {**GOOD_REPLIES, **(replies or {})}
        self.healthy = healthy
        self.calls: list[tuple[str, str]] = []

    def health_check(self) -> tuple[bool, str]:
        return (True, "ok") if self.healthy else (False, "judge is down")

    def chat(self, system: str, user: str, *, json_mode: bool = False) -> str:
        self.calls.append((system, user))
        for key, reply in self.replies.items():
            if key in system:
                if isinstance(reply, list):
                    reply = reply.pop(0) if len(reply) > 1 else reply[0]
                if isinstance(reply, Exception):
                    raise reply
                return reply
        raise AssertionError(f"FakeJudge has no reply for system prompt: {system[:60]!r}")

    def calls_for(self, key: str) -> int:
        return sum(1 for system, _ in self.calls if key in system)


class FakeRagasMetric:
    def __init__(self, value: Any = 0.8, reason: str | None = "fine", error: Exception | None = None,
                 delay: float = 0.0) -> None:
        self.value, self.reason, self.error, self.delay = value, reason, error, delay
        self.received: list[dict[str, Any]] = []

    async def ascore(self, **kwargs: Any) -> Any:
        import asyncio

        self.received.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return SimpleNamespace(value=self.value, reason=self.reason)


def sample(**overrides: Any) -> EvalSample:
    fields = {
        "sample_id": "s1",
        "question": "What is RAG?",
        "contexts": ["RAG combines information retrieval with language generation."],
        "context_ids": ["chunk_a"],
        "answer": "RAG combines retrieval and generation [1].",
        "ground_truth": None,
    }
    fields.update(overrides)
    return EvalSample(**fields)
