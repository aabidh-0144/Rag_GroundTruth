"""Live tests: real local models and a real Ollama judge. Skipped unless enabled.

    $env:GROUNDTRUTH_LIVE_TESTS = "1"
    .venv\\Scripts\\python.exe -m pytest -m live -v

Judge tests use the judge resolved from configs/evaluation.yaml (by default: the
generation model of configs/hybrid.yaml, including .env overrides). Expect each
test to take tens of seconds on a CPU.
"""

from __future__ import annotations

import os

import pytest

from groundtruth.evaluation.config import load_evaluation_config, resolve_judge

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("GROUNDTRUTH_LIVE_TESTS") != "1",
                       reason="set GROUNDTRUTH_LIVE_TESTS=1 to run against a real judge"),
]

GROUNDED = {
    "sample_id": "grounded",
    "question": "How many attention heads does the Transformer use?",
    "contexts": ["In this work we employ h = 8 parallel attention layers, or heads."],
    "answer": "The Transformer uses 8 parallel attention heads [1].",
}
HALLUCINATED = {
    **GROUNDED,
    "sample_id": "hallucinated",
    "answer": "The Transformer uses 8 attention heads and was trained for 9 years on Mars [1].",
}


@pytest.fixture(scope="module")
def judge_cfg():
    from groundtruth.llm import build_llm_client

    cfg = resolve_judge(load_evaluation_config().judge)
    healthy, message = build_llm_client(cfg, label="judge").health_check()
    if not healthy:
        pytest.skip(f"judge not available: {message}")
    return cfg


def test_live_mode1_scores_a_grounded_answer_above_a_hallucinated_one(judge_cfg, tmp_path):
    from groundtruth.evaluation.inputs import parse_samples
    from groundtruth.evaluation.orchestrator import run_evaluation
    from groundtruth.evaluation.reference_free.evaluator import ReferenceFreeEvaluator
    from groundtruth.evaluation.store import EvaluationStore

    evaluator = ReferenceFreeEvaluator(["faithfulness"], judge_cfg)
    run_evaluation(evaluator, parse_samples([GROUNDED, HALLUCINATED]), {"type": "live-test"},
                   store=EvaluationStore(tmp_path), progress=print)
    [evaluation] = list(tmp_path.iterdir())
    results = {r.sample_id: r for r in EvaluationStore(tmp_path).sample_results(evaluation.name)}
    grounded = results["grounded"].metrics["faithfulness"]
    hallucinated = results["hallucinated"].metrics["faithfulness"]
    assert grounded.status == hallucinated.status == "success", (grounded.error, hallucinated.error)
    assert grounded.score > hallucinated.score


def test_live_mode2_ragas_faithfulness_returns_a_score(judge_cfg):
    from groundtruth.evaluation.inputs import parse_samples
    from groundtruth.evaluation.ragas_mode.evaluator import RagasEvaluator

    cfg = load_evaluation_config()
    evaluator = RagasEvaluator(["faithfulness"], judge_cfg, cfg.ragas)
    out = []
    evaluator.evaluate(parse_samples([GROUNDED]), out.append)
    metric = out[0].metrics["faithfulness"]
    assert metric.status == "success", metric.error
    assert 0.0 <= metric.score <= 1.0


def test_real_semantic_similarity_runs_offline():
    """A real Ragas metric end to end: local bge-small embeddings, no LLM, no network.

    Needs no judge, but loads the embedding model (slow on a cold CPU), hence opt-in.
    """
    import asyncio

    from groundtruth.evaluation.ragas_mode import adapters

    emb = adapters.build_embeddings("BAAI/bge-small-en-v1.5")
    from ragas.metrics.collections import SemanticSimilarity

    metric = SemanticSimilarity(embeddings=emb)
    close = asyncio.run(metric.ascore(reference="RAG combines retrieval with generation.",
                                      response="RAG retrieves information and generates an answer."))
    far = asyncio.run(metric.ascore(reference="RAG combines retrieval with generation.",
                                    response="The capital of France is Paris."))
    assert 0.0 <= far.value < close.value <= 1.0
