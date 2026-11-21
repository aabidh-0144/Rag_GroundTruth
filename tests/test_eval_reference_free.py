"""Mode 1: the three judge metrics, the judge helper, and the evaluator."""

from __future__ import annotations

import pytest
from eval_fakes import ANSWER_RELEVANCE, CONTEXT_RELEVANCE, FAITHFULNESS, FakeJudge, sample

from groundtruth.config import GenerationConfig
from groundtruth.evaluation.judge import JudgeError, extract_json_object, judge_json
from groundtruth.evaluation.reference_free import METRICS, run_metric
from groundtruth.evaluation.reference_free.evaluator import ReferenceFreeEvaluator
from groundtruth.evaluation.reference_free.faithfulness import FaithfulnessVerdict

faith, relevance, context = METRICS["faithfulness"], METRICS["answer_relevance"], METRICS["context_relevance"]


# --- faithfulness -------------------------------------------------------------


def test_faithfulness_scores_supported_over_total_claims():
    judge = FakeJudge({FAITHFULNESS: '{"claims": ['
                       '{"claim": "a", "verdict": "supported", "evidence": [1]},'
                       '{"claim": "b", "verdict": "unsupported"},'
                       '{"claim": "c", "verdict": "contradicted"},'
                       '{"claim": "d", "verdict": "supported"}]}'})
    result = run_metric(faith, sample(), judge)
    assert result.status == "success"
    assert result.score == 0.5
    assert [c["claim"] for c in result.details["unsupported_claims"]] == ["b", "c"]


def test_faithfulness_skips_a_refusal_without_calling_the_judge():
    judge = FakeJudge()
    result = run_metric(faith, sample(answer="INSUFFICIENT_CONTEXT"), judge)
    assert result.status == "skipped" and result.score is None
    assert judge.calls == []


def test_faithfulness_skips_when_there_are_no_contexts():
    result = run_metric(faith, sample(contexts=[]), FakeJudge())
    assert result.status == "skipped"
    assert "no contexts" in result.reason


def test_faithfulness_skips_an_answer_with_no_factual_claims():
    result = run_metric(faith, sample(), FakeJudge({FAITHFULNESS: '{"claims": []}'}))
    assert result.status == "skipped"


def test_empty_answer_is_skipped_not_zero():
    result = run_metric(faith, sample(answer="   "), FakeJudge())
    assert result.status == "skipped" and result.score is None


# --- answer relevance --------------------------------------------------------


@pytest.mark.parametrize("rating, expected", [(1, 0.0), (3, 0.5), (5, 1.0)])
def test_answer_relevance_normalizes_the_1_to_5_rubric(rating, expected):
    result = run_metric(relevance, sample(), FakeJudge({ANSWER_RELEVANCE: f'{{"score": {rating}}}'}))
    assert result.score == expected


def test_answer_relevance_out_of_range_rating_is_repaired_then_failed():
    judge = FakeJudge({ANSWER_RELEVANCE: ['{"score": 9}', '{"score": 7}']})
    result = run_metric(relevance, sample(), judge)
    assert result.status == "failed"
    assert result.score is None
    assert judge.calls_for(ANSWER_RELEVANCE) == 2  # original + one repair


def test_answer_relevance_skips_a_refusal():
    assert run_metric(relevance, sample(answer="INSUFFICIENT_CONTEXT"), FakeJudge()).status == "skipped"


# --- context relevance -------------------------------------------------------


def test_context_relevance_averages_per_block_verdicts_and_keeps_chunk_ids():
    judge = FakeJudge({CONTEXT_RELEVANCE: '{"blocks": ['
                       '{"block": 1, "verdict": "relevant"},'
                       '{"block": 2, "verdict": "partial"},'
                       '{"block": 3, "verdict": "irrelevant"}]}'})
    s = sample(contexts=["a", "b", "c"], context_ids=["c1", "c2", "c3"])
    result = run_metric(context, s, judge)
    assert result.score == 0.5
    assert result.details["irrelevant_chunk_ids"] == ["c3"]
    assert [b["chunk_id"] for b in result.details["blocks"]] == ["c1", "c2", "c3"]


def test_context_relevance_rejects_a_verdict_that_skips_a_block():
    """A verdict missing block 2 would average over the wrong denominator."""
    incomplete = '{"blocks": [{"block": 1, "verdict": "relevant"}]}'
    complete = '{"blocks": [{"block": 1, "verdict": "relevant"}, {"block": 2, "verdict": "irrelevant"}]}'
    judge = FakeJudge({CONTEXT_RELEVANCE: [incomplete, complete]})
    result = run_metric(context, sample(contexts=["a", "b"], context_ids=[]), judge)
    assert result.status == "success" and result.score == 0.5
    assert judge.calls_for(CONTEXT_RELEVANCE) == 2


def test_context_relevance_still_scores_a_refusal():
    """Retrieval quality is independent of whether the generator refused."""
    assert run_metric(context, sample(answer="INSUFFICIENT_CONTEXT"), FakeJudge()).status == "success"


def test_context_relevance_skips_when_nothing_was_retrieved():
    assert run_metric(context, sample(contexts=[]), FakeJudge()).status == "skipped"


# --- judge helper ------------------------------------------------------------


def test_extract_json_handles_code_fences_and_surrounding_prose():
    reply = 'Sure! Here you go:\n```json\n{"claims": []}\n```\nHope that helps.'
    assert extract_json_object(reply) == '{"claims": []}'


def test_judge_repairs_malformed_json_once():
    judge = FakeJudge({FAITHFULNESS: ["not json at all", '{"claims": []}']})
    verdict = judge_json(judge, "fact-checker", "user", FaithfulnessVerdict)
    assert verdict.claims == []


def test_judge_raises_after_a_failed_repair():
    judge = FakeJudge({FAITHFULNESS: ["nope", "still nope"]})
    with pytest.raises(JudgeError, match="after one repair"):
        judge_json(judge, "fact-checker", "user", FaithfulnessVerdict)


def test_repair_prompt_repeats_the_original_task():
    """Otherwise the judge would 'fix' JSON without the contexts in front of it."""
    judge = FakeJudge({FAITHFULNESS: ["nope", '{"claims": []}']})
    judge_json(judge, "fact-checker", "ORIGINAL TASK TEXT", FaithfulnessVerdict)
    assert "ORIGINAL TASK TEXT" in judge.calls[1][1]


# --- failures are isolated ---------------------------------------------------


def test_llm_failure_marks_the_metric_failed_with_the_error():
    judge = FakeJudge({FAITHFULNESS: RuntimeError("Ollama generation failed after retries")})
    result = run_metric(faith, sample(), judge)
    assert result.status == "failed"
    assert result.score is None
    assert "failed after retries" in result.error


def test_one_failing_metric_does_not_affect_the_others():
    judge = FakeJudge({FAITHFULNESS: TimeoutError("judge timed out")})
    evaluator = ReferenceFreeEvaluator(list(METRICS), GenerationConfig(), client=judge)
    result = evaluator.score_sample(sample())
    assert result.metrics["faithfulness"].status == "failed"
    assert result.metrics["answer_relevance"].status == "success"
    assert result.metrics["context_relevance"].status == "success"
    # Generation group can't be summarised honestly with one of its metrics broken.
    assert result.generation_score is None
    assert result.retrieval_score == 1.0


# --- evaluator configuration -------------------------------------------------


def test_evaluator_runs_only_the_selected_metrics():
    judge = FakeJudge()
    evaluator = ReferenceFreeEvaluator(["context_relevance"], GenerationConfig(), client=judge)
    result = evaluator.score_sample(sample())
    assert list(result.metrics) == ["context_relevance"]
    assert judge.calls_for(FAITHFULNESS) == 0


def test_unknown_metric_is_rejected():
    with pytest.raises(ValueError, match="unknown reference-free metric"):
        ReferenceFreeEvaluator(["faithfulness", "bogus"], GenerationConfig(), client=FakeJudge())


def test_concurrent_evaluation_returns_every_sample():
    judge = FakeJudge()
    evaluator = ReferenceFreeEvaluator(["answer_relevance"], GenerationConfig(), concurrency=3, client=judge)
    seen = []
    evaluator.evaluate([sample(sample_id=f"s{i}") for i in range(6)], seen.append)
    assert sorted(r.sample_id for r in seen) == [f"s{i}" for i in range(6)]
