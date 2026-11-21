"""Aggregation rules, persistence, history, resume, and partial failures."""

from __future__ import annotations

import pytest
from eval_fakes import FAITHFULNESS, FakeJudge, sample

from groundtruth.config import GenerationConfig
from groundtruth.evaluation.aggregate import aggregate_metrics, sample_group_scores, summarize
from groundtruth.evaluation.orchestrator import run_evaluation
from groundtruth.evaluation.reference_free.evaluator import ReferenceFreeEvaluator
from groundtruth.evaluation.schemas import MetricResult, SampleResult
from groundtruth.evaluation.store import EvaluationStore

ALL = ["faithfulness", "answer_relevance", "context_relevance"]


def _evaluator(judge=None, metrics=ALL):
    return ReferenceFreeEvaluator(metrics, GenerationConfig(), client=judge or FakeJudge())


def _run(tmp_path, samples, judge=None, metrics=ALL, **kwargs):
    return run_evaluation(
        _evaluator(judge, metrics), samples, {"type": "test"},
        store=EvaluationStore(tmp_path), progress=lambda m: None, **kwargs,
    )


# --- aggregation rules -------------------------------------------------------


def _sample_result(**metrics: MetricResult) -> SampleResult:
    return sample_group_scores(SampleResult(sample_id="s", question="q", metrics=metrics))


def test_failed_and_skipped_metrics_are_never_averaged_in_as_zero():
    results = [
        _sample_result(faithfulness=MetricResult.success("faithfulness", 0.8, "")),
        _sample_result(faithfulness=MetricResult.failed("faithfulness", "boom")),
        _sample_result(faithfulness=MetricResult.skipped("faithfulness", "refusal")),
    ]
    agg = aggregate_metrics(results, ["faithfulness"])["faithfulness"]
    assert agg.mean == 0.8
    assert (agg.n_success, agg.n_failed, agg.n_skipped) == (1, 1, 1)


def test_overall_is_null_when_a_requested_metric_never_succeeded():
    results = [_sample_result(
        faithfulness=MetricResult.failed("faithfulness", "boom"),
        context_relevance=MetricResult.success("context_relevance", 1.0, ""),
    )]
    _, _, overall, status = summarize(aggregate_metrics(results, ["faithfulness", "context_relevance"]), 1, 1)
    assert overall is None
    assert status == "partial"


def test_an_all_skipped_metric_does_not_block_the_overall_score():
    """No ground truth -> reference metric skipped everywhere: not applicable, not broken."""
    results = [_sample_result(
        context_recall=MetricResult.skipped("context_recall", "no ground_truth"),
        faithfulness=MetricResult.success("faithfulness", 0.6, ""),
    )]
    _, generation, overall, status = summarize(aggregate_metrics(results, ["context_recall", "faithfulness"]), 1, 1)
    assert generation == 0.6 and overall == 0.6
    assert status == "completed"


def test_overall_is_the_mean_of_the_two_group_scores():
    results = [_sample_result(
        faithfulness=MetricResult.success("faithfulness", 1.0, ""),
        answer_relevance=MetricResult.success("answer_relevance", 0.5, ""),
        context_relevance=MetricResult.success("context_relevance", 0.25, ""),
    )]
    retrieval, generation, overall, _ = summarize(aggregate_metrics(results, ALL), 1, 1)
    assert (retrieval, generation, overall) == (0.25, 0.75, 0.5)


# --- end to end through the orchestrator --------------------------------------


def test_successful_evaluation_is_persisted_and_retrievable(tmp_path):
    result = _run(tmp_path, [sample(sample_id="a"), sample(sample_id="b")])
    store = EvaluationStore(tmp_path)

    assert result.status == "completed"
    assert result.overall_score == 1.0
    loaded = store.get(result.evaluation_id)
    assert loaded.to_dict() == result.to_dict()
    assert {r.sample_id for r in store.sample_results(result.evaluation_id)} == {"a", "b"}
    assert store.config(result.evaluation_id)["mode"] == "reference_free"


def test_partial_metric_failure_keeps_the_other_scores(tmp_path):
    judge = FakeJudge({FAITHFULNESS: RuntimeError("LLM down")})
    result = _run(tmp_path, [sample()], judge=judge)
    assert result.status == "partial"
    assert result.aggregates["faithfulness"].n_failed == 1
    assert result.aggregates["answer_relevance"].mean == 1.0
    assert result.overall_score is None


def test_unreachable_judge_fails_fast_without_creating_an_evaluation(tmp_path):
    with pytest.raises(RuntimeError, match="judge unavailable"):
        _run(tmp_path, [sample()], judge=FakeJudge(healthy=False))
    assert list(tmp_path.iterdir()) == []


def test_history_is_newest_first_and_filterable(tmp_path):
    first = _run(tmp_path, [sample()])
    second = _run(tmp_path, [sample()])
    store = EvaluationStore(tmp_path)
    assert [r.evaluation_id for r in store.history()] == [second.evaluation_id, first.evaluation_id]
    assert store.history(limit=1)[0].evaluation_id == second.evaluation_id
    assert store.history(mode="ragas") == []


def test_interrupted_evaluation_saves_finished_samples_and_resumes(tmp_path, monkeypatch):
    samples = [sample(sample_id=f"s{i}") for i in range(3)]
    evaluator = _evaluator()
    original = evaluator.score_sample
    calls = {"n": 0}

    def flaky(s):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return original(s)

    monkeypatch.setattr(evaluator, "score_sample", flaky)
    store = EvaluationStore(tmp_path)
    first = run_evaluation(evaluator, samples, {"type": "test"}, store=store, progress=lambda m: None)
    assert first.status == "partial"
    assert first.n_samples_evaluated == 1
    assert "resume" in first.errors[0]

    resumed = run_evaluation(
        _evaluator(), samples, {"type": "test"}, store=store,
        resume_id=first.evaluation_id, progress=lambda m: None,
    )
    assert resumed.evaluation_id == first.evaluation_id
    assert resumed.status == "completed"
    assert resumed.n_samples_evaluated == 3


def test_resume_refuses_an_evaluation_of_a_different_mode(tmp_path):
    first = _run(tmp_path, [sample()])
    store = EvaluationStore(tmp_path)

    class OtherMode(ReferenceFreeEvaluator):
        mode = "ragas"

    with pytest.raises(ValueError, match="cannot resume"):
        run_evaluation(OtherMode(ALL, GenerationConfig(), client=FakeJudge()), [sample()], {},
                       store=store, resume_id=first.evaluation_id, progress=lambda m: None)


@pytest.mark.parametrize("bad", ["../runs", "a/b", ""])
def test_store_rejects_ids_that_escape_the_evals_directory(tmp_path, bad):
    with pytest.raises(ValueError, match="invalid evaluation id"):
        EvaluationStore(tmp_path).get(bad)


def test_missing_evaluation_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        EvaluationStore(tmp_path).get("20990101_000000_reference_free")
