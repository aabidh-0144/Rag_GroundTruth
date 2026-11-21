"""Mode 2: the Ragas metric registry, applicability rules and the evaluator.

The registry test reads the *installed* Ragas (offline, no LLM). The evaluator
tests use fake metric objects, so no model is called. A real Ragas metric over
real embeddings is in test_eval_live.py (it loads a model, so it is opt-in).
"""

from __future__ import annotations

import inspect
import math

import pytest
from eval_fakes import FakeRagasMetric, sample

from groundtruth.config import GenerationConfig
from groundtruth.evaluation.config import RagasConfig
from groundtruth.evaluation.ragas_mode import adapters
from groundtruth.evaluation.ragas_mode.evaluator import RagasEvaluator
from groundtruth.evaluation.ragas_mode.metrics import (
    REFERENCE_BASED,
    REFERENCE_FREE,
    REGISTRY,
    SAMPLE_FIELD,
    validate_names,
)


# --- registry vs the installed Ragas ------------------------------------------


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_registry_matches_the_installed_ragas_api(name):
    """Catches a Ragas upgrade that renames a class or changes ascore() arguments."""
    adapters.import_ragas()
    from ragas.metrics import collections

    spec = REGISTRY[name]
    cls = getattr(collections, spec.ragas_class, None)
    assert cls is not None, f"ragas.metrics.collections has no {spec.ragas_class}"
    params = set(inspect.signature(cls.ascore).parameters) - {"self"}
    assert params == set(spec.inputs), f"{spec.ragas_class}.ascore takes {params}"

    init = inspect.signature(cls.__init__).parameters
    assert ("llm" in init) == spec.needs_llm
    assert ("embeddings" in init) == spec.needs_embeddings


def test_reference_free_and_reference_based_are_clearly_separated():
    assert set(REFERENCE_FREE) == {"faithfulness", "response_relevancy", "context_precision"}
    assert set(REFERENCE_BASED) == {
        "context_precision_with_reference", "context_recall", "answer_correctness", "answer_similarity",
    }
    for name in REFERENCE_BASED:
        assert "reference" in REGISTRY[name].inputs


def test_every_ascore_argument_maps_to_a_sample_field():
    for spec in REGISTRY.values():
        assert set(spec.inputs) <= set(SAMPLE_FIELD)


def test_unknown_ragas_metric_is_rejected():
    with pytest.raises(ValueError, match="unknown Ragas metric"):
        validate_names(["faithfulness", "ResponseRelevancy"])


def test_importing_ragas_disables_analytics_by_default(monkeypatch):
    monkeypatch.delenv("RAGAS_DO_NOT_TRACK", raising=False)
    adapters.import_ragas()
    import os

    assert os.environ["RAGAS_DO_NOT_TRACK"] == "true"


def test_ollama_judge_uses_its_openai_compatible_endpoint():
    url, key = adapters.openai_compatible_endpoint(GenerationConfig(base_url="http://localhost:11434/"))
    assert url == "http://localhost:11434/v1"
    assert key  # the openai client refuses an empty key


# --- evaluator ---------------------------------------------------------------


def _evaluate(metrics: dict, samples, timeout_s=30):
    evaluator = RagasEvaluator(
        list(metrics), GenerationConfig(), RagasConfig(timeout_s=timeout_s), metric_instances=metrics
    )
    out = []
    evaluator.evaluate(samples, out.append)
    return {r.sample_id: r for r in out}


def test_successful_ragas_evaluation_passes_the_right_arguments():
    faith = FakeRagasMetric(value=0.9, reason="9/10 claims supported")
    results = _evaluate({"faithfulness": faith}, [sample()])
    m = results["s1"].metrics["faithfulness"]
    assert (m.status, m.score, m.reason) == ("success", 0.9, "9/10 claims supported")
    assert set(faith.received[0]) == {"user_input", "response", "retrieved_contexts"}
    assert faith.received[0]["user_input"] == "What is RAG?"


def test_reference_based_metric_is_skipped_without_ground_truth_never_invented():
    recall = FakeRagasMetric()
    results = _evaluate({"context_recall": recall}, [sample(ground_truth=None)])
    m = results["s1"].metrics["context_recall"]
    assert m.status == "skipped" and m.score is None
    assert "ground_truth" in m.reason
    assert recall.received == []  # Ragas was never called with a made-up reference


def test_reference_based_metric_runs_when_ground_truth_is_present():
    recall = FakeRagasMetric(value=0.75)
    results = _evaluate({"context_recall": recall}, [sample(ground_truth="RAG = retrieval + generation.")])
    assert results["s1"].metrics["context_recall"].score == 0.75
    assert recall.received[0]["reference"] == "RAG = retrieval + generation."


def test_answer_judging_metrics_skip_refusals_but_reference_metrics_score_them():
    s = sample(answer="INSUFFICIENT_CONTEXT", ground_truth="the truth")
    results = _evaluate({"faithfulness": FakeRagasMetric(), "answer_correctness": FakeRagasMetric(value=0.1)}, [s])
    assert results["s1"].metrics["faithfulness"].status == "skipped"
    assert results["s1"].metrics["answer_correctness"].score == 0.1
    assert results["s1"].refusal is True


def test_empty_contexts_skip_context_metrics():
    results = _evaluate({"context_precision": FakeRagasMetric()}, [sample(contexts=[])])
    assert results["s1"].metrics["context_precision"].status == "skipped"


def test_nan_from_ragas_is_recorded_as_failed_not_as_a_score():
    results = _evaluate({"faithfulness": FakeRagasMetric(value=math.nan)}, [sample()])
    m = results["s1"].metrics["faithfulness"]
    assert m.status == "failed" and m.score is None
    assert "NaN" in m.error


def test_one_failing_ragas_metric_does_not_affect_the_others():
    results = _evaluate(
        {"faithfulness": FakeRagasMetric(error=RuntimeError("LLM refused JSON")),
         "response_relevancy": FakeRagasMetric(value=0.7)},
        [sample()],
    )
    assert results["s1"].metrics["faithfulness"].status == "failed"
    assert "LLM refused JSON" in results["s1"].metrics["faithfulness"].error
    assert results["s1"].metrics["response_relevancy"].score == 0.7


def test_slow_metric_times_out_as_failed():
    results = _evaluate({"faithfulness": FakeRagasMetric(delay=0.5)}, [sample()], timeout_s=0.05)
    m = results["s1"].metrics["faithfulness"]
    assert m.status == "failed" and "timed out" in m.error


def test_scores_outside_unit_range_are_clamped():
    results = _evaluate({"faithfulness": FakeRagasMetric(value=1.0000001)}, [sample()])
    assert results["s1"].metrics["faithfulness"].score == 1.0


def test_embeddings_only_metric_needs_no_judge_health_check():
    evaluator = RagasEvaluator(["answer_similarity"], GenerationConfig(), RagasConfig(),
                               metric_instances={"answer_similarity": FakeRagasMetric()})
    healthy, message = evaluator.health_check()
    assert healthy and "no LLM" in message
