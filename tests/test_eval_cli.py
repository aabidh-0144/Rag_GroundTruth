"""The `groundtruth.cli eval ...` commands, end to end, with a fake judge.

Exercises the real argument parsing, source loading, orchestrator and store -
only the judge's HTTP client is replaced.
"""

from __future__ import annotations

import json

import pytest
from eval_fakes import FakeJudge

import groundtruth.evaluation.reference_free.evaluator as rf_evaluator
import groundtruth.evaluation.store as store_mod
from groundtruth.cli import main


@pytest.fixture
def evals_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "EVALS_DIR", tmp_path / "evals")
    monkeypatch.setattr(rf_evaluator, "build_llm_client", lambda cfg, label="judge": FakeJudge())
    return tmp_path / "evals"


@pytest.fixture
def input_file(tmp_path):
    path = tmp_path / "samples.json"
    path.write_text(json.dumps([
        {"sample_id": "rag", "question": "What is RAG?", "contexts": ["RAG = retrieval + generation."],
         "answer": "RAG combines retrieval and generation [1]."},
        {"sample_id": "bolivia", "question": "Capital of Bolivia?", "contexts": ["RAG = retrieval + generation."],
         "answer": "INSUFFICIENT_CONTEXT"},
    ]), encoding="utf-8")
    return path


def test_reference_free_cli_evaluates_and_persists(evals_dir, input_file, capsys):
    assert main(["eval", "reference-free", "--input", str(input_file)]) == 0
    out = capsys.readouterr().out
    assert "completed" in out and "faithfulness" in out

    [evaluation] = [p for p in evals_dir.iterdir() if p.is_dir()]
    summary = json.loads((evaluation / "summary.json").read_text())
    assert summary["mode"] == "reference_free"
    assert summary["aggregates"]["faithfulness"]["n_skipped"] == 1  # the refusal
    assert summary["source"] == {"type": "file", "path": str(input_file)}


def test_metric_selection_from_the_command_line(evals_dir, input_file):
    main(["eval", "reference-free", "--input", str(input_file), "--metrics", "context_relevance"])
    [evaluation] = [p for p in evals_dir.iterdir() if p.is_dir()]
    assert list(json.loads((evaluation / "summary.json").read_text())["aggregates"]) == ["context_relevance"]


def test_results_and_history_commands_show_the_saved_evaluation(evals_dir, input_file, capsys):
    main(["eval", "reference-free", "--input", str(input_file)])
    evaluation_id = next(p.name for p in evals_dir.iterdir() if p.is_dir())
    capsys.readouterr()

    assert main(["eval", "history"]) == 0
    assert evaluation_id in capsys.readouterr().out

    assert main(["eval", "results", evaluation_id, "--samples"]) == 0
    out = capsys.readouterr().out
    assert "bolivia" in out and "(refusal)" in out and "skipped" in out


def test_invalid_input_file_is_a_clean_error(evals_dir, tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"contexts": ["no question here"]}), encoding="utf-8")
    assert main(["eval", "reference-free", "--input", str(bad)]) == 1
    assert "invalid sample #1" in capsys.readouterr().err


def test_ragas_reference_metrics_without_ground_truth_are_reported_skipped(evals_dir, input_file, monkeypatch):
    from eval_fakes import FakeRagasMetric

    import groundtruth.evaluation.ragas_mode.evaluator as ragas_evaluator

    monkeypatch.setattr(ragas_evaluator, "build_llm_client", lambda cfg, label="judge": FakeJudge())
    monkeypatch.setattr(ragas_evaluator.RagasEvaluator, "_build_instances",
                        lambda self: {m: FakeRagasMetric(value=0.9) for m in self.metrics})
    assert main(["eval", "ragas", "--input", str(input_file), "--metrics", "faithfulness,context_recall"]) == 0
    [evaluation] = [p for p in evals_dir.iterdir() if p.is_dir()]
    aggregates = json.loads((evaluation / "summary.json").read_text())["aggregates"]
    assert aggregates["context_recall"] == {"mean": None, "n_success": 0, "n_failed": 0, "n_skipped": 2}
    assert aggregates["faithfulness"]["mean"] == 0.9
