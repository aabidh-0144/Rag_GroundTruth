"""Streamlit pages render without errors, and dataset review persists.

Uses Streamlit's own headless test harness (AppTest) against temporary stores.
"""

from __future__ import annotations

import pytest
from eval_fakes import FakeJudge, sample

import groundtruth.evaluation.store as store_mod
import groundtruth.evaluation.synthetic.datasets as datasets_mod
from groundtruth.config import GenerationConfig
from groundtruth.evaluation.orchestrator import run_evaluation
from groundtruth.evaluation.reference_free.evaluator import ReferenceFreeEvaluator
from groundtruth.evaluation.synthetic.datasets import Dataset, DatasetSample, DatasetStore

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "EVALS_DIR", tmp_path / "evals")
    monkeypatch.setattr(datasets_mod, "DATASETS_DIR", tmp_path / "datasets")
    evaluator = ReferenceFreeEvaluator(
        ["faithfulness", "answer_relevance", "context_relevance"], GenerationConfig(), client=FakeJudge()
    )
    run_evaluation(evaluator, [sample()], {"type": "test"}, progress=lambda m: None)
    DatasetStore().save(Dataset("ds", {"kind": "synthetic"}, [
        DatasetSample("ds_001", "Q?", "A.", ["ctx"], [{"doc_id": "d", "start_char": 0, "end_char": 3,
                                                        "match": "chunk"}], "factual"),
    ]))


def _page(view: str):
    at = AppTest.from_string(f"from groundtruth.ui.views import {view}\n{view}.render()", default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("view", ["evaluate", "results", "datasets"])
def test_every_page_renders(seeded, view):
    _page(view)


def test_results_page_shows_the_group_scores(seeded):
    at = _page("results")
    assert [m.label for m in at.metric] == ["Retrieval", "Generation", "Overall"]


def test_verifying_a_sample_in_the_ui_persists_it(seeded):
    at = _page("datasets")
    next(b for b in at.button if b.label == "Verify").click().run()
    assert not at.exception
    assert DatasetStore().load("ds").samples[0].verification_status == "verified"
