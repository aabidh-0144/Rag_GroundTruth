"""Synthetic datasets: span anchoring, review workflow, joining with runs, generation.

Generation is tested with Ragas' TestsetGenerator and knowledge-graph steps mocked
at the module boundary - the real ones make many LLM calls.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml
from conftest import make_document

from groundtruth.evaluation.config import EvaluationConfig
from groundtruth.evaluation.synthetic import generator as gen
from groundtruth.evaluation.synthetic.datasets import (
    Dataset,
    DatasetSample,
    DatasetStore,
    select_samples,
    to_eval_samples,
)
from groundtruth.evaluation.synthetic.spans import SpanIndex
from groundtruth.config import GenerationConfig
from groundtruth.schemas import Chunk, make_chunk_id

DOC_TEXT = "Attention is all you need. Multi-head attention runs eight heads in parallel."


def _chunk(doc_id, start, end, text):
    return Chunk(chunk_id=make_chunk_id(doc_id, start, end), doc_id=doc_id, chunk_index=0,
                 start_char=start, end_char=end, page_start=1, page_end=1,
                 section_path="", n_tokens=10, text=text)


def _ds_sample(i, status="unverified", ground_truth="An answer."):
    return DatasetSample(sample_id=f"d_{i:03d}", question=f"Question {i}?", ground_truth=ground_truth,
                         reference_contexts=["ctx"], reference_spans=[], question_type="factual",
                         verification_status=status)


# --- span anchoring (invariant #2) --------------------------------------------


def test_reference_context_equal_to_a_chunk_anchors_to_that_chunk_span():
    doc = make_document(DOC_TEXT, doc_id="attn")
    chunk = _chunk("attn", 27, len(DOC_TEXT), DOC_TEXT[27:])
    span = SpanIndex([chunk], [doc]).anchor(chunk.text)
    assert span == {"doc_id": "attn", "start_char": 27, "end_char": len(DOC_TEXT), "match": "chunk"}
    assert doc.text[span["start_char"]:span["end_char"]] == chunk.text


def test_reference_context_inside_a_document_anchors_exactly():
    doc = make_document(DOC_TEXT, doc_id="attn")
    span = SpanIndex([], [doc]).anchor("eight heads")
    assert doc.text[span["start_char"]:span["end_char"]] == "eight heads"
    assert span["match"] == "exact"


def test_paraphrased_context_is_flagged_not_guessed():
    span = SpanIndex([], [make_document(DOC_TEXT)]).anchor("a sentence that is not in the corpus")
    assert span["match"] == "none" and span["doc_id"] is None


# --- review workflow -----------------------------------------------------------


def _store_with(tmp_path, samples):
    store = DatasetStore(tmp_path)
    store.save(Dataset("ds", {"kind": "synthetic"}, samples))
    return store


def test_generated_samples_start_unverified(tmp_path):
    store = _store_with(tmp_path, [_ds_sample(1)])
    loaded = store.load("ds").samples[0]
    assert loaded.verification_status == "unverified"
    assert loaded.origin == "synthetic"


def test_verify_reject_and_counts_are_persisted(tmp_path):
    store = _store_with(tmp_path, [_ds_sample(1), _ds_sample(2)])
    store.review("ds", "d_001", "verify")
    store.review("ds", "d_002", "reject", note="ambiguous")
    dataset = store.load("ds")
    assert dataset.counts() == {"unverified": 0, "verified": 1, "rejected": 1}
    assert dataset.samples[1].review_note == "ambiguous"
    assert json.loads((tmp_path / "ds" / "manifest.json").read_text())["counts"]["verified"] == 1


def test_a_sample_without_ground_truth_cannot_be_verified(tmp_path):
    store = _store_with(tmp_path, [_ds_sample(1, ground_truth=None)])
    with pytest.raises(ValueError, match="without a ground truth"):
        store.review("ds", "d_001", "verify")


def test_edit_keeps_the_original_and_resets_verification(tmp_path):
    store = _store_with(tmp_path, [_ds_sample(1, status="verified")])
    edited = store.review("ds", "d_001", "edit", ground_truth="A better answer.")
    assert edited.ground_truth == "A better answer."
    assert edited.original == {"question": "Question 1?", "ground_truth": "An answer."}
    assert edited.origin == "synthetic_edited"
    assert edited.verification_status == "unverified"


def test_an_edit_that_changes_nothing_is_rejected(tmp_path):
    store = _store_with(tmp_path, [_ds_sample(1)])
    with pytest.raises(ValueError, match="changes nothing"):
        store.review("ds", "d_001", "edit", question="Question 1?")


def test_rejected_samples_are_never_selected_for_evaluation():
    ds = Dataset("ds", {}, [_ds_sample(1, "verified"), _ds_sample(2, "unverified"), _ds_sample(3, "rejected")])
    assert [s.sample_id for s in select_samples(ds)] == ["d_001", "d_002"]
    assert [s.sample_id for s in select_samples(ds, verified_only=True)] == ["d_001"]


def test_dataset_ids_cannot_escape_the_datasets_directory(tmp_path):
    with pytest.raises(ValueError, match="invalid dataset id"):
        DatasetStore(tmp_path).load("../evals")


# --- joining a dataset with a RAG run ------------------------------------------


def _write_run(tmp_path, answered_ids):
    run = tmp_path / "runs" / "20260101_000000_hybrid"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump({"generation": {"context_chunks": 1}}), encoding="utf-8")
    rows = [{"query_id": qid, "question": "q", "answer": f"answer for {qid}",
             "retrieved": [{"chunk_id": "c1", "text": "retrieved text"}]} for qid in answered_ids]
    (run / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return run


def test_dataset_ground_truth_is_joined_with_run_answers_and_labelled(tmp_path):
    ds = Dataset("ds", {}, [_ds_sample(1, "verified"), _ds_sample(2, "unverified")])
    samples = to_eval_samples(ds, _write_run(tmp_path, ["d_001", "d_002"]))
    assert samples[0].ground_truth == "An answer."
    assert samples[0].answer == "answer for d_001"
    assert samples[0].contexts == ["retrieved text"]
    # Synthetic vs human-verified stays visible on every evaluated sample.
    assert [s.origin["verification_status"] for s in samples] == ["verified", "unverified"]


def test_join_fails_clearly_when_the_run_lacks_a_dataset_question(tmp_path):
    ds = Dataset("ds", {}, [_ds_sample(1), _ds_sample(2)])
    with pytest.raises(ValueError, match="no answer for dataset sample 'd_002'"):
        to_eval_samples(ds, _write_run(tmp_path, ["d_001"]))


# --- distribution --------------------------------------------------------------


def test_distribution_is_parsed_and_normalized():
    assert gen.parse_distribution("factual=2,multihop=2") == {"factual": 0.5, "multihop": 0.5}


@pytest.mark.parametrize("text, message", [
    ("factual=1,trivia=1", "unknown question type"),
    ("factual=-1,reasoning=1", "negative"),
    ("factual=0", "positive weight"),
    ("factual", "expected type=weight"),
])
def test_bad_distributions_are_rejected(text, message):
    with pytest.raises(ValueError, match=message):
        gen.parse_distribution(text)


# --- generation (Ragas mocked at the boundary) ---------------------------------


class _FakeSynthesizer:
    def __init__(self, supported: bool) -> None:
        self.supported = supported

    def get_node_clusters(self, kg):
        return ["cluster"] if self.supported else []


def test_unsupported_question_types_are_dropped_and_recorded():
    classes = {
        "SingleHopSpecificQuerySynthesizer": lambda llm: _FakeSynthesizer(True),
        "MultiHopAbstractQuerySynthesizer": lambda llm: _FakeSynthesizer(False),
        "MultiHopSpecificQuerySynthesizer": lambda llm: _FakeSynthesizer(False),
    }
    import ragas.testset.synthesizers.multi_hop as multi_hop
    import ragas.testset.synthesizers.single_hop.specific as single_hop

    patches = {
        (single_hop, "SingleHopSpecificQuerySynthesizer"): classes["SingleHopSpecificQuerySynthesizer"],
        (multi_hop, "MultiHopAbstractQuerySynthesizer"): classes["MultiHopAbstractQuerySynthesizer"],
        (multi_hop, "MultiHopSpecificQuerySynthesizer"): classes["MultiHopSpecificQuerySynthesizer"],
    }
    originals = {key: getattr(*key) for key in patches}
    try:
        for (module, name), fake in patches.items():
            setattr(module, name, fake)
        distribution, dropped = gen._build_distribution(
            {"factual": 0.5, "reasoning": 0.25, "multihop": 0.25}, llm=None, kg=None, progress=lambda m: None
        )
    finally:
        for (module, name), original in originals.items():
            setattr(module, name, original)
    assert [w for _, w in distribution] == [1.0]  # renormalized to the one usable type
    assert set(dropped) == {"reasoning", "multihop"}


def test_generate_dataset_saves_unverified_anchored_samples(tmp_path, monkeypatch):
    doc = make_document(DOC_TEXT, doc_id="attn")
    chunk = _chunk("attn", 0, 26, DOC_TEXT[:26])

    generated = SimpleNamespace(samples=[SimpleNamespace(
        synthesizer_name="single_hop_specific_query_synthesizer",
        eval_sample=SimpleNamespace(user_input="What is all you need?", reference="Attention.",
                                    reference_contexts=[chunk.text]),
    )])

    class FakeTestsetGenerator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, testset_size, query_distribution, run_config):
            assert testset_size == 1
            assert run_config.max_workers == 1  # not Ragas' default of 16
            return generated

    import ragas.testset

    monkeypatch.setattr(ragas.testset, "TestsetGenerator", FakeTestsetGenerator)
    monkeypatch.setattr(gen.adapters, "build_ragas_llm", lambda cfg: "llm")
    monkeypatch.setattr(gen.adapters, "build_embeddings", lambda name: "embeddings")
    monkeypatch.setattr(gen, "_build_knowledge_graph", lambda chunks, llm, emb, rc: "kg")
    monkeypatch.setattr(gen, "_build_distribution", lambda w, llm, kg, progress: ([("synth", 1.0)], {}))

    dataset = gen.generate_dataset(
        EvaluationConfig(), GenerationConfig(model="qwen3:14b"), size=1,
        chunks=[chunk], documents=[doc], store=DatasetStore(tmp_path), progress=lambda m: None,
    )
    [s] = dataset.samples
    assert s.question == "What is all you need?" and s.ground_truth == "Attention."
    assert s.question_type == "factual"
    assert s.verification_status == "unverified" and s.origin == "synthetic"
    assert s.reference_spans[0]["match"] == "chunk"
    assert s.generator_model == "ollama:qwen3:14b"
    saved = DatasetStore(tmp_path).load(dataset.dataset_id)
    assert saved.manifest["counts"] == {"unverified": 1, "verified": 0, "rejected": 0}


def test_generation_needs_an_indexed_corpus(tmp_path):
    with pytest.raises(ValueError, match="no chunks"):
        gen.generate_dataset(EvaluationConfig(), GenerationConfig(), size=1, chunks=[], documents=[],
                             store=DatasetStore(tmp_path))
