"""Evaluation inputs: payload validation and loading samples from a RAG run."""

from __future__ import annotations

import json

import pytest
import yaml

from groundtruth.evaluation.inputs import load_run, load_samples_file, parse_samples


def test_parses_the_documented_example_payload():
    [s] = parse_samples({
        "question": "What is Retrieval-Augmented Generation?",
        "contexts": ["RAG combines information retrieval with language generation."],
        "answer": "RAG is a technique that retrieves relevant information and uses an LLM.",
    })
    assert s.sample_id == "s0001"
    assert s.ground_truth is None


@pytest.mark.parametrize("key", ["ground_truth", "reference"])
def test_accepts_ground_truth_under_either_name(key):
    [s] = parse_samples({"question": "q", "answer": "a", key: "the truth"})
    assert s.ground_truth == "the truth"


def test_blank_ground_truth_is_treated_as_absent_not_as_a_reference():
    [s] = parse_samples({"question": "q", "answer": "a", "ground_truth": "   "})
    assert s.ground_truth is None


def test_blank_contexts_are_dropped():
    [s] = parse_samples({"question": "q", "contexts": ["real", "", "  "], "answer": "a"})
    assert s.contexts == ["real"]


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"contexts": ["c"], "answer": "a"}, "question"),
        ({"question": "   ", "answer": "a"}, "blank"),
        ({"question": "q", "contexts": "not a list"}, "contexts"),
        (["not an object"], "expected a JSON object"),
        ([], "no samples"),
    ],
)
def test_invalid_payloads_are_rejected_with_a_clear_error(payload, message):
    with pytest.raises(ValueError, match=message):
        parse_samples(payload)


def test_duplicate_sample_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        parse_samples([{"sample_id": "x", "question": "q"}, {"sample_id": "x", "question": "q2"}])


def test_loads_jsonl_files(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text('{"question": "a?"}\n{"question": "b?"}\n', encoding="utf-8")
    assert [s.question for s in load_samples_file(path)] == ["a?", "b?"]


def test_missing_input_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_samples_file(tmp_path / "nope.json")


def _write_run(tmp_path, context_chunks=2):
    run = tmp_path / "20260101_000000_hybrid"
    run.mkdir()
    (run / "config.yaml").write_text(
        yaml.safe_dump({"generation": {"context_chunks": context_chunks}}), encoding="utf-8"
    )
    (run / "manifest.json").write_text(
        json.dumps({"config_name": "hybrid", "index_fingerprint": "abc123"}), encoding="utf-8"
    )
    retrieved = [{"chunk_id": f"c{i}", "text": f"chunk {i}"} for i in range(5)]
    rows = [
        {"query_id": "q1", "question": "Q1?", "retrieved": retrieved, "answer": "A1 [1]", "error": None},
        {"query_id": "q2", "question": "Q2?", "retrieved": retrieved, "answer": "", "error": "timeout"},
    ]
    (run / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return run


def test_run_contexts_are_only_what_the_generator_saw(tmp_path):
    """Faithfulness must be judged against the chunks in the prompt, not all of top-k."""
    samples, source = load_run(_write_run(tmp_path, context_chunks=2))
    assert samples[0].contexts == ["chunk 0", "chunk 1"]
    assert samples[0].context_ids == ["c0", "c1"]
    assert source["index_fingerprint"] == "abc123"
    assert source["context_chunks"] == 2


def test_run_generation_errors_are_carried_into_the_sample_origin(tmp_path):
    samples, _ = load_run(_write_run(tmp_path))
    assert samples[1].answer == ""
    assert samples[1].origin["generation_error"] == "timeout"


def test_missing_run_dir_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="run directory"):
        load_run(tmp_path / "nope")
