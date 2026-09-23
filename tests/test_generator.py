"""Generator tests.

Ollama is stubbed at the HTTP boundary. These cover the parts that shape the
evaluation - prompt construction, the refusal path, and reasoning-trace stripping -
without needing a model running.
"""

from __future__ import annotations

import pytest
import requests

from groundtruth.config import GenerationConfig
from groundtruth.generator import (
    REFUSAL,
    OllamaGenerator,
    _strip_thinking,
    build_generator,
    format_context,
)
from groundtruth.schemas import RetrievedChunk


def make_chunks(n: int = 3) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"chunk{i}",
            text=f"Passage number {i}.",
            score=1.0 - i * 0.1,
            rank=i + 1,
            metadata={"doc_id": f"paper_{i}", "section_path": f"Section {i}"},
        )
        for i in range(n)
    ]


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self) -> dict:
        return self._payload


# --- context formatting ----------------------------------------------------


def test_context_blocks_are_numbered_from_one():
    """The prompt asks for citations like [1]; the numbering must match."""
    rendered = format_context(make_chunks(3))
    assert "[1]" in rendered and "[2]" in rendered and "[3]" in rendered
    assert "[0]" not in rendered


def test_context_labels_each_block_with_its_source():
    rendered = format_context(make_chunks(2))
    assert "paper_0" in rendered
    assert "Section 1" in rendered


def test_context_includes_the_chunk_text():
    assert "Passage number 0." in format_context(make_chunks(1))


def test_context_handles_missing_metadata():
    chunk = RetrievedChunk(chunk_id="c", text="body", score=1.0, rank=1, metadata={})
    rendered = format_context([chunk])
    assert "unknown" in rendered and "body" in rendered


# --- generation ------------------------------------------------------------


def test_generate_returns_the_model_message(monkeypatch):
    generator = OllamaGenerator(GenerationConfig())
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: FakeResponse({"message": {"content": "The answer [1]."}})
    )
    assert generator.generate("q?", make_chunks()) == "The answer [1]."


def test_generate_refuses_when_nothing_was_retrieved():
    """A retrieval miss must not reach the model at all - it has no context to use."""
    assert OllamaGenerator(GenerationConfig()).generate("q?", []) == REFUSAL


def test_generate_sends_deterministic_options(monkeypatch):
    """Reproducibility is the point of a regression framework."""
    captured: dict = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(json)
        return FakeResponse({"message": {"content": "ok"}})

    monkeypatch.setattr(requests, "post", fake_post)
    OllamaGenerator(GenerationConfig(seed=7, num_ctx=2048)).generate("q?", make_chunks())

    assert captured["options"]["temperature"] == 0.0
    assert captured["options"]["seed"] == 7
    assert captured["options"]["num_ctx"] == 2048
    assert captured["stream"] is False


def test_generate_sends_system_and_user_messages(monkeypatch):
    captured: dict = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(json)
        return FakeResponse({"message": {"content": "ok"}})

    monkeypatch.setattr(requests, "post", fake_post)
    OllamaGenerator(GenerationConfig()).generate("What is X?", make_chunks())

    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["system", "user"]
    assert REFUSAL in captured["messages"][0]["content"]
    assert "What is X?" in captured["messages"][1]["content"]


def test_generate_retries_then_raises(monkeypatch):
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", flaky)
    monkeypatch.setattr("time.sleep", lambda s: None)

    with pytest.raises(RuntimeError, match="failed after retries"):
        OllamaGenerator(GenerationConfig(max_retries=2)).generate("q?", make_chunks())

    assert calls["n"] == 3  # initial attempt plus two retries


def test_generate_recovers_after_a_transient_failure(monkeypatch):
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.ConnectionError("transient")
        return FakeResponse({"message": {"content": "recovered"}})

    monkeypatch.setattr(requests, "post", flaky)
    monkeypatch.setattr("time.sleep", lambda s: None)

    assert OllamaGenerator(GenerationConfig()).generate("q?", make_chunks()) == "recovered"


# --- reasoning traces ------------------------------------------------------


def test_thinking_blocks_are_stripped():
    """qwen3 emits a scratchpad; a faithfulness judge must not score it as the answer."""
    assert _strip_thinking("<think>hmm, let me consider</think>\nThe answer is X.") == "The answer is X."


def test_multiline_thinking_blocks_are_stripped():
    assert _strip_thinking("<think>\nline one\nline two\n</think>Answer.") == "Answer."


def test_text_without_thinking_is_unchanged():
    assert _strip_thinking("A plain answer [1].") == "A plain answer [1]."


def test_answer_that_is_only_a_thinking_block_is_not_emptied():
    """Better to surface the raw output than to silently return an empty answer."""
    assert _strip_thinking("<think>only thinking</think>") != ""


# --- health check ----------------------------------------------------------


def test_health_check_passes_when_the_model_is_present(monkeypatch):
    monkeypatch.setattr(
        requests, "get", lambda *a, **k: FakeResponse({"models": [{"name": "qwen3:4b"}]})
    )
    healthy, message = OllamaGenerator(GenerationConfig(model="qwen3:4b")).health_check()
    assert healthy and message == "ok"


def test_health_check_reports_a_missing_model(monkeypatch):
    monkeypatch.setattr(
        requests, "get", lambda *a, **k: FakeResponse({"models": [{"name": "llama3.2:3b"}]})
    )
    healthy, message = OllamaGenerator(GenerationConfig(model="qwen3:4b")).health_check()
    assert not healthy
    assert "ollama pull qwen3:4b" in message


def test_health_check_matches_a_bare_model_name(monkeypatch):
    """A config naming 'qwen3' should match an installed 'qwen3:4b'."""
    monkeypatch.setattr(
        requests, "get", lambda *a, **k: FakeResponse({"models": [{"name": "qwen3:4b"}]})
    )
    healthy, _ = OllamaGenerator(GenerationConfig(model="qwen3")).health_check()
    assert healthy


def test_health_check_reports_an_unreachable_server(monkeypatch):
    def refuse(*a, **k):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "get", refuse)
    healthy, message = OllamaGenerator(GenerationConfig()).health_check()
    assert not healthy
    assert "not reachable" in message


def test_unsupported_provider_is_rejected():
    with pytest.raises(ValueError, match="provider"):
        build_generator(GenerationConfig(provider="openai"))
