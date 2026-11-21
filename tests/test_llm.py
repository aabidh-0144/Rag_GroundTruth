"""The shared LLM transport: JSON mode and judge-aware error messages.

Generator behaviour over the same transport is covered by test_generator.py.
"""

from __future__ import annotations

import pytest
import requests

from groundtruth.config import GenerationConfig
from groundtruth.llm import GroqChatClient, OllamaChatClient, build_llm_client


class _Response:
    def __init__(self, payload, status=200):
        self._payload, self.status_code, self.headers = payload, status, {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self._payload


def _capture(monkeypatch, reply):
    captured = {}

    def fake_post(url, json=None, timeout=None, headers=None):
        captured.update(json)
        return _Response(reply)

    monkeypatch.setattr(requests, "post", fake_post)
    return captured


def test_ollama_json_mode_constrains_the_output_format(monkeypatch):
    captured = _capture(monkeypatch, {"message": {"content": "{}"}})
    OllamaChatClient(GenerationConfig()).chat("sys", "user", json_mode=True)
    assert captured["format"] == "json"


def test_ollama_plain_chat_sends_no_format(monkeypatch):
    captured = _capture(monkeypatch, {"message": {"content": "hi"}})
    OllamaChatClient(GenerationConfig()).chat("sys", "user")
    assert "format" not in captured


def test_groq_json_mode_uses_response_format(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    captured = _capture(monkeypatch, {"choices": [{"message": {"content": "{}"}}]})
    GroqChatClient(GenerationConfig(provider="groq")).chat("sys", "user", json_mode=True)
    assert captured["response_format"] == {"type": "json_object"}


def test_reasoning_trace_is_stripped_from_judge_replies(monkeypatch):
    _capture(monkeypatch, {"message": {"content": "<think>hmm</think>{\"score\": 5}"}})
    assert OllamaChatClient(GenerationConfig()).chat("s", "u", json_mode=True) == '{"score": 5}'


def test_truncation_error_names_the_judge_settings_for_a_judge(monkeypatch):
    _capture(monkeypatch, {"message": {"content": "<think>never finishes"}})
    with pytest.raises(RuntimeError, match=r"judge\.num_predict"):
        OllamaChatClient(GenerationConfig(), label="judge").chat("s", "u")


def test_factory_dispatches_and_rejects_unknown_providers():
    assert isinstance(build_llm_client(GenerationConfig(provider="ollama")), OllamaChatClient)
    assert isinstance(build_llm_client(GenerationConfig(provider="groq")), GroqChatClient)
    with pytest.raises(ValueError, match="Unsupported"):
        build_llm_client(GenerationConfig(provider="openai"))
