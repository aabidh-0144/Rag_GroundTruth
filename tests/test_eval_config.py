"""Evaluation config: loading, and how the judge model is resolved."""

from __future__ import annotations

import pytest

from groundtruth.evaluation.config import JudgeConfig, load_evaluation_config, resolve_judge

GENERATION_YAML = "name: t\ngeneration:\n  provider: ollama\n  model: qwen3:14b\n"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("GROUNDTRUTH_JUDGE_PROVIDER", "GROUNDTRUTH_JUDGE_MODEL",
                "GROUNDTRUTH_GENERATION_PROVIDER", "GROUNDTRUTH_GENERATION_MODEL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def generation_config(tmp_path):
    path = tmp_path / "gen.yaml"
    path.write_text(GENERATION_YAML, encoding="utf-8")
    return str(path)


def test_the_checked_in_evaluation_config_loads():
    cfg = load_evaluation_config()
    assert cfg.reference_free.metrics == ["faithfulness", "answer_relevance", "context_relevance"]
    assert cfg.judge.inherit_from_generation is True
    assert cfg.concurrency == 1


def test_unknown_keys_are_errors_not_silently_ignored(tmp_path):
    path = tmp_path / "evaluation.yaml"
    path.write_text("judge:\n  modle: llama3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown config key 'modle'"):
        load_evaluation_config(path)


def test_judge_inherits_the_generation_model_by_default(generation_config):
    judge = resolve_judge(JudgeConfig(generation_config=generation_config))
    assert (judge.provider, judge.model) == ("ollama", "qwen3:14b")


def test_judge_follows_the_generation_env_override(generation_config, monkeypatch):
    """Whatever .env points generation at, an inheriting judge uses too."""
    monkeypatch.setenv("GROUNDTRUTH_GENERATION_MODEL", "qwen3:8b")
    assert resolve_judge(JudgeConfig(generation_config=generation_config)).model == "qwen3:8b"


def test_an_explicit_judge_model_overrides_inheritance(generation_config):
    judge = resolve_judge(JudgeConfig(generation_config=generation_config, model="llama3.1:8b"))
    assert (judge.provider, judge.model) == ("ollama", "llama3.1:8b")


def test_judge_env_override_wins_over_the_config_file(generation_config, monkeypatch):
    monkeypatch.setenv("GROUNDTRUTH_JUDGE_MODEL", "gemma3:12b")
    judge = resolve_judge(JudgeConfig(generation_config=generation_config, model="llama3.1:8b"))
    assert judge.model == "gemma3:12b"


def test_judge_limits_always_apply_even_when_inheriting(generation_config):
    judge = resolve_judge(JudgeConfig(generation_config=generation_config, num_ctx=8192, temperature=0.0))
    assert judge.num_ctx == 8192
    assert judge.temperature == 0.0
    assert judge.think is False


def test_unknown_judge_provider_is_rejected(generation_config):
    with pytest.raises(ValueError, match="Unknown judge provider"):
        resolve_judge(JudgeConfig(generation_config=generation_config, provider="openai"))


def test_concurrency_below_one_is_rejected(tmp_path):
    path = tmp_path / "evaluation.yaml"
    path.write_text("concurrency: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="concurrency"):
        load_evaluation_config(path)
