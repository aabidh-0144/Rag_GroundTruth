"""Where evaluation samples come from.

Every source is validated and normalized into `EvalSample`s, so the evaluators
never see raw, untrusted payloads:

* `load_samples_file(path)` - a .json file (one object, or a list) or a .jsonl file
    {"question": ..., "contexts": [...], "answer": ..., "ground_truth": ...}
* `load_run(run_dir)`       - a RAG run produced by `groundtruth.cli run`
* `load_dataset_samples()`  - lives in synthetic/datasets.py (needs answers joined in)

From a run, `contexts` are the first `generation.context_chunks` retrieved chunks -
exactly what the generator was shown - so faithfulness is judged against the same
evidence the answer was written from, not the full top-k list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from groundtruth.evaluation.schemas import EvalSample
from groundtruth.utils import read_json, read_jsonl


class InputSample(BaseModel):
    """The accepted shape of one sample. `reference` is accepted as an alias."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    sample_id: str | None = None
    question: str = Field(min_length=1)
    contexts: list[str] = []
    context_ids: list[str] = []
    answer: str = ""
    ground_truth: str | None = Field(default=None, alias="reference")

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value

    @field_validator("contexts")
    @classmethod
    def _drop_blank_contexts(cls, value: list[str]) -> list[str]:
        return [c for c in value if c and c.strip()]

    @field_validator("ground_truth")
    @classmethod
    def _blank_ground_truth_is_none(cls, value: str | None) -> str | None:
        # An empty string is "no reference", never a reference to compare against.
        return value if value and value.strip() else None


def _to_sample(raw: dict[str, Any], index: int, origin: dict[str, Any]) -> EvalSample:
    # Accept both spellings; pydantic's alias handling covers `reference`.
    if "ground_truth" in raw and "reference" not in raw:
        raw = {**raw, "reference": raw["ground_truth"]}
    try:
        parsed = InputSample.model_validate(raw)
    except ValidationError as exc:
        errors = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'sample'}: {e['msg']}" for e in exc.errors()
        )
        raise ValueError(f"invalid sample #{index + 1}: {errors}") from exc
    return EvalSample(
        sample_id=parsed.sample_id or f"s{index + 1:04d}",
        question=parsed.question,
        contexts=parsed.contexts,
        context_ids=parsed.context_ids,
        answer=parsed.answer,
        ground_truth=parsed.ground_truth,
        origin=origin,
    )


def _check_unique_ids(samples: list[EvalSample]) -> list[EvalSample]:
    seen: set[str] = set()
    for s in samples:
        if s.sample_id in seen:
            raise ValueError(f"duplicate sample_id {s.sample_id!r}")
        seen.add(s.sample_id)
    return samples


def parse_samples(payload: Any, origin: dict[str, Any] | None = None) -> list[EvalSample]:
    """Validate an in-memory payload: one sample dict, or a list of them."""
    rows = payload if isinstance(payload, list) else [payload]
    if not rows:
        raise ValueError("no samples to evaluate")
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"invalid sample #{i + 1}: expected a JSON object")
    return _check_unique_ids([_to_sample(row, i, origin or {}) for i, row in enumerate(rows)])


def load_samples_file(path: str | Path) -> list[EvalSample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"input file not found: {path}")
    payload = read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)
    return parse_samples(payload, origin={"file": str(path)})


def load_run(run_dir: str | Path) -> tuple[list[EvalSample], dict[str, Any]]:
    """Samples from a `groundtruth.cli run` directory, plus provenance for the summary."""
    run_dir = Path(run_dir)
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"{results_path} not found - is {run_dir} a run directory?")

    config_path = run_dir / "config.yaml"
    run_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    n_context = int(((run_cfg or {}).get("generation") or {}).get("context_chunks", 5))

    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}

    samples = []
    for i, row in enumerate(read_jsonl(results_path)):
        context_chunks = row.get("retrieved", [])[:n_context]
        raw = {
            "sample_id": row.get("query_id"),
            "question": row.get("question", ""),
            "contexts": [c.get("text", "") for c in context_chunks],
            "context_ids": [c.get("chunk_id", "") for c in context_chunks],
            "answer": row.get("answer") or "",
        }
        origin = {"run": run_dir.name}
        if row.get("error"):
            origin["generation_error"] = row["error"]
        samples.append(_to_sample(raw, i, origin))

    if not samples:
        raise ValueError(f"{results_path} contains no results")
    source = {
        "type": "run",
        "run_id": run_dir.name,
        "config_name": manifest.get("config_name"),
        "index_fingerprint": manifest.get("index_fingerprint"),
        "context_chunks": n_context,
        "generation_enabled": manifest.get("generation_enabled"),
    }
    return _check_unique_ids(samples), source
