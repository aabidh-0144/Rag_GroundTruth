"""Saving and loading evaluations - same layout philosophy as runs/.

    evals/<evaluation_id>/
        config.yaml      what was requested: mode, metrics, judge, source (no secrets)
        results.jsonl    one SampleResult per line, appended as each sample finishes
        summary.json     the EvaluationResult (written at the end, and on interruption)

`results.jsonl` is append-only so an interrupted evaluation keeps every finished
sample and `--resume <evaluation_id>` can skip them. History is a scan of
`evals/*/summary.json`: instant at this project's scale, and greppable.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import yaml

from groundtruth.config import ROOT
from groundtruth.evaluation.schemas import EvaluationResult, SampleResult
from groundtruth.utils import iter_jsonl, read_json, write_json

EVALS_DIR = ROOT / "evals"


class EvaluationStore:
    def __init__(self, root: Path | None = None) -> None:
        # Looked up at call time (not bound as a default) so tests can redirect it.
        self.root = root if root is not None else EVALS_DIR

    # --- ids & paths -------------------------------------------------------------

    def new_id(self, mode: str) -> str:
        base = f"{dt.datetime.now():%Y%m%d_%H%M%S}_{mode}"
        candidate, n = base, 1
        while (self.root / candidate).exists():
            n += 1
            candidate = f"{base}_{n}"
        return candidate

    def path(self, evaluation_id: str) -> Path:
        # Refuse anything that could step outside evals/ (e.g. "../runs").
        if not evaluation_id or Path(evaluation_id).name != evaluation_id:
            raise ValueError(f"invalid evaluation id {evaluation_id!r}")
        return self.root / evaluation_id

    # --- writing -----------------------------------------------------------------

    def create(self, evaluation_id: str, config: dict[str, Any]) -> Path:
        directory = self.path(evaluation_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        return directory

    def append_result(self, evaluation_id: str, result: SampleResult) -> None:
        with (self.path(evaluation_id) / "results.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    def write_summary(self, result: EvaluationResult) -> None:
        write_json(self.path(result.evaluation_id) / "summary.json", result.to_dict())

    # --- reading -----------------------------------------------------------------

    def config(self, evaluation_id: str) -> dict[str, Any]:
        path = self.path(evaluation_id) / "config.yaml"
        if not path.exists():
            raise FileNotFoundError(f"evaluation {evaluation_id!r} not found in {self.root}")
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def sample_results(self, evaluation_id: str) -> list[SampleResult]:
        path = self.path(evaluation_id) / "results.jsonl"
        if not path.exists():
            return []
        # Last write wins, so a resumed sample is never counted twice.
        by_id: dict[str, SampleResult] = {}
        for row in iter_jsonl(path):
            by_id[row["sample_id"]] = SampleResult.from_dict(row)
        return list(by_id.values())

    def get(self, evaluation_id: str) -> EvaluationResult:
        path = self.path(evaluation_id) / "summary.json"
        if not path.exists():
            if (self.path(evaluation_id) / "config.yaml").exists():
                raise FileNotFoundError(
                    f"evaluation {evaluation_id!r} has no summary yet - it may still be running, "
                    f"or it was interrupted (resume with --resume {evaluation_id})"
                )
            raise FileNotFoundError(f"evaluation {evaluation_id!r} not found in {self.root}")
        return EvaluationResult.from_dict(read_json(path))

    def history(self, limit: int | None = None, mode: str | None = None) -> list[EvaluationResult]:
        if not self.root.exists():
            return []
        results = []
        for summary in self.root.glob("*/summary.json"):
            try:
                result = EvaluationResult.from_dict(read_json(summary))
            except (ValueError, KeyError, TypeError):
                continue  # a hand-edited or foreign file must not break the listing
            if mode is None or result.mode == mode:
                results.append(result)
        # started_at has 1s resolution; the id breaks ties (a same-second id gets a
        # "_2" suffix, which sorts after the first), keeping newest-first exact.
        results.sort(key=lambda r: (r.started_at, r.evaluation_id), reverse=True)
        return results[:limit] if limit else results
