"""End-to-end RAG pipeline: question in, cited answer plus full retrieval trace out.

`RagPipeline.answer()` is the unit the evaluation layer will call once per golden
dataset question. It returns a `QueryResult` rather than a string precisely because
the eval layer needs to score retrieval and generation separately.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import yaml

from groundtruth.config import RUNS_DIR, Config
from groundtruth.embedder import Embedder
from groundtruth.generator import build_generator
from groundtruth.index import load_index_manifest
from groundtruth.retrievers import build_retriever
from groundtruth.schemas import QueryResult
from groundtruth.utils import Timer, write_json, write_jsonl


class RagPipeline:
    """Retriever + generator, wired from a config."""

    def __init__(self, cfg: Config, generate: bool = True) -> None:
        self.cfg = cfg
        self.embedder = Embedder(cfg.embedding)
        self.retriever = build_retriever(cfg, self.embedder)
        self.generator = build_generator(cfg.generation) if generate else None

        self._check_index_compatibility()

    def _check_index_compatibility(self) -> None:
        """Fail loudly when the config and the built index disagree (invariant #7).

        Silently retrieving from an index built with a different embedding model
        would produce plausible-looking but meaningless numbers - the worst possible
        failure for an evaluation framework.
        """
        try:
            manifest = load_index_manifest()
        except FileNotFoundError:
            return  # `index` has not run yet; the retriever will report that itself.

        if manifest.get("index_fingerprint") != self.cfg.index_fingerprint():
            raise RuntimeError(
                "Index/config mismatch.\n"
                f"  index was built with: {manifest.get('embedding_model')} "
                f"(fingerprint {manifest.get('index_fingerprint')})\n"
                f"  this config expects:  {self.cfg.embedding.model} "
                f"(fingerprint {self.cfg.index_fingerprint()})\n"
                "Chunking or embedding settings changed. Re-run `chunk` and `index`."
            )

    def warmup(self) -> None:
        """Load every model before the first timed query.

        Model loading costs tens of seconds on CPU. Left lazy, it is charged to
        whichever query happens to run first, which inflates the mean and can
        single-handedly determine the p95 over a 100-question set.
        """
        warmup = getattr(self.retriever, "warmup", None)
        if callable(warmup):
            warmup()

    def answer(self, question: str, query_id: str = "adhoc") -> QueryResult:
        timer = Timer()
        error: str | None = None
        answer = ""

        retrieved, retrieval_timings = self.retriever.retrieve(question, self.cfg.retrieval.k)
        timer.merge(retrieval_timings)

        if self.generator is not None:
            context = retrieved[: self.cfg.generation.context_chunks]
            try:
                with timer("generate"):
                    answer = self.generator.generate(question, context)
            except Exception as exc:
                # One bad generation must not abort a 100-question batch; record it
                # so the eval layer can count failures rather than silently skip them.
                error = f"{type(exc).__name__}: {exc}"

        timings = timer.total()
        timings["total_ms"] = round(sum(v for k, v in timings.items() if k.endswith("_ms")), 2)

        return QueryResult(
            query_id=query_id,
            question=question,
            retrieved=retrieved,
            answer=answer,
            timings=timings,
            config_name=self.cfg.name,
            error=error,
        )


def run_batch(
    cfg: Config,
    queries: list[dict],
    generate: bool = True,
    run_id: str | None = None,
) -> Path:
    """Run every query through the pipeline and write a run directory.

    The three artifacts - `config.yaml`, `results.jsonl`, `manifest.json` - are the
    complete, self-describing record of an experiment. `manifest.json` carries the
    corpus fingerprint so two runs can be proven comparable before their metrics are
    placed side by side.
    """
    pipeline = RagPipeline(cfg, generate=generate)

    if generate and pipeline.generator is not None:
        healthy, message = pipeline.generator.health_check()
        if not healthy:
            raise RuntimeError(message)

    print("  warming up models (excluded from timings)...")
    pipeline.warmup()

    run_id = run_id or f"{dt.datetime.now():%Y%m%d_%H%M%S}_{cfg.name}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[QueryResult] = []
    for i, query in enumerate(queries, start=1):
        question = query["question"]
        query_id = str(query.get("query_id", f"q{i:04d}"))
        print(f"  [{i}/{len(queries)}] {query_id}: {question[:70]}")
        result = pipeline.answer(question, query_id=query_id)
        results.append(result)
        if result.error:
            print(f"        ! {result.error}")

    write_jsonl(run_dir / "results.jsonl", [r.to_dict() for r in results])
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8"
    )

    latencies = sorted(r.timings.get("total_ms", 0.0) for r in results)
    index_manifest = {}
    try:
        index_manifest = load_index_manifest()
    except FileNotFoundError:
        pass

    write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "config_name": cfg.name,
            "config_fingerprint": cfg.fingerprint(),
            "index_fingerprint": cfg.index_fingerprint(),
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "n_queries": len(results),
            "n_errors": sum(1 for r in results if r.error),
            "generation_enabled": generate,
            "index": index_manifest,
            "latency_ms": {
                "mean": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
                "p50": _percentile(latencies, 50),
                "p95": _percentile(latencies, 95),
                "max": round(latencies[-1], 2) if latencies else 0.0,
            },
        },
    )

    return run_dir


def _percentile(sorted_values: list[float], pct: int) -> float:
    """Nearest-rank percentile. Exact and dependency-free; p95 over ~100 queries
    does not need interpolation, and nearest-rank is the easier definition to defend
    in a report."""
    if not sorted_values:
        return 0.0
    index = max(0, min(len(sorted_values) - 1, int(round(pct / 100 * len(sorted_values) + 0.5)) - 1))
    return round(sorted_values[index], 2)
