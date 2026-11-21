"""`groundtruth.cli eval ...` - the evaluation commands.

Kept out of cli.py so the RAG commands and the evaluation commands each stay in
one readable place. cli.py calls `register(subparsers)`.

    eval reference-free   Mode 1 - LLM-judge metrics, no references needed
    eval ragas            Mode 2 - Ragas metrics (reference-free and reference-based)
    eval generate-dataset Ragas synthetic dataset from the indexed corpus
    eval review-dataset   verify / reject / edit dataset samples in the terminal
    eval datasets         list datasets and their review status
    eval results <id>     one evaluation's summary (and per-sample results)
    eval history          previous evaluations, newest first

Sources for both modes (pick one):
    --input FILE    a .json object/list or .jsonl of {question, contexts, answer[, ground_truth]}
    --run DIR       a RAG run from `groundtruth.cli run`
    --dataset ID    a dataset's ground truth, joined with answers from --run DIR or
                    from a fresh pipeline run (--generate-answers --config ...)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from groundtruth.config import ROOT, RUNS_DIR


def _split(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def _resolve_run_dir(value: str) -> Path:
    path = Path(value)
    if not path.exists() and (RUNS_DIR / value).exists():
        path = RUNS_DIR / value  # accept a bare run id
    return path


def _load_source(args: argparse.Namespace) -> tuple[list[Any], dict[str, Any]]:
    from groundtruth.evaluation.inputs import load_run, load_samples_file

    chosen = [bool(args.input), bool(args.run) and not args.dataset, bool(args.dataset)]
    if sum(chosen) != 1:
        raise ValueError("choose exactly one source: --input FILE, --run DIR, or --dataset ID")

    if args.input:
        return load_samples_file(args.input), {"type": "file", "path": str(args.input)}
    if not args.dataset:
        return load_run(_resolve_run_dir(args.run))

    from groundtruth.evaluation.synthetic.datasets import (
        DatasetStore,
        answer_dataset,
        to_eval_samples,
    )

    dataset = DatasetStore().load(args.dataset)
    if getattr(args, "generate_answers", False):
        if not args.config:
            raise ValueError("--generate-answers needs --config (e.g. configs/hybrid.yaml)")
        print(f"generating answers for dataset {args.dataset} with {args.config} ...")
        run_dir = answer_dataset(dataset, args.config, verified_only=args.verified_only)
    elif args.run:
        run_dir = _resolve_run_dir(args.run)
    else:
        raise ValueError("--dataset needs answers: pass --run DIR or --generate-answers --config ...")

    samples = to_eval_samples(dataset, run_dir, verified_only=args.verified_only)
    _, run_source = load_run(run_dir)
    statuses = sorted({s.origin.get("verification_status") for s in samples})
    return samples, {
        **run_source,
        "type": "dataset",
        "dataset_id": dataset.dataset_id,
        "verified_only": args.verified_only,
        "verification_statuses_included": statuses,
    }


def print_summary(result: Any) -> None:
    print(f"\n=== evaluation {result.evaluation_id} ({result.mode}) - {result.status} ===")
    print(f"samples evaluated: {result.n_samples_evaluated}/{result.n_samples}   "
          f"judge: {result.judge.get('provider')}:{result.judge.get('model')}   "
          f"time: {result.execution_time_s}s")
    print(f"\n  {'metric':34s} {'mean':>7s} {'ok':>4s} {'fail':>5s} {'skip':>5s}")
    for name, agg in result.aggregates.items():
        mean = "-" if agg.mean is None else f"{agg.mean:.3f}"
        print(f"  {name:34s} {mean:>7s} {agg.n_success:>4d} {agg.n_failed:>5d} {agg.n_skipped:>5d}")

    def fmt(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.3f}"

    print(f"\n  retrieval score:  {fmt(result.retrieval_score)}")
    print(f"  generation score: {fmt(result.generation_score)}")
    print(f"  overall score:    {fmt(result.overall_score)}")
    for err in result.errors:
        print(f"  ! {err}")
    print(f"\nsaved to {Path('evals') / result.evaluation_id}")


def _run_mode(args: argparse.Namespace, mode: str) -> int:
    from groundtruth.evaluation.config import load_evaluation_config, resolve_judge
    from groundtruth.evaluation.orchestrator import run_evaluation

    cfg = load_evaluation_config(args.eval_config)
    judge = resolve_judge(cfg.judge)
    samples, source = _load_source(args)

    if mode == "reference_free":
        from groundtruth.evaluation.reference_free.evaluator import ReferenceFreeEvaluator

        evaluator = ReferenceFreeEvaluator(
            _split(args.metrics) or cfg.reference_free.metrics, judge, cfg.concurrency
        )
    else:
        from groundtruth.evaluation.ragas_mode.evaluator import RagasEvaluator

        evaluator = RagasEvaluator(_split(args.metrics) or cfg.ragas.metrics, judge, cfg.ragas, cfg.concurrency)

    result = run_evaluation(
        evaluator,
        samples,
        source,
        resume_id=args.resume,
        extra_config={"evaluation_config": cfg.to_dict()},
    )
    print_summary(result)
    return 0 if result.status != "failed" else 1


def cmd_reference_free(args: argparse.Namespace) -> int:
    return _run_mode(args, "reference_free")


def cmd_ragas(args: argparse.Namespace) -> int:
    return _run_mode(args, "ragas")


def cmd_generate_dataset(args: argparse.Namespace) -> int:
    from groundtruth.evaluation.config import load_evaluation_config, resolve_judge
    from groundtruth.evaluation.synthetic.generator import generate_dataset, parse_distribution

    cfg = load_evaluation_config(args.eval_config)
    judge = resolve_judge(cfg.judge)
    distribution = parse_distribution(args.distribution) if args.distribution else None
    dataset = generate_dataset(cfg, judge, size=args.size, distribution=distribution)

    print(f"\ndataset {dataset.dataset_id}: {len(dataset.samples)} sample(s), all unverified")
    for s in dataset.samples:
        anchored = sum(sp["match"] != "none" for sp in s.reference_spans)
        print(f"  {s.sample_id} [{s.question_type}] {s.question[:80]}")
        print(f"      reference contexts anchored to spans: {anchored}/{len(s.reference_spans)}")
    print(f"\nreview it: groundtruth.cli eval review-dataset {dataset.dataset_id}  (or in `groundtruth.cli ui`)")
    return 0


def cmd_review_dataset(args: argparse.Namespace) -> int:
    from groundtruth.evaluation.synthetic.datasets import DatasetStore

    store = DatasetStore()
    if args.sample:
        if not args.action:
            raise ValueError("--sample needs --action verify|reject|unverify|edit")
        sample = store.review(
            args.dataset_id, args.sample, args.action,
            note=args.note, question=args.question, ground_truth=args.ground_truth,
        )
        print(f"{sample.sample_id}: {sample.verification_status} ({sample.origin})")
        return 0

    # Interactive pass over everything not yet reviewed.
    dataset = store.load(args.dataset_id)
    todo = [s for s in dataset.samples if s.verification_status == "unverified"]
    print(f"{len(todo)} unverified sample(s) in {args.dataset_id}. "
          "[v]erify  [r]eject  [e]dit  [s]kip  [q]uit")
    for s in todo:
        print(f"\n--- {s.sample_id}  [{s.question_type}]  origin={s.origin}")
        print(f"Q:  {s.question}")
        print(f"GT: {s.ground_truth}")
        for i, ctx in enumerate(s.reference_contexts, 1):
            print(f"  ctx{i}: {' '.join(ctx.split())[:300]}")
        choice = input("> ").strip().lower()[:1]
        if choice == "q":
            break
        if choice == "v":
            store.review(args.dataset_id, s.sample_id, "verify")
        elif choice == "r":
            store.review(args.dataset_id, s.sample_id, "reject", note=input("reason: ") or None)
        elif choice == "e":
            q = input("new question (blank = keep): ").strip() or None
            gt = input("new ground truth (blank = keep): ").strip() or None
            store.review(args.dataset_id, s.sample_id, "edit", question=q, ground_truth=gt)
            if input("verify the edited sample? [y/N] ").strip().lower() == "y":
                store.review(args.dataset_id, s.sample_id, "verify")
    print(f"\ncounts: {store.load(args.dataset_id).counts()}")
    return 0


def cmd_datasets(args: argparse.Namespace) -> int:
    from groundtruth.evaluation.synthetic.datasets import DatasetStore

    manifests = DatasetStore().list_datasets()
    if not manifests:
        print("no datasets yet - create one with: groundtruth.cli eval generate-dataset")
        return 0
    for m in manifests:
        print(f"{m.get('dataset_id')}  {m.get('kind', '?'):10s} {m.get('counts')}")
    return 0


def cmd_results(args: argparse.Namespace) -> int:
    from groundtruth.evaluation.store import EvaluationStore

    store = EvaluationStore()
    result = store.get(args.evaluation_id)
    print_summary(result)
    if args.samples:
        for r in store.sample_results(args.evaluation_id):
            print(f"\n[{r.sample_id}] {r.question[:90]}" + ("  (refusal)" if r.refusal else ""))
            for name, m in r.metrics.items():
                score = "-" if m.score is None else f"{m.score:.3f}"
                print(f"   {name:30s} {m.status:8s} {score:>6s}  {(m.error or m.reason)[:110]}")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    from groundtruth.evaluation.store import EvaluationStore

    results = EvaluationStore().history(limit=args.limit, mode=args.mode)
    if not results:
        print("no evaluations yet")
        return 0
    print(f"{'evaluation_id':36s} {'mode':15s} {'status':10s} {'samples':>8s} {'overall':>8s}")
    for r in results:
        overall = "n/a" if r.overall_score is None else f"{r.overall_score:.3f}"
        print(f"{r.evaluation_id:36s} {r.mode:15s} {r.status:10s} "
              f"{r.n_samples_evaluated:>4d}/{r.n_samples:<3d} {overall:>8s}")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    import subprocess

    app = ROOT / "src" / "groundtruth" / "ui" / "app.py"
    print(f"starting Streamlit: {app}")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])


def _add_source_args(p: argparse.ArgumentParser, dataset: bool) -> None:
    p.add_argument("--input", help="JSON / JSONL file of samples")
    p.add_argument("--run", help="run directory (or run id) from `groundtruth.cli run`")
    if dataset:
        p.add_argument("--dataset", help="dataset id (ground truth) - needs --run or --generate-answers")
        p.add_argument("--generate-answers", action="store_true",
                       help="run the RAG pipeline over the dataset questions first")
        p.add_argument("--config", help="experiment config for --generate-answers")
        p.add_argument("--verified-only", action="store_true",
                       help="only human-verified dataset samples")
    else:
        p.set_defaults(dataset=None, verified_only=False, generate_answers=False, config=None)
    p.add_argument("--metrics", help="comma-separated metric names (default: configs/evaluation.yaml)")
    p.add_argument("--resume", metavar="EVALUATION_ID", help="continue an interrupted evaluation")
    p.add_argument("--eval-config", help="evaluation config (default: configs/evaluation.yaml)")


def register(subparsers: Any) -> None:
    p_eval = subparsers.add_parser("eval", help="evaluate RAG quality (Mode 1 / Mode 2 / datasets)")
    sub = p_eval.add_subparsers(dest="eval_command", required=True)

    p = sub.add_parser("reference-free", help="Mode 1: LLM-judge metrics, no references needed")
    _add_source_args(p, dataset=True)
    p.set_defaults(func=cmd_reference_free)

    p = sub.add_parser("ragas", help="Mode 2: Ragas metrics")
    _add_source_args(p, dataset=True)
    p.set_defaults(func=cmd_ragas)

    p = sub.add_parser("generate-dataset", help="generate a synthetic dataset with Ragas")
    p.add_argument("--size", type=int, help="number of samples (default: configs/evaluation.yaml)")
    p.add_argument("--distribution", help="e.g. factual=0.5,reasoning=0.25,multihop=0.25")
    p.add_argument("--eval-config", help="evaluation config (default: configs/evaluation.yaml)")
    p.set_defaults(func=cmd_generate_dataset)

    p = sub.add_parser("review-dataset", help="verify / reject / edit dataset samples")
    p.add_argument("dataset_id")
    p.add_argument("--sample", help="review one sample non-interactively")
    p.add_argument("--action", choices=["verify", "reject", "unverify", "edit"])
    p.add_argument("--note")
    p.add_argument("--question", help="new question (with --action edit)")
    p.add_argument("--ground-truth", help="new ground truth (with --action edit)")
    p.set_defaults(func=cmd_review_dataset)

    p = sub.add_parser("datasets", help="list datasets")
    p.set_defaults(func=cmd_datasets)

    p = sub.add_parser("results", help="show one evaluation")
    p.add_argument("evaluation_id")
    p.add_argument("--samples", action="store_true", help="also show per-sample metric results")
    p.set_defaults(func=cmd_results)

    p = sub.add_parser("history", help="list previous evaluations")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--mode", choices=["reference_free", "ragas"])
    p.set_defaults(func=cmd_history)

    p_ui = subparsers.add_parser("ui", help="open the Streamlit evaluation app")
    p_ui.set_defaults(func=cmd_ui)
