"""Evaluation datasets: storage, human review, and turning samples into EvalSamples.

    datasets/<dataset_id>/
        samples.jsonl    one DatasetSample per line
        manifest.json    how it was generated, corpus fingerprint, counts by status

Generated data and human-verified data stay distinguishable on every sample:

    origin               "synthetic"        produced by Ragas, untouched
                         "synthetic_edited" produced by Ragas, then edited by a human
    verification_status  "unverified"       nobody has checked it (the default)
                         "verified"         a human confirmed question + ground truth
                         "rejected"         a human marked it unusable; never evaluated

Nothing is ever marked verified automatically. `datasets/` is not gitignored, so a
reviewed dataset can be committed as a regression suite.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from groundtruth.config import ROOT
from groundtruth.evaluation.schemas import EvalSample
from groundtruth.utils import read_json, read_jsonl, write_json, write_jsonl

DATASETS_DIR = ROOT / "datasets"

VerificationStatus = Literal["unverified", "verified", "rejected"]
ReviewAction = Literal["verify", "reject", "unverify", "edit"]
STATUSES: tuple[str, ...] = ("unverified", "verified", "rejected")


@dataclass(slots=True)
class DatasetSample:
    sample_id: str
    question: str
    ground_truth: str | None
    reference_contexts: list[str]
    reference_spans: list[dict[str, Any]]
    question_type: str
    origin: str = "synthetic"
    verification_status: VerificationStatus = "unverified"
    generator_model: str | None = None
    created_at: str = ""
    reviewed_at: str | None = None
    review_note: str | None = None
    # Pre-edit values, kept so an edit never destroys what Ragas generated.
    original: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DatasetSample:
        return cls(**d)


@dataclass(slots=True)
class Dataset:
    dataset_id: str
    manifest: dict[str, Any]
    samples: list[DatasetSample] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {s: sum(x.verification_status == s for x in self.samples) for s in STATUSES}


class DatasetStore:
    def __init__(self, root: Path | None = None) -> None:
        # Looked up at call time (not bound as a default) so tests can redirect it.
        self.root = root if root is not None else DATASETS_DIR

    def path(self, dataset_id: str) -> Path:
        if not dataset_id or Path(dataset_id).name != dataset_id:
            raise ValueError(f"invalid dataset id {dataset_id!r}")
        return self.root / dataset_id

    def new_id(self) -> str:
        base = f"{dt.datetime.now():%Y%m%d_%H%M%S}_synthetic"
        candidate, n = base, 1
        while (self.root / candidate).exists():
            n += 1
            candidate = f"{base}_{n}"
        return candidate

    def save(self, dataset: Dataset) -> None:
        directory = self.path(dataset.dataset_id)
        directory.mkdir(parents=True, exist_ok=True)
        write_jsonl(directory / "samples.jsonl", [s.to_dict() for s in dataset.samples])
        manifest = {**dataset.manifest, "dataset_id": dataset.dataset_id, "counts": dataset.counts()}
        write_json(directory / "manifest.json", manifest)

    def load(self, dataset_id: str) -> Dataset:
        directory = self.path(dataset_id)
        if not (directory / "samples.jsonl").exists():
            raise FileNotFoundError(f"dataset {dataset_id!r} not found in {self.root}")
        manifest = read_json(directory / "manifest.json") if (directory / "manifest.json").exists() else {}
        samples = [DatasetSample.from_dict(row) for row in read_jsonl(directory / "samples.jsonl")]
        return Dataset(dataset_id=dataset_id, manifest=manifest, samples=samples)

    def list_datasets(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        manifests = []
        for path in sorted(self.root.glob("*/manifest.json"), reverse=True):
            try:
                manifests.append(read_json(path))
            except ValueError:
                continue
        return manifests

    def review(
        self,
        dataset_id: str,
        sample_id: str,
        action: ReviewAction,
        note: str | None = None,
        question: str | None = None,
        ground_truth: str | None = None,
    ) -> DatasetSample:
        """Apply one human review action and persist it.

        - verify / reject / unverify set the status
        - edit changes question and/or ground_truth, marks origin
          "synthetic_edited" and keeps the pre-edit text in `original`. An edit
          does not verify: the human confirms separately with `verify`.
        """
        dataset = self.load(dataset_id)
        sample = next((s for s in dataset.samples if s.sample_id == sample_id), None)
        if sample is None:
            raise ValueError(f"sample {sample_id!r} not in dataset {dataset_id!r}")

        if action == "verify":
            if not (sample.ground_truth and sample.ground_truth.strip()):
                raise ValueError("cannot verify a sample without a ground truth - edit it first")
            sample.verification_status = "verified"
        elif action == "reject":
            sample.verification_status = "rejected"
        elif action == "unverify":
            sample.verification_status = "unverified"
        elif action == "edit":
            new_question = question.strip() if question is not None else sample.question
            new_truth = ground_truth.strip() if ground_truth is not None else sample.ground_truth
            if not new_question:
                raise ValueError("question must not be blank")
            if (new_question, new_truth) == (sample.question, sample.ground_truth):
                raise ValueError("edit changes nothing")
            if sample.original is None:
                sample.original = {"question": sample.question, "ground_truth": sample.ground_truth}
            sample.question, sample.ground_truth = new_question, new_truth
            sample.origin = "synthetic_edited"
            # The content changed, so any earlier verification no longer holds.
            sample.verification_status = "unverified"
        else:
            raise ValueError(f"unknown review action {action!r}")

        sample.reviewed_at = dt.datetime.now().isoformat(timespec="seconds")
        if note is not None:
            sample.review_note = note
        self.save(dataset)
        return sample


def select_samples(dataset: Dataset, verified_only: bool = False) -> list[DatasetSample]:
    """Samples eligible for evaluation. Rejected ones never are."""
    wanted = {"verified"} if verified_only else {"verified", "unverified"}
    return [s for s in dataset.samples if s.verification_status in wanted]


def to_eval_samples(
    dataset: Dataset,
    run_dir: Path,
    verified_only: bool = False,
) -> list[EvalSample]:
    """Join a dataset's ground truth with a RAG run's answers and contexts.

    The run must have been produced over this dataset's questions with
    `query_id == sample_id` (which `answer_dataset()` guarantees).
    """
    from groundtruth.evaluation.inputs import load_run

    run_samples, _ = load_run(run_dir)
    by_id = {s.sample_id: s for s in run_samples}
    samples = []
    for ds in select_samples(dataset, verified_only):
        answered = by_id.get(ds.sample_id)
        if answered is None:
            raise ValueError(
                f"run {run_dir.name} has no answer for dataset sample {ds.sample_id!r}; "
                "regenerate answers with --generate-answers"
            )
        answered.ground_truth = ds.ground_truth
        answered.origin = {
            **answered.origin,
            "dataset": dataset.dataset_id,
            "question_type": ds.question_type,
            "sample_origin": ds.origin,
            "verification_status": ds.verification_status,
        }
        samples.append(answered)
    if not samples:
        which = "verified" if verified_only else "non-rejected"
        raise ValueError(f"dataset {dataset.dataset_id!r} has no {which} samples to evaluate")
    return samples


def answer_dataset(dataset: Dataset, config_path: str, verified_only: bool = False) -> Path:
    """Run the RAG pipeline over a dataset's questions; returns the new run dir."""
    from groundtruth.config import load_config
    from groundtruth.pipeline import run_batch

    selected = select_samples(dataset, verified_only)
    if not selected:
        raise ValueError(f"dataset {dataset.dataset_id!r} has no samples to answer")
    queries = [{"query_id": s.sample_id, "question": s.question} for s in selected]
    return run_batch(load_config(config_path), queries, generate=True)
