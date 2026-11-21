"""Evaluate page: choose a mode, a source and metrics, then launch the job."""

from __future__ import annotations

import datetime as dt
import json

import streamlit as st

from groundtruth.evaluation.config import load_evaluation_config
from groundtruth.evaluation.inputs import parse_samples
from groundtruth.evaluation.ragas_mode.metrics import REGISTRY
from groundtruth.evaluation.reference_free import METRICS
from groundtruth.evaluation.synthetic.datasets import DatasetStore
from groundtruth.ui.common import INPUTS_DIR, experiment_configs, jobs, launch, run_dirs, show_job

EXAMPLE = [
    {
        "question": "What is Retrieval-Augmented Generation?",
        "contexts": ["RAG combines information retrieval with language generation."],
        "answer": "RAG retrieves relevant information and uses an LLM to generate answers.",
    }
]


def _metric_label(name: str) -> str:
    spec = REGISTRY.get(name)
    return f"{name}  (needs ground truth)" if spec and spec.requires_reference else name


def render() -> None:
    st.title("Evaluate")
    cfg = load_evaluation_config()

    mode = st.segmented_control(
        "Mode", ["Reference-free (Mode 1)", "Ragas (Mode 2)"], default="Reference-free (Mode 1)"
    ) or "Reference-free (Mode 1)"
    is_ragas = mode.startswith("Ragas")
    command = "ragas" if is_ragas else "reference-free"
    st.caption(
        "Ragas metrics, including reference-based ones that need a ground truth."
        if is_ragas
        else "LLM-judge faithfulness, answer relevance and context relevance - no references needed."
    )

    available = list(REGISTRY) if is_ragas else list(METRICS)
    defaults = cfg.ragas.metrics if is_ragas else cfg.reference_free.metrics
    metrics = st.multiselect(
        "Metrics", available, default=[m for m in defaults if m in available], format_func=_metric_label
    )

    source = st.radio("Source", ["A RAG run", "A dataset", "Paste JSON"], horizontal=True)
    args: list[str] | None = None

    if source == "A RAG run":
        runs = run_dirs()
        if not runs:
            st.warning("No runs yet. Create one with `groundtruth.cli run --config ... --queries ...`.")
        else:
            run = st.selectbox("Run", runs)
            args = ["--run", run]

    elif source == "A dataset":
        datasets = DatasetStore().list_datasets()
        if not datasets:
            st.warning("No datasets yet - generate one on the Datasets page.")
        else:
            labels = {m["dataset_id"]: f"{m['dataset_id']}  {m.get('counts')}" for m in datasets}
            dataset_id = st.selectbox("Dataset", list(labels), format_func=labels.get)
            verified_only = st.checkbox("Only human-verified samples", value=False)
            answers = st.radio(
                "Answers", ["Generate now with an experiment config", "Use an existing run"], horizontal=True
            )
            args = ["--dataset", dataset_id] + (["--verified-only"] if verified_only else [])
            if answers.startswith("Generate"):
                config = st.selectbox("Experiment config", experiment_configs(),
                                      index=experiment_configs().index("configs/hybrid.yaml")
                                      if "configs/hybrid.yaml" in experiment_configs() else 0)
                args += ["--generate-answers", "--config", config]
            else:
                runs = run_dirs()
                if runs:
                    args += ["--run", st.selectbox("Run with this dataset's answers", runs)]
                else:
                    st.warning("No runs available.")
                    args = None

    else:
        text = st.text_area(
            "Samples (one JSON object or a list)", json.dumps(EXAMPLE, indent=2), height=220
        )
        try:
            parse_samples(json.loads(text))
            valid = True
        except (ValueError, json.JSONDecodeError) as exc:
            st.error(f"Invalid input: {exc}")
            valid = False
        if valid:
            INPUTS_DIR.mkdir(parents=True, exist_ok=True)
            path = INPUTS_DIR / f"{dt.datetime.now():%Y%m%d_%H%M%S}.json"
            args = ["--input", str(path)]
            st.session_state["pending_input"] = (path, text)

    can_run = bool(args) and bool(metrics)
    if st.button("Run evaluation", type="primary", disabled=not can_run):
        pending = st.session_state.pop("pending_input", None)
        if pending:
            pending[0].write_text(pending[1], encoding="utf-8")
        launch("eval", ["eval", command, *args, "--metrics", ",".join(metrics)])

    st.caption(
        "Runs `groundtruth.cli eval` in the background - the page stays usable, and finished "
        "samples are saved even if the job is stopped. Local judging is slow: expect tens of "
        "seconds per metric per sample."
    )

    for job in reversed(jobs("eval")):
        show_job(job)
