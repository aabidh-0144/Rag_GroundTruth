"""Datasets page: generate a synthetic dataset, then review it sample by sample."""

from __future__ import annotations

import streamlit as st

from groundtruth.evaluation.config import load_evaluation_config
from groundtruth.evaluation.synthetic.datasets import DatasetStore
from groundtruth.ui.common import badge, jobs, launch, show_job


def _generate_form() -> None:
    cfg = load_evaluation_config()
    with st.form("generate"):
        st.markdown("Ragas reads your indexed chunks and writes questions + reference answers.")
        size = st.number_input("Samples", min_value=1, max_value=200, value=cfg.synthetic.size)
        c1, c2, c3 = st.columns(3)
        dist = cfg.synthetic.distribution
        factual = c1.slider("Factual (single-hop)", 0.0, 1.0, float(dist.get("factual", 0.5)))
        reasoning = c2.slider("Reasoning (multi-hop abstract)", 0.0, 1.0, float(dist.get("reasoning", 0.25)))
        multihop = c3.slider("Multi-hop (specific)", 0.0, 1.0, float(dist.get("multihop", 0.25)))
        submitted = st.form_submit_button("Generate dataset", type="primary")
    st.caption(
        "Slow on a local model: building the knowledge graph makes many LLM calls before the "
        "first question appears. Question types the corpus can't support are dropped and noted."
    )
    if submitted:
        if factual + reasoning + multihop == 0:
            st.error("At least one question type needs a weight above 0.")
            return
        distribution = f"factual={factual},reasoning={reasoning},multihop={multihop}"
        launch("dataset", ["eval", "generate-dataset", "--size", str(int(size)), "--distribution", distribution])


def _review(store: DatasetStore, dataset_id: str) -> None:
    dataset = store.load(dataset_id)
    counts = dataset.counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("Unverified", counts["unverified"])
    c2.metric("Verified", counts["verified"])
    c3.metric("Rejected", counts["rejected"])
    dropped = dataset.manifest.get("question_types_dropped")
    if dropped:
        st.caption(f"Question types dropped at generation: {dropped}")

    show = st.pills(
        "Show", ["unverified", "verified", "rejected"], selection_mode="multi", default=["unverified"]
    )
    for s in dataset.samples:
        if s.verification_status not in (show or []):
            continue
        with st.container(border=True):
            st.markdown(
                f"**{s.sample_id}** {badge(s.verification_status)} &nbsp; `{s.question_type}` "
                f"&nbsp; origin `{s.origin}`"
            )
            st.markdown(f"**Question:** {s.question}")
            st.markdown(f"**Ground truth:** {s.ground_truth or '_(none)_'}")
            with st.expander(f"Reference contexts ({len(s.reference_contexts)})"):
                for ctx, span in zip(s.reference_contexts, s.reference_spans):
                    where = (f"{span['doc_id']} chars {span['start_char']}-{span['end_char']}"
                             if span["match"] != "none" else "not found in corpus")
                    st.caption(f"anchored: {span['match']} · {where}")
                    st.text(ctx[:1500])
            if s.original:
                st.caption(f"Edited. Original question: {s.original.get('question')}")

            key = s.sample_id
            b1, b2, b3, _ = st.columns([1, 1, 1, 4])
            if b1.button("Verify", key=f"v_{key}", disabled=s.verification_status == "verified"):
                _apply(store, dataset_id, key, "verify")
            if b2.button("Reject", key=f"r_{key}", disabled=s.verification_status == "rejected"):
                _apply(store, dataset_id, key, "reject")
            if b3.button("Reset", key=f"u_{key}", disabled=s.verification_status == "unverified"):
                _apply(store, dataset_id, key, "unverify")
            with st.popover("Edit"):
                question = st.text_area("Question", s.question, key=f"q_{key}")
                truth = st.text_area("Ground truth", s.ground_truth or "", key=f"g_{key}")
                note = st.text_input("Note", s.review_note or "", key=f"n_{key}")
                if st.button("Save edit", key=f"e_{key}"):
                    _apply(store, dataset_id, key, "edit", question=question, ground_truth=truth, note=note)


def _apply(store: DatasetStore, dataset_id: str, sample_id: str, action: str, **kwargs) -> None:
    try:
        store.review(dataset_id, sample_id, action, **kwargs)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.rerun()


def render() -> None:
    st.title("Datasets")
    st.caption(
        "Synthetic samples start **unverified**. Only verified samples should be treated as ground "
        "truth; rejected ones are never evaluated. An edited sample goes back to unverified."
    )
    tab_review, tab_generate = st.tabs(["Review", "Generate"])
    with tab_generate:
        _generate_form()
        for job in reversed(jobs("dataset")):
            show_job(job)
    with tab_review:
        store = DatasetStore()
        datasets = store.list_datasets()
        if not datasets:
            st.info("No datasets yet - use the Generate tab.")
            return
        labels = {m["dataset_id"]: f"{m['dataset_id']}  {m.get('counts')}" for m in datasets}
        dataset_id = st.selectbox("Dataset", list(labels), format_func=labels.get)
        if dataset_id:
            _review(store, dataset_id)
