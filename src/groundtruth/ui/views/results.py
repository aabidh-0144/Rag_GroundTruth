"""Results page: history, one evaluation in detail, and side-by-side comparison."""

from __future__ import annotations

import streamlit as st

from groundtruth.evaluation.store import EvaluationStore
from groundtruth.ui.common import badge, fmt_score


def _history_rows(results) -> list[dict]:
    return [
        {
            "evaluation_id": r.evaluation_id,
            "mode": r.mode,
            "status": r.status,
            "samples": f"{r.n_samples_evaluated}/{r.n_samples}",
            "retrieval": r.retrieval_score,
            "generation": r.generation_score,
            "overall": r.overall_score,
            "judge": f"{r.judge.get('provider')}:{r.judge.get('model')}",
            "source": r.source.get("run_id") or r.source.get("dataset_id") or r.source.get("type"),
            "started": r.started_at,
        }
        for r in results
    ]


def _detail(store: EvaluationStore, evaluation_id: str) -> None:
    result = store.get(evaluation_id)
    st.subheader(evaluation_id)
    st.markdown(
        f"{badge(result.status)} &nbsp; mode `{result.mode}` &nbsp; judge "
        f"`{result.judge.get('provider')}:{result.judge.get('model')}` &nbsp; "
        f"{result.execution_time_s}s"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Retrieval", fmt_score(result.retrieval_score))
    c2.metric("Generation", fmt_score(result.generation_score))
    c3.metric("Overall", fmt_score(result.overall_score))
    st.caption(f"How the overall score is computed: {result.overall_score_method}")
    for err in result.errors:
        st.warning(err)

    st.dataframe(
        [{"metric": n, **a.to_dict()} for n, a in result.aggregates.items()],
        hide_index=True, width="stretch",
    )

    st.markdown("#### Samples")
    show = st.pills(
        "Show samples with", ["failed", "skipped", "success"], selection_mode="multi",
        default=["failed", "skipped", "success"],
    )
    for sample in store.sample_results(evaluation_id):
        statuses = {m.status for m in sample.metrics.values()}
        if not statuses & set(show or []):
            continue
        title = f"{sample.sample_id} — {sample.question[:90]}" + ("  (refusal)" if sample.refusal else "")
        with st.expander(title):
            st.caption(
                f"retrieval {fmt_score(sample.retrieval_score)} · "
                f"generation {fmt_score(sample.generation_score)}"
                + (f" · origin {sample.origin}" if sample.origin else "")
            )
            for name, m in sample.metrics.items():
                st.markdown(f"**{name}** {badge(m.status)} &nbsp; {fmt_score(m.score)}")
                st.write(m.error or m.reason)
                if m.details:
                    with st.popover("details"):
                        st.json(m.details)


def _compare(results) -> None:
    ids = st.multiselect(
        "Compare evaluations", [r.evaluation_id for r in results], max_selections=4,
        help="Only compare evaluations over the same samples - check the source column.",
    )
    if len(ids) < 2:
        return
    chosen = [r for r in results if r.evaluation_id in ids]
    if len({r.source.get("index_fingerprint") for r in chosen}) > 1:
        st.warning("These evaluations come from runs over different indexes - the comparison is not like-for-like.")
    metrics = sorted({m for r in chosen for m in r.aggregates})
    rows = []
    for name in metrics + ["retrieval", "generation", "overall"]:
        row = {"metric": name}
        for r in chosen:
            if name in r.aggregates:
                row[r.evaluation_id] = r.aggregates[name].mean
            else:
                row[r.evaluation_id] = getattr(r, f"{name}_score", None)
        rows.append(row)
    st.dataframe(rows, hide_index=True, width="stretch")


def render() -> None:
    st.title("Results")
    store = EvaluationStore()
    results = store.history()
    if not results:
        st.info("No evaluations yet - start one on the Evaluate page.")
        return

    st.dataframe(_history_rows(results), hide_index=True, width="stretch")
    tab_detail, tab_compare = st.tabs(["Inspect one", "Compare"])
    with tab_detail:
        evaluation_id = st.selectbox("Evaluation", [r.evaluation_id for r in results])
        if evaluation_id:
            _detail(store, evaluation_id)
    with tab_compare:
        _compare(results)
