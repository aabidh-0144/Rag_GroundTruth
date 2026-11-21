"""GroundTruth evaluation app - `groundtruth.cli ui` (or `streamlit run` this file).

Three pages, one file each under views/:

  Evaluate  run Mode 1 / Mode 2 on a RAG run, a dataset, or pasted JSON
  Results   history, per-sample drill-down, side-by-side comparison
  Datasets  generate synthetic datasets and review them (verify / reject / edit)

The app never computes scores itself: evaluations and dataset generation run as
`groundtruth.cli` subprocesses (see common.py), and the pages read what those
write to evals/ and datasets/.
"""

from __future__ import annotations

import streamlit as st

from groundtruth.ui.views import datasets, evaluate, results

st.set_page_config(page_title="GroundTruth", layout="wide")

page = st.navigation(
    [
        st.Page(evaluate.render, title="Evaluate", url_path="evaluate", default=True),
        st.Page(results.render, title="Results", url_path="results"),
        st.Page(datasets.render, title="Datasets", url_path="datasets"),
    ]
)
page.run()
