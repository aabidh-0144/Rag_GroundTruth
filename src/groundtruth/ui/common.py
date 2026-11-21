"""Helpers shared by the Streamlit pages.

Long jobs (evaluations, dataset generation) are run as `groundtruth.cli`
subprocesses, never inside the Streamlit script: a local-CPU evaluation can take
many minutes, and running it in-process would freeze the page and die with the
browser tab. The page starts the process, then polls its log file and the
evaluation's results.jsonl. The CLI stays the single place the work is done.

    evals/_jobs/<timestamp>_<kind>.log   stdout/stderr of each launched job
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from groundtruth.config import ROOT, RUNS_DIR
from groundtruth.evaluation.store import EVALS_DIR

JOBS_DIR = EVALS_DIR / "_jobs"
INPUTS_DIR = EVALS_DIR / "_inputs"

STATUS_COLOR = {
    "success": "green", "skipped": "gray", "failed": "red",
    "completed": "green", "partial": "orange",
    "verified": "green", "unverified": "gray", "rejected": "red",
}


@dataclass
class Job:
    kind: str
    args: list[str]
    log_path: Path
    process: subprocess.Popen

    @property
    def running(self) -> bool:
        return self.process.poll() is None

    @property
    def returncode(self) -> int | None:
        return self.process.poll()

    def log_tail(self, lines: int = 40) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return ""
        return "\n".join(text.splitlines()[-lines:])


def launch(kind: str, args: list[str]) -> Job:
    """Start `python -m groundtruth.cli <args>` in the background."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = JOBS_DIR / f"{dt.datetime.now():%Y%m%d_%H%M%S}_{kind}.log"
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "groundtruth.cli", *args],
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
    )
    job = Job(kind=kind, args=args, log_path=log_path, process=process)
    st.session_state.setdefault("jobs", []).append(job)
    return job


def jobs(kind: str | None = None) -> list[Job]:
    return [j for j in st.session_state.get("jobs", []) if kind is None or j.kind == kind]


def show_job(job: Job) -> None:
    """Live log panel for one job; refreshes itself while the job runs."""

    @st.fragment(run_every=2 if job.running else None)
    def panel() -> None:
        if job.running:
            st.info(f"Running: `groundtruth.cli {' '.join(job.args)}`")
        elif job.returncode == 0:
            st.success("Finished. See the Results page.")
        else:
            st.error(f"Exited with code {job.returncode} - see the log below.")
        st.code(job.log_tail() or "(starting...)", language=None)
        if job.running and st.button("Stop", key=f"stop_{job.log_path.name}"):
            job.process.terminate()
            st.rerun()

    panel()


def run_dirs() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    return sorted((p.name for p in RUNS_DIR.iterdir() if (p / "results.jsonl").exists()), reverse=True)


def experiment_configs() -> list[str]:
    return sorted(
        str(p.relative_to(ROOT)).replace("\\", "/")
        for p in (ROOT / "configs").glob("*.yaml")
        if p.name != "evaluation.yaml"
    )


def fmt_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def badge(status: str) -> str:
    """Inline coloured label using Streamlit's markdown colour syntax."""
    return f":{STATUS_COLOR.get(status, 'gray')}-badge[{status}]"
