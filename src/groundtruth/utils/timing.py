"""Per-stage latency instrumentation.

p95 latency is a phase-3 deliverable, so every stage is timed from the start rather
than retrofitted later (invariant #6).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class Timer:
    """Accumulates named stage durations in milliseconds.

    >>> t = Timer()
    >>> with t("embed"):
    ...     pass
    >>> "embed_ms" in t.timings
    True
    """

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @contextmanager
    def __call__(self, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            # Accumulate rather than overwrite so a stage entered twice (e.g. a
            # reranker re-invoking a base retriever) reports total time spent.
            self.timings[f"{stage}_ms"] = self.timings.get(f"{stage}_ms", 0.0) + elapsed

    def merge(self, other: dict[str, float]) -> None:
        for key, value in other.items():
            self.timings[key] = self.timings.get(key, 0.0) + value

    def total(self) -> dict[str, float]:
        """Timings rounded for stable on-disk output."""
        return {k: round(v, 2) for k, v in self.timings.items()}
