"""Result container returned by :meth:`Evaluator.run`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class EvalResult:
    """Outcome of a single evaluation run.

    Attributes:
        summary: aggregate metric values (one scalar per metric).
        per_sample: one row per example with prediction, label and any
            per-metric breakdowns. ``None`` until Milestone 2.
        meta: run metadata (tag, notes, throughput, timing, ...).
    """

    summary: pd.Series
    per_sample: Optional[pd.DataFrame] = None
    meta: dict[str, Any] = field(default_factory=dict)
    # Non-scalar metric outputs (e.g. ConfusionMatrix -> DataFrame), keyed
    # by metric name. Not persisted to the SQL summary; in-memory only.
    artifacts: dict[str, Any] = field(default_factory=dict)
    # Back-reference to the store, set by Evaluator.run so compare_to() works.
    _store: Any = field(default=None, repr=False, compare=False)

    @property
    def run_id(self) -> Optional[str]:
        return self.meta.get("run_id")

    def compare_to(self, baseline_ref: str, threshold: float = 0.0):
        """Compare this run against a stored baseline run.

        Returns a :class:`~torchcheck.regression.RegressionReport`. Requires
        this result to have been persisted (so both runs live in the store).
        """
        from .regression import RegressionComparator

        if self._store is None:
            raise RuntimeError(
                "compare_to() needs a run store; this result was not persisted "
                "(store=False or persist=False)"
            )
        if self.run_id is None:
            raise RuntimeError("this result has no run_id; it was not persisted")
        return RegressionComparator(self._store).compare(
            baseline_ref, self.run_id, threshold=threshold
        )

    @property
    def tag(self) -> Optional[str]:
        return self.meta.get("tag")

    def __repr__(self) -> str:
        parts = ", ".join(f"{k}={v:.4g}" for k, v in self.summary.items())
        return f"EvalResult(tag={self.tag!r}, {parts})"
