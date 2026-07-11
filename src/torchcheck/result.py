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
    # Back-reference to the store, set by Evaluator.run so compare_to() works.
    _store: Any = field(default=None, repr=False, compare=False)

    @property
    def run_id(self) -> Optional[str]:
        return self.meta.get("run_id")

    @property
    def tag(self) -> Optional[str]:
        return self.meta.get("tag")

    def __repr__(self) -> str:
        parts = ", ".join(f"{k}={v:.4g}" for k, v in self.summary.items())
        return f"EvalResult(tag={self.tag!r}, {parts})"
