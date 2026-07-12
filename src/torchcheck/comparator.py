"""Regression detection: aggregate metric deltas + per-sample flips."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .metrics.base import higher_is_better

# Column in a per-sample frame that encodes correctness (0/1). Accuracy's
# per-sample output is exactly this.
CORRECTNESS_COL = "accuracy"


def _df_to_md(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Render a DataFrame as a GitHub markdown table (no tabulate dep)."""
    def fmt(v):
        if isinstance(v, float):
            return format(v, floatfmt)
        return str(v)

    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [
        "| " + " | ".join(fmt(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *rows])


@dataclass
class RegressionReport:
    """Comparison of a candidate run against a baseline run.

    Attributes:
        baseline_ref / candidate_ref: how the two runs were referenced.
        threshold: minimum degradation magnitude to flag a metric.
        metrics: per-metric table (baseline, candidate, delta, pct_change,
            direction, degradation, regressed).
        flips: per-sample table of examples that flipped correctness, with
            ``transition`` in {"correct->incorrect", "incorrect->correct"}.
            ``None`` when neither run has a correctness column.
    """

    baseline_ref: str
    candidate_ref: str
    threshold: float
    metrics: pd.DataFrame
    flips: Optional[pd.DataFrame] = None

    @property
    def regressed(self) -> bool:
        """True if any metric degraded past the threshold."""
        return bool(self.metrics["regressed"].any())

    @property
    def regressions(self) -> pd.DataFrame:
        return self.metrics[self.metrics["regressed"]]

    def newly_incorrect(self) -> pd.DataFrame:
        """Samples that went correct -> incorrect (the key debug view)."""
        if self.flips is None:
            return pd.DataFrame()
        return self.flips[self.flips["transition"] == "correct->incorrect"]

    def newly_correct(self) -> pd.DataFrame:
        if self.flips is None:
            return pd.DataFrame()
        return self.flips[self.flips["transition"] == "incorrect->correct"]

    def __repr__(self) -> str:
        n_reg = int(self.metrics["regressed"].sum())
        n_flip = 0 if self.flips is None else len(self.newly_incorrect())
        verdict = "REGRESSED" if self.regressed else "ok"
        return (
            f"RegressionReport({self.baseline_ref!r} -> {self.candidate_ref!r}: "
            f"{verdict}, {n_reg} metric(s) down, {n_flip} sample(s) newly wrong)"
        )

    def to_markdown(self) -> str:
        lines = [
            f"# Regression report: `{self.baseline_ref}` -> `{self.candidate_ref}`",
            "",
            f"**Verdict:** {'REGRESSED' if self.regressed else 'no regression'} "
            f"(threshold={self.threshold})",
            "",
            "## Metric deltas",
            "",
        ]
        tbl = self.metrics.copy()
        tbl["regressed"] = tbl["regressed"].map({True: "yes", False: ""})
        lines.append(_df_to_md(tbl))
        ni = self.newly_incorrect()
        lines += ["", f"## Per-sample flips (correct -> incorrect): {len(ni)}", ""]
        if len(ni):
            lines.append(_df_to_md(ni.head(20)))
        else:
            lines.append("_none_")
        return "\n".join(lines)


def _metric_table(
    baseline: dict[str, float], candidate: dict[str, float], threshold: float
) -> pd.DataFrame:
    rows = []
    for name in sorted(set(baseline) & set(candidate)):
        b = float(baseline[name])
        c = float(candidate[name])
        delta = c - b
        hib = higher_is_better(name)
        # positive degradation == worse
        degradation = -delta if hib else delta
        pct = (delta / abs(b) * 100.0) if b != 0 else float("nan")
        rows.append(
            {
                "metric": name,
                "baseline": b,
                "candidate": c,
                "delta": delta,
                "pct_change": pct,
                "direction": "higher_better" if hib else "lower_better",
                "degradation": degradation,
                "regressed": degradation > threshold,
            }
        )
    return pd.DataFrame(rows)


def _flip_table(
    base_ps: Optional[pd.DataFrame], cand_ps: Optional[pd.DataFrame]
) -> Optional[pd.DataFrame]:
    if base_ps is None or cand_ps is None:
        return None
    if CORRECTNESS_COL not in base_ps.columns or CORRECTNESS_COL not in cand_ps.columns:
        return None
    n = min(len(base_ps), len(cand_ps))
    b = base_ps.iloc[:n].reset_index(drop=True)
    c = cand_ps.iloc[:n].reset_index(drop=True)

    b_correct = b[CORRECTNESS_COL].to_numpy() > 0.5
    c_correct = c[CORRECTNESS_COL].to_numpy() > 0.5
    changed = b_correct != c_correct
    if not changed.any():
        return pd.DataFrame(
            columns=[
                "sample_id",
                "label",
                "baseline_prediction",
                "candidate_prediction",
                "transition",
            ]
        )

    idx = np.nonzero(changed)[0]
    transition = np.where(
        b_correct[idx] & ~c_correct[idx],
        "correct->incorrect",
        "incorrect->correct",
    )
    out = pd.DataFrame(
        {
            "sample_id": idx,
            "label": c["label"].to_numpy()[idx] if "label" in c else np.nan,
            "baseline_prediction": b["prediction"].to_numpy()[idx] if "prediction" in b else np.nan,
            "candidate_prediction": c["prediction"].to_numpy()[idx] if "prediction" in c else np.nan,
            "transition": transition,
        }
    )
    # newly-incorrect first -- that's what the user came to see
    out["_order"] = (out["transition"] == "correct->incorrect").map({True: 0, False: 1})
    return out.sort_values(["_order", "sample_id"]).drop(columns="_order").reset_index(drop=True)


class RegressionComparator:
    """Builds :class:`RegressionReport`s from runs in a :class:`RunStore`."""

    def __init__(self, store):
        self.store = store

    def compare(
        self,
        baseline_ref: str,
        candidate_ref: str,
        threshold: float = 0.0,
    ) -> RegressionReport:
        base = self.store.resolve(baseline_ref)
        cand = self.store.resolve(candidate_ref)
        if base is None:
            raise KeyError(f"no run found for baseline ref {baseline_ref!r}")
        if cand is None:
            raise KeyError(f"no run found for candidate ref {candidate_ref!r}")

        metrics_tbl = _metric_table(base["summary"], cand["summary"], threshold)
        base_ps = self.store.load_per_sample(base["run_id"])
        cand_ps = self.store.load_per_sample(cand["run_id"])
        flips = _flip_table(base_ps, cand_ps)

        return RegressionReport(
            baseline_ref=baseline_ref,
            candidate_ref=candidate_ref,
            threshold=threshold,
            metrics=metrics_tbl,
            flips=flips,
        )
