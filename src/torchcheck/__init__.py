"""torchcheck: structured, reproducible evaluation for PyTorch models.

One class per file:

- :class:`~torchcheck.evaluator.Evaluator`   -- runs metrics over a model + dataloader
- :class:`~torchcheck.result.EvalResult`     -- what a run gives you back
- :class:`~torchcheck.store.RunStore`        -- saves runs so you can compare later
- :class:`~torchcheck.comparator.RegressionComparator` -- baseline vs candidate
- ``torchcheck.metrics``                     -- the metric classes
"""

from typing import Optional

import pandas as pd

from . import metrics
from .comparator import RegressionComparator, RegressionReport
from .evaluator import Evaluator
from .metrics import Metric
from .result import EvalResult
from .store import RunStore

__version__ = "0.1.0"


def history(tag: Optional[str] = None, store: Optional[RunStore | str] = None) -> pd.DataFrame:
    """Return a DataFrame of past runs (optionally filtered by tag)."""
    if not isinstance(store, RunStore):
        store = RunStore(store) if store is not None else RunStore()
    return store.history(tag=tag)


__all__ = [
    "Evaluator",
    "Metric",
    "EvalResult",
    "RunStore",
    "RegressionComparator",
    "RegressionReport",
    "metrics",
    "history",
    "__version__",
]
