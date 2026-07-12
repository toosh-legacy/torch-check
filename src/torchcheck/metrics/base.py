"""The Metric base class and shared helpers.

A ``Metric`` receives the *raw model outputs* as ``predictions`` (logits for
classification, continuous values for regression) plus integer/float
``targets``, both as ``numpy`` arrays. Metrics interpret the outputs
themselves -- e.g. classification metrics argmax internally -- so that
metrics needing the full logits (TopK, ConfusionMatrix) work off the same
collected array.

Most metrics return a scalar from :meth:`compute`. A metric may instead
return a richer object (e.g. ``ConfusionMatrix`` returns a ``DataFrame``);
such metrics set ``scalar = False`` and the evaluator routes their output
into ``EvalResult.artifacts`` rather than the summary Series.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def to_labels(predictions: np.ndarray) -> np.ndarray:
    """Collapse (N, C) logits/probabilities to (N,) class labels.

    1-D inputs are assumed to already be class labels and returned as-is.
    """
    predictions = np.asarray(predictions)
    if predictions.ndim >= 2 and predictions.shape[-1] > 1:
        return predictions.argmax(axis=-1)
    return predictions.reshape(-1).astype(np.int64)


class Metric(ABC):
    """Base class for all metrics.

    Subclasses set a ``name`` (used as the column/key in results) and
    implement :meth:`compute`. Override :meth:`per_sample` to expose a
    per-example breakdown (e.g. per-sample correctness or squared error).
    """

    name: str
    #: Whether a larger value means a better model. Regression error
    #: metrics (MSE/MAE) override this to False.
    higher_is_better: bool = True
    #: Whether :meth:`compute` returns a scalar (True) or a richer artifact
    #: like a DataFrame (False, e.g. ConfusionMatrix).
    scalar: bool = True

    @abstractmethod
    def compute(self, predictions: np.ndarray, targets: np.ndarray):
        """Return the aggregate value of this metric."""

    def per_sample(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Return one value per sample. Override where meaningful."""
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a per-sample breakdown"
        )


#: Metric names where a smaller value is better. Used by the comparator,
#: which only has stored metric names (not Metric objects).
LOWER_IS_BETTER = {"mse", "mae"}


def higher_is_better(name: str) -> bool:
    """Whether a larger value of the named metric is better (default True)."""
    return name not in LOWER_IS_BETTER
