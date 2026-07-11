"""Pluggable metric definitions.

A ``Metric`` receives the *raw model outputs* as ``predictions`` (logits for
classification, continuous values for regression) plus integer/float
``targets``, both as ``numpy`` arrays. Metrics interpret the outputs
themselves -- e.g. classification metrics argmax internally -- so that
metrics needing the full logits (TopK, ConfusionMatrix) work off the same
collected array.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def _to_labels(predictions: np.ndarray) -> np.ndarray:
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
    #: Whether a larger value means a better model. Regression metrics
    #: (MSE/MAE) override this to False.
    higher_is_better: bool = True

    @abstractmethod
    def compute(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Return the aggregate scalar value of this metric."""

    def per_sample(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Return one value per sample. Override where meaningful."""
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a per-sample breakdown"
        )


class Accuracy(Metric):
    name = "accuracy"

    def compute(self, predictions, targets):
        return float(self.per_sample(predictions, targets).mean())

    def per_sample(self, predictions, targets):
        labels = _to_labels(predictions)
        targets = np.asarray(targets).reshape(-1)
        return (labels == targets).astype(np.float64)


class F1(Metric):
    """F1 score for classification.

    ``average`` is one of ``"macro"``, ``"micro"``, ``"weighted"``.
    """

    def __init__(self, average: str = "macro"):
        if average not in {"macro", "micro", "weighted"}:
            raise ValueError(f"unsupported average: {average!r}")
        self.average = average
        self.name = "f1"

    def _per_class_stats(self, labels, targets, classes):
        tp = np.zeros(len(classes))
        fp = np.zeros(len(classes))
        fn = np.zeros(len(classes))
        support = np.zeros(len(classes))
        for i, c in enumerate(classes):
            pred_c = labels == c
            true_c = targets == c
            tp[i] = np.sum(pred_c & true_c)
            fp[i] = np.sum(pred_c & ~true_c)
            fn[i] = np.sum(~pred_c & true_c)
            support[i] = np.sum(true_c)
        return tp, fp, fn, support

    def compute(self, predictions, targets):
        labels = _to_labels(predictions)
        targets = np.asarray(targets).reshape(-1)
        classes = np.unique(np.concatenate([labels, targets]))

        if self.average == "micro":
            tp = np.sum(labels == targets)
            # micro precision == recall == accuracy for single-label
            total = len(targets)
            return float(tp / total) if total else 0.0

        tp, fp, fn, support = self._per_class_stats(labels, targets, classes)
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
            recall = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
            denom = precision + recall
            f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)

        if self.average == "macro":
            return float(f1.mean())
        # weighted
        total = support.sum()
        return float(np.sum(f1 * support) / total) if total else 0.0


class MSE(Metric):
    name = "mse"
    higher_is_better = False

    def compute(self, predictions, targets):
        return float(self.per_sample(predictions, targets).mean())

    def per_sample(self, predictions, targets):
        predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
        targets = np.asarray(targets, dtype=np.float64).reshape(-1)
        return (predictions - targets) ** 2


#: Metric names where a smaller value is better. Used by the regression
#: comparator, which only has stored metric names (not Metric objects).
LOWER_IS_BETTER = {"mse", "mae"}


def higher_is_better(name: str) -> bool:
    """Whether a larger value of the named metric is better (default True)."""
    return name not in LOWER_IS_BETTER
