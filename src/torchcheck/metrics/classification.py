"""Classification metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Metric, to_labels


def _prf_per_class(labels, targets, classes):
    """Per-class precision, recall, f1 and support."""
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
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        recall = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    return precision, recall, f1, support


def _average(values, support, average):
    if average == "macro":
        return float(values.mean()) if len(values) else 0.0
    total = support.sum()
    return float(np.sum(values * support) / total) if total else 0.0


class Accuracy(Metric):
    name = "accuracy"

    def compute(self, predictions, targets):
        return float(self.per_sample(predictions, targets).mean())

    def per_sample(self, predictions, targets):
        labels = to_labels(predictions)
        targets = np.asarray(targets).reshape(-1)
        return (labels == targets).astype(np.float64)


class _AveragedClfMetric(Metric):
    """Shared base for Precision/Recall/F1 with an ``average`` mode."""

    _index = 0  # which of (precision, recall, f1) to pick

    def __init__(self, average: str = "macro"):
        if average not in {"macro", "micro", "weighted"}:
            raise ValueError(f"unsupported average: {average!r}")
        self.average = average

    def compute(self, predictions, targets):
        labels = to_labels(predictions)
        targets = np.asarray(targets).reshape(-1)
        if self.average == "micro":
            # single-label micro P == R == F1 == accuracy
            total = len(targets)
            return float(np.sum(labels == targets) / total) if total else 0.0
        classes = np.unique(np.concatenate([labels, targets]))
        prf = _prf_per_class(labels, targets, classes)
        values = prf[self._index]
        support = prf[3]
        return _average(values, support, self.average)


class Precision(_AveragedClfMetric):
    name = "precision"
    _index = 0


class Recall(_AveragedClfMetric):
    name = "recall"
    _index = 1


class F1(_AveragedClfMetric):
    name = "f1"
    _index = 2


class TopKAccuracy(Metric):
    """Fraction of samples whose true label is within the top-k logits."""

    def __init__(self, k: int = 5):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.name = f"top{k}_accuracy"

    def _hits(self, predictions, targets):
        predictions = np.asarray(predictions)
        if predictions.ndim < 2:
            raise ValueError("TopKAccuracy requires 2-D logits/probabilities")
        targets = np.asarray(targets).reshape(-1)
        k = min(self.k, predictions.shape[1])
        topk = np.argpartition(-predictions, kth=k - 1, axis=1)[:, :k]
        return (topk == targets[:, None]).any(axis=1).astype(np.float64)

    def compute(self, predictions, targets):
        return float(self._hits(predictions, targets).mean())

    def per_sample(self, predictions, targets):
        return self._hits(predictions, targets)


class ConfusionMatrix(Metric):
    """Confusion matrix as a DataFrame (rows = true, cols = predicted)."""

    name = "confusion_matrix"
    scalar = False

    def compute(self, predictions, targets):
        labels = to_labels(predictions)
        targets = np.asarray(targets).reshape(-1)
        classes = np.unique(np.concatenate([labels, targets]))
        idx = {c: i for i, c in enumerate(classes)}
        mat = np.zeros((len(classes), len(classes)), dtype=int)
        for t, p in zip(targets, labels):
            mat[idx[t], idx[p]] += 1
        return pd.DataFrame(
            mat,
            index=pd.Index(classes, name="true"),
            columns=pd.Index(classes, name="pred"),
        )
