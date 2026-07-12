"""Regression metrics."""

from __future__ import annotations

import numpy as np

from .base import Metric


class MSE(Metric):
    name = "mse"
    higher_is_better = False

    def compute(self, predictions, targets):
        return float(self.per_sample(predictions, targets).mean())

    def per_sample(self, predictions, targets):
        predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
        targets = np.asarray(targets, dtype=np.float64).reshape(-1)
        return (predictions - targets) ** 2


class MAE(Metric):
    name = "mae"
    higher_is_better = False

    def compute(self, predictions, targets):
        return float(self.per_sample(predictions, targets).mean())

    def per_sample(self, predictions, targets):
        predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
        targets = np.asarray(targets, dtype=np.float64).reshape(-1)
        return np.abs(predictions - targets)


class R2(Metric):
    """Coefficient of determination."""

    name = "r2"

    def compute(self, predictions, targets):
        predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
        targets = np.asarray(targets, dtype=np.float64).reshape(-1)
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - targets.mean()) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
