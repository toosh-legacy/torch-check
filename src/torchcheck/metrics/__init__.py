"""Built-in metrics.

Import them from here regardless of which file they live in::

    from torchcheck import metrics
    metrics.Accuracy(), metrics.F1(average="macro"), metrics.MSE()

Write your own by subclassing :class:`~torchcheck.metrics.base.Metric`.
"""

from .base import LOWER_IS_BETTER, Metric, higher_is_better, to_labels
from .classification import (
    Accuracy,
    ConfusionMatrix,
    F1,
    Precision,
    Recall,
    TopKAccuracy,
)
from .regression import MAE, MSE, R2

__all__ = [
    "Metric",
    "to_labels",
    "higher_is_better",
    "LOWER_IS_BETTER",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "TopKAccuracy",
    "ConfusionMatrix",
    "MSE",
    "MAE",
    "R2",
]
