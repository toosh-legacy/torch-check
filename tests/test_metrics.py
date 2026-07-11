"""Metric correctness, cross-checked against sklearn where applicable."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

from torchcheck import metrics


@pytest.fixture
def clf_data():
    rng = np.random.default_rng(0)
    n, c = 300, 4
    logits = rng.normal(size=(n, c))
    targets = rng.integers(0, c, size=n)
    return logits, targets


def test_accuracy_matches_sklearn(clf_data):
    logits, targets = clf_data
    preds = logits.argmax(1)
    assert metrics.Accuracy().compute(logits, targets) == pytest.approx(
        accuracy_score(targets, preds)
    )


@pytest.mark.parametrize("avg", ["macro", "micro", "weighted"])
def test_f1_matches_sklearn(clf_data, avg):
    logits, targets = clf_data
    preds = logits.argmax(1)
    got = metrics.F1(average=avg).compute(logits, targets)
    expected = f1_score(targets, preds, average=avg)
    assert got == pytest.approx(expected)


def test_mse_matches_sklearn():
    rng = np.random.default_rng(1)
    pred = rng.normal(size=200)
    tgt = rng.normal(size=200)
    assert metrics.MSE().compute(pred, tgt) == pytest.approx(
        mean_squared_error(tgt, pred)
    )


def test_accuracy_per_sample():
    logits = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    targets = np.array([1, 1, 1])
    ps = metrics.Accuracy().per_sample(logits, targets)
    assert ps.tolist() == [1.0, 0.0, 1.0]


def test_f1_rejects_bad_average():
    with pytest.raises(ValueError):
        metrics.F1(average="bogus")


def test_precision_recall_topk_match_sklearn(clf_data):
    from sklearn.metrics import (
        precision_score,
        recall_score,
        top_k_accuracy_score,
    )

    logits, targets = clf_data
    preds = logits.argmax(1)
    n_classes = logits.shape[1]
    assert metrics.Precision("macro").compute(logits, targets) == pytest.approx(
        precision_score(targets, preds, average="macro", zero_division=0)
    )
    assert metrics.Recall("weighted").compute(logits, targets) == pytest.approx(
        recall_score(targets, preds, average="weighted", zero_division=0)
    )
    assert metrics.TopKAccuracy(3).compute(logits, targets) == pytest.approx(
        top_k_accuracy_score(targets, logits, k=3, labels=range(n_classes))
    )


def test_r2_mae_match_sklearn():
    from sklearn.metrics import mean_absolute_error, r2_score

    rng = np.random.default_rng(2)
    pred, tgt = rng.normal(size=150), rng.normal(size=150)
    assert metrics.R2().compute(pred, tgt) == pytest.approx(r2_score(tgt, pred))
    assert metrics.MAE().compute(pred, tgt) == pytest.approx(
        mean_absolute_error(tgt, pred)
    )
    assert metrics.MAE().higher_is_better is False


def test_confusion_matrix_is_dataframe():
    logits = np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    targets = np.array([0, 1, 1, 1])
    cm = metrics.ConfusionMatrix().compute(logits, targets)
    assert isinstance(cm, pd.DataFrame)
    assert metrics.ConfusionMatrix().scalar is False
    assert cm.loc[1, 0] == 1  # one true-1 predicted-0
    assert cm.values.sum() == 4
