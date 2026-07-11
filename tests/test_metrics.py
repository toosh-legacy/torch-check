"""Metric correctness, cross-checked against sklearn where applicable."""

import numpy as np
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
