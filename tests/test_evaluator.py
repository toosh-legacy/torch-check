"""Evaluator loop behaviour."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from torchcheck import Evaluator, metrics


def _toy_setup(n=120, n_features=6, n_classes=3, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n, n_features)
    y = torch.randint(0, n_classes, (n,))
    model = torch.nn.Linear(n_features, n_classes)
    loader = DataLoader(TensorDataset(X, y), batch_size=32)
    return model, loader


def test_run_returns_summary_series():
    model, loader = _toy_setup()
    ev = Evaluator(model, loader, [metrics.Accuracy(), metrics.F1()], device="cpu")
    result = ev.run(tag="t1")
    assert set(result.summary.index) == {"accuracy", "f1"}
    assert result.meta["n_samples"] == 120
    assert 0.0 <= result.summary["accuracy"] <= 1.0


def test_run_sets_eval_mode_and_no_grad():
    model, loader = _toy_setup()
    model.train()  # deliberately leave in train mode
    ev = Evaluator(model, loader, [metrics.Accuracy()], device="cpu")
    ev.run()
    assert not model.training  # evaluator flipped it to eval


def test_requires_at_least_one_metric():
    model, loader = _toy_setup()
    try:
        Evaluator(model, loader, [], device="cpu")
        assert False, "expected ValueError"
    except ValueError:
        pass
