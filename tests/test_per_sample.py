"""Per-sample DataFrame + throughput diagnostics."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from torchcheck import Evaluator, metrics


def _setup(n=100, d=6, c=3, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n, d)
    y = torch.randint(0, c, (n,))
    model = torch.nn.Linear(d, c)
    return model, DataLoader(TensorDataset(X, y), batch_size=25)


def test_per_sample_shape_and_columns():
    model, loader = _setup()
    r = Evaluator(model, loader, [metrics.Accuracy(), metrics.F1()], "cpu", store=False).run()
    assert r.per_sample.shape[0] == 100
    # F1 has no per_sample -> excluded; accuracy included
    assert "accuracy" in r.per_sample.columns
    assert "f1" not in r.per_sample.columns
    assert {"prediction", "label"} <= set(r.per_sample.columns)


def test_per_sample_accuracy_mean_equals_aggregate():
    model, loader = _setup()
    r = Evaluator(model, loader, [metrics.Accuracy()], "cpu", store=False).run()
    assert abs(r.per_sample["accuracy"].mean() - r.summary["accuracy"]) < 1e-9


def test_throughput_and_timing_present():
    model, loader = _setup()
    r = Evaluator(model, loader, [metrics.Accuracy()], "cpu", store=False).run()
    assert r.meta["throughput_samples_per_sec"] > 0
    assert r.meta["inference_seconds"] > 0
    assert "timestamp" in r.meta
