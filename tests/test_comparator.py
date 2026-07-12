"""Regression detection: metric deltas + per-sample flips."""

import pandas as pd
import pytest

from torchcheck.comparator import _flip_table, _metric_table


def test_metric_table_higher_is_better_regression():
    tbl = _metric_table({"accuracy": 0.90}, {"accuracy": 0.80}, threshold=0.02)
    row = tbl.iloc[0]
    assert row["delta"] == pytest.approx(-0.10)
    assert row["degradation"] == pytest.approx(0.10)
    assert bool(row["regressed"]) is True


def test_metric_table_improvement_not_flagged():
    tbl = _metric_table({"accuracy": 0.80}, {"accuracy": 0.90}, threshold=0.0)
    assert bool(tbl.iloc[0]["regressed"]) is False


def test_metric_table_threshold_gates_small_drop():
    tbl = _metric_table({"f1": 0.90}, {"f1": 0.895}, threshold=0.02)
    assert bool(tbl.iloc[0]["regressed"]) is False  # 0.005 < 0.02


def test_metric_table_lower_is_better_mse():
    # MSE going up is a regression; going down is an improvement.
    up = _metric_table({"mse": 0.10}, {"mse": 0.20}, threshold=0.0).iloc[0]
    down = _metric_table({"mse": 0.20}, {"mse": 0.10}, threshold=0.0).iloc[0]
    assert up["direction"] == "lower_better"
    assert bool(up["regressed"]) is True
    assert bool(down["regressed"]) is False


def test_flip_table_detects_transitions():
    base = pd.DataFrame(
        {"prediction": [1, 1, 0, 2], "label": [1, 1, 1, 2], "accuracy": [1.0, 1.0, 0.0, 1.0]}
    )
    cand = pd.DataFrame(
        {"prediction": [0, 1, 1, 0], "label": [1, 1, 1, 2], "accuracy": [0.0, 1.0, 1.0, 0.0]}
    )
    flips = _flip_table(base, cand)
    ni = flips[flips["transition"] == "correct->incorrect"]
    nc = flips[flips["transition"] == "incorrect->correct"]
    assert set(ni["sample_id"]) == {0, 3}
    assert set(nc["sample_id"]) == {2}


def test_flip_table_none_without_correctness_col():
    df = pd.DataFrame({"prediction": [1.0], "label": [1.0]})  # regression-style
    assert _flip_table(df, df) is None


def test_compare_to_end_to_end(tmp_path):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from torchcheck import Evaluator, RunStore, metrics

    torch.manual_seed(0)
    X = torch.randn(200, 6)
    w = torch.randn(6, 3)
    y = (X @ w).argmax(1)
    loader = DataLoader(TensorDataset(X, y), batch_size=64)

    good = torch.nn.Linear(6, 3)
    opt = torch.optim.Adam(good.parameters(), lr=0.1)
    lf = torch.nn.CrossEntropyLoss()
    for _ in range(200):
        opt.zero_grad(); lf(good(X), y).backward(); opt.step()

    import copy
    bad = copy.deepcopy(good)
    with torch.no_grad():
        bad.weight.add_(torch.randn_like(bad.weight))

    store = RunStore(tmp_path / "s")
    ms = [metrics.Accuracy(), metrics.F1()]
    Evaluator(good, loader, ms, "cpu", store=store).run(tag="base")
    r_bad = Evaluator(bad, loader, ms, "cpu", store=store).run(tag="cand")

    report = r_bad.compare_to("base", threshold=0.02)
    assert report.regressed
    assert len(report.newly_incorrect()) > 0
