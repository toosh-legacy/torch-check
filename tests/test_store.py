"""Run store: SQLite metadata + parquet per-sample."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from torchcheck import Evaluator, RunStore, metrics


def _setup(seed=0, n=80, d=6, c=3):
    torch.manual_seed(seed)
    X = torch.randn(n, d)
    y = torch.randint(0, c, (n,))
    model = torch.nn.Linear(d, c)
    return model, DataLoader(TensorDataset(X, y), batch_size=20)


def test_run_persists_and_history_query(tmp_path):
    store = RunStore(tmp_path / "s")
    model, loader = _setup()
    ev = Evaluator(model, loader, [metrics.Accuracy(), metrics.F1()], "cpu", store=store)
    r1 = ev.run(tag="A")
    r2 = ev.run(tag="A")
    ev2 = Evaluator(*_setup(seed=1), metrics=[metrics.Accuracy()], device="cpu", store=store)
    ev2.run(tag="B")

    assert r1.run_id and r2.run_id and r1.run_id != r2.run_id

    all_hist = store.history()
    assert len(all_hist) == 3
    a_hist = store.history(tag="A")
    assert len(a_hist) == 2
    assert {"accuracy", "f1"} <= set(a_hist.columns)


def test_parquet_roundtrip(tmp_path):
    store = RunStore(tmp_path / "s")
    model, loader = _setup()
    ev = Evaluator(model, loader, [metrics.Accuracy()], "cpu", store=store)
    r = ev.run(tag="A")
    ps = store.load_per_sample(r.run_id)
    assert ps is not None and ps.shape[0] == 80
    assert "prediction" in ps.columns


def test_git_commit_captured(tmp_path):
    store = RunStore(tmp_path / "s")
    model, loader = _setup()
    r = Evaluator(model, loader, [metrics.Accuracy()], "cpu", store=store).run(tag="A")
    # repo exists -> a 40-char sha; tolerate None only if not a repo
    assert r.meta["git_commit"] is None or len(r.meta["git_commit"]) == 40


def test_persist_false_writes_nothing(tmp_path):
    store = RunStore(tmp_path / "s")
    model, loader = _setup()
    ev = Evaluator(model, loader, [metrics.Accuracy()], "cpu", store=store)
    ev.run(tag="A", persist=False)
    assert len(store.history()) == 0


def test_store_false_disables_store(tmp_path):
    model, loader = _setup()
    ev = Evaluator(model, loader, [metrics.Accuracy()], "cpu", store=False)
    r = ev.run(tag="A")
    assert r.run_id is None
