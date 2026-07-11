"""Milestone 3: persist runs (SQLite + parquet) and query history()."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader, TensorDataset

import torchcheck
from torchcheck import Evaluator, RunStore, metrics


def make_model_and_loader(seed, train_steps):
    torch.manual_seed(seed)
    n, d, c = 400, 8, 3
    X = torch.randn(n, d)
    w = torch.randn(d, c)
    y = (X @ w).argmax(1)
    flip = torch.rand(n) < 0.25
    y[flip] = torch.randint(0, c, (int(flip.sum()),))
    model = torch.nn.Sequential(torch.nn.Linear(d, 32), torch.nn.ReLU(), torch.nn.Linear(32, c))
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    lf = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(train_steps):
        opt.zero_grad(); lf(model(X), y).backward(); opt.step()
    return model, DataLoader(TensorDataset(X, y), batch_size=64)


def main():
    store = RunStore(Path(tempfile.mkdtemp()) / "store")

    # Two runs of the same model tag with different training budgets.
    for steps in (10, 150):
        model, loader = make_model_and_loader(seed=0, train_steps=steps)
        ev = Evaluator(model, loader, [metrics.Accuracy(), metrics.F1()], "cpu", store=store)
        r = ev.run(tag="mymodel", notes=f"{steps} train steps")
        print(f"saved run_id={r.run_id}  git={r.meta['git_commit'][:8]}  acc={r.summary['accuracy']:.3f}")

    print("\n=== torchcheck.history(tag='mymodel') ===")
    hist = torchcheck.history(tag="mymodel", store=store)
    cols = ["run_id", "tag", "dataset_id", "n_samples", "accuracy", "f1", "throughput"]
    print(hist[cols].to_string(index=False))

    print("\n=== parquet round-trip (per_sample of newest run) ===")
    newest = hist.iloc[0]["run_id"]
    ps = store.load_per_sample(newest)
    print(ps.head(5).to_string())
    print(f"loaded per_sample shape: {ps.shape}")


if __name__ == "__main__":
    main()
