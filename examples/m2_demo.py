"""Milestone 2: per_sample DataFrame + throughput diagnostics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader, TensorDataset

from torchcheck import Evaluator, metrics


def main():
    torch.manual_seed(0)
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
    for _ in range(120):
        opt.zero_grad(); lf(model(X), y).backward(); opt.step()

    loader = DataLoader(TensorDataset(X, y), batch_size=64)
    ev = Evaluator(model, loader, [metrics.Accuracy(), metrics.F1()], device="cpu")
    r = ev.run(tag="v1")

    print("=== summary ===")
    print(r.summary)
    print("\n=== per_sample (head) ===")
    print(r.per_sample.head(8))
    print(f"\nper_sample shape: {r.per_sample.shape}, columns: {list(r.per_sample.columns)}")
    print("\n=== throughput / timing ===")
    print(f"n_samples          : {r.meta['n_samples']}")
    print(f"inference_seconds  : {r.meta['inference_seconds']:.4f}")
    print(f"throughput (samp/s): {r.meta['throughput_samples_per_sec']:.1f}")
    print(f"timestamp          : {r.meta['timestamp']}")

    # per-sample accuracy column mean must equal aggregate accuracy
    assert abs(r.per_sample["accuracy"].mean() - r.summary["accuracy"]) < 1e-9
    print("\nOK: per_sample 'accuracy' mean == aggregate accuracy")


if __name__ == "__main__":
    main()
