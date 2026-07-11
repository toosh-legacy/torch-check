"""Milestone 4: catch a deliberately-degraded checkpoint.

Trains a good model, then makes a worse 'checkpoint' by damaging its
weights, and shows compare_to() flagging both the metric drop and the
specific samples that flipped correct -> incorrect.
"""

import copy
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader, TensorDataset

from torchcheck import Evaluator, RunStore, metrics


def main():
    torch.manual_seed(0)
    n, d, c = 500, 8, 4
    X = torch.randn(n, d)
    w = torch.randn(d, c)
    y = (X @ w).argmax(1)
    flip = torch.rand(n) < 0.15
    y[flip] = torch.randint(0, c, (int(flip.sum()),))

    good = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Linear(64, c))
    opt = torch.optim.Adam(good.parameters(), lr=0.03)
    lf = torch.nn.CrossEntropyLoss()
    good.train()
    for _ in range(300):
        opt.zero_grad(); lf(good(X), y).backward(); opt.step()

    # Degraded "checkpoint": same architecture, damaged final layer.
    bad = copy.deepcopy(good)
    with torch.no_grad():
        bad[-1].weight.mul_(0.3)
        bad[-1].weight.add_(torch.randn_like(bad[-1].weight) * 0.8)

    loader = DataLoader(TensorDataset(X, y), batch_size=128)
    store = RunStore(Path(tempfile.mkdtemp()) / "store")
    mset = [metrics.Accuracy(), metrics.F1(average="macro")]

    r_v2 = Evaluator(good, loader, mset, "cpu", store=store).run(tag="checkpoint_v2")
    r_v3 = Evaluator(bad, loader, mset, "cpu", store=store).run(tag="checkpoint_v3")
    print(f"v2 acc={r_v2.summary['accuracy']:.3f}  v3 acc={r_v3.summary['accuracy']:.3f}\n")

    report = r_v3.compare_to("checkpoint_v2", threshold=0.02)
    print(report, "\n")
    print("=== metric deltas ===")
    print(report.metrics.to_string(index=False))

    ni = report.newly_incorrect()
    print(f"\n=== samples flipped correct -> incorrect: {len(ni)} ===")
    print(ni.head(10).to_string(index=False))
    print(f"\n(also {len(report.newly_correct())} flipped incorrect -> correct)")

    assert report.regressed, "should have detected a regression"
    print("\nOK: regression + per-sample flips detected")


if __name__ == "__main__":
    main()
