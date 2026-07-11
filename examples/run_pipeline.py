"""Milestone 7: full pipeline via the Python API on real digit images.

Trains a good (v2) and under-trained (v3) checkpoint, evaluates both with
torchcheck, and prints the regression report that catches the quality drop.

Run: python examples/run_pipeline.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from examples.digits_pipeline import build_model, test_loader, train_loader
from torchcheck import Evaluator, RunStore, metrics


def train(epochs, seed=0):
    torch.manual_seed(seed)
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader(seed=seed):
            opt.zero_grad(); lossf(model(xb), yb).backward(); opt.step()
    return model


def main():
    store = RunStore(Path(tempfile.mkdtemp()) / "store")
    mset = [
        metrics.Accuracy(),
        metrics.F1(average="macro"),
        metrics.Precision(average="macro"),
        metrics.Recall(average="macro"),
        metrics.TopKAccuracy(k=3),
        metrics.ConfusionMatrix(),
    ]
    loader = test_loader()

    good = train(epochs=40)
    r_v2 = Evaluator(good, loader, mset, "cpu", store=store).run(
        tag="checkpoint_v2", notes="40 epochs"
    )
    bad = train(epochs=2)
    r_v3 = Evaluator(bad, loader, mset, "cpu", store=store).run(
        tag="checkpoint_v3", notes="2 epochs, under-trained"
    )

    print("=== summaries ===")
    print("v2:", r_v2.summary.to_dict())
    print("v3:", r_v3.summary.to_dict())

    report = r_v3.compare_to("checkpoint_v2", threshold=0.02)
    print("\n" + report.to_markdown())

    print("\n=== v3 confusion matrix artifact (collapsed to class 3) ===")
    print(r_v3.artifacts["confusion_matrix"])


if __name__ == "__main__":
    main()
