"""Train two checkpoints of intentionally different quality.

- ``digits_v2.pt``: well-trained (many epochs)  -> the good baseline.
- ``digits_v3.pt``: under-trained (few epochs)   -> the regressed candidate.

Run: python examples/train_checkpoints.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from examples.digits_pipeline import build_model, test_loader, train_loader

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"


def train(epochs: int, seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = build_model()
    loader = train_loader(seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()
    return model


@torch.no_grad()
def quick_acc(model) -> float:
    model.eval()
    correct = total = 0
    for xb, yb in test_loader():
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / total


def main():
    CKPT_DIR.mkdir(exist_ok=True)

    good = train(epochs=40)
    torch.save(good.state_dict(), CKPT_DIR / "digits_v2.pt")
    print(f"digits_v2 (40 epochs) test acc = {quick_acc(good):.4f}")

    bad = train(epochs=2)
    torch.save(bad.state_dict(), CKPT_DIR / "digits_v3.pt")
    print(f"digits_v3 ( 2 epochs) test acc = {quick_acc(bad):.4f}")


if __name__ == "__main__":
    main()
