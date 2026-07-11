"""Milestone 1 smoke demo: tiny model + synthetic data -> pandas summary."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader, TensorDataset

from torchcheck import Evaluator, metrics


def main():
    torch.manual_seed(0)

    # Synthetic 3-class problem: 8 features -> 3 logits.
    n, n_features, n_classes = 500, 8, 3
    X = torch.randn(n, n_features)
    true_w = torch.randn(n_features, n_classes)
    y = (X @ true_w).argmax(dim=1)  # learnable structure
    # Flip 20% of labels so the model can't hit 100% -> F1 != accuracy.
    flip = torch.rand(n) < 0.20
    y[flip] = torch.randint(0, n_classes, (int(flip.sum()),))

    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, n_classes),
    )

    # Quick train so accuracy is meaningfully above chance.
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    lossf = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(200):
        opt.zero_grad()
        loss = lossf(model(X), y)
        loss.backward()
        opt.step()

    loader = DataLoader(TensorDataset(X, y), batch_size=64)
    evaluator = Evaluator(
        model=model,
        dataloader=loader,
        metrics=[metrics.Accuracy(), metrics.F1(average="macro")],
        device="cpu",
    )
    result = evaluator.run(tag="synthetic_v1", notes="M1 smoke test")

    print("=== result repr ===")
    print(result)
    print("\n=== summary (pandas Series) ===")
    print(result.summary)
    print("\n=== meta ===")
    print(result.meta)

    # Cross-check numpy metrics against sklearn on the same outputs.
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1).numpy()
    sk_acc = accuracy_score(y.numpy(), preds)
    sk_f1 = f1_score(y.numpy(), preds, average="macro")
    print("\n=== sklearn cross-check ===")
    print(f"sklearn accuracy={sk_acc:.6f}  f1_macro={sk_f1:.6f}")
    assert abs(sk_acc - result.summary["accuracy"]) < 1e-9
    assert abs(sk_f1 - result.summary["f1"]) < 1e-9
    print("OK: torchcheck matches sklearn to 1e-9")


if __name__ == "__main__":
    main()
