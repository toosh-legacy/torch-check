"""A small CNN on the scikit-learn handwritten-digits dataset.

1,797 real 8x8 images, 10 classes -- an MNIST stand-in that ships inside
scikit-learn, so this runs fully offline with no torchvision download. Swap
``_tensors()`` for ``torchvision.datasets.MNIST`` and nothing else here has
to change.

This module owns the *model and the data*. The evaluation itself lives in
``run_digits_sim.py``.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class DigitCNN(torch.nn.Module):
    """Tiny CNN for 1x8x8 digit images."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),  # 16 x 4 x 4
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),  # 32 x 2 x 2
        )
        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 2 * 2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def _tensors(seed: int = 0):
    """Deterministic train/test split as (Xtr, ytr, Xte, yte) tensors."""
    digits = load_digits()
    X = (digits.images / 16.0).astype(np.float32)[:, None, :, :]  # N,1,8,8
    y = digits.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    return (
        torch.from_numpy(Xtr),
        torch.from_numpy(ytr),
        torch.from_numpy(Xte),
        torch.from_numpy(yte),
    )


def train_loader(batch_size: int = 128, seed: int = 0) -> DataLoader:
    Xtr, ytr, _, _ = _tensors(seed=seed)
    return DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)


def holdout_loader(batch_size: int = 256, seed: int = 0) -> DataLoader:
    """Held-out test set. Not shuffled -- per-sample flip detection compares
    the two runs row by row, so both runs must see the same order.

    (Named ``holdout_`` rather than ``test_`` so pytest doesn't try to collect
    it as a test case.)"""
    _, _, Xte, yte = _tensors(seed=seed)
    return DataLoader(TensorDataset(Xte, yte), batch_size=batch_size)


def train(epochs: int, seed: int = 0) -> DigitCNN:
    """Train a DigitCNN for ``epochs`` passes. Fewer epochs -> worse model,
    which is how the simulation manufactures a regression to detect."""
    torch.manual_seed(seed)
    model = DigitCNN()
    loader = train_loader(seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return model
