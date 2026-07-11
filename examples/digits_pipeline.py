"""A small CNN on the sklearn handwritten-digits dataset (8x8, 10 classes).

Real image data, bundled with scikit-learn, so it works fully offline with
no torchvision. These functions are the code the eval configs reference:
``build_model`` constructs the architecture and ``test_loader`` builds the
held-out DataLoader.
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


def build_model(n_classes: int = 10) -> DigitCNN:
    """Config entry point: construct the model architecture."""
    return DigitCNN(n_classes=n_classes)


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


def test_loader(batch_size: int = 256, seed: int = 0) -> DataLoader:
    """Config entry point: held-out test DataLoader (deterministic)."""
    _, _, Xte, yte = _tensors(seed=seed)
    return DataLoader(TensorDataset(Xte, yte), batch_size=batch_size)


def train_loader(batch_size: int = 128, seed: int = 0) -> DataLoader:
    Xtr, ytr, _, _ = _tensors(seed=seed)
    return DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
