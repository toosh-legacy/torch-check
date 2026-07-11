"""Deterministic model + loader used by the CLI/API parity test.

Seeding inside ``build_model`` guarantees identical weights every call, so
the API path and CLI path evaluate the exact same model on the exact same
data -- any difference in results would be a real bug.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset


def build_model():
    torch.manual_seed(1234)
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16, 3))


def loader(batch_size: int = 20):
    torch.manual_seed(0)
    X = torch.randn(60, 16)
    y = torch.randint(0, 3, (60,))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)
