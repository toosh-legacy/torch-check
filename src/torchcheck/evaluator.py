"""The core evaluation loop."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .metrics import Metric
from .result import EvalResult


def _resolve_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _unpack_batch(batch: Any) -> tuple[Any, Any]:
    """Split a dataloader batch into (inputs, targets).

    Accepts the idiomatic ``(inputs, targets)`` tuple/list produced by
    ``TensorDataset`` and most torchvision datasets.
    """
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return batch[0], batch[1]
    raise ValueError(
        "Evaluator expects each batch to be a (inputs, targets) pair; "
        f"got {type(batch).__name__}. Wrap your dataset accordingly."
    )


class Evaluator:
    """Run a set of metrics over a model + dataloader.

    Accepts real ``nn.Module`` / ``DataLoader`` objects. The inference loop
    sets ``model.eval()``, runs under ``torch.no_grad()``, moves batches to
    the target device and collects raw outputs for metric computation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: Iterable,
        metrics: Sequence[Metric],
        device: Optional[str] = None,
    ):
        if not metrics:
            raise ValueError("at least one metric is required")
        self.model = model
        self.dataloader = dataloader
        self.metrics = list(metrics)
        self.device = _resolve_device(device)

    def _collect(self) -> tuple[np.ndarray, np.ndarray]:
        """Run inference and return (outputs, targets) as numpy arrays."""
        self.model.to(self.device)
        self.model.eval()

        outputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        with torch.no_grad():
            for batch in self.dataloader:
                inputs, y = _unpack_batch(batch)
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device, non_blocking=True)
                out = self.model(inputs)
                # Detach + move to CPU once per batch; no per-element sync.
                outputs.append(out.detach().to("cpu").numpy())
                targets.append(np.asarray(y))

        out_arr = np.concatenate(outputs, axis=0)
        tgt_arr = np.concatenate([np.atleast_1d(t) for t in targets], axis=0)
        return out_arr, tgt_arr

    def run(self, tag: Optional[str] = None, notes: Optional[str] = None) -> EvalResult:
        outputs, targets = self._collect()

        summary = {}
        for metric in self.metrics:
            summary[metric.name] = metric.compute(outputs, targets)
        summary_series = pd.Series(summary, name=tag)

        meta = {
            "tag": tag,
            "notes": notes,
            "n_samples": int(len(targets)),
            "device": str(self.device),
        }
        return EvalResult(summary=summary_series, meta=meta)
