"""The core evaluation loop."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .metrics import Metric, _to_labels
from .result import EvalResult

if TYPE_CHECKING:
    from .store import RunStore


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


def _derive_predictions(outputs: np.ndarray) -> np.ndarray:
    """Human-facing prediction column: class label for classification,
    raw value for regression/single-output."""
    if outputs.ndim >= 2 and outputs.shape[-1] > 1:
        return _to_labels(outputs)
    return outputs.reshape(-1)


class Evaluator:
    """Run a set of metrics over a model + dataloader.

    Accepts real ``nn.Module`` / ``DataLoader`` objects. The inference loop
    sets ``model.eval()``, runs under ``torch.no_grad()``, moves batches to
    the target device and collects raw outputs for metric computation.
    Inference wall-time is measured to report throughput (samples/sec).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: Iterable,
        metrics: Sequence[Metric],
        device: Optional[str] = None,
        store: "Optional[RunStore | str]" = None,
    ):
        if not metrics:
            raise ValueError("at least one metric is required")
        self.model = model
        self.dataloader = dataloader
        self.metrics = list(metrics)
        self.device = _resolve_device(device)
        # store=None -> default on-disk store; store=False -> no persistence.
        from .store import RunStore

        if store is False:
            self.store = None
        elif store is None or isinstance(store, (str, Path)):
            self.store = RunStore(store) if isinstance(store, (str, Path)) else RunStore()
        else:
            self.store = store

    def _collect(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Run inference; return (outputs, targets, inference_seconds)."""
        self.model.to(self.device)
        self.model.eval()

        outputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        cuda = self.device.type == "cuda"

        if cuda:
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with torch.no_grad():
            for batch in self.dataloader:
                inputs, y = _unpack_batch(batch)
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device, non_blocking=True)
                out = self.model(inputs)
                # One CPU transfer per batch; no per-element sync points.
                outputs.append(out.detach().to("cpu").numpy())
                targets.append(np.asarray(y))
        if cuda:
            torch.cuda.synchronize(self.device)
        inference_seconds = time.perf_counter() - start

        out_arr = np.concatenate(outputs, axis=0)
        tgt_arr = np.concatenate([np.atleast_1d(t) for t in targets], axis=0)
        return out_arr, tgt_arr, inference_seconds

    def _build_per_sample(
        self, outputs: np.ndarray, targets: np.ndarray
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "prediction": _derive_predictions(outputs),
                "label": targets.reshape(-1),
            }
        )
        for metric in self.metrics:
            try:
                values = metric.per_sample(outputs, targets)
            except NotImplementedError:
                continue
            df[metric.name] = np.asarray(values).reshape(-1)
        df.index.name = "sample_id"
        return df

    def run(
        self,
        tag: Optional[str] = None,
        notes: Optional[str] = None,
        dataset_id: Optional[str] = None,
        persist: bool = True,
    ) -> EvalResult:
        outputs, targets, inference_seconds = self._collect()
        n = int(len(targets))

        summary = {}
        artifacts = {}
        for m in self.metrics:
            value = m.compute(outputs, targets)
            if getattr(m, "scalar", True):
                summary[m.name] = value
            else:
                artifacts[m.name] = value
        summary_series = pd.Series(summary, name=tag)

        per_sample = self._build_per_sample(outputs, targets)

        throughput = n / inference_seconds if inference_seconds > 0 else float("nan")
        meta = {
            "tag": tag,
            "notes": notes,
            "n_samples": n,
            "device": str(self.device),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inference_seconds": inference_seconds,
            "throughput_samples_per_sec": throughput,
        }
        result = EvalResult(
            summary=summary_series, per_sample=per_sample, meta=meta, artifacts=artifacts
        )
        result._store = self.store  # let result.compare_to() reach the store

        if persist and self.store is not None:
            from .store import _dataset_id

            ds = dataset_id if dataset_id is not None else _dataset_id(self.dataloader)
            self.store.save(result, dataset_id=ds)
        return result
