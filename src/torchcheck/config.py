"""Load an ``eval_config.yaml`` into a ready-to-run :class:`Evaluator`.

Models and dataloaders are not YAML-expressible, so the config *references
code*: a ``module:attribute`` target plus kwargs. This is the single code
path the CLI uses, so ``torchcheck run --config ...`` and the Python API
produce identical results.

Example config::

    model:
      target: examples.mnist_pipeline:build_model   # module:callable
      args: {}
      checkpoint: checkpoints/model_v3.pt           # optional state_dict
    dataloader:
      target: examples.mnist_pipeline:test_loader
      args: {batch_size: 256}
    metrics:
      - {name: accuracy}
      - {name: f1, args: {average: macro}}
      - {name: top_k_accuracy, args: {k: 3}}
    device: cpu
    tag: checkpoint_v3
    notes: "after LR schedule change"
    store: .torchcheck
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from .evaluator import Evaluator
from . import metrics as _metrics

METRIC_REGISTRY = {
    "accuracy": _metrics.Accuracy,
    "precision": _metrics.Precision,
    "recall": _metrics.Recall,
    "f1": _metrics.F1,
    "top_k_accuracy": _metrics.TopKAccuracy,
    "topk": _metrics.TopKAccuracy,
    "confusion_matrix": _metrics.ConfusionMatrix,
    "mse": _metrics.MSE,
    "mae": _metrics.MAE,
    "r2": _metrics.R2,
}


def _import_target(target: str) -> Any:
    """Resolve a ``"module.path:attribute"`` reference to the object."""
    if ":" not in target:
        raise ValueError(
            f"target {target!r} must be of the form 'module.path:attribute'"
        )
    # Resolve targets relative to the invocation directory so user code
    # (e.g. examples/ or a project package) is importable via the CLI.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    module_path, attr = target.split(":", 1)
    module = importlib.import_module(module_path)
    obj = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a mapping")
    return cfg


def build_model(cfg: dict) -> torch.nn.Module:
    mcfg = cfg["model"]
    factory = _import_target(mcfg["target"])
    obj = factory(**mcfg.get("args", {})) if callable(factory) else factory

    checkpoint = mcfg.get("checkpoint")
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, torch.nn.Module):
            obj = state
        else:
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            obj.load_state_dict(state)
    if not isinstance(obj, torch.nn.Module):
        raise TypeError(f"model target did not produce an nn.Module: {type(obj)}")
    return obj


def build_dataloader(cfg: dict):
    dcfg = cfg["dataloader"]
    factory = _import_target(dcfg["target"])
    return factory(**dcfg.get("args", {})) if callable(factory) else factory


def build_metrics(cfg: dict) -> list:
    specs = cfg.get("metrics") or [{"name": "accuracy"}]
    built = []
    for spec in specs:
        if isinstance(spec, str):
            spec = {"name": spec}
        name = spec["name"].lower()
        if name not in METRIC_REGISTRY:
            raise KeyError(
                f"unknown metric {name!r}; known: {sorted(METRIC_REGISTRY)}"
            )
        built.append(METRIC_REGISTRY[name](**spec.get("args", {})))
    return built


def build_evaluator(cfg: dict) -> tuple[Evaluator, dict]:
    """Return an Evaluator plus the kwargs for its ``.run()`` call."""
    model = build_model(cfg)
    dataloader = build_dataloader(cfg)
    metric_objs = build_metrics(cfg)
    evaluator = Evaluator(
        model=model,
        dataloader=dataloader,
        metrics=metric_objs,
        device=cfg.get("device"),
        store=cfg.get("store"),  # None -> default store
    )
    run_kwargs = {
        "tag": cfg.get("tag"),
        "notes": cfg.get("notes"),
        "dataset_id": cfg.get("dataset_id"),
    }
    return evaluator, run_kwargs


def run_from_config(path: str | Path, **overrides):
    """Build and run an evaluation from a config file.

    ``overrides`` (e.g. ``tag=...``) take precedence over the config's own
    values. This is the exact function the CLI ``run`` command calls.
    """
    cfg = load_config(path)
    evaluator, run_kwargs = build_evaluator(cfg)
    run_kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return evaluator.run(**run_kwargs)
