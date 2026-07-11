"""torchcheck: structured, reproducible evaluation for PyTorch models."""

from . import metrics
from .evaluator import Evaluator
from .metrics import Metric
from .result import EvalResult

__version__ = "0.1.0"

__all__ = ["Evaluator", "Metric", "EvalResult", "metrics", "__version__"]
