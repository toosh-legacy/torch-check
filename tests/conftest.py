"""Make ``src/torchcheck`` importable without installing the package first,
so a plain ``pytest`` from the repo root just works."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
