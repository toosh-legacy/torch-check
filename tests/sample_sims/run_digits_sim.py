"""End-to-end simulation: catch a real quality drop between two checkpoints.

Trains a well-trained model and an under-trained one, evaluates both with
torchcheck, then asks the comparator what got worse -- both in aggregate and
sample by sample.

Run it:

    python tests/sample_sims/run_digits_sim.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `torchcheck` and `digits_cnn` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from digits_cnn import holdout_loader, train  # noqa: E402

from torchcheck import Evaluator, RunStore, metrics  # noqa: E402

STORE_DIR = Path(__file__).resolve().parent / ".torchcheck"


def main() -> None:
    store = RunStore(STORE_DIR)
    loader = holdout_loader()
    ms = [
        metrics.Accuracy(),
        metrics.F1(average="macro"),
        metrics.TopKAccuracy(k=3),
        metrics.ConfusionMatrix(),
    ]

    print("training v1 (40 epochs, the good baseline)...")
    good = train(epochs=40)
    r_good = Evaluator(good, loader, ms, device="cpu", store=store).run(
        tag="v1", notes="40 epochs"
    )
    print(" ", r_good)

    print("training v2 (2 epochs, deliberately under-trained)...")
    bad = train(epochs=2)
    r_bad = Evaluator(bad, loader, ms, device="cpu", store=store).run(
        tag="v2", notes="2 epochs -- expect a regression"
    )
    print(" ", r_bad)

    report = r_bad.compare_to("v1", threshold=0.02)
    print()
    print(report.to_markdown())

    print()
    print("confusion matrix of the bad model (where it collapsed):")
    print(r_bad.artifacts["confusion_matrix"])


if __name__ == "__main__":
    main()
