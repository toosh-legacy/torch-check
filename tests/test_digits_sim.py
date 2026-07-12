"""The CNN simulation, run as a test so the worked example can't rot.

Same flow as ``sample_sims/run_digits_sim.py`` but with far fewer epochs to
keep the suite fast.
"""

import pytest

from sample_sims.digits_cnn import holdout_loader, train
from torchcheck import Evaluator, RunStore, metrics


@pytest.mark.slow
def test_undertrained_checkpoint_is_caught_as_a_regression(tmp_path):
    store = RunStore(tmp_path / "store")
    loader = holdout_loader()
    ms = [metrics.Accuracy(), metrics.F1(average="macro")]

    good = Evaluator(train(epochs=8), loader, ms, "cpu", store=store).run(tag="v1")
    bad = Evaluator(train(epochs=1), loader, ms, "cpu", store=store).run(tag="v2")

    assert good.summary["accuracy"] > bad.summary["accuracy"]

    report = bad.compare_to("v1", threshold=0.02)
    assert report.regressed
    assert "accuracy" in set(report.regressions["metric"])

    # The point of the tool: name the samples that went correct -> incorrect.
    newly_wrong = report.newly_incorrect()
    assert len(newly_wrong) > 0
    assert (newly_wrong["baseline_prediction"] == newly_wrong["label"]).all()
    assert (newly_wrong["candidate_prediction"] != newly_wrong["label"]).all()
