"""CLI behaviour + the CLI/API parity guarantee (Milestone 5)."""

import textwrap

import numpy as np
from typer.testing import CliRunner

from torchcheck import RunStore
from torchcheck.cli import app
from torchcheck.config import run_from_config

runner = CliRunner()


def _write_config(tmp_path, tag, store_dir):
    cfg = tmp_path / f"{tag}.yaml"
    cfg.write_text(
        textwrap.dedent(
            f"""
            model:
              target: parity_pipeline:build_model
            dataloader:
              target: parity_pipeline:loader
              args: {{batch_size: 20}}
            metrics:
              - {{name: accuracy}}
              - {{name: f1, args: {{average: macro}}}}
              - {{name: precision, args: {{average: macro}}}}
            device: cpu
            tag: {tag}
            store: {store_dir.as_posix()}
            """
        )
    )
    return cfg


def test_cli_and_api_produce_identical_results(tmp_path):
    store_dir = tmp_path / "store"
    cfg = _write_config(tmp_path, "parity", store_dir)

    # API path
    api_result = run_from_config(cfg, tag="api_run")

    # CLI path (same config, in-process typer invocation)
    res = runner.invoke(app, ["run", "--config", str(cfg), "--tag", "cli_run"])
    assert res.exit_code == 0, res.output

    store = RunStore(store_dir)
    api = store.history(tag="api_run").iloc[0]
    cli = store.history(tag="cli_run").iloc[0]

    for metric in ["accuracy", "f1", "precision"]:
        assert np.isclose(api[metric], cli[metric]), metric
        # and both equal the in-memory API result
        assert np.isclose(api[metric], api_result.summary[metric])


def test_cli_history_and_compare(tmp_path):
    store_dir = tmp_path / "store"
    cfg = _write_config(tmp_path, "parity", store_dir)
    runner.invoke(app, ["run", "--config", str(cfg), "--tag", "base"])
    runner.invoke(app, ["run", "--config", str(cfg), "--tag", "cand"])

    h = runner.invoke(app, ["history", "--store", str(store_dir)])
    assert h.exit_code == 0
    assert "base" in h.output and "cand" in h.output

    # identical models -> no regression -> exit 0
    c = runner.invoke(app, ["compare", "base", "cand", "--store", str(store_dir)])
    assert c.exit_code == 0
    assert "RegressionReport" in c.output
