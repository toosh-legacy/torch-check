"""Thin typer CLI over the same Evaluator/config/store code path as the API.

No evaluation logic lives here -- every command delegates to the library so
CLI and API results are identical for the same config.
"""

from __future__ import annotations

from typing import Optional

import typer

from .config import run_from_config
from .regression import RegressionComparator
from .store import RunStore

app = typer.Typer(
    add_completion=False,
    help="Structured, reproducible evaluation for PyTorch models.",
)


def _store(path: Optional[str]) -> RunStore:
    return RunStore(path) if path else RunStore()


@app.command()
def run(
    config: str = typer.Option(..., "--config", "-c", help="Path to eval_config.yaml"),
    tag: Optional[str] = typer.Option(None, "--tag", help="Override the config's tag"),
    notes: Optional[str] = typer.Option(None, "--notes", help="Override notes"),
):
    """Run an evaluation from a config file (same path as the Python API)."""
    result = run_from_config(config, tag=tag, notes=notes)
    typer.echo(f"run_id: {result.run_id}")
    typer.echo(f"tag:    {result.tag}")
    typer.echo(f"git:    {result.meta.get('git_commit')}")
    typer.echo(
        f"throughput: {result.meta.get('throughput_samples_per_sec', float('nan')):.1f} samp/s"
    )
    typer.echo("\nsummary:")
    for name, value in result.summary.items():
        typer.echo(f"  {name:20s} {value:.6f}")


@app.command()
def history(
    tag: Optional[str] = typer.Argument(None, help="Filter by model tag"),
    store: Optional[str] = typer.Option(None, "--store", help="Store directory"),
):
    """List past runs (newest first)."""
    hist = _store(store).history(tag=tag)
    if hist.empty:
        typer.echo("no runs found")
        raise typer.Exit()
    keep = [c for c in ["run_id", "tag", "timestamp", "n_samples"] if c in hist.columns]
    metric_cols = [c for c in hist.columns if c not in
                   {"run_id", "tag", "notes", "git_commit", "dataset_id",
                    "timestamp", "n_samples", "device", "inference_seconds", "throughput"}]
    typer.echo(hist[keep + metric_cols].to_string(index=False))


@app.command()
def compare(
    baseline: str = typer.Argument(..., help="Baseline run_id or tag"),
    candidate: str = typer.Argument(..., help="Candidate run_id or tag"),
    threshold: float = typer.Option(0.0, "--threshold", "-t", help="Degradation to flag"),
    store: Optional[str] = typer.Option(None, "--store", help="Store directory"),
):
    """Compare two runs: metric deltas + samples that flipped correct->incorrect."""
    report = RegressionComparator(_store(store)).compare(baseline, candidate, threshold)
    typer.echo(str(report))
    typer.echo("\n" + report.metrics.to_string(index=False))
    ni = report.newly_incorrect()
    typer.echo(f"\nnewly incorrect samples: {len(ni)}")
    if len(ni):
        typer.echo(ni.head(15).to_string(index=False))
    if report.regressed:
        raise typer.Exit(code=1)  # non-zero so CI can gate on it


@app.command()
def report(
    run: str = typer.Argument(..., help="run_id or tag to report on"),
    baseline: Optional[str] = typer.Option(None, "--baseline", "-b", help="Compare against this run"),
    fmt: str = typer.Option("md", "--format", "-f", help="Output format: md"),
    store: Optional[str] = typer.Option(None, "--store", help="Store directory"),
):
    """Render a markdown report for a run (optionally vs a baseline)."""
    st = _store(store)
    resolved = st.resolve(run)
    if resolved is None:
        typer.echo(f"no run found for {run!r}")
        raise typer.Exit(code=1)

    if baseline:
        rep = RegressionComparator(st).compare(baseline, run)
        typer.echo(rep.to_markdown())
        return

    lines = [f"# Report: `{run}`", "", f"- run_id: `{resolved['run_id']}`",
             f"- tag: `{resolved['tag']}`", f"- git: `{resolved['git_commit']}`",
             f"- dataset: `{resolved['dataset_id']}`", f"- n_samples: {resolved['n_samples']}",
             "", "## Metrics", ""]
    for name, value in resolved["summary"].items():
        lines.append(f"- **{name}**: {value:.6f}")
    typer.echo("\n".join(lines))


if __name__ == "__main__":
    app()
