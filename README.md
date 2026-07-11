# torchcheck

**Structured, reproducible, versioned evaluation for PyTorch models.**

ML engineers need a repeatable way to answer *"did this new checkpoint actually
get better, and where did it get worse?"* `torchcheck` replaces ad-hoc notebook
checks with versioned evaluation runs, persistent history, and automatic
regression detection — including the single most useful debugging view: **which
exact samples flipped from correct to incorrect** between two checkpoints.

It works as both an importable Python API and a CLI. The CLI is a thin wrapper
over the same code path as the API, so identical configs produce identical
results.

---

## Install

```bash
pip install -e .          # editable install from a clone
# provides both `import torchcheck` and the `torchcheck` CLI command
```

Requires Python ≥ 3.10, `torch`, `numpy`, `pandas`, `pyarrow`, `pyyaml`, `typer`.

---

## Quickstart (Python API)

```python
from torchcheck import Evaluator, metrics

evaluator = Evaluator(
    model=my_model,          # a real nn.Module
    dataloader=test_loader,  # a real torch DataLoader yielding (inputs, targets)
    metrics=[metrics.Accuracy(), metrics.F1(average="macro"), metrics.TopKAccuracy(k=3)],
    device="cuda",           # or "cpu"; None auto-detects
)

result = evaluator.run(tag="checkpoint_v3", notes="after LR schedule change")

result.summary        # pandas Series: {"accuracy": 0.91, "f1": 0.88, ...}
result.per_sample     # pandas DataFrame: one row/sample, prediction + label + per-metric breakdown
result.meta["throughput_samples_per_sec"]

report = result.compare_to("checkpoint_v2", threshold=0.02)   # -> RegressionReport
report.regressed                # bool
report.newly_incorrect()        # DataFrame of samples that went correct -> incorrect
```

Every `run()` is automatically persisted (SQLite metadata + parquet per-sample
data). Query the history for trend analysis:

```python
import torchcheck
torchcheck.history(tag="checkpoint_v3")   # DataFrame of past runs, newest first
```

---

## CLI

```bash
torchcheck run --config eval_config.yaml --tag checkpoint_v3
torchcheck history                       # or: torchcheck history checkpoint_v3
torchcheck compare checkpoint_v2 checkpoint_v3 --threshold 0.02
torchcheck report checkpoint_v3 --baseline checkpoint_v2 --format md
```

`torchcheck compare` exits with a **non-zero code when a regression is detected**,
so you can gate CI on it.

### Config format

Models and dataloaders aren't naturally YAML-expressible, so the config
*references code* (a `module:attribute` target plus kwargs) rather than trying to
replace it:

```yaml
model:
  target: examples.digits_pipeline:build_model   # module:callable
  checkpoint: examples/checkpoints/digits_v3.pt  # optional state_dict
dataloader:
  target: examples.digits_pipeline:test_loader
  args: { batch_size: 256 }
metrics:
  - { name: accuracy }
  - { name: f1, args: { average: macro } }
  - { name: top_k_accuracy, args: { k: 3 } }
device: cpu
tag: checkpoint_v3
notes: "under-trained"
store: .torchcheck
```

Targets are resolved relative to the directory you invoke `torchcheck` from.

---

## Metrics

Built-in (all computed in numpy, cross-checked against scikit-learn):

| Task | Metrics |
| --- | --- |
| Classification | `Accuracy`, `Precision`, `Recall`, `F1` (`average=macro/micro/weighted`), `TopKAccuracy(k)`, `ConfusionMatrix` |
| Regression | `MSE`, `MAE`, `R2` |

`ConfusionMatrix` returns a `DataFrame` (not a scalar); non-scalar metrics land in
`result.artifacts` instead of the summary Series.

Metrics are pluggable — subclass `Metric`:

```python
class MyMetric(Metric):
    name = "my_metric"
    higher_is_better = True

    def compute(self, predictions, targets) -> float:
        ...
    def per_sample(self, predictions, targets):   # optional
        ...
```

`predictions` are the **raw model outputs** (logits for classification); metrics
argmax internally as needed, so metrics that need the full logits (`TopK`,
`ConfusionMatrix`) work off the same collected array.

---

## End-to-end example: catching a real quality drop

`examples/` contains a small CNN trained on the scikit-learn handwritten-digits
dataset (1,797 real 8×8 images, 10 classes — bundled with scikit-learn, so it
runs fully offline with no torchvision download). We train two checkpoints of
deliberately different quality:

| checkpoint | training | test accuracy |
| --- | --- | --- |
| `checkpoint_v2` | 40 epochs (good) | **0.9704** |
| `checkpoint_v3` | 2 epochs (under-trained) | **0.2352** |

Reproduce:

```bash
python examples/train_checkpoints.py          # writes examples/checkpoints/*.pt
torchcheck run --config examples/eval_v2.yaml
torchcheck run --config examples/eval_v3.yaml
torchcheck compare checkpoint_v2 checkpoint_v3 --threshold 0.02
# ...or run the whole thing via the Python API:
python examples/run_pipeline.py
```

### Aggregate regression (real output)

```
RegressionReport('checkpoint_v2' -> 'checkpoint_v3': REGRESSED, 5 metric(s) down, 399 sample(s) newly wrong)
```

| metric | baseline | candidate | delta | pct_change | direction | degradation | regressed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| accuracy | 0.9704 | 0.2352 | -0.7352 | -75.76% | higher_better | 0.7352 | yes |
| f1 | 0.9704 | 0.1983 | -0.7722 | -79.57% | higher_better | 0.7722 | yes |
| precision | 0.9727 | 0.4889 | -0.4838 | -49.74% | higher_better | 0.4838 | yes |
| recall | 0.9700 | 0.2341 | -0.7359 | -75.87% | higher_better | 0.7359 | yes |
| top3_accuracy | 0.9926 | 0.7259 | -0.2667 | -26.87% | higher_better | 0.2667 | yes |

### Per-sample regression (the useful part)

399 samples flipped correct → incorrect. The breakdown immediately shows the
failure mode — the under-trained model collapsed to almost always predicting
class **3**:

| sample_id | label | baseline_prediction | candidate_prediction | transition |
| --- | --- | --- | --- | --- |
| 0 | 1 | 1 | 8 | correct->incorrect |
| 2 | 5 | 5 | 3 | correct->incorrect |
| 3 | 6 | 6 | 3 | correct->incorrect |
| 4 | 9 | 9 | 3 | correct->incorrect |

The `ConfusionMatrix` artifact confirms it — nearly every true class is dumped
into the predicted-3 column:

```
pred  0  1   2   3  4  5  6   7   8  9
true
0     0  0   0  54  0  0  0   0   0  0
3     0  0   0  55  0  0  0   0   0  0
6     0  0   0  54  0  0  0   0   0  0
```

This is the "did it get better, and *where* did it get worse" question answered
concretely, not just as a scalar.

---

## Design notes

- **Framework-idiomatic:** accepts real `nn.Module` / `DataLoader` objects, not
  custom wrappers. Batches are the standard `(inputs, targets)` pairs.
- **Inference loop:** `model.eval()` + `torch.no_grad()`, device placement, one
  CPU transfer per batch (no per-element sync), CUDA-synced timing for accurate
  throughput.
- **Storage:** SQLite for queryable run metadata (tag, git commit, dataset id,
  timestamp, throughput, summary metrics); parquet for the large per-sample
  frames so the SQL table stays lean.
- **Provenance:** every run records its git commit hash (when in a repo), so a
  stored result is traceable to the code that produced it.

## Out of scope (v1)

Distributed / multi-GPU evaluation, TensorFlow/JAX backends, a hosted dashboard,
LLM-judge qualitative grading, and AutoML/HPO integration are intentionally not
built.

---

## What's still rough / what I'd do differently at scale

Honest self-assessment — this is a portfolio piece, not battle-tested infra:

- **Whole-dataset materialization.** `Evaluator` concatenates every output and
  target into one in-memory numpy array before computing metrics. Fine for the
  test sets here; it would blow up on ImageNet-scale eval. At scale I'd move to
  streaming/online metric accumulation (update per batch, never hold all outputs)
  and only spill per-sample data to parquet incrementally.
- **Batch contract is rigid.** Only `(inputs, targets)` tuples are supported.
  Real models take dict batches, multiple inputs, masks, etc. I'd add a
  user-supplied `unpack` / `forward` hook rather than hard-coding the shape.
- **Per-sample alignment is positional.** Flip detection assumes both runs
  iterate the same dataset in the same order. That's true for a non-shuffled
  loader but silently wrong otherwise. A real version would key per-sample rows
  on a stable sample id from the dataset, not row position.
- **Metric direction is a hard-coded name set** (`{"mse","mae"}` are
  lower-is-better). Custom metrics carry a `higher_is_better` flag on the class,
  but the comparator only has stored *names*, so a custom lower-is-better metric
  wouldn't be judged correctly after persistence. I'd store direction alongside
  the value in the run record.
- **Artifacts aren't persisted.** `ConfusionMatrix` and other non-scalar outputs
  live only on the in-memory result, not in the store. I'd write them to parquet
  next to the per-sample data.
- **No concurrency story.** SQLite with default settings is fine for a single
  user; concurrent writers (a CI fleet hammering one store) would need WAL mode
  and retry logic, or a real DB.
- **Config imports are `eval`-adjacent.** `module:attr` resolution executes
  arbitrary user code by design. That's appropriate for a local dev tool but
  would need sandboxing before accepting configs from untrusted sources.
- **Device handling is single-device.** No sharding, no autocast/AMP toggles, no
  channels-last — deliberately, per the v1 scope, but that's the obvious next
  frontier.

---

## Tests

```bash
pip install -e ".[dev]"
pytest        # 30 tests: metric correctness vs sklearn, evaluator behaviour,
              # store round-trips, regression logic, and CLI==API parity.
```

## License

MIT — see [LICENSE](LICENSE).
