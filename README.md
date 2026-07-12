# torchcheck

**Structured, reproducible evaluation for PyTorch models.**

Answering *"did this new checkpoint actually get better, and where did it get
worse?"* usually means ad-hoc notebook cells that get thrown away. `torchcheck`
replaces them with versioned evaluation runs, persistent history, and automatic
regression detection — including the single most useful debugging view: **which
exact samples flipped from correct to incorrect** between two checkpoints.

Pure Python API. No CLI, no config files.

---

## Layout

```
src/torchcheck/
  evaluator.py     Evaluator             -- runs metrics over a model + dataloader
  result.py        EvalResult            -- what a run gives you back
  store.py         RunStore              -- saves runs so you can compare later
  comparator.py    RegressionComparator  -- baseline vs candidate, + RegressionReport
  metrics/
    base.py            Metric (the ABC you subclass)
    classification.py  Accuracy, Precision, Recall, F1, TopKAccuracy, ConfusionMatrix
    regression.py      MSE, MAE, R2

tests/
  test_*.py            the test suite
  sample_sims/         worked end-to-end examples on a real CNN
```

## Install

```bash
pip install -e ".[dev]"     # editable install from a clone, with test deps
```

Requires Python ≥ 3.10, `torch`, `numpy`, `pandas`, `pyarrow`.

---

## Quickstart

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
result.per_sample     # pandas DataFrame: one row/sample, prediction + label + breakdown
result.meta["throughput_samples_per_sec"]

report = result.compare_to("checkpoint_v2", threshold=0.02)   # -> RegressionReport
report.regressed                # bool
report.newly_incorrect()        # DataFrame of samples that went correct -> incorrect
print(report.to_markdown())
```

Every `run()` is persisted by default (SQLite metadata + parquet per-sample
data) so a later run can compare against it. Pass `store=False` to skip that.
Query the history for trend analysis:

```python
import torchcheck
torchcheck.history(tag="checkpoint_v3")   # DataFrame of past runs, newest first
```

---

## Metrics

Built-in (computed in numpy, cross-checked against scikit-learn in the tests):

| Task | Metrics |
| --- | --- |
| Classification | `Accuracy`, `Precision`, `Recall`, `F1` (`average=macro/micro/weighted`), `TopKAccuracy(k)`, `ConfusionMatrix` |
| Regression | `MSE`, `MAE`, `R2` |

`ConfusionMatrix` returns a `DataFrame` rather than a scalar; non-scalar metrics
land in `result.artifacts` instead of the summary Series.

Metrics are pluggable — subclass `Metric`:

```python
from torchcheck import Metric

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

## Worked example: catching a real quality drop

`tests/sample_sims/` trains a small CNN on scikit-learn's handwritten-digits set
(1,797 real 8×8 images, 10 classes — an MNIST stand-in that needs no download),
then compares a well-trained checkpoint against a deliberately under-trained one.

```bash
python tests/sample_sims/run_digits_sim.py     # ~30s on CPU
```

| checkpoint | training | test accuracy |
| --- | --- | --- |
| `v1` | 40 epochs (good) | **0.9704** |
| `v2` | 2 epochs (under-trained) | **0.2352** |

### Aggregate regression

| metric | baseline | candidate | delta | pct_change | direction | regressed |
| --- | --- | --- | --- | --- | --- | --- |
| accuracy | 0.9704 | 0.2352 | -0.7352 | -75.76% | higher_better | yes |
| f1 | 0.9704 | 0.1983 | -0.7722 | -79.57% | higher_better | yes |
| top3_accuracy | 0.9926 | 0.7259 | -0.2667 | -26.87% | higher_better | yes |

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
into the predicted-3 column. That's the "did it get better, and *where* did it
get worse" question answered concretely, not just as a scalar.

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

## What's still rough

Honest self-assessment — this is a learning/portfolio project, not battle-tested
infra:

- **Whole-dataset materialization.** `Evaluator` concatenates every output and
  target into one in-memory numpy array before computing metrics. Fine for the
  test sets here; it would blow up on ImageNet-scale eval. At scale: streaming
  metric accumulation, spilling per-sample data to parquet incrementally.
- **Batch contract is rigid.** Only `(inputs, targets)` tuples are supported.
  Real models take dict batches, multiple inputs, masks. A user-supplied
  `unpack` hook would beat hard-coding the shape.
- **Per-sample alignment is positional.** Flip detection assumes both runs
  iterate the same dataset in the same order. True for a non-shuffled loader,
  silently wrong otherwise. Should key on a stable sample id from the dataset.
- **Metric direction is a hard-coded name set** (`{"mse","mae"}` are
  lower-is-better). Custom metrics carry a `higher_is_better` flag on the class,
  but the comparator only sees stored *names*, so a custom lower-is-better metric
  wouldn't be judged correctly after persistence. Direction should be stored
  alongside the value.
- **Artifacts aren't persisted.** `ConfusionMatrix` and other non-scalar outputs
  live only on the in-memory result, not in the store.
- **No concurrency story.** SQLite with default settings is fine for one user;
  concurrent writers would need WAL mode and retry logic.
- **Single-device.** No sharding, no AMP toggles, no channels-last.

Distributed evaluation, TensorFlow/JAX backends, and a hosted dashboard are
deliberately out of scope.

---

## Tests

```bash
pytest                    # 29 tests
pytest -m "not slow"      # skip the CNN training sim
```

Covers: metric correctness against scikit-learn, evaluator behaviour, store
round-trips, regression logic, and the full CNN simulation end to end.

## License

MIT — see [LICENSE](LICENSE).
