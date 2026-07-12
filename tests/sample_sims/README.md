# Sample simulations

Realistic end-to-end uses of `torchcheck` on an actual CNN, as opposed to the
toy tensors in the unit tests. These double as the project's worked examples.

| file | what it is |
| --- | --- |
| `digits_cnn.py` | The model and the data: a small CNN on scikit-learn's handwritten-digits set (1,797 real 8×8 images, 10 classes). An MNIST stand-in that needs no download — swap `_tensors()` for `torchvision.datasets.MNIST` to use the real thing. |
| `run_digits_sim.py` | The simulation: train a good checkpoint and an under-trained one, evaluate both, and print the regression report. |

Run it:

```bash
python tests/sample_sims/run_digits_sim.py
```

Takes ~30s on CPU. It writes its run history to `tests/sample_sims/.torchcheck/`
(gitignored) — delete that folder to start fresh.

`test_digits_sim.py` in the parent folder runs a fast version of the same flow
as a real test, so the example can't silently rot.

## What you should see

The 2-epoch model collapses to predicting class 3 for nearly everything. Test
accuracy falls from **0.9704** to **0.2352**, every aggregate metric drops hard,
and — the useful part — the report names the **399** exact test samples that
flipped from correct to incorrect.

That per-sample view is the whole point: a scalar tells you the model got
worse, this tells you *where*.
