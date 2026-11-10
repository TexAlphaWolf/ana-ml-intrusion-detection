# Contributing

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy && pytest -n auto --cov
idsbench demo --quick --out /tmp/demo       # the whole benchmark on synthetic data, about a minute
```

Rules that keep the numbers meaningful:

- **Never tune on the test rows.** Thresholds are chosen on the validation rows and the analysis model is chosen by
  validation average precision (`tests/test_experiment.py` checks both). A new model or a new feature is judged on the
  same splits as the others.
- **Synthetic data is for tests only.** `idsbench.synth` makes files in the layout of the real ones, with the same
  defects, so the loaders and the pipeline can be tested without the datasets. Its numbers are never quoted as results.
- **Hygiene before splitting.** A new loader must go through `clean()` (non-finite values, conflicting labels, duplicates)
  and report the counts.
- Claims in the README and docs are checked against `docs/report.json` by `tests/test_docs.py`. Re-run
  `idsbench run --unsw datasets/unsw --cic datasets/cic --out docs`, then update the README text if a number moved.
