# Changelog

## 1.0.0

First release.

- Loaders for UNSW-NB15 (training and testing partition files) and CIC-IDS2017 (MachineLearningCSV and GeneratedLabelledFlows).
- Data hygiene with a count for each step: non-finite values, rows with the same features and different labels, exact
  duplicates (also between the UNSW training and testing files), constant columns, repeated columns.
- Six detectors on the same preprocessing: a prior-only baseline, logistic regression, decision tree, random forest,
  histogram gradient boosting and an isolation forest fitted on normal traffic only.
- Splits: random (optimistic), the official UNSW partition, CIC-IDS2017 by day (attacks unseen in training), and
  transfer between the two datasets on the features they share.
- Metrics: average precision, ROC AUC, and an operating point whose threshold is chosen on validation rows for 1% false
  alarms, with bootstrap intervals, recall by attack family, permutation importance and the origin of false alarms.
- `idsbench check`, `run`, `demo`, `synth`, `report`; Markdown report and summary figure; Docker image.
- Measured results of a full run on the real datasets in `docs/` (`report.json`, `evaluation.md`, `summary.png`, `analysis.md`).
