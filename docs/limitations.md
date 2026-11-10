# Limitations

What this benchmark does not show, and where its numbers can mislead.

## The datasets

- **Both are old and made in a lab.** UNSW-NB15 was captured in 2015 and CIC-IDS2017 in 2017, on small networks with
  scripted users and attack tools. Real networks have more varied normal traffic, different attacks and encrypted
  protocols the datasets hardly contain. A good score here does not predict a good score on a real network.
- **The labels are not perfect.** Both datasets have known labelling problems (CIC-IDS2017 in particular: flows
  labelled as attacks that are not, attack flows labelled benign, and the time windows of some attacks). This benchmark
  removes rows whose features are identical with different labels, but it cannot find the others. No score is higher
  than the label quality allows.
- **Flows are built by each dataset's own tool** (Argus and Bro for UNSW-NB15, CICFlowMeter for CIC-IDS2017) with
  different definitions and timeouts. The transfer experiment compares the two on the few features they share and is
  partly a comparison of the tools.
- **Attack tools leave fingerprints.** A given tool uses a given port, a given packet size, a given timing. A model
  can learn the fingerprint of the tool, not the attack. The `no-port` experiment removes the most obvious one; the
  others are not removed and the permutation importance table shows which columns matter.
- **One run of each experiment.** There is one fixed seed, one split of each kind. The intervals in the report are over
  the test rows (how much the number would move with other rows of the same traffic), not over splits, seeds or networks.

## Where the UNSW-NB15 numbers come from

The UNSW-NB15 results in this repository were computed from a NumPy export of the training and testing partition files
(integer codes for `proto`, `service` and `state`, 32-bit floats, and the same 42 features and 9 attack families),
converted to CSV outside this repository, not from the official CSV files. The row counts match the
official ones, and the loader reads the official CSVs as well, but the numbers may differ slightly on them, and the
tables of false alarms by `service` show codes, not names. Rows in the testing file whose categories never appear in
the training file carry a code of -1 in the export.

## The method

- **Fixed hyper-parameters.** No model was tuned. A tuned model would score higher, and by an amount that depends on
  the effort spent, which is why it was not done; the ranking of the models could change with tuning.
- **Large training sets are subsampled** (500,000 rows by default); the report says when it happened. On CIC-IDS2017 this means
  500,000 of up to 1,997,237 rows, and the ranking of the models on the by-day split moved with the sample size in a
  smaller test run ([analysis](analysis.md)).
- **The threshold targets 1% false alarms.** A different target changes recall, precision and F1 (average precision and
  ROC AUC do not depend on it). 1% of normal flows is still a very large number of alerts on a real network: the
  tables show the precision that results.
- **Detection delay, drift over weeks, adversarial evasion and the cost of a missed attack are not measured.**
- **Speed numbers are of this code on one machine.**

## What the benchmark is for

Showing, with the numbers, that a score depends on how the data was cleaned and divided; that supervised models are
good at the attacks they were shown and weak at the ones they were not; and that a threshold chosen on past traffic does
not carry over to new traffic. It is not a recommendation to deploy any of the models.
