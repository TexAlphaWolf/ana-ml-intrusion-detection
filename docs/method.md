# Method

What is measured, and every choice that could change the numbers.

## 1. Data hygiene (before any split)

Applied by `clean.py` to each dataset, in this order, and counted in the report:

1. **Non-finite values** (`Infinity` and empty cells in the CIC rate columns): the row is removed. Two flows with zero
   duration have no rate; the alternative (inventing one) would put made-up numbers in the features.
2. **Same features, different label**: all rows of the group are removed. No model can learn them and they put a
   ceiling on every score that has nothing to do with the model. (Decided on the yes/no label, not on the family.)
3. **Exact duplicates**: the first copy is kept. A copy in the test set of a row in the training set is a free point.
   For UNSW-NB15 the training and testing files are joined before this step, so a test row that repeats a training row
   is removed from the test; the report says how many.
4. **Constant columns** are dropped, and so are repeated columns (CIC has `Fwd Header Length` twice) and the columns
   that identify a flow (addresses, times, ids).

The result is therefore not the "official" test set row for row, and a score here will not equal a published score that
did not do this. That is intended.

## 2. Splits (`splits.py`)

| Id | Training / threshold | Test | What it shows |
|---|---|---|---|
| `random` | 60% / 20% of all rows, family by family | the remaining 20% | the usual, optimistic score: train and test are the same traffic |
| `unsw-official` | UNSW training file, 80% fit / 20% threshold | UNSW testing file | the dataset's own split |
| `cic-by-day` | Monday-Wednesday, 80% / 20% | Thursday-Friday | attacks that were never seen in training (bot, port scan, DDoS, web attacks, infiltration) |
| `cic-by-day-no-port` | the same | the same | the same, without the destination port (a strong shortcut: a given attack tool uses a given port) |
| `transfer-*` | one dataset, 80% / 20% | all of the other | whether anything learned on one lab dataset works on another |

Transfer can only use what both datasets measure: duration (seconds), packets and bytes in each direction, and the
packets per second, bytes per second and mean packet size derived from them. The two datasets define their flows in
different ways (different tools, different timeouts), so a difference here is partly a difference in definitions, and
the transfer result is the least reliable one of the report.

## 3. Detectors (`models.py`)

All use the same pipeline: numeric columns pass unchanged to the trees and are `log(1+|x|)`-scaled and standardised for
logistic regression and the isolation forest; categorical columns (UNSW `proto`, `service`, `state`) are one-hot
encoded, with values seen in less than 0.1% of rows grouped together and unseen values ignored.

| Name | Model |
|---|---|
| `dummy` | the class prior: the floor |
| `logreg` | logistic regression |
| `tree` | decision tree, depth 14, 10 rows per leaf |
| `forest` | random forest, 150 trees, 2 rows per leaf |
| `boosting` | histogram gradient boosting, 200 iterations |
| `isolation` | isolation forest, 150 trees, **fitted on normal traffic only**: it never sees an attack label |

Hyper-parameters are fixed, not searched. They are sensible defaults, and tuning each model on its own would make the
comparison depend on how much effort each got. Training sets larger than `--max-train` rows (500,000 by default) are
subsampled uniformly; the report says when that happened.

## 4. The alert threshold and the numbers

A detector outputs a score. A real system has to turn it into "alert or not". Here the threshold is the score above
which **at most 1% of the normal rows of the validation set** would alert (`threshold_for_fpr`), and it is then used
**unchanged on the test set**. Reported on the test set:

- **Average precision** (area under the precision-recall curve) and **ROC AUC**: threshold-free. Average precision
  depends on the share of attacks in the test set, which differs between experiments (it is in the report).
- **Recall** (share of attacks that alert), **false-alarm rate** (share of normal rows that alert), **precision**, **F1**
  at the chosen threshold.
- **Best possible recall**: the recall a threshold chosen on the test rows themselves would give at 1% false alarms. The
  distance between this and the real recall is how much the threshold chosen on validation did not carry over to
  new traffic.
- **95% intervals** for average precision, recall and false-alarm rate: percentile bootstrap over the test rows (50
  resamples; 10 with `--quick`). They say how much the number would move with other rows from the same traffic; they do
  not say how it would move with another network.
- **Recall by attack family**, and whether the family was in the training set.
- **Permutation importance** of the model chosen for the analysis: how much average precision drops when one column is
  shuffled in a sample of 20,000 test rows.
- **Where the false alarms come from**: the service (UNSW) or destination port (CIC) with most false alarms.

The model used for the analysis is chosen by **validation** average precision among the supervised models, never by
the test rows.

## 5. What is not measured

Detection delay, throughput on a live network, behaviour over time (concept drift beyond the day split), adversarial
attacks on the model, and cost-sensitive thresholds (what a missed attack costs against a false alarm). Prediction
speed in the report is the speed of this code on this machine, not of a deployment.
