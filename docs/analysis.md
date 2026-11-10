# What the numbers show

Every figure here is in [`report.json`](report.json) (the full run: 711 s, seed 1000, at most 500,000 training rows per
experiment) and in the generated [`evaluation.md`](evaluation.md); the tests check the ones quoted in the README.
"Recall" and "false alarms" are measured on test rows with a threshold chosen on validation rows for 1% false alarms.

## 1. The data is not as large as it looks

UNSW-NB15 has 257,673 rows in its two partition files; 102,645 (39.8%) are exact duplicates and 1,758 more have the
same features as a row with the other label, leaving 153,270. Another 8,541 rows of the testing file are identical to
a training row, so a model that memorised the training file would score on them for free; they are removed from the
test. CIC-IDS2017 loses 324,542 duplicates, 6,790 conflicting rows and 2,867 rows with infinite or missing values
(2,830,743 down to 2,496,544), and 8 columns that never change.

## 2. A random split is the optimistic number

| | Random | Realistic |
|---|---|---|
| CIC-IDS2017, boosting | AP 1.000, recall 100.0% | AP 0.807, recall 59.6% (by day) |
| CIC-IDS2017, forest | AP 1.000, recall 100.0% | AP 0.691, recall 59.6% |
| UNSW-NB15, boosting | AP 0.978, recall 78.3% | AP 0.956, recall 82.1% (official files) |

On CIC the random split mixes the same attacks into training and test, so the model needs only to remember them. By
day, seven attack families appear only in the test days. Boosting catches 99% of the DDoS (128,011 flows, 57.6% of the
test attacks) and 3% of the port scan (90,130 flows, 40.6%); the DDoS alone is 96% of the attacks it catches. The recall
of 59.6% says almost nothing about attacks in general: it is one family. `bot` (1,948 flows) and `infiltration`
(36) are never caught by boosting, while the three web attacks are caught almost in full (99.7% to 100%, on 1,470,
652 and 21 flows).

On UNSW the gap is smaller on average precision but appears in the threshold: see the next section.

## 3. The 1% threshold does not hold

The threshold is picked on validation rows from the same distribution as training. On the random splits it holds
(0.99% to 1.14% of normal test rows alert for the supervised models). On the UNSW testing file it does not: logistic
regression gave 2.72%, the tree 3.25%, the forest 4.04% and boosting 4.35%. On CIC by day the supervised models gave
1.18%, 0.31%, 3.94% and 1.46%. A threshold fixed on past traffic does not carry over to new traffic, and the table of
false alarms by destination port shows part of why: on CIC by day, port 80 produces 8,295 false alarms among its 83,030
normal flows (10.0%), and on port 21 651 of 1,438 normal flows alert (45.3%).

The `best possible recall` column (what an oracle threshold on the test rows would give at the same false-alarm rate)
is 71.0% for boosting on UNSW official against 82.1% reached, and 58.4% on CIC by day against 59.6%: the scores carry
the information, the threshold is what does not carry over. The single tree shows 0.0% on UNSW official because its
scores are too coarse for any threshold to reach 1% false alarms.

## 4. What the models learn

Permutation importance (the drop in average precision when a column is shuffled) on the model chosen by validation
average precision:

- UNSW official, boosting: `sttl` 0.208, `service` 0.128, `dbytes` 0.040. The time to live of the source packets
  plausibly reflects how the lab generated its traffic rather than the attacks themselves; the ablation below measures
  how much depends on it.
- CIC by day, boosting: `destination_port` 0.237, `bwd_packet_length_mean` 0.140.

Two ablations test this. Without `sttl`, `dttl` and `ct_state_ttl`, the trees barely move (boosting 0.956 to 0.949,
forest 0.946 to 0.941), but logistic regression falls from 0.892 to 0.758 and its recall from 58.0% to 30.0%; the linear
model has the most to lose. Without the destination port, the ranking changes: the
forest rises to 0.859, boosting falls from 0.807 to 0.700, the single tree falls from 0.589 to 0.393 with 0.3% recall.

A smaller run of the same code (60,000 training rows, not kept in the repository) gave boosting 0.664 and forest 0.718
on CIC by day, against 0.807 and 0.691 here. The ranking of the models on a split with unseen attacks depends on
details as small as the training sample. The intervals in the report cover only the test rows, so they understate this.

## 5. The isolation forest

Fitted on normal traffic only, it needs no attack labels, and that is its appeal. Its average precision is 0.615 on
the random UNSW split (prior 0.443), 0.524 on UNSW official, 0.446 on CIC by day (prior 0.220), with recall of 1.6%,
4.7% and 0.0%. It never reaches the supervised models. The benchmark does not show why; one reading is that in these datasets being
"unusual" is not what separates the attacks from normal traffic.

## 6. Transfer between the datasets

Fitting on one dataset and testing on the other, on the 8 features they share (duration in seconds, packets and bytes
in each direction, packets and bytes per second, mean packet size):

- UNSW to CIC: the prior is 0.170; logistic regression 0.216, tree 0.208, forest 0.155, boosting 0.175. Boosting and
  the tree alert on 33.9% and 34.3% of normal flows. The isolation forest scores 0.378 (ROC AUC 0.777) with 29.5%
  recall at 6.96% false alarms, the only detector clearly above the prior.
- CIC to UNSW: the prior is 0.443; logistic regression 0.509, tree 0.455, forest 0.465, boosting 0.481, isolation
  0.380 with 18.56% false alarms.

The oracle recall is 3.3% or less for every supervised model in both directions: the scores themselves do not separate the
classes. The two datasets use different flow tools and different attacks (see [limitations](limitations.md)), so this
is also a comparison of tools, but a model that cannot cross is not a model of attacks.

## 7. Speed

On 12 cores (4 jobs for the forests), in Docker: boosting fits 91,963 UNSW rows in 2.3 s and 500,000 CIC rows in 9.2 s and scores about 566,675
CIC rows per second; the forest takes 62.3 s on the same CIC rows and scores 438,335 rows per second.
