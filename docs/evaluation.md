# Evaluation

Run on Linux x86_64, Python 3.14.10, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.4; seed 1000; 711 s; at most 500,000 training rows per run; command `idsbench run --out /out --jobs 4 --unsw /datasets/unsw --cic /datasets/cic`.

## 1. The data, and what was removed

| Dataset | Rows read | Not finite | Same features, different label | Exact duplicates | Rows kept | Features kept |
|---|---|---|---|---|---|---|
| unsw | 257,673 | 0 | 1,758 | 102,645 | 153,270 | 42 |
| cic | 2,830,743 | 2,867 | 6,790 | 324,542 | 2,496,544 | 69 |

- **unsw** classes after cleaning: benign 85,308, exploits 25,806, fuzzers 18,982, reconnaissance 8,646, generic 7,305, dos 4,190, shellcode 1,450, analysis 836, backdoor 583, worms 164.
- **unsw**: 8,541 rows of the testing file were identical to a training row (removed from the test).
- **cic** classes after cleaning: benign 2,071,536, dos hulk 172,697, ddos 128,011, portscan 90,130, dos goldeneye 10,282, ftp patator 5,931, dos slowloris 5,372, dos slowhttptest 5,228, ssh patator 3,219, bot 1,948, web attack brute force 1,470, web attack xss 652, infiltration 36, web attack sql injection 21, heartbleed 11.

## 2. Results by experiment

### 2.1 `unsw-random`

Random 60/20/20 split of all rows, family by family (optimistic: train and test come from the same traffic).

Rows: train 91,963, validation 30,654, test 30,653. Share of attacks: train 44.3%, test 44.3%. Features: 42.

Alert threshold: the score above which at most 1% of the normal validation rows alert. Recall and false alarms are measured on the test rows with that threshold; "best possible recall" is what a threshold chosen on the test rows themselves would give at the same false-alarm rate.

| Model | Average precision | ROC AUC | Recall | False alarms | Precision | F1 | Best possible recall | Fit (s) | Rows/s |
|---|---|---|---|---|---|---|---|---|---|
| dummy | 0.443 (0.438-0.448) | 0.500 | 0.0% | 0.00% | 0.0% | 0.000 | 0.0% | 0.2 | 1,368,595 |
| logreg | 0.948 (0.944-0.950) | 0.960 | 51.1% | 1.06% | 97.5% | 0.671 | 50.5% | 18.3 | 449,504 |
| tree | 0.961 (0.959-0.964) | 0.973 | 71.9% | 1.04% | 98.2% | 0.830 | 71.0% | 1.5 | 1,320,594 |
| forest | 0.977 (0.976-0.978) | 0.981 | 76.8% | 1.14% | 98.2% | 0.862 | 75.6% | 8.9 | 190,999 |
| boosting | 0.978 (0.977-0.979) | 0.981 | 78.3% | 1.10% | 98.3% | 0.871 | 77.8% | 2.3 | 316,306 |
| isolation | 0.615 (0.605-0.624) | 0.692 | 1.6% | 0.99% | 55.9% | 0.031 | 1.6% | 0.4 | 139,789 |

Recall by attack family at the same threshold ("seen in training" says whether the family had rows in the training set):

| Attack family | Test rows | Seen in training | logreg | tree | forest | boosting | isolation |
|---|---|---|---|---|---|---|---|
| exploits | 5,161 | yes | 73% | 91% | 94% | 95% | 0% |
| fuzzers | 3,797 | yes | 11% | 22% | 32% | 35% | 0% |
| reconnaissance | 1,729 | yes | 41% | 96% | 98% | 99% | 0% |
| generic | 1,461 | yes | 91% | 97% | 98% | 99% | 14% |
| dos | 838 | yes | 60% | 88% | 92% | 94% | 0% |
| shellcode | 290 | yes | 3% | 83% | 85% | 84% | 0% |
| analysis | 167 | yes | 47% | 51% | 53% | 54% | 1% |
| backdoor | 116 | yes | 61% | 84% | 90% | 95% | 0% |
| worms | 33 | yes | 88% | 97% | 100% | 100% | 0% |

Most important columns for `boosting` (drop in average precision when one column is shuffled (test rows, 3 repeats)):

- `sttl`: 0.2794 (± 0.0026)
- `service`: 0.0561 (± 0.0013)
- `ct_dst_src_ltm`: 0.0425 (± 0.0011)
- `dbytes`: 0.0204 (± 0.0006)
- `ct_state_ttl`: 0.0182 (± 0.0007)
- `dmean`: 0.0104 (± 0.0005)
- `sbytes`: 0.0091 (± 0.0005)
- `smean`: 0.0089 (± 0.0002)
- `ct_srv_dst`: 0.0068 (± 0.0004)
- `dload`: 0.0037 (± 0.0003)

False alarms of `boosting` by `service` (false alarms of normal rows with that value): `0` 130 of 11400, `5` 52 of 1779, `3` 3 of 395, `7` 1 of 1, `8` 1 of 1.

### 2.2 `unsw-official`

The dataset's own partition: training file for fitting (80%) and for choosing the alert threshold (20%), testing file for the result.

Rows: train 80,505, validation 20,125, test 52,640. Share of attacks: train 48.7%, test 36.0%. Features: 42.

Alert threshold: the score above which at most 1% of the normal validation rows alert. Recall and false alarms are measured on the test rows with that threshold; "best possible recall" is what a threshold chosen on the test rows themselves would give at the same false-alarm rate.

| Model | Average precision | ROC AUC | Recall | False alarms | Precision | F1 | Best possible recall | Fit (s) | Rows/s |
|---|---|---|---|---|---|---|---|---|---|
| dummy | 0.360 (0.356-0.365) | 0.500 | 0.0% | 0.00% | 0.0% | 0.000 | 0.0% | 0.2 | 1,867,883 |
| logreg | 0.892 (0.889-0.896) | 0.932 | 58.0% | 2.72% | 92.3% | 0.713 | 46.9% | 10.8 | 655,398 |
| tree | 0.885 (0.879-0.889) | 0.951 | 72.1% | 3.25% | 92.6% | 0.811 | 0.0% | 1.1 | 1,696,084 |
| forest | 0.946 (0.945-0.948) | 0.966 | 77.6% | 4.04% | 91.5% | 0.840 | 67.2% | 5.7 | 178,914 |
| boosting | 0.956 (0.954-0.957) | 0.971 | 82.1% | 4.35% | 91.4% | 0.865 | 71.0% | 2.2 | 282,373 |
| isolation | 0.524 (0.515-0.530) | 0.697 | 4.7% | 1.01% | 72.2% | 0.088 | 4.6% | 0.4 | 160,387 |

Recall by attack family at the same threshold ("seen in training" says whether the family had rows in the training set):

| Attack family | Test rows | Seen in training | logreg | tree | forest | boosting | isolation |
|---|---|---|---|---|---|---|---|
| exploits | 7,085 | yes | 71% | 81% | 92% | 94% | 0% |
| fuzzers | 4,331 | yes | 8% | 29% | 25% | 37% | 0% |
| generic | 3,381 | yes | 93% | 95% | 98% | 99% | 26% |
| reconnaissance | 2,224 | yes | 58% | 90% | 95% | 98% | 0% |
| dos | 1,186 | yes | 59% | 75% | 89% | 92% | 0% |
| shellcode | 364 | yes | 47% | 80% | 86% | 90% | 0% |
| analysis | 294 | yes | 80% | 84% | 80% | 85% | 0% |
| backdoor | 70 | yes | 49% | 77% | 90% | 91% | 3% |
| worms | 43 | yes | 77% | 67% | 91% | 100% | 0% |

Most important columns for `boosting` (drop in average precision when one column is shuffled (test rows, 3 repeats)):

- `sttl`: 0.2080 (± 0.0003)
- `service`: 0.1284 (± 0.0031)
- `dbytes`: 0.0400 (± 0.0010)
- `ct_dst_src_ltm`: 0.0294 (± 0.0005)
- `ct_srv_dst`: 0.0212 (± 0.0008)
- `ct_state_ttl`: 0.0189 (± 0.0004)
- `smean`: 0.0161 (± 0.0001)
- `sbytes`: 0.0148 (± 0.0001)
- `ct_srv_src`: 0.0144 (± 0.0005)
- `dmean`: 0.0137 (± 0.0006)

False alarms of `boosting` by `service` (false alarms of normal rows with that value): `0` 1136 of 24398, `5` 307 of 3739, `3` 17 of 753, `8` 2 of 2, `2` 1 of 2991.

### 2.3 `unsw-official-no-ttl`

The dataset's own partition: training file for fitting (80%) and for choosing the alert threshold (20%), testing file for the result.

Rows: train 80,505, validation 20,125, test 52,640. Share of attacks: train 48.7%, test 36.0%. Features: 39 (without sttl, dttl, ct_state_ttl).

Alert threshold: the score above which at most 1% of the normal validation rows alert. Recall and false alarms are measured on the test rows with that threshold; "best possible recall" is what a threshold chosen on the test rows themselves would give at the same false-alarm rate.

| Model | Average precision | ROC AUC | Recall | False alarms | Precision | F1 | Best possible recall | Fit (s) | Rows/s |
|---|---|---|---|---|---|---|---|---|---|
| dummy | 0.360 (0.356-0.365) | 0.500 | 0.0% | 0.00% | 0.0% | 0.000 | 0.0% | 0.2 | 2,040,129 |
| logreg | 0.758 (0.751-0.764) | 0.861 | 30.0% | 3.27% | 83.8% | 0.442 | 19.2% | 15.6 | 440,816 |
| tree | 0.839 (0.833-0.844) | 0.931 | 70.7% | 4.85% | 89.2% | 0.789 | 0.0% | 1.6 | 1,506,401 |
| forest | 0.941 (0.939-0.943) | 0.962 | 77.4% | 4.40% | 90.8% | 0.836 | 65.9% | 6.9 | 219,306 |
| boosting | 0.949 (0.947-0.951) | 0.966 | 78.9% | 4.07% | 91.6% | 0.848 | 68.8% | 1.9 | 341,785 |
| isolation | 0.523 (0.517-0.531) | 0.658 | 3.7% | 0.50% | 80.8% | 0.071 | 5.3% | 0.4 | 175,337 |

Recall by attack family at the same threshold ("seen in training" says whether the family had rows in the training set):

| Attack family | Test rows | Seen in training | logreg | tree | forest | boosting | isolation |
|---|---|---|---|---|---|---|---|
| exploits | 7,085 | yes | 25% | 85% | 91% | 92% | 0% |
| fuzzers | 4,331 | yes | 5% | 18% | 25% | 28% | 0% |
| generic | 3,381 | yes | 73% | 93% | 98% | 98% | 21% |
| reconnaissance | 2,224 | yes | 25% | 86% | 95% | 97% | 0% |
| dos | 1,186 | yes | 27% | 81% | 89% | 89% | 0% |
| shellcode | 364 | yes | 37% | 75% | 89% | 89% | 0% |
| analysis | 294 | yes | 68% | 85% | 80% | 81% | 0% |
| backdoor | 70 | yes | 26% | 63% | 93% | 89% | 1% |
| worms | 43 | yes | 42% | 88% | 93% | 93% | 0% |

Most important columns for `forest` (drop in average precision when one column is shuffled (test rows, 3 repeats)):

- `ackdat`: 0.0727 (± 0.0018)
- `tcprtt`: 0.0458 (± 0.0011)
- `synack`: 0.0408 (± 0.0009)
- `service`: 0.0230 (± 0.0002)
- `dbytes`: 0.0206 (± 0.0012)
- `ct_dst_src_ltm`: 0.0193 (± 0.0007)
- `sbytes`: 0.0161 (± 0.0004)
- `dinpkt`: 0.0151 (± 0.0015)
- `proto`: 0.0131 (± 0.0000)
- `state`: 0.0108 (± 0.0002)

False alarms of `forest` by `service` (false alarms of normal rows with that value): `0` 1228 of 24398, `5` 237 of 3739, `2` 8 of 2991, `3` 5 of 753, `8` 2 of 2.

### 2.4 `cic-random`

Random 60/20/20 split of all rows, family by family (optimistic: train and test come from the same traffic).

Rows: train 500,000 (of 1,497,928 available), validation 499,307, test 499,309. Share of attacks: train 17.0%, test 17.0%. Features: 69.

Alert threshold: the score above which at most 1% of the normal validation rows alert. Recall and false alarms are measured on the test rows with that threshold; "best possible recall" is what a threshold chosen on the test rows themselves would give at the same false-alarm rate.

| Model | Average precision | ROC AUC | Recall | False alarms | Precision | F1 | Best possible recall | Fit (s) | Rows/s |
|---|---|---|---|---|---|---|---|---|---|
| dummy | 0.170 (0.169-0.171) | 0.500 | 0.0% | 0.00% | 0.0% | 0.000 | 0.0% | 0.0 | 17,409,748 |
| logreg | 0.982 (0.981-0.983) | 0.996 | 95.6% | 1.00% | 95.2% | 0.954 | 95.7% | 42.9 | 1,219,502 |
| tree | 0.997 (0.997-0.997) | 0.999 | 99.6% | 1.00% | 95.3% | 0.974 | 99.6% | 12.9 | 10,745,667 |
| forest | 1.000 (1.000-1.000) | 1.000 | 100.0% | 1.01% | 95.3% | 0.976 | 100.0% | 62.3 | 438,335 |
| boosting | 1.000 (1.000-1.000) | 1.000 | 100.0% | 0.99% | 95.4% | 0.976 | 100.0% | 9.2 | 566,675 |
| isolation | 0.456 (0.453-0.459) | 0.880 | 1.0% | 0.97% | 18.0% | 0.020 | 1.1% | 0.9 | 126,098 |

Recall by attack family at the same threshold ("seen in training" says whether the family had rows in the training set):

| Attack family | Test rows | Seen in training | logreg | tree | forest | boosting | isolation |
|---|---|---|---|---|---|---|---|
| dos hulk | 34,540 | yes | 97% | 100% | 100% | 100% | 0% |
| ddos | 25,602 | yes | 100% | 100% | 100% | 100% | 0% |
| portscan | 18,026 | yes | 97% | 100% | 100% | 100% | 0% |
| dos goldeneye | 2,057 | yes | 89% | 100% | 100% | 100% | 0% |
| ftp patator | 1,186 | yes | 97% | 100% | 100% | 100% | 0% |
| dos slowloris | 1,075 | yes | 91% | 99% | 100% | 100% | 21% |
| dos slowhttptest | 1,045 | yes | 76% | 93% | 100% | 100% | 58% |
| ssh patator | 644 | yes | 9% | 99% | 100% | 100% | 0% |
| bot | 389 | yes | 0% | 53% | 100% | 100% | 1% |
| web attack brute force | 294 | yes | 0% | 94% | 100% | 100% | 0% |
| web attack xss | 131 | yes | 0% | 95% | 100% | 100% | 0% |
| infiltration | 7 | yes | 0% | 14% | 86% | 71% | 57% |
| web attack sql injection | 4 | yes | 25% | 50% | 100% | 100% | 0% |
| heartbleed | 2 | yes | 0% | 100% | 100% | 100% | 100% |

Most important columns for `boosting` (drop in average precision when one column is shuffled (test rows, 3 repeats)):

- `init_win_bytes_forward`: 0.0173 (± 0.0008)
- `destination_port`: 0.0167 (± 0.0010)
- `bwd_header_length`: 0.0029 (± 0.0001)
- `average_packet_size`: 0.0019 (± 0.0001)
- `fwd_iat_min`: 0.0018 (± 0.0001)
- `flow_iat_min`: 0.0006 (± 0.0000)
- `bwd_packet_length_std`: 0.0005 (± 0.0000)
- `max_packet_length`: 0.0005 (± 0.0001)
- `init_win_bytes_backward`: 0.0004 (± 0.0001)
- `total_length_of_bwd_packets`: 0.0002 (± 0.0001)

False alarms of `boosting` by `destination_port` (false alarms of normal rows with that value): `80.0` 1797 of 43302, `443.0` 496 of 93684, `8080.0` 183 of 272, `22.0` 157 of 1302, `21.0` 68 of 666.

### 2.5 `cic-by-day`

Monday to Wednesday for fitting (80%) and for choosing the alert threshold (20%); Thursday and Friday for the result: several attack families appear only there.

Rows: train 500,000 (of 1,188,343 available), validation 297,084, test 1,011,117. Share of attacks: train 13.6%, test 22.0%. Features: 69.

Alert threshold: the score above which at most 1% of the normal validation rows alert. Recall and false alarms are measured on the test rows with that threshold; "best possible recall" is what a threshold chosen on the test rows themselves would give at the same false-alarm rate.

| Model | Average precision | ROC AUC | Recall | False alarms | Precision | F1 | Best possible recall | Fit (s) | Rows/s |
|---|---|---|---|---|---|---|---|---|---|
| dummy | 0.220 (0.219-0.221) | 0.500 | 0.0% | 0.00% | 0.0% | 0.000 | 0.0% | 0.1 | 8,327,154 |
| logreg | 0.672 (0.670-0.674) | 0.704 | 48.7% | 1.18% | 92.1% | 0.637 | 47.4% | 11.9 | 1,058,986 |
| tree | 0.589 (0.588-0.591) | 0.598 | 37.8% | 0.31% | 97.2% | 0.545 | 37.8% | 13.9 | 6,141,790 |
| forest | 0.691 (0.690-0.694) | 0.789 | 59.6% | 3.94% | 81.0% | 0.687 | 47.1% | 66.3 | 479,937 |
| boosting | 0.807 (0.806-0.808) | 0.913 | 59.6% | 1.46% | 92.0% | 0.723 | 58.4% | 9.9 | 704,114 |
| isolation | 0.446 (0.445-0.447) | 0.845 | 0.0% | 3.30% | 0.4% | 0.001 | 0.0% | 1.2 | 175,705 |

Recall by attack family at the same threshold ("seen in training" says whether the family had rows in the training set):

| Attack family | Test rows | Seen in training | logreg | tree | forest | boosting | isolation |
|---|---|---|---|---|---|---|---|
| ddos | 128,011 | **no** | 83% | 65% | 98% | 99% | 0% |
| portscan | 90,130 | **no** | 2% | 2% | 5% | 3% | 0% |
| bot | 1,948 | **no** | 0% | 0% | 2% | 0% | 1% |
| web attack brute force | 1,470 | **no** | 0% | 0% | 93% | 100% | 0% |
| web attack xss | 652 | **no** | 4% | 3% | 98% | 100% | 0% |
| infiltration | 36 | **no** | 0% | 0% | 86% | 0% | 64% |
| web attack sql injection | 21 | **no** | 52% | 24% | 90% | 100% | 0% |

Most important columns for `boosting` (drop in average precision when one column is shuffled (test rows, 3 repeats)):

- `destination_port`: 0.2371 (± 0.0051)
- `bwd_packet_length_mean`: 0.1402 (± 0.0019)
- `bwd_packet_length_std`: 0.1142 (± 0.0014)
- `init_win_bytes_forward`: 0.1004 (± 0.0012)
- `bwd_packet_length_max`: 0.0933 (± 0.0010)
- `bwd_packet_length_min`: 0.0586 (± 0.0007)
- `fwd_packet_length_mean`: 0.0345 (± 0.0003)
- `average_packet_size`: 0.0329 (± 0.0004)
- `flow_bytes_s`: 0.0306 (± 0.0002)
- `packet_length_mean`: 0.0299 (± 0.0005)

False alarms of `boosting` by `destination_port` (false alarms of normal rows with that value): `80.0` 8295 of 83030, `21.0` 651 of 1438, `22.0` 315 of 2787, `0.0` 134 of 649, `1.0` 112 of 112.

### 2.6 `cic-by-day-no-port`

Monday to Wednesday for fitting (80%) and for choosing the alert threshold (20%); Thursday and Friday for the result: several attack families appear only there.

Rows: train 500,000 (of 1,188,343 available), validation 297,084, test 1,011,117. Share of attacks: train 13.6%, test 22.0%. Features: 68 (without destination_port).

Alert threshold: the score above which at most 1% of the normal validation rows alert. Recall and false alarms are measured on the test rows with that threshold; "best possible recall" is what a threshold chosen on the test rows themselves would give at the same false-alarm rate.

| Model | Average precision | ROC AUC | Recall | False alarms | Precision | F1 | Best possible recall | Fit (s) | Rows/s |
|---|---|---|---|---|---|---|---|---|---|
| dummy | 0.220 (0.219-0.221) | 0.500 | 0.0% | 0.00% | 0.0% | 0.000 | 0.0% | 0.1 | 5,335,263 |
| logreg | 0.616 (0.614-0.618) | 0.818 | 35.1% | 0.61% | 94.2% | 0.512 | 36.1% | 43.2 | 715,059 |
| tree | 0.393 (0.391-0.394) | 0.476 | 0.3% | 0.78% | 10.6% | 0.006 | 0.3% | 17.0 | 4,006,109 |
| forest | 0.859 (0.858-0.861) | 0.955 | 44.6% | 3.03% | 80.6% | 0.574 | 37.9% | 61.1 | 492,361 |
| boosting | 0.700 (0.698-0.702) | 0.869 | 38.7% | 2.66% | 80.4% | 0.522 | 37.8% | 9.4 | 635,573 |
| isolation | 0.438 (0.437-0.440) | 0.828 | 7.2% | 3.41% | 37.2% | 0.120 | 1.1% | 1.1 | 161,942 |

Recall by attack family at the same threshold ("seen in training" says whether the family had rows in the training set):

| Attack family | Test rows | Seen in training | logreg | tree | forest | boosting | isolation |
|---|---|---|---|---|---|---|---|
| ddos | 128,011 | **no** | 60% | 0% | 75% | 65% | 12% |
| portscan | 90,130 | **no** | 1% | 0% | 2% | 1% | 0% |
| bot | 1,948 | **no** | 0% | 0% | 0% | 0% | 0% |
| web attack brute force | 1,470 | **no** | 0% | 0% | 88% | 92% | 0% |
| web attack xss | 652 | **no** | 0% | 0% | 94% | 94% | 0% |
| infiltration | 36 | **no** | 0% | 3% | 86% | 6% | 72% |
| web attack sql injection | 21 | **no** | 0% | 24% | 81% | 100% | 0% |

Most important columns for `boosting` (drop in average precision when one column is shuffled (test rows, 3 repeats)):

- `bwd_packet_length_mean`: 0.1256 (± 0.0016)
- `total_length_of_bwd_packets`: 0.0740 (± 0.0012)
- `packet_length_std`: 0.0460 (± 0.0003)
- `fwd_header_length`: 0.0399 (± 0.0045)
- `bwd_packet_length_std`: 0.0304 (± 0.0017)
- `bwd_packets_s`: 0.0245 (± 0.0014)
- `fwd_iat_min`: 0.0228 (± 0.0002)
- `psh_flag_count`: 0.0184 (± 0.0002)
- `init_win_bytes_backward`: 0.0167 (± 0.0010)
- `idle_min`: 0.0148 (± 0.0020)

### 2.7 `transfer-unsw-to-cic`

Fit and choose the threshold on unsw, test on all of cic, using only the 8 features both have (the attack families are different: only 'attack or not' can be compared).

Rows: train 122,616, validation 30,654, test 2,496,544. Share of attacks: train 44.3%, test 17.0%. Features: 8.

Alert threshold: the score above which at most 1% of the normal validation rows alert. Recall and false alarms are measured on the test rows with that threshold; "best possible recall" is what a threshold chosen on the test rows themselves would give at the same false-alarm rate.

| Model | Average precision | ROC AUC | Recall | False alarms | Precision | F1 | Best possible recall | Fit (s) | Rows/s |
|---|---|---|---|---|---|---|---|---|---|
| dummy | 0.170 (0.170-0.171) | 0.500 | 0.0% | 0.00% | 0.0% | 0.000 | 0.0% | 0.0 | 75,447,054 |
| logreg | 0.216 (0.216-0.217) | 0.477 | 27.7% | 10.58% | 35.0% | 0.309 | 1.7% | 1.5 | 9,219,224 |
| tree | 0.208 (0.207-0.208) | 0.575 | 44.6% | 34.32% | 21.0% | 0.286 | 0.0% | 0.5 | 29,103,114 |
| forest | 0.155 (0.154-0.155) | 0.475 | 0.0% | 7.06% | 0.0% | 0.000 | 0.0% | 5.9 | 496,647 |
| boosting | 0.175 (0.175-0.176) | 0.490 | 46.0% | 33.91% | 21.8% | 0.295 | 0.1% | 1.0 | 547,534 |
| isolation | 0.378 (0.376-0.379) | 0.777 | 29.5% | 6.96% | 46.5% | 0.361 | 3.1% | 0.4 | 235,054 |

Most important columns for `boosting` (drop in average precision when one column is shuffled (test rows, 3 repeats)):

- `bwd_bytes`: 0.0421 (± 0.0005)
- `mean_packet`: 0.0352 (± 0.0008)
- `duration`: 0.0101 (± 0.0002)
- `bytes_per_s`: -0.0001 (± 0.0002)
- `bwd_packets`: -0.0030 (± 0.0005)
- `packets_per_s`: -0.0280 (± 0.0009)
- `fwd_packets`: -0.0286 (± 0.0003)
- `fwd_bytes`: -0.0338 (± 0.0001)

### 2.8 `transfer-cic-to-unsw`

Fit and choose the threshold on cic, test on all of unsw, using only the 8 features both have (the attack families are different: only 'attack or not' can be compared).

Rows: train 500,000 (of 1,997,237 available), validation 499,307, test 153,270. Share of attacks: train 17.0%, test 44.3%. Features: 8.

Alert threshold: the score above which at most 1% of the normal validation rows alert. Recall and false alarms are measured on the test rows with that threshold; "best possible recall" is what a threshold chosen on the test rows themselves would give at the same false-alarm rate.

| Model | Average precision | ROC AUC | Recall | False alarms | Precision | F1 | Best possible recall | Fit (s) | Rows/s |
|---|---|---|---|---|---|---|---|---|---|
| dummy | 0.443 (0.441-0.445) | 0.500 | 0.0% | 0.00% | 0.0% | 0.000 | 0.0% | 0.0 | 38,172,873 |
| logreg | 0.509 (0.506-0.513) | 0.570 | 0.0% | 0.13% | 9.2% | 0.000 | 0.4% | 1.8 | 3,313,645 |
| tree | 0.455 (0.451-0.458) | 0.487 | 0.3% | 0.10% | 72.2% | 0.007 | 0.4% | 2.6 | 16,756,986 |
| forest | 0.465 (0.463-0.468) | 0.473 | 5.0% | 1.59% | 71.7% | 0.094 | 1.9% | 27.6 | 581,242 |
| boosting | 0.481 (0.477-0.484) | 0.527 | 2.5% | 0.76% | 72.1% | 0.047 | 3.3% | 1.5 | 1,125,792 |
| isolation | 0.380 (0.377-0.382) | 0.447 | 4.6% | 18.56% | 16.4% | 0.071 | 0.2% | 0.5 | 242,355 |

Most important columns for `forest` (drop in average precision when one column is shuffled (test rows, 3 repeats)):

- `fwd_packets`: 0.0373 (± 0.0024)
- `bwd_packets`: 0.0283 (± 0.0012)
- `fwd_bytes`: 0.0047 (± 0.0005)
- `mean_packet`: 0.0028 (± 0.0012)
- `bwd_bytes`: -0.0071 (± 0.0029)
- `bytes_per_s`: -0.0329 (± 0.0015)
- `packets_per_s`: -0.0354 (± 0.0022)
- `duration`: -0.0506 (± 0.0024)

