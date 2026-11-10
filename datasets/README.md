# Where the data goes

The two datasets are public research data. They are **not stored in this repository**: they are large (CIC-IDS2017 is
about 1 GB of CSV), and they belong to the people who made them. Download them from their owners and put the files in
two folders; nothing in this folder is committed (`datasets/*` is ignored except this file).

```
datasets/
  unsw/   UNSW_NB15_training-set.csv   UNSW_NB15_testing-set.csv
  cic/    Monday-WorkingHours.pcap_ISCX.csv   Tuesday-WorkingHours.pcap_ISCX.csv   Wednesday-workingHours.pcap_ISCX.csv
          Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv   Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
          Friday-WorkingHours-Morning.pcap_ISCX.csv   Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
          Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

## UNSW-NB15

Made by the Australian Centre for Cyber Security (UNSW Canberra). Use the **partition** files "training-set" and
"testing-set" (about 175,000 and 82,000 rows, 42 features, 9 attack families). Search for "UNSW-NB15 dataset" on the
UNSW research site; the same two files are mirrored on several data portals. Cite: Moustafa and Slay, "UNSW-NB15: a
comprehensive data set for network intrusion detection systems", MilCIS 2015.

## CIC-IDS2017

Made by the Canadian Institute for Cybersecurity (University of New Brunswick). Use the **MachineLearningCSV**
package (eight files, 78 features, about 2.8 million flows, 14 attack types). The other package ("GeneratedLabelledFlows")
also loads: the columns that identify a flow (addresses, times, ids) are dropped. Cite: Sharafaldin, Lashkari and Ghorbani,
"Toward generating a new intrusion detection dataset and intrusion traffic characterization", ICISSP 2018.

## Check the files before a long run

```bash
idsbench check datasets/unsw datasets/cic
```

It reads the files, says which dataset it recognised, how many rows were removed and why, the columns it will use, and
the number of rows of each class. If a file is not what the loader expects, the message says what it found instead.
