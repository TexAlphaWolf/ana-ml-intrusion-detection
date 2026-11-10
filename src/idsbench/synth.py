"""Synthetic stand-ins for the two datasets, written in the exact file layout of the real ones.

They exist to test the loaders and the pipeline, and to let anyone run the whole thing in a minute without downloading
gigabytes. They are NOT data about real attacks: the numbers they give mean nothing about intrusion detection and are
never quoted as results. They do contain the defects of the real files on purpose (duplicate rows and columns,
`Infinity`, empty cells, a broken dash in the labels, spaces in column names).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

UNSW_COLUMNS = [
    "id",
    "dur",
    "proto",
    "service",
    "state",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
    "attack_cat",
    "label",
]

# family -> (share of the attacks, how different it is from normal traffic: 0 identical ... 1 very different)
UNSW_FAMILIES = {
    "Generic": (0.30, 0.9),
    "Exploits": (0.25, 0.45),
    "Fuzzers": (0.15, 0.5),
    "DoS": (0.10, 0.35),
    "Reconnaissance": (0.09, 0.6),
    "Analysis": (0.04, 0.15),
    "Backdoor": (0.03, 0.15),
    "Shellcode": (0.03, 0.4),
    "Worms": (0.01, 0.5),
}

CIC_COLUMNS = (
    " Destination Port; Flow Duration; Total Fwd Packets; Total Backward Packets;Total Length of Fwd Packets; "
    "Total Length of Bwd Packets; Fwd Packet Length Max; Fwd Packet Length Min; Fwd Packet Length Mean; "
    "Fwd Packet Length Std;Bwd Packet Length Max; Bwd Packet Length Min; Bwd Packet Length Mean; Bwd Packet Length Std;"
    "Flow Bytes/s; Flow Packets/s; Flow IAT Mean; Flow IAT Std; Flow IAT Max; Flow IAT Min;Fwd IAT Total; Fwd IAT Mean; "
    "Fwd IAT Std; Fwd IAT Max; Fwd IAT Min;Bwd IAT Total; Bwd IAT Mean; Bwd IAT Std; Bwd IAT Max; Bwd IAT Min;"
    "Fwd PSH Flags; Bwd PSH Flags; Fwd URG Flags; Bwd URG Flags; Fwd Header Length; Bwd Header Length;Fwd Packets/s; "
    "Bwd Packets/s; Min Packet Length; Max Packet Length; Packet Length Mean; Packet Length Std; Packet Length Variance;"
    "FIN Flag Count; SYN Flag Count; RST Flag Count; PSH Flag Count; ACK Flag Count; URG Flag Count; CWE Flag Count; "
    "ECE Flag Count; Down/Up Ratio; Average Packet Size; Avg Fwd Segment Size; Avg Bwd Segment Size; Fwd Header Length.1;"
    "Fwd Avg Bytes/Bulk; Fwd Avg Packets/Bulk; Fwd Avg Bulk Rate; Bwd Avg Bytes/Bulk; Bwd Avg Packets/Bulk;"
    "Bwd Avg Bulk Rate;Subflow Fwd Packets; Subflow Fwd Bytes; Subflow Bwd Packets; Subflow Bwd Bytes;"
    "Init_Win_bytes_forward; Init_Win_bytes_backward; act_data_pkt_fwd; min_seg_size_forward;Active Mean; Active Std; "
    "Active Max; Active Min;Idle Mean; Idle Std; Idle Max; Idle Min; Label"
)
# the real files put a space in front of most names, not of all; this keeps the same irregularity
_NO_SPACE = {
    "Total Length of Fwd Packets", "Bwd Packet Length Max", "Flow Bytes/s", "Fwd IAT Total", "Bwd IAT Total",
    "Fwd PSH Flags", "Fwd Packets/s", "FIN Flag Count", "Fwd Avg Bytes/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
    "Init_Win_bytes_forward", "Active Mean", "Idle Mean",
}  # fmt: skip


def cic_columns() -> list[str]:
    """The 78 feature names and the label, with the spaces of the original files."""
    names: list[str] = []
    for group in CIC_COLUMNS.split(";"):
        names.extend(n for n in group.split("; ") if n.strip())
    return [n.strip() if n.strip() in _NO_SPACE else " " + n.strip() for n in names]


# family -> (day, share of the day's flows, how different, destination port)
CIC_PLAN = {
    "BENIGN": ("monday", 1.0, 0.0, 0),
    "FTP-Patator": ("tuesday", 0.03, 0.8, 21),
    "SSH-Patator": ("tuesday", 0.02, 0.7, 22),
    "DoS Hulk": ("wednesday", 0.25, 0.85, 80),
    "DoS GoldenEye": ("wednesday", 0.03, 0.6, 80),
    "DoS slowloris": ("wednesday", 0.02, 0.55, 80),
    "DoS Slowhttptest": ("wednesday", 0.02, 0.5, 80),
    "Heartbleed": ("wednesday", 0.001, 0.9, 444),
    "Web Attack \x96 Brute Force": ("thursday", 0.02, 0.35, 80),
    "Web Attack \x96 XSS": ("thursday", 0.01, 0.3, 80),
    "Web Attack \x96 Sql Injection": ("thursday", 0.002, 0.3, 80),
    "Infiltration": ("thursday", 0.003, 0.25, 444),
    "Bot": ("friday", 0.015, 0.45, 8080),
    "PortScan": ("friday", 0.12, 0.7, 0),
    "DDoS": ("friday", 0.15, 0.8, 80),
}
CIC_DAY_FILES = {
    "monday": ["Monday-WorkingHours.pcap_ISCX.csv"],
    "tuesday": ["Tuesday-WorkingHours.pcap_ISCX.csv"],
    "wednesday": ["Wednesday-workingHours.pcap_ISCX.csv"],
    "thursday": ["Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"],
    "friday": ["Friday-WorkingHours-Morning.pcap_ISCX.csv"],
}


def _shift(rng: np.random.Generator, n_cols: int, name: str, strength: float) -> np.ndarray:
    """A fixed shift (in log units) for each column, for one family: what makes it different from normal traffic."""
    seed = sum(map(ord, name))
    r = np.random.default_rng(seed)
    pick = r.random(n_cols) < 0.5
    return np.where(pick, r.normal(0, 1.6, n_cols) * strength, 0.0)


def make_unsw(n: int, seed: int) -> pd.DataFrame:
    """n rows of a table with the columns of the UNSW-NB15 partition files."""
    rng = np.random.default_rng(seed)
    fams = ["Normal", *UNSW_FAMILIES]
    share = np.array([0.55, *[0.45 * v[0] for v in UNSW_FAMILIES.values()]])
    cat = rng.choice(len(fams), size=n, p=share / share.sum())
    family = np.array(fams, dtype=object)[cat]
    strength = np.array([0.0, *[v[1] for v in UNSW_FAMILIES.values()]])[cat]

    num_cols = [c for c in UNSW_COLUMNS if c not in ("id", "proto", "service", "state", "attack_cat", "label")]
    shifts = {f: _shift(rng, len(num_cols), f, 1.0) for f in fams}
    mu = np.random.default_rng(98).normal(2.0, 1.2, len(num_cols))  # the same "normal traffic" in every file
    data = {}
    for j, c in enumerate(num_cols):
        s = np.array([shifts[f][j] for f in family]) * strength
        data[c] = np.round(np.exp(rng.normal(mu[j] + s, 1.0)), 4)
    df = pd.DataFrame(data)
    for c in ("is_ftp_login", "is_sm_ips_ports"):
        df[c] = rng.integers(0, 2, n)
    protos = np.array(["tcp", "udp", "icmp", "arp", "ospf", "sctp", "gre", "ipv6-icmp"], dtype=object)
    p = np.where(strength > 0.5, rng.choice(len(protos), n, p=[0.5, 0.3, 0.1, 0.04, 0.02, 0.02, 0.01, 0.01]), 0)
    p = np.where(strength == 0, rng.choice(2, n, p=[0.7, 0.3]), p)
    df.insert(0, "id", np.arange(1, n + 1))
    df["proto"] = protos[p]
    df["service"] = rng.choice(
        ["-", "dns", "http", "ftp", "smtp", "ssh", "ssl"], n, p=[0.4, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05]
    )
    df["state"] = rng.choice(["FIN", "INT", "CON", "REQ", "RST"], n, p=[0.5, 0.2, 0.2, 0.05, 0.05])
    df["attack_cat"] = family
    df["label"] = (family != "Normal").astype(int)
    # the real tables hold exact repeats too
    dup = np.flatnonzero(rng.random(n) < 0.02)
    if len(dup):
        src = rng.choice(np.setdiff1d(np.arange(n), dup), len(dup))
        cols = [c for c in df.columns if c != "id"]
        df.loc[dup, cols] = df.loc[src, cols].to_numpy()
    return df[UNSW_COLUMNS]


def write_unsw(folder: Path, rows: int = 6000, seed: int = 1) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    test_rows = max(100, rows * 3 // 7)
    out = []
    for name, n, s in (("UNSW_NB15_training-set.csv", rows, seed), ("UNSW_NB15_testing-set.csv", test_rows, seed + 1)):
        path = folder / name
        make_unsw(n, s).to_csv(path, index=False)
        out.append(path)
    return out


def make_cic_day(day: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    plan = {k: v for k, v in CIC_PLAN.items() if v[0] == day or k == "BENIGN"}
    names = list(plan)
    share = np.array([plan[k][1] if k != "BENIGN" else 1.0 for k in names])
    if len(names) > 1:
        share[0] = 1.0 - share[1:].sum()
    fam_idx = rng.choice(len(names), size=n, p=share / share.sum())
    family = np.array(names, dtype=object)[fam_idx]
    strength = np.array([plan[k][2] for k in names])[fam_idx]
    port = np.array([plan[k][3] for k in names])[fam_idx]
    cols = cic_columns()
    feature_cols = cols[1:-1]
    mu = np.random.default_rng(99).normal(3.0, 1.5, len(feature_cols))  # the same "normal traffic" in every file
    shifts = {k: _shift(rng, len(feature_cols), k, 1.0) for k in names}
    data = {}
    for j, c in enumerate(feature_cols):
        s = np.array([shifts[k][j] for k in family]) * strength
        data[c] = np.round(np.exp(rng.normal(mu[j] + s, 1.0)), 3)
    df = pd.DataFrame(data)
    web = rng.choice([80, 443, 53, 8080, 22, 445, 3389], n, p=[0.3, 0.35, 0.15, 0.05, 0.05, 0.05, 0.05])
    df.insert(0, cols[0], np.where(port > 0, port, web))
    fwd = next(c for c in df.columns if c.strip() == "Fwd Header Length")
    df[" Fwd Header Length.1"] = df[fwd]
    # flows with zero duration make the rate columns infinite or empty in the real files
    zero = rng.random(n) < 0.004
    rate = next(c for c in df.columns if c.strip() == "Flow Bytes/s")
    pps = next(c for c in df.columns if c.strip() == "Flow Packets/s")
    df[rate] = df[rate].astype(float)
    df[pps] = df[pps].astype(float)
    df.loc[zero, rate] = np.inf
    df.loc[zero, pps] = np.inf
    empty = rng.random(n) < 0.001
    df.loc[empty, rate] = np.nan
    df[cols[-1]] = family
    dup = rng.random(n) < 0.05
    src = rng.choice(np.flatnonzero(~dup), int(dup.sum()))
    df.iloc[np.flatnonzero(dup)] = df.iloc[src].to_numpy()
    return df[cols]


def write_cic(folder: Path, rows_per_day: int = 3000, seed: int = 1) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for i, (day, files) in enumerate(CIC_DAY_FILES.items()):
        df = make_cic_day(day, rows_per_day, seed + i)
        path = folder / files[0]
        text = df.to_csv(index=False).replace(",inf", ",Infinity")
        path.write_bytes(text.encode("latin-1"))
        out.append(path)
    return out
