from __future__ import annotations

import numpy as np

from idsbench.clean import row_hash
from idsbench.splits import (
    COMMON,
    TEST_DAYS,
    TRAIN_DAYS,
    by_day_split,
    common_view,
    official_split,
    random_split,
    realistic_split,
    stratified_parts,
    transfer_split,
)


def test_stratified_parts_cover_everything_once_and_keep_every_family_on_each_side():
    fam = np.array(["a"] * 100 + ["b"] * 10 + ["c"] * 3 + ["d"] * 2, dtype=object)
    parts = stratified_parts(fam, (0.6, 0.2, 0.2), seed=1)
    allidx = np.concatenate(parts)
    assert sorted(allidx) == list(range(len(fam)))
    for p in parts:
        assert {"a", "b"} <= set(fam[p])
    assert {"c"} <= set(fam[parts[2]])  # three rows: one for each side
    assert set(fam[parts[0]]) >= {"d"} and not set(fam[parts[1]]) & {"d"}  # two rows all go to the first part


def test_stratified_parts_are_deterministic_and_seeded():
    fam = np.array(["a", "b"] * 50, dtype=object)
    a = stratified_parts(fam, (0.5, 0.5), 7)
    b = stratified_parts(fam, (0.5, 0.5), 7)
    c = stratified_parts(fam, (0.5, 0.5), 8)
    assert all((x == y).all() for x, y in zip(a, b, strict=True))
    assert not (a[0] == c[0]).all()


def test_random_split_has_no_row_in_two_parts(unsw):
    s = random_split(unsw, 1)
    assert len(s.train) + len(s.val) + len(s.test) == len(unsw)
    h = [set(row_hash(d.X).tolist()) for d in (s.train, s.val, s.test)]
    assert not h[0] & h[2] and not h[0] & h[1]  # duplicates were removed before splitting
    assert abs(s.train.y.mean() - s.test.y.mean()) < 0.03


def test_official_split_uses_the_two_files(unsw):
    s = official_split(unsw, 1)
    assert set(s.test.group) == {"test"} and set(s.train.group) == {"train"} == set(s.val.group)
    assert len(s.train) + len(s.val) == (unsw.group == "train").sum()
    assert s.train.cat_cols == unsw.cat_cols


def test_by_day_split_has_attacks_the_model_never_saw(cic):
    s = by_day_split(cic, 1)
    assert set(s.train.group) <= set(TRAIN_DAYS) and set(s.test.group) == set(TEST_DAYS)
    unseen = set(s.test.family[s.test.y]) - set(s.train.family) - set(s.val.family)
    assert {"bot", "portscan", "ddos"} <= unseen
    assert realistic_split(cic, 1).protocol == "by-day"


def test_realistic_split_picks_by_dataset(unsw):
    assert realistic_split(unsw, 1).protocol == "official"


def test_common_view_converts_units_and_derives_the_same_columns(unsw, cic):
    u, c = common_view(unsw), common_view(cic)
    assert list(u.X.columns) == COMMON == list(c.X.columns)
    # CIC counts microseconds: after conversion the duration is the raw value / 1e6
    raw = cic.X["flow_duration"].to_numpy(dtype=np.float64)
    assert np.allclose(c.X["duration"].to_numpy(dtype=np.float64), raw / 1e6, rtol=1e-5)
    pk = c.X["fwd_packets"] + c.X["bwd_packets"]
    assert (c.X["packets_per_s"] >= 0).all() and np.isfinite(c.X.to_numpy()).all()
    assert np.allclose(
        c.X["mean_packet"].to_numpy(dtype=np.float64),
        (c.X["fwd_bytes"] + c.X["bwd_bytes"]).to_numpy(dtype=np.float64) / pk.clip(lower=1).to_numpy(dtype=np.float64),
        rtol=1e-4,
    )


def test_transfer_split_trains_on_one_and_tests_on_all_of_the_other(unsw, cic):
    s = transfer_split(unsw, cic, 1)
    assert len(s.test) == len(cic) and len(s.train) + len(s.val) == len(unsw)
    assert s.train.cat_cols == [] and s.protocol == "transfer unsw->cic"
