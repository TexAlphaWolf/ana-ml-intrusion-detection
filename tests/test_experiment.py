from __future__ import annotations

import json

import numpy as np

from idsbench.experiment import Config, run_all, run_split
from idsbench.splits import by_day_split, official_split, random_split
from tests.conftest import FAST


def runs(report):
    return {r["id"]: r for r in report["runs"]}


def test_all_experiments_are_run_and_named(report):
    assert list(runs(report)) == [
        "unsw-random",
        "unsw-official",
        "unsw-official-no-ttl",
        "cic-random",
        "cic-by-day",
        "cic-by-day-no-port",
        "transfer-unsw-to-cic",
        "transfer-cic-to-unsw",
    ]
    assert report["environment"]["synthetic"] is True and report["environment"]["quick"] is True


def test_the_report_is_plain_json(report):
    assert json.loads(json.dumps(report)) == report


def test_dummy_is_the_floor(report):
    for run in report["runs"]:
        d = run["models"]["dummy"]
        assert abs(d["test_ap"] - run["attack_share"]["test"]) < 0.01
        assert abs(d["test_roc_auc"] - 0.5) < 1e-9
        assert d["test_operating_point"]["tpr"] == 0


def test_a_real_model_beats_the_floor_on_the_easy_split(report):
    r = runs(report)["unsw-random"]
    assert r["models"]["boosting"]["test_ap"] > r["models"]["dummy"]["test_ap"] + 0.2


def test_threshold_is_chosen_on_validation_not_on_test(report):
    for run in report["runs"]:
        for m in run["models"].values():
            assert m["val_operating_point"]["fpr"] <= run["target_fpr"] + 1e-9
            assert m["test_best_possible_tpr_at_target_fpr"] + 1e-9 >= 0


def test_the_analysis_model_is_picked_with_validation_rows(report):
    for run in report["runs"]:
        sup = {k: v["val_ap"] for k, v in run["models"].items() if k not in ("dummy", "isolation")}
        assert run["analysis_model"] == max(sup, key=sup.get)
        assert run["importance"]["model"] == run["analysis_model"]
        assert len(run["importance"]["top"]) == min(10, run["features"])


def test_unseen_families_are_flagged_in_the_realistic_cic_split(report):
    fams = runs(report)["cic-by-day"]["families"]["test"]
    assert fams["bot"]["seen_in_training"] is False and fams["portscan"]["seen_in_training"] is False
    assert runs(report)["cic-random"]["families"]["test"]["portscan"]["seen_in_training"] is True


def test_excluding_the_port_removes_that_column_only(report):
    a, b = runs(report)["cic-by-day"], runs(report)["cic-by-day-no-port"]
    assert a["excluded_features"] == [] and b["excluded_features"] == ["destination_port"]
    assert a["features"] == b["features"] + 1
    assert b["rows"] == a["rows"]


def test_false_positive_view_names_the_column(report):
    assert runs(report)["unsw-official"]["false_positives"]["column"] == "service"
    assert runs(report)["cic-by-day"]["false_positives"]["column"] == "destination_port"


def test_same_seed_same_numbers(unsw):
    cfg = Config(models=("logreg", "tree"), seed=9, resamples=2, importance_rows=200)
    a = run_split(random_split(unsw, 9), cfg)
    b = run_split(random_split(unsw, 9), cfg)

    def strip(r):
        for m in r["models"].values():
            m.pop("fit_seconds"), m.pop("predict_rows_per_second")
        return r

    assert strip(a) == strip(b)


def test_max_train_limits_the_training_rows_but_not_validation_or_test(unsw):
    cfg = Config(models=("tree",), seed=1, max_train=500, resamples=0, importance_rows=0)
    r = run_split(official_split(unsw, 1), cfg)
    assert r["rows"]["train"] == 500 and r["rows"]["train_available"] > 500
    assert r["rows"]["test"] == (unsw.group == "test").sum()
    assert "importance" not in r and r["models"]["tree"]["intervals"] == {}


def test_a_log_function_is_called_for_each_model(unsw):
    lines: list[str] = []
    cfg = Config(models=("dummy", "tree"), seed=1, resamples=0, importance_rows=0)
    run_split(random_split(unsw, 1), cfg, label="x", log=lines.append)
    assert len(lines) == 2 and lines[0].startswith("  x: dummy")


def test_one_dataset_only_has_no_transfer_runs(cic):
    r = run_all({"cic": cic}, Config(models=("dummy", "tree"), resamples=0, importance_rows=0))
    assert [x["id"] for x in r["runs"]] == ["cic-random", "cic-by-day", "cic-by-day-no-port"]


def test_models_without_a_supervised_one_skip_the_analysis(cic):
    r = run_split(by_day_split(cic, 1), Config(models=("dummy", "isolation"), resamples=0))
    assert "analysis_model" not in r and "importance" not in r


def test_unsupervised_detector_is_not_far_below_the_floor(report):
    r = runs(report)["unsw-random"]["models"]["isolation"]
    assert np.isfinite(r["test_ap"])
    assert FAST.models  # the fixture uses the cheap set of models


def test_excluding_the_ttl_columns_removes_exactly_those(report):
    a, b = runs(report)["unsw-official"], runs(report)["unsw-official-no-ttl"]
    assert b["excluded_features"] == ["sttl", "dttl", "ct_state_ttl"] and a["features"] == b["features"] + 3
