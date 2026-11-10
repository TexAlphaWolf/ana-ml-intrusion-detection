"""The numbers quoted in the README must be the ones in docs/report.json, and the documented things must exist."""

from __future__ import annotations

import json
import re
from pathlib import Path

from idsbench.cli import _parser
from idsbench.report import render_markdown

ROOT = Path(__file__).resolve().parent.parent
REPORT = json.loads((ROOT / "docs" / "report.json").read_text(encoding="utf-8"))
README = (ROOT / "README.md").read_text(encoding="utf-8")
RUNS = {r["id"]: r for r in REPORT["runs"]}


def ap(run: str, model: str) -> float:
    return RUNS[run]["models"][model]["test_ap"]


def op(run: str, model: str) -> dict[str, float]:
    return RUNS[run]["models"][model]["test_operating_point"]


def test_report_is_a_full_run_on_real_data():
    env = REPORT["environment"]
    assert env["quick"] is False and env["synthetic"] is False and env["sample"] is None
    assert env["seed"] >= 1000 and env["max_train"] == 500_000 and env["resamples"] == 50
    assert set(RUNS) == {
        "unsw-random",
        "unsw-official",
        "unsw-official-no-ttl",
        "cic-random",
        "cic-by-day",
        "cic-by-day-no-port",
        "transfer-unsw-to-cic",
        "transfer-cic-to-unsw",
    }


def test_evaluation_md_is_rebuilt_from_the_json():
    assert (ROOT / "docs" / "evaluation.md").read_text(encoding="utf-8") == render_markdown(REPORT)


def test_the_data_table():
    for name, label in (("unsw", "UNSW-NB15"), ("cic", "CIC-IDS2017")):
        h = REPORT["data"][name]["hygiene"]
        row = (
            f"| {label} | {h['rows_in']:,} | {h['non_finite']:,} | {h['conflicting_labels']:,} | {h['duplicates']:,} "
            f"| {h['rows_out']:,} | {h['features']} |"
        )
        assert row in README, row
    h = REPORT["data"]["unsw"]["hygiene"]
    assert f"{h['test_rows_identical_to_a_training_row']:,} rows of the UNSW testing file" in README
    assert f"{h['duplicates']:,} of {h['rows_in']:,} UNSW rows ({h['duplicates'] / h['rows_in']:.1%})" in README


def test_results_table_in_the_readme():
    labels = {
        "UNSW random": "unsw-random",
        "UNSW official": "unsw-official",
        "UNSW official, no TTL": "unsw-official-no-ttl",
        "CIC random": "cic-random",
        "CIC by day": "cic-by-day",
        "CIC by day, no port": "cic-by-day-no-port",
        "Transfer UNSW to CIC": "transfer-unsw-to-cic",
        "Transfer CIC to UNSW": "transfer-cic-to-unsw",
    }
    for label, run in labels.items():
        cells = [f"{ap(run, m):.3f}" for m in ("dummy", "logreg", "tree", "forest", "boosting", "isolation")]
        assert f"| {label} | " + " | ".join(cells) + " |" in README, label


def test_headline_claims():
    cb = RUNS["cic-by-day"]
    b = cb["models"]["boosting"]
    assert (
        op("cic-random", "boosting")["tpr"] > 0.9995
        and f"recall **{op('cic-random', 'boosting')['tpr']:.1%}**" in README
    )
    assert f"recall **{op('cic-by-day', 'boosting')['tpr']:.1%}**" in README
    assert f"average precision {ap('cic-by-day', 'boosting'):.3f}" in README
    assert f"recall 78.3% at {op('unsw-random', 'boosting')['fpr']:.2%} false alarms" in README
    assert (
        f"recall {op('unsw-official', 'boosting')['tpr']:.1%}, but **{op('unsw-official', 'boosting')['fpr']:.2%}**"
        in README
    )
    # the recall by day is one family
    fam = cb["families"]["test"]
    total = sum(v["rows"] for v in fam.values())
    caught = {k: v["recall"] * v["rows"] for k, v in b["per_family_recall"].items()}
    assert b["per_family_recall"]["ddos"]["recall"] > 0.985 and "catches 99% of the DDoS" in README
    assert round(caught["ddos"] / sum(caught.values()), 2) == 0.96 and "96% of everything it catches" in README
    assert (
        round(fam["portscan"]["rows"] / total, 2) == 0.41
        and round(b["per_family_recall"]["portscan"]["recall"], 2) == 0.03
    )
    assert b["per_family_recall"]["bot"]["recall"] == 0.0 and "`bot` 0%" in README
    assert all(not v["seen_in_training"] for v in fam.values())
    # the threshold does not carry over
    sup = ("logreg", "tree", "forest", "boosting")
    off = [op("unsw-official", m)["fpr"] for m in sup]
    day = [op("cic-by-day", m)["fpr"] for m in sup]
    assert f"{min(off):.2%} to\n   {max(off):.2%} on the UNSW test file" in README
    assert f"{min(day):.2%} to {max(day):.2%} on CIC by day" in README
    # ablations
    assert (
        f"recall from {op('unsw-official', 'logreg')['tpr']:.1%} to {op('unsw-official-no-ttl', 'logreg')['tpr']:.1%}"
        in README
    )
    assert (
        f"forest {ap('cic-by-day-no-port', 'forest'):.3f} on average precision, boosting {ap('cic-by-day-no-port', 'boosting'):.3f}"
        in README
    )
    assert RUNS["unsw-official"]["importance"]["top"][0][0] == "sttl"
    assert RUNS["cic-by-day"]["importance"]["top"][0][0] == "destination_port"
    # the isolation forest and the transfer
    assert (
        f"{op('unsw-random', 'isolation')['tpr']:.1%}\n   recall on a random UNSW split and {op('cic-by-day', 'isolation')['tpr']:.1%} on CIC by day"
        in README
    )
    for run in ("transfer-unsw-to-cic", "transfer-cic-to-unsw"):
        prior = ap(run, "dummy")
        for m in ("tree", "forest", "boosting"):
            assert abs(ap(run, m) - prior) < 0.06, (run, m)
        for m in sup:
            assert RUNS[run]["models"][m]["test_best_possible_tpr_at_target_fpr"] < 0.034


def test_analysis_numbers():
    text = (ROOT / "docs" / "analysis.md").read_text(encoding="utf-8")
    fp = RUNS["cic-by-day"]["false_positives"]["top"]
    assert f"port 80 produces {fp['80.0'][0]:,} false alarms among its {fp['80.0'][1]:,}" in text
    assert f"port 21 {fp['21.0'][0]} of {fp['21.0'][1]:,} normal" in text
    env = REPORT["environment"]
    assert f"{env['seconds']:.0f} s" in text
    sp = RUNS["unsw-random"]["models"]["boosting"]["fit_seconds"]
    cr = RUNS["cic-random"]["models"]["boosting"]
    assert f"{sp:.1f} s" in text and f"{cr['fit_seconds']:.1f} s" in text
    forest = RUNS["cic-random"]["models"]["forest"]
    assert f"{round(cr['predict_rows_per_second']):,}" in text.replace("\n", " ")
    assert f"{round(forest['predict_rows_per_second']):,}" in text and f"{forest['fit_seconds']:.1f} s" in text


def test_documented_commands_exist():
    sub = next(a for a in _parser()._actions if a.dest == "cmd").choices
    for name in ("check", "run", "demo", "report"):
        assert name in sub and f"idsbench {name}" in README


def test_links_point_to_files():
    for doc in ("README.md", "docs/analysis.md", "docs/limitations.md"):
        for target in re.findall(r"\]\(((?!http)[^)#]+)\)", (ROOT / doc).read_text(encoding="utf-8")):
            base = ROOT if doc == "README.md" else (ROOT / doc).parent
            assert (base / target).exists(), (doc, target)


def test_scaffold_is_gone():
    assert "work in progress" not in README.lower() and "scaffold" not in README
    assert (ROOT / "docs" / "summary.png").stat().st_size > 20_000
