from __future__ import annotations

import json
from pathlib import Path

import pytest

from idsbench.cli import main
from idsbench.report import render_markdown, summary_figure


def test_markdown_has_every_section_and_the_banner(report):
    md = render_markdown(report)
    assert md.startswith("# Evaluation")
    assert "SYNTHETIC DATA" in md
    assert "## 1. The data, and what was removed" in md and "## 2. Results by experiment" in md
    for run in report["runs"]:
        assert f"`{run['id']}`" in md
    assert "| boosting |" in md and "Seen in training" in md and "**no**" in md


def test_markdown_for_real_data_has_no_banner(report):
    real = json.loads(json.dumps(report))
    real["environment"]["synthetic"] = False
    real["environment"].pop("command", None)
    real["environment"]["max_train"] = None
    assert "SYNTHETIC" not in render_markdown(real)


def test_markdown_numbers_are_the_report_numbers(report):
    md = render_markdown(report)
    r = next(x for x in report["runs"] if x["id"] == "unsw-official")
    assert f"{r['models']['boosting']['test_ap']:.3f}" in md
    assert f"{r['rows']['test']:,}" in md


def test_the_figure_is_written(report, tmp_path: Path):
    out = tmp_path / "sub" / "s.png"
    summary_figure(report, out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and out.stat().st_size > 20_000


def test_the_figure_works_with_one_dataset_only(report, tmp_path: Path):
    one = json.loads(json.dumps(report))
    one["runs"] = [r for r in one["runs"] if r["id"].startswith("unsw")]
    summary_figure(one, tmp_path / "one.png")
    assert (tmp_path / "one.png").exists()


def test_cli_synth_check_run_report(folders, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    d = tmp_path / "u"
    assert main(["synth", "unsw", str(d), "--rows", "500"]) == 0
    assert "UNSW_NB15_training-set.csv" in capsys.readouterr().out
    assert main(["check", str(folders["unsw"]), str(folders["cic"])]) == 0
    out = capsys.readouterr().out
    assert "unsw" in out and "cic" in out and "dos hulk" in out and "categorical: proto, service, state" in out

    outdir = tmp_path / "o"
    code = main(
        [
            "run",
            "--unsw",
            str(folders["unsw"]),
            "--out",
            str(outdir),
            "--models",
            "dummy,tree",
            "--quick",
            "--sample",
            "0.5",
        ]
    )
    assert code == 0
    rep = json.loads((outdir / "report.json").read_text())
    assert rep["environment"]["quick"] is True and rep["environment"]["synthetic"] is False
    assert [r["id"] for r in rep["runs"]] == ["unsw-random", "unsw-official", "unsw-official-no-ttl"]
    assert (outdir / "evaluation.md").read_text().startswith("# Evaluation")
    assert (outdir / "summary.png").exists()

    md = tmp_path / "again.md"
    assert main(["report", str(outdir / "report.json"), "-o", str(md), "--figure", str(tmp_path / "f.png")]) == 0
    assert md.read_text() == (outdir / "evaluation.md").read_text()


def test_cli_demo_runs_without_any_download(tmp_path: Path):
    out = tmp_path / "demo"
    assert main(["demo", "--rows", "600", "--out", str(out), "--models", "dummy,tree", "--quick"]) == 0
    rep = json.loads((out / "report.json").read_text())
    assert rep["environment"]["synthetic"] is True and len(rep["runs"]) == 8


def test_cli_errors_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["check", str(tmp_path / "nope")]) == 2
    assert "not a folder" in capsys.readouterr().err
    assert main(["run", "--out", str(tmp_path)]) == 2
    assert main(["run", "--unsw", str(tmp_path / "nope"), "--out", str(tmp_path)]) == 2
    (tmp_path / "x.csv").write_text("a\n1\n")
    assert main(["check", str(tmp_path)]) == 2
    with pytest.raises(SystemExit):
        main(["bogus"])


def test_cli_rejects_an_unknown_model(folders, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["run", "--unsw", str(folders["unsw"]), "--models", "svm", "--out", str(tmp_path)]) == 2
    assert "unknown model" in capsys.readouterr().err


def test_cli_version(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and capsys.readouterr().out.startswith("idsbench ")
