"""Command line: check the data, run the benchmark, try it on synthetic data, rebuild the report."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from idsbench import __version__
from idsbench.dataset import Dataset, DatasetError
from idsbench.experiment import Config, run_all
from idsbench.loaders import detect, load
from idsbench.models import MODEL_NAMES
from idsbench.report import render_markdown, summary_figure
from idsbench.synth import write_cic, write_unsw


def _log(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def cmd_check(args: argparse.Namespace) -> int:
    for folder in args.folders:
        path = Path(folder)
        kind = detect(path)
        print(f"{path}: {kind or 'not recognised'}")
        if kind is None:
            return 2
        ds = load(path, kind)
        h = ds.hygiene
        print(
            f"  rows read {h['rows_in']:,}, kept {h['rows_out']:,} ({h['duplicates']:,} duplicates, "
            f"{h['non_finite']:,} not finite, {h['conflicting_labels']:,} with conflicting labels)"
        )
        print(f"  {h['features']} features" + (f", categorical: {', '.join(ds.cat_cols)}" if ds.cat_cols else ""))
        print(f"  groups: {', '.join(f'{g} {int((ds.group == g).sum()):,}' for g in sorted(set(ds.group)))}")
        for fam, n in ds.family_counts().items():
            print(f"    {fam}: {n:,}")
    return 0


def _config(args: argparse.Namespace) -> Config:
    models = tuple(args.models.split(",")) if args.models else MODEL_NAMES
    unknown = [m for m in models if m not in MODEL_NAMES]
    if unknown:
        raise DatasetError(f"unknown model {unknown}; choose from {', '.join(MODEL_NAMES)}")
    cfg = Config(
        models=models,
        seed=args.seed,
        max_train=args.max_train or None,
        n_jobs=args.jobs,
        sample=getattr(args, "sample", None),
    )
    if args.quick:
        cfg.max_train = min(cfg.max_train or 60_000, 60_000)
        cfg.resamples, cfg.importance_rows = 10, 3000
    return cfg


def _finish(report: dict[str, object], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (out / "evaluation.md").write_text(render_markdown(report), encoding="utf-8")
    summary_figure(report, out / "summary.png")
    print(f"written: {out / 'report.json'}, {out / 'evaluation.md'}, {out / 'summary.png'}")


def cmd_run(args: argparse.Namespace) -> int:
    datasets: dict[str, Dataset] = {}
    for kind, folder in (("unsw", args.unsw), ("cic", args.cic)):
        if folder:
            _log(f"reading {kind} from {folder}")
            datasets[kind] = load(Path(folder), kind, sample=args.sample, seed=args.seed)
    if not datasets:
        _log("give --unsw FOLDER and/or --cic FOLDER (or try `idsbench demo`)")
        return 2
    cfg = _config(args)
    report = run_all(datasets, cfg, quick=args.quick, log=_log)
    _finish(report, Path(args.out))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    cfg = _config(args)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_unsw(root / "unsw", rows=args.rows, seed=args.seed)
        write_cic(root / "cic", rows_per_day=args.rows // 2, seed=args.seed)
        datasets = {"unsw": load(root / "unsw", "unsw"), "cic": load(root / "cic", "cic")}
        report = run_all(datasets, cfg, quick=args.quick, synthetic=True, log=_log)
    _finish(report, Path(args.out))
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    folder = Path(args.folder)
    files = write_unsw(folder, args.rows, args.seed) if args.kind == "unsw" else write_cic(folder, args.rows, args.seed)
    for f in files:
        print(f)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    out = Path(args.output)
    out.write_text(render_markdown(report), encoding="utf-8")
    if args.figure:
        summary_figure(report, Path(args.figure))
    print(f"written: {out}")
    return 0


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="idsbench", description="Benchmark machine-learning intrusion detectors.")
    ap.add_argument("--version", action="version", version=f"idsbench {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--out", default="idsbench-out", help="folder for report.json, evaluation.md, summary.png")
        p.add_argument("--models", help=f"comma separated, default all: {','.join(MODEL_NAMES)}")
        p.add_argument("--seed", type=int, default=1000)
        p.add_argument("--max-train", type=int, default=500_000, help="most training rows per run (0 = all)")
        p.add_argument("--jobs", type=int, default=1, help="CPU cores for the forests")
        p.add_argument("--quick", action="store_true", help="smaller training sets and fewer resamples")

    p = sub.add_parser("check", help="read the files of a dataset folder and describe what was found")
    p.add_argument("folders", nargs="+")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("run", help="the benchmark on real data")
    p.add_argument(
        "--unsw", metavar="FOLDER", help="folder with UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv"
    )
    p.add_argument("--cic", metavar="FOLDER", help="folder with the CIC-IDS2017 MachineLearningCSV files")
    p.add_argument("--sample", type=float, help="read only this fraction of each file (0-1), for a fast try")
    common(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser(
        "demo", help="the whole benchmark on synthetic data: tests the software, says nothing about attacks"
    )
    p.add_argument("--rows", type=int, default=6000)
    common(p)
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("synth", help="write synthetic files in the layout of a dataset")
    p.add_argument("kind", choices=["unsw", "cic"])
    p.add_argument("folder")
    p.add_argument("--rows", type=int, default=6000, help="rows (unsw) or rows per day (cic)")
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("report", help="rebuild the Markdown (and the figure) from a report.json")
    p.add_argument("report")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--figure")
    p.set_defaults(fn=cmd_report)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except DatasetError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
