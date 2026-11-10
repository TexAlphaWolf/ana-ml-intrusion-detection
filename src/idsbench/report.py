"""Turn the result of a run (the JSON) into a Markdown document and into figures. No new measurement happens here."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "dummy": "#8a8a8a",
    "logreg": "#0072b2",
    "tree": "#009e73",
    "forest": "#d55e00",
    "boosting": "#cc79a7",
    "isolation": "#e69f00",
}
STYLES = {"dummy": ":", "logreg": "-", "tree": "-", "forest": "-", "boosting": "-", "isolation": "--"}
BASE_RUNS = ("unsw-official", "cic-by-day")


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def _runs(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["id"]: r for r in report["runs"]}


def _num(n: int | float) -> str:
    return f"{int(n):,}"


def model_table(run: dict[str, Any]) -> list[str]:
    lines = [
        "| Model | Average precision | ROC AUC | Recall | False alarms | Precision | F1 | Best possible recall | Fit (s) | Rows/s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in run["models"].items():
        op, iv = m["test_operating_point"], m["intervals"]
        ci = f" ({iv['ap'][0]:.3f}-{iv['ap'][1]:.3f})" if iv else ""
        lines.append(
            f"| {name} | {m['test_ap']:.3f}{ci} | {m['test_roc_auc']:.3f} | {pct(op['tpr'])} | {pct(op['fpr'], 2)} | "
            f"{pct(op['precision'])} | {op['f1']:.3f} | {pct(m['test_best_possible_tpr_at_target_fpr'])} | "
            f"{m['fit_seconds']:.1f} | {_num(m['predict_rows_per_second'])} |"
        )
    return lines


def family_table(run: dict[str, Any]) -> list[str]:
    best = run.get("analysis_model")
    if not best:
        return []
    fams = run["families"]["test"]
    models = [m for m in run["models"] if m != "dummy"]
    head = "| Attack family | Test rows | Seen in training | " + " | ".join(models) + " |"
    lines = [head, "|---|---|---|" + "---|" * len(models)]
    for fam, info in sorted(fams.items(), key=lambda t: -t[1]["rows"]):
        cells = []
        for m in models:
            rec = run["models"][m]["per_family_recall"].get(fam)
            cells.append(pct(rec["recall"], 0) if rec else "-")
        lines.append(
            f"| {fam} | {_num(info['rows'])} | {'yes' if info['seen_in_training'] else '**no**'} | "
            + " | ".join(cells)
            + " |"
        )
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    env = report["environment"]
    out = ["# Evaluation", ""]
    if env.get("synthetic"):
        out += [
            "> **SYNTHETIC DATA.** This report was produced on generated stand-in data (`idsbench demo`). The numbers test the",
            "> software and say nothing about real intrusion detection.",
            "",
        ]
    cmd = f"; command `idsbench {env['command']}`" if env.get("command") else ""
    part = f"; only {env['sample']:.0%} of the rows of each file were read" if env.get("sample") else ""
    limit = part + f"; at most {_num(env['max_train'])} training rows per run" if env.get("max_train") else ""
    out += [
        f"Run on {env['system']}, Python {env['python']}, scikit-learn {env['scikit_learn']}, pandas {env['pandas']}, "
        f"numpy {env['numpy']}; seed {env['seed']}; {env['seconds']:.0f} s{limit}{cmd}.",
        "",
        "## 1. The data, and what was removed",
        "",
        "| Dataset | Rows read | Not finite | Same features, different label | Exact duplicates | Rows kept | Features kept |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, d in report["data"].items():
        h = d["hygiene"]
        out.append(
            f"| {name} | {_num(h['rows_in'])} | {_num(h['non_finite'])} | {_num(h['conflicting_labels'])} | "
            f"{_num(h['duplicates'])} | {_num(h['rows_out'])} | {h['features']} |"
        )
    out.append("")
    for name, d in report["data"].items():
        fams = ", ".join(f"{k} {_num(v)}" for k, v in d["families"].items())
        out.append(f"- **{name}** classes after cleaning: {fams}.")
        empty = d["hygiene"].get("rows_without_label")
        if empty:
            out.append(f"- **{name}**: {_num(empty)} rows without a label were skipped while reading.")
        extra = d["hygiene"].get("test_rows_identical_to_a_training_row")
        if extra is not None:
            out.append(
                f"- **{name}**: {_num(extra)} rows of the testing file were identical to a training row (removed from the test)."
            )
    out += ["", "## 2. Results by experiment", ""]
    for i, run in enumerate(report["runs"], start=1):
        r, s = run["rows"], run["attack_share"]
        out += [
            f"### 2.{i} `{run['id']}`",
            "",
            run["description"][0].upper() + run["description"][1:] + ".",
            "",
            f"Rows: train {_num(r['train'])}"
            + (f" (of {_num(r['train_available'])} available)" if r["train_available"] != r["train"] else "")
            + f", validation {_num(r['val'])}, test {_num(r['test'])}. Share of attacks: train {pct(s['train'])}, "
            f"test {pct(s['test'])}. Features: {run['features']}"
            + (f" (without {', '.join(run['excluded_features'])})" if run["excluded_features"] else "")
            + ".",
            "",
            f"Alert threshold: the score above which at most {pct(run['target_fpr'], 0)} of the normal validation rows alert. "
            'Recall and false alarms are measured on the test rows with that threshold; "best possible recall" is what a '
            "threshold chosen on the test rows themselves would give at the same false-alarm rate.",
            "",
            *model_table(run),
            "",
        ]
        fam = family_table(run)
        if fam and "transfer" not in run["protocol"]:
            out += [
                'Recall by attack family at the same threshold ("seen in training" says whether the family had rows in the training set):',
                "",
                *fam,
                "",
            ]
        imp = run.get("importance")
        if imp:
            out += [f"Most important columns for `{imp['model']}` ({imp['metric']}):", ""]
            out += [f"- `{c}`: {m:.4f} (± {s:.4f})" for c, m, s in imp["top"]]
            out.append("")
        fp = run.get("false_positives")
        if fp and fp.get("top"):
            top = ", ".join(f"`{k}` {a} of {b}" for k, (a, b) in fp["top"].items())
            out += [
                f"False alarms of `{run['analysis_model']}` by `{fp['column']}` (false alarms of normal rows with that value): {top}.",
                "",
            ]
    return "\n".join(out) + "\n"


def _fig_pr(ax: Any, run: dict[str, Any]) -> None:
    for name, m in run["models"].items():
        c = m["curve"]
        if c["recall"]:
            ax.plot(
                c["recall"],
                c["precision"],
                STYLES[name],
                color=COLORS[name],
                lw=1.8,
                label=f"{name} ({m['test_ap']:.2f})",
            )
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(run["id"], fontsize=10)
    ax.legend(fontsize=7, loc="lower left", title="model (average precision)", title_fontsize=7)
    ax.grid(alpha=0.25)


def _fig_family(ax: Any, run: dict[str, Any]) -> None:
    best = run["analysis_model"]
    fams = sorted(run["families"]["test"].items(), key=lambda t: t[1]["rows"])
    names = [f for f, _ in fams]
    rec = [run["models"][best]["per_family_recall"][f]["recall"] for f in names]
    cols = ["#0072b2" if run["families"]["test"][f]["seen_in_training"] else "#d55e00" for f in names]
    ax.barh(names, rec, color=cols)
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"recall of {best} at the chosen threshold")
    ax.set_title(f"{run['id']}: blue seen in training, orange never seen", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=0.25)


def _fig_gap(ax: Any, runs: dict[str, dict[str, Any]]) -> None:
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for d, rid in (("unsw", "unsw-official"), ("cic", "cic-by-day")):
        a, b = runs.get(f"{d}-random"), runs.get(rid)
        if a and b:
            pairs.append((d, a, b))
    if not pairs:
        ax.axis("off")
        return
    models = [m for m in pairs[0][1]["models"] if m != "dummy"]
    width = 0.8 / (2 * len(pairs))
    for k, (d, a, b) in enumerate(pairs):
        for j, (run, tag, alpha) in enumerate(((a, "random split", 0.4), (b, "realistic split", 1.0))):
            offset = (2 * k + j - len(pairs) + 0.5) * width
            ax.bar(
                [i + offset for i in range(len(models))],
                [run["models"][m]["test_operating_point"]["tpr"] for m in models],
                width,
                color=[COLORS[m] for m in models],
                alpha=alpha,
                edgecolor="black",
                linewidth=0.4,
                hatch="//" if d == "cic" else None,
                label=f"{d}, {tag}",
            )
    ax.set_xticks(range(len(models)), models, fontsize=8)
    ax.set_ylabel("recall at the threshold for 1% false alarms")
    ax.set_ylim(0, 1)
    ax.set_title("random split (pale) vs realistic split (solid); hatched = CIC-IDS2017", fontsize=9)
    ax.grid(axis="y", alpha=0.25)


def summary_figure(report: dict[str, Any], path: Path) -> None:
    runs = _runs(report)
    base = [runs[r] for r in BASE_RUNS if r in runs] or [r for r in report["runs"] if "transfer" not in r["protocol"]][
        :2
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=110)
    panels = list(axes.flat)
    for ax, run in zip(panels[:2], base, strict=False):
        _fig_pr(ax, run)
    for ax in panels[len(base) : 2]:
        ax.axis("off")
    family_run = next((r for r in reversed(base) if r.get("analysis_model") and r["families"]["test"]), None)
    if family_run:
        _fig_family(panels[2], family_run)
    else:
        panels[2].axis("off")
    _fig_gap(panels[3], runs)
    title = "idsbench" + ("  (SYNTHETIC data: not a result)" if report["environment"].get("synthetic") else "")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
