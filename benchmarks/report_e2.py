"""E2 report: does a label-free statistic track downstream accuracy? (review section 5, WP 3.2)

    python benchmarks/report_e2.py --out benchmarks/results/e2/REPORT.md

Reads ``<results>/e2/<dataset>/c*.json`` written by ``benchmarks/e2.py``. Every monitor is
sign-adjusted by its direction in the registry, so that a positive correlation always means "the
monitor's better values go with higher accuracy".

- **Within a run (stopping signal):** Spearman correlation between the monitor and the test
  metric over the evaluation epochs of one run; median and quartiles over the runs. Regret: the
  test metric's best value in the run minus its value at the epoch where the monitor is best
  (first best); for comparison, the same for the final epoch.
- **Across runs (selection signal):** Spearman correlation over configurations between the
  monitor and the test metric, both at the final epoch. Regret: the best configuration's test
  metric minus that of the configuration the monitor ranks first.

``val_knn_accuracy`` uses labels (the labelled reference); ``train_loss`` is the training loss of
the evaluation epoch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
TARGETS = ("test_knn_accuracy", "test_linear_probe")
SHORT = {"test_knn_accuracy": "kNN", "test_linear_probe": "probe"}
MONITOR_ORDER = ("uniformity", "isotropy", "k_skewness", "knn_overlap", "effective_rank",
                 "val_loss", "train_loss", "val_knn_accuracy")
LABELLED = {"val_knn_accuracy"}


def load_runs(ds_dir: Path):
    runs, failed = [], []
    for path in sorted(ds_dir.glob("c*.json")):
        rec = json.loads(path.read_text())
        (failed if "error" in rec else runs).append(rec)
    ref_path = ds_dir / "_reference.json"
    ref = json.loads(ref_path.read_text()) if ref_path.exists() else {}
    return runs, failed, ref


def _signed(rec, monitor) -> np.ndarray:
    v = np.asarray(rec["history"]["evals"][monitor], dtype=float)
    return v if rec["directions"][monitor] == "max" else -v


def _rho(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return np.nan
    return float(spearmanr(a[ok], b[ok]).statistic)


def _first_best(v: np.ndarray) -> int:
    v = np.where(np.isfinite(v), v, -np.inf)
    return int(np.argmax(v))


def within_run(runs, monitors, target):
    """Per monitor: (rhos over runs, regrets over runs); plus the final-epoch regrets."""
    out = {}
    for m in monitors:
        rhos, regrets = [], []
        for rec in runs:
            acc = np.asarray(rec["history"]["evals"][target], float)
            s = _signed(rec, m)
            rhos.append(_rho(s, acc))
            regrets.append(acc.max() - acc[_first_best(s)])
        out[m] = (np.array(rhos), np.array(regrets))
    final = np.array([max(r["history"]["evals"][target]) - r["history"]["evals"][target][-1]
                      for r in runs])
    return out, final


def across_runs(runs, monitors, target):
    acc = np.array([r["history"]["evals"][target][-1] for r in runs])
    out = {}
    for m in monitors:
        s = np.array([_signed(r, m)[-1] for r in runs])
        out[m] = (_rho(s, acc), acc.max() - acc[_first_best(s)])
    return out


def _q(x):
    x = x[np.isfinite(x)]
    if not len(x):
        return "n/a"
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return f"{med:+.2f} [{q1:+.2f}, {q3:+.2f}]"


def _name(m):
    return f"{m} (labels)" if m in LABELLED else m


def dataset_section(name, runs, failed, ref) -> tuple[str, dict]:
    monitors = [m for m in MONITOR_ORDER if m in runs[0]["history"]["evals"]]
    lines = [f"## {name}", ""]
    knn_final = np.array([r["history"]["evals"]["test_knn_accuracy"][-1] for r in runs])
    knn_peak = np.array([max(r["history"]["evals"]["test_knn_accuracy"]) for r in runs])
    probe_final = np.array([r["history"]["evals"]["test_linear_probe"][-1] for r in runs])
    r0 = runs[0]
    lines += [
        f"{len(runs)} runs ({len(failed)} failed), {r0['epochs']} epochs, evaluated every "
        f"{r0['eval_every']}; downstream on {ref.get('n_pool', '?')} training / "
        f"{ref.get('n_test', '?')} test rows.",
        "",
        "| | min | median | max | raw |",
        "|---|---|---|---|---|",
        f"| test kNN, final epoch | {knn_final.min():.3f} | {np.median(knn_final):.3f} | "
        f"{knn_final.max():.3f} | {ref.get('raw_test_knn_accuracy', np.nan):.3f} |",
        f"| test kNN, best epoch | {knn_peak.min():.3f} | {np.median(knn_peak):.3f} | "
        f"{knn_peak.max():.3f} | |",
        f"| test probe (C = 1), final epoch | {probe_final.min():.3f} | "
        f"{np.median(probe_final):.3f} | {probe_final.max():.3f} | "
        f"{ref.get('raw_test_linear_probe', np.nan):.3f} |",
        "",
    ]
    summary = {}
    for target in TARGETS:
        w, final_regret = within_run(runs, monitors, target)
        a = across_runs(runs, monitors, target)
        t = SHORT[target]
        lines += [
            f"### Test {t}",
            "",
            "| Monitor | within-run ρ, median [IQR] | runs with ρ > 0.5 | stop-at-best regret, "
            "mean (max) | across-run ρ | selection regret |",
            "|---|---|---|---|---|---|",
        ]
        for m in monitors:
            rhos, reg = w[m]
            ok = rhos[np.isfinite(rhos)]
            frac = f"{(ok > 0.5).sum()}/{len(rhos)}"
            rho_a, sel_reg = a[m]
            lines.append(f"| {_name(m)} | {_q(rhos)} | {frac} | {reg.mean():.3f} ({reg.max():.3f}) "
                         f"| {rho_a:+.2f} | {sel_reg:.3f} |")
            summary[(target, m)] = (np.nanmedian(rhos) if len(ok) else np.nan, rho_a)
        lines += [f"| final epoch (no stopping) | | | {final_regret.mean():.3f} "
                  f"({final_regret.max():.3f}) | | |", ""]
    if failed:
        lines += ["Failed runs: " + ", ".join(f"{r['config']} ({r['error'][:80]})" for r in failed),
                  ""]
    return "\n".join(lines), summary


def overview(summaries: dict[str, dict]) -> str:
    names = list(summaries)
    lines = ["## Overview: median within-run ρ (stopping) / across-run ρ (selection), test kNN",
             "", "| Monitor | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    monitors = [m for m in MONITOR_ORDER
                if any(("test_knn_accuracy", m) in s for s in summaries.values())]
    for m in monitors:
        cells = []
        for n in names:
            w, a = summaries[n].get(("test_knn_accuracy", m), (np.nan, np.nan))
            cells.append(f"{w:+.2f} / {a:+.2f}")
        lines.append(f"| {_name(m)} | " + " | ".join(cells) + " |")
    lines += ["", "The plan's rule (WP 3.3): the default monitor becomes the label-free statistic "
              "with the best within-run correlation; if none has a median ρ above 0.5, early "
              "stopping is disabled by default.", ""]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default=str(HERE / "results"))
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--out", default=None, help="Write Markdown here instead of stdout")
    args = p.parse_args(argv)
    root = Path(args.results) / "e2"
    names = args.datasets or sorted(p.name for p in root.iterdir() if p.is_dir())
    sections, summaries = [], {}
    for name in names:
        runs, failed, ref = load_runs(root / name)
        if not runs:
            sections.append(f"## {name}\n\nNo completed runs.\n")
            continue
        text, summaries[name] = dataset_section(name, runs, failed, ref)
        sections.append(text)
    doc = "\n".join(["# Experiment e2", "", __doc__.split("\n\n", 2)[2].strip(), "",
                     overview(summaries), *sections])
    if args.out:
        Path(args.out).write_text(doc)
    else:
        sys.stdout.write(doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
