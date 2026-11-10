"""Turn benchmark result files into Markdown tables.

    python benchmarks/report.py --experiment e1 [--datasets ...] [--out report.md]

For each dataset it prints:

1. **All configurations**: test-split mean ± sd over seeds for the main metrics.
2. **Selected on validation**: for each method family (PCA, whitening, all-but-the-top, ...),
   the configuration with the best validation score on the primary metric, and its test score,
   with the paired difference to ``raw`` and its bootstrap interval over test points. This is
   the selection rule of WP 3.1: no hyperparameter is chosen on test data.

The primary metric is cosine kNN accuracy for labelled sets, the STS Spearman correlation for
sentence-similarity sets, and trustworthiness otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from run import decode_per_point  # noqa: E402

from embedkit.evaluation import compare  # noqa: E402

COLUMNS = ("sts_spearman", "knn_accuracy", "knn_accuracy_euclidean", "linear_probe",
           "linear_probe_c1", "recall_at_10", "trustworthiness", "k_skewness", "effective_rank")


def load_results(exp_dir: Path, dataset: str) -> tuple[dict[str, pd.DataFrame], dict, list]:
    """Per stage, one DataFrame of all rows; plus per-method metadata and failed methods."""
    frames: dict[str, list] = {"val": [], "test": []}
    meta, failed = {}, []
    for path in sorted((exp_dir / dataset).glob("*.json")):
        rec = json.loads(path.read_text())
        if "error" in rec:
            failed.append((rec["method"], rec["error"]))
            continue
        meta[rec["method"]] = dict(family=rec.get("family") or rec["method"],
                                   params=rec.get("params", {}))
        for stage, rows in rec["stages"].items():
            for r in rows:
                r = dict(r)
                r["per_point"] = decode_per_point(r.get("per_point"))
                frames[stage].append(r)
    return ({k: pd.DataFrame(v) for k, v in frames.items() if v}, meta, failed)


def fmt(mean, sd, n):
    return f"{mean:.3f}" if n <= 1 or not np.isfinite(sd) else f"{mean:.3f} ± {sd:.3f}"


def all_configurations_table(test: pd.DataFrame) -> str:
    cols = [c for c in COLUMNS if c in set(test["metric"])]
    g = test.groupby(["method", "metric"])["value"]
    stats = pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1), "n": g.count()}).reset_index()
    dims = test.groupby("method")["dim"].first()
    lines = ["| Method | d | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    order = _method_order(list(dims.index))
    for m in order:
        cells = []
        for c in cols:
            row = stats[(stats.method == m) & (stats.metric == c)]
            cells.append("" if row.empty else fmt(row["mean"].iloc[0], row["sd"].iloc[0],
                                                  int(row["n"].iloc[0])))
        lines.append(f"| {m} | {dims[m]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _method_order(methods):
    def key(m):
        base, _, k = m.partition("-")
        return (0 if m == "raw" else 1, base, int(k) if k.isdigit() else 0, m)

    return sorted(methods, key=key)


def selection_table(stages, meta, primary: str) -> str:
    val, test = stages.get("val"), stages.get("test")
    if val is None or test is None or primary not in set(val["metric"]):
        return "_No validation results for the primary metric._"
    v = val[val.metric == primary].groupby("method")["value"].mean()
    families: dict[str, list[str]] = {}
    for m in v.index:
        families.setdefault(meta[m]["family"], []).append(m)
    chosen = {fam: max(ms, key=lambda m: (v[m], -len(m))) for fam, ms in families.items()}
    cmp = compare(test[test.metric == primary], reference="raw")
    lines = [f"| Family | Selected on validation | d | val {primary} | test {primary} | "
             "test − raw [95% CI] |", "|---|---|---|---|---|---|"]
    for fam in sorted(chosen, key=lambda f: -cmp.set_index("method").loc[chosen[f], "mean"]):
        m = chosen[fam]
        row = cmp[cmp.method == m].iloc[0]
        diff = "" if m == "raw" else (
            f"{row['diff']:+.3f}" + (f" [{row['diff_lo']:+.3f}, {row['diff_hi']:+.3f}]"
                                     if np.isfinite(row["diff_lo"]) else "")
        )
        lines.append(f"| {fam} | {m} | {int(row['dim'])} | {v[m]:.3f} | "
                     f"{fmt(row['mean'], row['sd'], int(row['n_seeds']))} | {diff} |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--experiment", required=True)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--results", default=str(HERE / "results"))
    p.add_argument("--out", default=None, help="Write Markdown here instead of stdout")
    args = p.parse_args(argv)

    exp_dir = Path(args.results) / args.experiment
    names = args.datasets or sorted(d.name for d in exp_dir.iterdir() if d.is_dir())
    parts = [f"# Experiment {args.experiment}\n"]
    for name in names:
        stages, meta, failed = load_results(exp_dir, name)
        if "test" not in stages:
            parts.append(f"## {name}\n\n_No results._\n")
            continue
        test = stages["test"]
        present = set(test["metric"])
        primary = next(m for m in ("knn_accuracy", "sts_spearman", "trustworthiness")
                       if m in present)
        parts.append(f"## {name}\n")
        parts.append(f"### Selected on validation (primary metric: {primary})\n")
        parts.append(selection_table(stages, meta, primary) + "\n")
        parts.append("### All configurations (test split, mean ± sd over seeds)\n")
        parts.append(all_configurations_table(test) + "\n")
        if failed:
            parts.append("Failed: " + "; ".join(f"{m} ({e[:80]})" for m, e in failed) + "\n")
    text = "\n".join(parts)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
