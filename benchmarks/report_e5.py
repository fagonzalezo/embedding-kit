"""Summarise experiment E5 (controlled damage) into CSV tables and a Markdown report.

    python benchmarks/report_e5.py [--results benchmarks/results/e5]

Detection (``<results>/_detect/*.json``, from ``benchmarks/e5.py``): every damaged variant is
compared with its clean base, per metric. A calibrated statistic detects the damage when its z
moves by more than ``max(3, 3 x sd)``, and a property when it moves by more than ``3 x sd``, with
``sd`` the spread over the clean report on 5 random 80% row subsets (the noise floor). The
expected signatures are the table in ``docs/IMPLEMENTATION_PLAN.md`` (WP 4.5), fixed before the
runs.

Repair (``<results>/<dataset>/<method>.json``, from ``run.py --experiment e5``): test accuracy
of each training-free baseline on each variant. A damage is recovered when the best baseline's
cosine kNN accuracy is within 1 point of raw on the clean set. Methods are grouped by family,
since ``abtt-<d // 100>`` changes name with the padded dimension.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from report_e3 import CALIBRATED, md, rec_types  # noqa: E402

PROPS = {
    "mean_vector_ratio": "Geometry/mean_vector_ratio",
    "mean_cosine": "Geometry/mean_cosine",
    "top1_share": "Geometry/top_1_variance_share",
    "top3_share": "Geometry/top_3_variance_share",
    "pr_normalized": "Geometry/participation_ratio_normalized",
    "spectrum_alpha": "Geometry/spectrum_alpha",
    "scale_dev": "ID/scale_max_deviation",
    "twonn": "ID/TwoNN",
    "distance_cv": "Geometry/distance_cv",
    "neighbor_consistency": "Geometry/neighbor_consistency",
    "k_skewness": "Hubness/k_skewness",
    "kernel_gap": "Kernel/spectral_gap",
    "kernel_eff_rank": "Kernel/effective_rank",
}
# Expected signatures (plan, WP 4.5): property -> sign of the change.
EXPECTED = {
    "mean": {"mean_vector_ratio": +1, "mean_cosine": +1},
    "nuis": {"top1_share": +1, "top3_share": +1, "pr_normalized": -1, "spectrum_alpha": +1},
    "aniso": {"pr_normalized": -1, "spectrum_alpha": +1},
    "noise": {"twonn": +1, "distance_cv": -1, "neighbor_consistency": -1, "scale_dev": +1},
    "pad": {"twonn": +1, "top1_share": -1, "top3_share": -1},
    "domain": {"kernel_gap": +1, "top1_share": +1},
}
ACC = ("knn_accuracy", "knn_accuracy_euclidean", "linear_probe_c1")


# ------------------------------------------------------------------------------------ detection
def detect_frame(root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((root / "_detect").glob("*.json")):
        r = json.loads(f.read_text())
        if "error" in r:
            print(f"[error] {f.name}: {r['error'][:120]}")
            continue
        variant = r["dataset"].partition("__")[2]
        row = dict(base=r["base"], metric=r["metric"], variant=variant or "clean",
                   damage=r.get("damage"), level=r.get("damage_level"),
                   recs=",".join(rec_types(r["recommendations"])) or "none")
        for k, c in r["calibration"].items():
            row[f"z_{k}"] = c["z"]
        for name, key in PROPS.items():
            row[name] = r["scalars"].get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def detection(df: pd.DataFrame) -> pd.DataFrame:
    stats = [f"z_{k}" for k in CALIBRATED] + list(PROPS)
    out = []
    for (base, metric), g in df.groupby(["base", "metric"]):
        clean = g[g.variant == "clean"]
        floor = g[g.variant.str.startswith("sub80")]
        if clean.empty:
            continue
        c = clean.iloc[0]
        sd = floor[stats].astype(float).std() if len(floor) >= 2 else pd.Series(np.nan, stats)
        for _, v in g[g.damage.notna()].iterrows():
            delta = v[stats].astype(float) - c[stats].astype(float)
            thr = 3 * sd.fillna(0)
            for s_ in stats:
                if s_.startswith("z_"):
                    thr[s_] = max(3.0, thr[s_])
            moved = delta.abs() > thr
            expected = EXPECTED.get(v.damage, {})
            hits = [p for p, sign in expected.items()
                    if moved.get(p, False) and np.sign(delta[p]) == sign]
            other = [s_ for s_ in stats if moved[s_] and s_ not in expected]
            added = sorted(set(v.recs.split(",")) - set(c.recs.split(",")) - {"none"})
            removed = sorted(set(c.recs.split(",")) - set(v.recs.split(",")) - {"none"})
            out.append(dict(base=base, metric=metric, variant=v.variant, damage=v.damage,
                            expected=f"{len(hits)}/{len(expected)}",
                            expected_hits=", ".join(hits) or "-",
                            other_moved=", ".join(other) or "-",
                            recs_added=", ".join(added) or "-",
                            recs_removed=", ".join(removed) or "-"))
    return pd.DataFrame(out)


def floor_table(df: pd.DataFrame) -> pd.DataFrame:
    stats = [f"z_{k}" for k in CALIBRATED] + list(PROPS)
    f = df[df.variant.str.startswith("sub80")]
    return f.groupby(["base", "metric"])[stats].std().T


# --------------------------------------------------------------------------------------- repair
def repair_frame(root: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for f in sorted(d.glob("*.json")):
            r = json.loads(f.read_text())
            if "error" in r:
                print(f"[error] {d.name}/{f.name}: {r['error'][:120]}")
                continue
            fam = r["family"]
            if fam == "abtt":
                nr = r["params"]["n_remove"]
                fam = f"abtt-{nr}" if nr in (1, 3) else "abtt-d/100"
            base, _, variant = r["dataset"].partition("__")
            for row in r["stages"].get("test", []):
                rows.append(dict(base=base, variant=variant or "clean", family=fam,
                                 metric=row["metric"], value=row["value"]))
    return pd.DataFrame(rows)


def repair_tables(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    out, summary = [], []
    for base, g in df.groupby("base"):
        acc = g[g.metric == "knn_accuracy"].pivot_table(index="variant", columns="family",
                                                        values="value")
        clean_raw = acc.loc["clean", "raw"] if "clean" in acc.index else np.nan
        out += [f"### {base}: cosine kNN accuracy (clean raw {clean_raw:.3f})", "",
                md(acc, ".3f"), ""]
        for variant, row in acc.iterrows():
            best = row.drop("raw", errors="ignore").idxmax()
            summary.append(dict(base=base, variant=variant, raw=row.get("raw"),
                                best_family=best, best=row[best],
                                recovered=bool(row[best] >= clean_raw - 0.01)))
    return out, pd.DataFrame(summary)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default=str(HERE / "results" / "e5"))
    args = p.parse_args(argv)
    root = Path(args.results)
    lines = ["# E5: controlled damage of a good embedding", "",
             "Generated by `benchmarks/report_e5.py`; per-run files stay on Occidata.", ""]
    if (root / "_detect").exists():
        df = detect_frame(root)
        df.to_csv(root / "summary_detect.csv", index=False)
        det = detection(df)
        det.to_csv(root / "summary_detection.csv", index=False)
        lines += ["## Detection", "",
                  "`expected` counts the expected signals (plan, WP 4.5) that moved the expected "
                  "way beyond the noise floor; `other_moved` lists the statistics that moved "
                  "beyond it without being expected.", ""]
        for metric, g in det.groupby("metric"):
            lines += [f"### Metric: {metric}", "",
                      md(g.drop(columns=["metric"]).set_index(["base", "variant"])), ""]
        lines += ["### Noise floor: sd over 5 random 80% row subsets of the clean set", "",
                  md(floor_table(df), ".3g"), ""]
    rep = repair_frame(root) if root.exists() else pd.DataFrame()
    if len(rep):
        rep.to_csv(root / "summary_repair.csv", index=False)
        part, summary = repair_tables(rep)
        summary.to_csv(root / "summary_recovery.csv", index=False)
        lines += ["## Repair", "", "Recovered = best baseline within 1 point of clean raw "
                  "(cosine kNN accuracy).", "", md(summary.set_index(["base", "variant"]), ".3f"),
                  ""] + part
        for m in ACC[1:]:
            t = rep[rep.metric == m].pivot_table(index=["base", "variant"], columns="family",
                                                 values="value")
            lines += [f"### {m}", "", md(t, ".3f"), ""]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {root / 'REPORT.md'}")


if __name__ == "__main__":
    main()
