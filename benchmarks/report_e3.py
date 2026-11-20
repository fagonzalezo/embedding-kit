"""Summarise experiment E3 (``benchmarks/e3.py``) into CSV tables and a Markdown report.

    python benchmarks/report_e3.py [--results benchmarks/results/e3]

Reads ``<results>/{id,gauss,subsample}/*.json`` (whatever parts exist) and writes, next to
them, ``summary_id.csv``, ``summary_gauss.csv``, ``summary_subsample.csv`` and ``REPORT.md``.
The analysis follows the criteria fixed in ``docs/dev/IMPLEMENTATION_PLAN.md`` (WP 4.5) before the
runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CALIBRATED = ("k_skewness", "antihub_ratio", "distance_cv", "neighbor_consistency",
              "uniformity")
REC_TYPES = {
    "More hubness than the null": "hubness",
    "Distances are more concentrated": "concentration",
    "The kNN graph is less stable": "consistency",
    "More antihubs than the null": "antihub",
    "Low intrinsic-to-ambient": "id_low",
    "Moderate intrinsic-to-ambient": "id_moderate",
}
# Scalars whose dependence on n is reported (subsample part), as "<category>/<metric>".
TRACKED = [
    "Hubness/k_skewness", "Hubness/antihub_ratio", "Hubness/hub_ratio",
    "Geometry/distance_cv", "Geometry/relative_contrast_per_query",
    "Geometry/participation_ratio_normalized", "Geometry/isoscore", "Geometry/effective_rank",
    "Geometry/spectrum_alpha", "Geometry/top_1_variance_share",
    "Geometry/top_10_variance_share", "Geometry/mean_vector_ratio", "Geometry/mean_cosine",
    "Geometry/neighbor_consistency", "Geometry/uniformity", "Kernel/effective_rank",
    "Kernel/spectral_gap", "ID/TwoNN", "ID/MLE", "ID/lPCA", "ID/MOM",
    "ID/scale_max_deviation",
    "Summary/suggested_k", "Summary/suggested_target_dim",
]


def load(dir_: Path) -> list[dict]:
    out = []
    for f in sorted(dir_.glob("*.json")):
        r = json.loads(f.read_text())
        if "error" in r:
            print(f"[error] {f.name}: {r['error'][:120]}")
            continue
        out.append(r)
    return out


def rec_types(recs: list[str]) -> list[str]:
    types = sorted({t for r in recs for prefix, t in REC_TYPES.items() if r.startswith(prefix)})
    return types


def md(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    """A Markdown table of ``df`` with its index (no dependency on ``tabulate``)."""
    def fmt(v):
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else format(float(v), floatfmt)
        return str(v)

    idx = df.index.names if any(df.index.names) else [""]
    head = [" / ".join(str(n) for n in idx if n is not None)] + [
        " ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, row in zip(df.index, df.itertuples(index=False)):
        key = " / ".join(map(str, i)) if isinstance(i, tuple) else str(i)
        lines.append("| " + " | ".join([key] + [fmt(v) for v in row]) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------ id
def scale_deviation(r: dict) -> float:
    """``scale_max_deviation`` of a record; computed from ``scale_curve`` for records written
    before it existed (the first E3 run, which stored ``scale_plateau``)."""
    if r.get("scale_max_deviation") is not None:
        return float(r["scale_max_deviation"])
    curve = {int(k): v for k, v in (r.get("scale_curve") or {}).items()
             if v is not None and v == v}
    if len(curve) < 2:
        return float("nan")
    ref = curve[max(curve)]
    return float(max(abs(v / ref - 1.0) for v in curve.values()))


def id_frame(records) -> pd.DataFrame:
    rows = []
    for r in records:
        s = r["settings"]
        for m, est in r["estimates"].items():
            lo, hi = r["intervals"][m]
            rows.append(dict(
                manifold=s["manifold"], true_id=s["true_id"], sweep=s["sweep"], n=s["n"],
                D=s["D"], noise=s["noise"], map=s["map"], seed=s["seed"], method=m,
                estimate=est, se=r["standard_errors"][m], lo=lo, hi=hi,
                covered=(lo <= s["true_id"] <= hi) if lo == lo else np.nan,
                scale_dev=scale_deviation(r), seconds=r["seconds"][m]))
    df = pd.DataFrame(rows)
    df["rel_bias"] = (df.estimate - df.true_id) / df.true_id
    return df


def id_report(df: pd.DataFrame) -> list[str]:
    out = ["## Intrinsic dimension (part `id`)", ""]
    g = df.groupby(["manifold", "true_id", "sweep", "n", "D", "noise", "map", "method"])
    cells = g.agg(mean=("estimate", "mean"), sd=("estimate", "std"), se=("se", "mean"),
                  coverage=("covered", "mean"), rel_bias=("rel_bias", "mean"),
                  seconds=("seconds", "mean"), seeds=("seed", "nunique")).reset_index()
    cells["bias"] = cells["mean"] - cells["true_id"]

    centre = cells[cells.sweep == "centre"]
    t = centre.pivot_table(index=["manifold", "true_id"], columns="method", values="mean")
    out += ["### Centre cell (n = 5,000, D = 256, no noise, linear): mean estimate over 5 seeds",
            "", md(t.sort_index(level=1)), ""]
    t = centre.pivot_table(index=["manifold", "true_id"], columns="method", values="sd")
    out += ["Standard deviation over seeds:", "", md(t.sort_index(level=1)), ""]

    wp42 = centre[centre.manifold.isin(["sphere10", "sphere20", "cube10", "cube20"])]
    t = wp42.pivot_table(index="manifold", columns="method", values="bias")
    out += ["### WP 4.2 hand-over: bias at ID 10 and 20 (centre cell)", "", md(t), ""]

    for sweep, col in (("n", "n"), ("D", "D"), ("noise", "noise"), ("map", "map")):
        sub = cells[(cells.sweep == sweep) | (cells.sweep == "centre")]
        t = sub.pivot_table(index=col, columns="method", values="rel_bias", aggfunc="median")
        out += [f"### Relative bias against {col} (median over manifolds; other factors at the "
                "centre)", "", md(t), ""]
    sub = cells[cells.sweep.isin(["centre", "n"])]
    t = sub.pivot_table(index="true_id", columns="method", values="rel_bias", aggfunc="median")
    out += ["### Relative bias by true ID (centre and n sweep, median over cells)", "", md(t), ""]

    boot = cells[cells.se.notna() & (cells.seeds >= 3)]
    ratio = (boot.se / boot.sd).replace([np.inf, -np.inf], np.nan)
    t = boot.assign(ratio=ratio).groupby("method").agg(
        median_se_over_sd=("ratio", "median"), coverage=("coverage", "mean"),
        cells=("ratio", "size"))
    out += ["### Bootstrap: se against the sd over seeds, and coverage of the true ID", "",
            "Coverage is the share of (cell, seed) intervals containing the true ID; it mixes "
            "sampling error with bias, which the intervals do not include.", "", md(t), ""]

    plat = df.drop_duplicates(["manifold", "sweep", "n", "D", "noise", "map", "seed"])
    t = plat.groupby("noise").scale_dev.describe(percentiles=[0.1, 0.5, 0.9])[
        ["count", "10%", "50%", "90%"]]
    out += ["### TwoNN scale curve: largest relative deviation from the full sample, by noise",
            "", md(t, ".3g"), ""]
    t = df.groupby(["method", "n"]).seconds.median().unstack()
    out += ["### Median seconds per fit (bootstrap included where run)", "", md(t, ".1f"), ""]
    return out, cells


# --------------------------------------------------------------------------------------- gauss
def gauss_frame(records) -> pd.DataFrame:
    rows = []
    for r in records:
        s = r["settings"]
        row = dict(s, n_outside_null=r["n_outside_null"],
                   recs=",".join(rec_types(r["recommendations"])) or "none")
        for k, c in r["calibration"].items():
            row[f"z_{k}"] = c["z"]
            row[f"obs_{k}"] = c["observed"]
        for k in TRACKED:
            row[k] = r["scalars"].get(k)
        rows.append(row)
    return pd.DataFrame(rows)


def gauss_report(df: pd.DataFrame) -> list[str]:
    out = ["## Gaussian families (part `gauss`)", ""]
    inside = pd.DataFrame({k: (df[f"z_{k}"].abs() < 3) for k in CALIBRATED})
    t = inside.mean().to_frame("share |z| < 3").T
    out += ["### Calibration: share of cells with |z| < 3 (WP 4.1 bar: 0.95)", "",
            "All families except the `_l2` ones are drawn from the null family itself.", ""]
    null_fams = ~df.family.str.endswith("_l2")
    for by in ("family", "n", "d", "k", "metric"):
        frame = inside.assign(**{by: df[by]})
        frame = frame[null_fams] if by != "family" else frame
        out += [f"By {by}:", "", md(frame.groupby(by).mean()), ""]
    worst = inside[null_fams].assign(**df[null_fams][["family", "n", "d", "k", "metric"]])
    worst = worst.groupby(["family", "n", "d", "k", "metric"]).mean()
    worst = worst[(worst < 0.8).any(axis=1)]
    out += ["### Null-family cells where a statistic is inside +-3 in fewer than 4 of 5 seeds",
            "", md(worst) if len(worst) else "None.", ""]

    flat = df[(df.family == "flat") & (df.k == 10) & (df.metric == "euclidean")]
    cols = [c for c in TRACKED if c in flat and flat[c].notna().any()]
    t = flat.groupby(["n", "d"])[cols].mean().T
    out += ["### Null table: isotropic Gaussian, k = 10, Euclidean (mean over 5 seeds)", "",
            "How the raw statistics move with n and d on structureless data.", "", md(t, ".3g"),
            ""]
    types = sorted(set(REC_TYPES.values()))
    fa = pd.DataFrame({t_: df.recs.str.contains(t_) for t_ in types}).assign(
        family=df.family, metric=df.metric)
    out += ["### Recommendation firing rate (null families: false alarms)", "",
            md(fa.groupby(["family", "metric"]).mean()), ""]
    return out


# ----------------------------------------------------------------------------------- subsample
def subsample_frame(records) -> pd.DataFrame:
    rows = []
    for r in records:
        s = r["settings"]
        row = dict(dataset=s["dataset"], n=s["n"], seed=s["seed"],
                   recs=",".join(rec_types(r["recommendations"])) or "none")
        for k, c in r["calibration"].items():
            row[f"z_{k}"] = c["z"]
            row[f"sign_{k}"] = int(np.sign(c["z"])) if abs(c["z"]) > 3 else 0
        for k in TRACKED:
            row[k] = r["scalars"].get(k)
        rows.append(row)
    return pd.DataFrame(rows)


def reading(group: pd.DataFrame) -> dict:
    """Majority reading over the subsets of one (dataset, n)."""
    out = {}
    types = sorted(set(REC_TYPES.values()))
    for t in types:
        out[f"rec_{t}"] = bool(group.recs.str.contains(t).mean() > 0.5)
    for k in CALIBRATED:
        out[f"sign_{k}"] = int(group[f"sign_{k}"].mode().iloc[0])
    return out


def subsample_report(df: pd.DataFrame) -> list[str]:
    out = ["## Subsampling stability (part `subsample`; phase 4 exit criterion)", "",
           "Reading = majority over the 5 subsets of: each recommendation type, the sign of each "
           "calibrated z beyond +-3 (the plateau flag was dropped from the report after the "
           "first run; its replacement, `ID/scale_max_deviation`, is in the tables below). "
           "`yelp_mpnet` has exactly 20,000 "
           "training rows, so its five 20k subsets are identical (fluorescence, 21,446 rows, "
           "nearly so).", ""]
    readings = {(ds, n): reading(g) for (ds, n), g in df.groupby(["dataset", "n"])}
    rows = []
    for ds in sorted(df.dataset.unique()):
        a, b = readings.get((ds, 2000)), readings.get((ds, 20000))
        if a is None or b is None:
            continue
        diff = [k for k in a if a[k] != b[k]]
        rows.append(dict(dataset=ds, agree=not diff, differs=", ".join(diff) or "-"))
    if rows:
        out += ["### n = 2,000 against n = 20,000", "", md(pd.DataFrame(rows).set_index("dataset")),
                ""]
    t = pd.DataFrame(readings).T
    out += ["### Majority reading by dataset and n", "", md(t), ""]

    med = df.groupby(["dataset", "n"])[[c for c in TRACKED if c in df]].median()
    rel = []
    for ds in med.index.get_level_values(0).unique():
        m = med.loc[ds]
        if 20000 in m.index and 2000 in m.index:
            base = m.loc[20000].astype(float)
            rel.append(((m.loc[2000].astype(float) - base) / base.abs().replace(0, np.nan))
                       .rename(ds))
    if rel:
        t = pd.concat(rel, axis=1)
        t["median"] = t.median(axis=1)
        out += ["### Relative change from n = 20,000 to n = 2,000 (median over subsets)", "",
                md(t, ".2f"), ""]
    sd = df.groupby(["dataset", "n"])[[c for c in TRACKED if c in df]].std()
    out += ["### Spread over subsets (sd) at each n", "", md(sd.T, ".3g"), ""]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default=str(HERE / "results" / "e3"))
    args = p.parse_args(argv)
    root = Path(args.results)
    lines = ["# E3: calibration of the diagnostics on data with known answers", "",
             "Generated by `benchmarks/report_e3.py`; per-cell files stay on Occidata.", ""]
    if (root / "id").exists():
        df = id_frame(load(root / "id"))
        df.to_csv(root / "summary_id.csv", index=False)
        part, _ = id_report(df)
        lines += part
    if (root / "gauss").exists():
        df = gauss_frame(load(root / "gauss"))
        df.to_csv(root / "summary_gauss.csv", index=False)
        lines += gauss_report(df)
    if (root / "subsample").exists():
        df = subsample_frame(load(root / "subsample"))
        df.to_csv(root / "summary_subsample.csv", index=False)
        lines += subsample_report(df)
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {root / 'REPORT.md'}")


if __name__ == "__main__":
    main()
