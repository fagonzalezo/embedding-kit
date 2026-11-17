"""Experiment E4 analysis (WP 7.1): does a reading of the report predict which fix helps?

    python benchmarks/report_e4.py [--results benchmarks/results] [--out benchmarks/results/e4]

Implements the analysis fixed in ``docs/IMPLEMENTATION_PLAN.md`` ("WP 7.1 design and criteria,
fixed before the runs") on the natural sets of E4 (``<results>/e4``): the reports
(``e4/_detect``, ``e4.py``) and the training-free interventions (``run.py --experiment e4``).

For each rule (signal, intervention, outcome):

* the intervention's gain on a set is the test gain over ``raw`` of its validation-selected
  member (``e4_data.gains_frame``), with the paired bootstrap interval over test points;
* condition 1: Spearman correlation between signal and gain over the sets, with a 95% interval
  from ``N_BOOT`` bootstrap resamples of *source* data sets (all encoders of a source together),
  excluding 0;
* condition 2: where the rule fires, the gain is significantly positive (interval above 0) on at
  least half of the sets.

A rule is kept, for an intervention and a primary outcome, when both hold. The cosine report
goes with cosine kNN, the Euclidean report with Euclidean kNN and the probe. Properties (H1-H4)
fire above the median of the signal over the sets. The E5 damaged sets are a check (reported,
not decisive). Writes ``REPORT.md`` and ``summary_*.csv`` to ``--out``.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e4_data  # noqa: E402

N_BOOT = 2_000
SEED = 0
Z = 3.0
DEFAULT_SHARE = 0.75  # an intervention significantly better than raw on this share of sets
OUTCOMES = ("knn_accuracy", "knn_accuracy_euclidean", "linear_probe_c1")
OUTCOME_REPORT = {"knn_accuracy": "cosine", "knn_accuracy_euclidean": "euclidean",
                  "linear_probe_c1": "euclidean"}
OUTCOME_LABEL = {"knn_accuracy": "cosine kNN", "knn_accuracy_euclidean": "Euclidean kNN",
                 "linear_probe_c1": "probe (C = 1)"}

# Intervention groups: a predicate on the gains rows; the member is selected on validation.
INTERVENTIONS = {
    "hubness": lambda g: g.family.str.startswith("hub_"),
    "pca": lambda g: g.family == "pca",
    "pca_auto": lambda g: g.family == "pca_auto",
    "pca_whiten": lambda g: g.family == "pca_whiten",
    "zca": lambda g: g.family == "zca",
    "center_l2": lambda g: g.family == "center_l2",
    "abtt-1": lambda g: g.method == "abtt-1",
    "abtt": lambda g: g.family == "abtt",
    "standardize": lambda g: g.family == "standardize",
}
# The same groups among E5's fixed-setting methods (no validation stage there).
E5_MEMBERS = {
    "hubness": ("raw+mp", "raw+csls"), "pca": ("pca-128",), "pca_whiten": ("pca_whiten-128",),
    "zca": ("zca",), "center_l2": ("center_l2",), "abtt-1": ("abtt-1",),
    "abtt": ("abtt-1", "abtt-3"), "standardize": ("standardize",),
}


@dataclass(frozen=True)
class Rule:
    key: str
    name: str
    signal: str  # description
    fire: str  # description of the firing point
    interventions: tuple
    primary: tuple  # primary outcomes
    report: str | None = None  # force one report (R2: Euclidean); None = paired with outcome
    threshold: float | None = None  # firing point on the signal; None = median over the sets

    def values(self, diag: pd.DataFrame) -> pd.Series:
        if self.key == "R1":
            return diag["z_k_skewness"]
        if self.key == "R2":
            return diag["z_antihub_ratio"]
        if self.key == "R3":
            return -diag["z_distance_cv"]
        if self.key == "R4":
            return -diag["z_neighbor_consistency"]
        if self.key == "R5":
            return -(diag["ID/TwoNN"] / diag["d"])
        col = {"H1": "Geometry/mean_vector_ratio", "H2": "Geometry/top_3_variance_share",
               "H3": "Geometry/spectrum_alpha", "H4": "Geometry/top_1_coordinate_share"}
        return diag[col[self.key]]


KNN = ("knn_accuracy", "knn_accuracy_euclidean")
RULES = (
    Rule("R1", "hubness (report)", "z_k_skewness", "> 3", ("hubness",), KNN, threshold=Z),
    Rule("R2", "antihubs (report)", "z_antihub_ratio, Euclidean report", "> 3", ("hubness",),
         ("knn_accuracy_euclidean",), report="euclidean", threshold=Z),
    Rule("R3", "concentration (report)", "-z_distance_cv", "> 3", ("pca", "hubness"), KNN,
         threshold=Z),
    Rule("R4", "neighbour consistency (report)", "-z_neighbor_consistency", "> 3", ("pca",),
         KNN, threshold=Z),
    Rule("R5", "ID/D (report)", "-(TwoNN / D)", "ID/D < 0.3", ("pca_auto", "pca"), KNN,
         threshold=-0.3),
    Rule("H1", "dominant mean (review 4.4)", "mean_vector_ratio", "> median",
         ("center_l2", "abtt-1"), ("knn_accuracy",)),
    Rule("H2", "dominant top PCs (review 4.4)", "top_3_variance_share", "> median", ("abtt",),
         KNN),
    Rule("H3", "anisotropy (review 4.4)", "spectrum_alpha", "> median", ("zca", "pca_whiten"),
         KNN),
    Rule("H4", "rogue coordinates (WP 4.3)", "top_1_coordinate_share", "> median",
         ("standardize",), KNN),
)


def source_of(dataset: str) -> str:
    """``cifar100_dinov2`` -> ``cifar100``: the source data set, the bootstrap cluster."""
    return dataset.rsplit("_", 1)[0] if "_" in dataset else dataset


# ----------------------------------------------------------------------------------- gains
def selected_gains(gains: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, intervention, outcome): the validation-selected member's test gain."""
    rows = []
    g = gains[gains.metric.isin(OUTCOMES) & (gains.method != "raw")]
    for name, pred in INTERVENTIONS.items():
        sub = g[pred(g)]
        for (ds, metric), grp in sub.groupby(["dataset", "metric"]):
            val = grp["val_mean"] if "val_mean" in grp else pd.Series(np.nan, index=grp.index)
            best = grp.loc[val.idxmax()] if val.notna().any() else grp.iloc[0]
            rows.append(dict(dataset=ds, source=source_of(ds), intervention=name, metric=metric,
                             method=best.method, n_members=len(grp), gain=best.gain,
                             gain_lo=best.gain_lo, gain_hi=best.gain_hi,
                             test_mean=best.test_mean, raw_test_mean=best.raw_test_mean,
                             significant_up=bool(best.gain_lo > 0),
                             significant_down=bool(best.gain_hi < 0)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------- tests
def cluster_spearman(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                     n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """Spearman rho and a 95% percentile interval from resampling whole groups."""
    rho = spearmanr(x, y).statistic if len(x) > 2 else np.nan
    uniq = np.unique(groups)
    idx = {u: np.flatnonzero(groups == u) for u in uniq}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        take = np.concatenate([idx[u] for u in rng.choice(uniq, size=len(uniq))])
        xs, ys = x[take], y[take]
        if len(np.unique(xs)) > 1 and len(np.unique(ys)) > 1:
            boots.append(spearmanr(xs, ys).statistic)
    if len(boots) < n_boot / 2:
        return float(rho), np.nan, np.nan
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(rho), float(lo), float(hi)


def evaluate_rules(diag: pd.DataFrame, sel: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    """One row per (rule, intervention, outcome) with both conditions and the decision."""
    rows = []
    for rule in RULES:
        for interv in rule.interventions:
            for metric in OUTCOMES:
                report = rule.report or OUTCOME_REPORT[metric]
                d = diag[diag.metric == report].set_index("dataset")
                s = sel[(sel.intervention == interv) & (sel.metric == metric)].set_index("dataset")
                common = sorted(set(d.index) & set(s.index))
                if not common:
                    continue
                x = rule.values(d.loc[common]).to_numpy(dtype=float)
                y = s.loc[common, "gain"].to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(y)
                names = np.asarray(common)[ok]
                x, y = x[ok], y[ok]
                groups = np.asarray([source_of(n) for n in names])
                rho, lo, hi = cluster_spearman(x, y, groups, n_boot=n_boot)
                thr = rule.threshold if rule.threshold is not None else float(np.median(x))
                fires = x > thr
                sig = s.loc[names, "significant_up"].to_numpy(dtype=bool)
                n_fire, n_sig = int(fires.sum()), int((fires & sig).sum())
                cond1 = bool(np.isfinite(lo) and lo > 0)
                cond2 = bool(n_fire > 0 and n_sig >= math.ceil(n_fire / 2))
                primary = metric in rule.primary
                rows.append(dict(
                    rule=rule.key, name=rule.name, signal=rule.signal, fires_at=rule.fire,
                    threshold=thr, intervention=interv, outcome=metric, report=report,
                    primary=primary, n_sets=len(x), n_sources=len(np.unique(groups)),
                    rho=rho, rho_lo=lo, rho_hi=hi, cond1=cond1, n_fire=n_fire,
                    n_fire_sig_up=n_sig,
                    n_nofire_sig_up=int((~fires & sig).sum()), n_nofire=int((~fires).sum()),
                    median_gain_fire=float(np.median(y[fires])) if n_fire else np.nan,
                    median_gain_nofire=float(np.median(y[~fires])) if (~fires).any() else np.nan,
                    cond2=cond2, kept=bool(primary and cond1 and cond2),
                ))
    return pd.DataFrame(rows)


def candidate_defaults(sel: pd.DataFrame) -> pd.DataFrame:
    """Interventions significantly better than raw on most sets, whatever the signal."""
    t = (sel.groupby(["intervention", "metric"])
         .agg(n_sets=("gain", "size"), n_up=("significant_up", "sum"),
              n_down=("significant_down", "sum"), median_gain=("gain", "median"))
         .reset_index())
    t["share_up"] = t.n_up / t.n_sets
    t["candidate"] = t.share_up >= DEFAULT_SHARE
    return t


def e5_check(root: Path, rules: pd.DataFrame) -> pd.DataFrame:
    """For each kept rule: on the E5 damaged sets, by damage kind, how often it fires and the
    gain of its intervention's fixed-setting members (reported, not decisive)."""
    kept = rules[rules.kept]
    if kept.empty:
        return pd.DataFrame()
    diag = e4_data.diagnostics_frame(root, "e5")
    gains = e4_data.gains_frame(root, "e5")
    if diag.empty or gains.empty:
        return pd.DataFrame()
    diag = diag[diag.damage.notna()]
    by_key = {r.key: r for r in RULES}
    rows = []
    for _, k in kept.iterrows():
        rule = by_key[k.rule]
        d = diag[diag.metric == k.report].set_index("dataset")
        x = rule.values(d)
        fires = x > k.threshold
        members = E5_MEMBERS.get(k.intervention, ())
        g = gains[gains.method.isin(members) & (gains.metric == k.outcome)]
        for kind, idx in d.groupby("damage").groups.items():
            gk = g[g.dataset.isin(idx)]
            rows.append(dict(rule=k.rule, intervention=k.intervention, outcome=k.outcome,
                             damage=kind, n_sets=len(idx), fire_rate=float(fires[idx].mean()),
                             members=", ".join(members),
                             median_gain=float(gk.gain.median()) if len(gk) else np.nan,
                             share_sig_up=float((gk.gain_lo > 0).mean()) if len(gk) else np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------- render
def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "(none)"
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else format(v, floatfmt))
    cols = list(out.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in out.itertuples(index=False)]
    return "\n".join(lines)


def render(diag, sel, rules, defaults, e5, n_boot: int = N_BOOT) -> str:
    n_sets = sel.dataset.nunique()
    sources = sorted({source_of(s) for s in sel.dataset})
    lines = [
        "# E4: does a reading of the report predict which fix helps?", "",
        "Generated by `benchmarks/report_e4.py`; criteria fixed before the runs in "
        "`docs/IMPLEMENTATION_PLAN.md` (WP 7.1). Per-run files stay on Occidata.", "",
        f"{n_sets} natural sets from {len(sources)} source data sets ({', '.join(sources)}); "
        f"{len(RULES)} rules on up to {len(OUTCOMES)} outcomes, no multiplicity correction "
        f"({int(rules.primary.sum())} primary (rule, intervention, outcome) tests). "
        f"Spearman intervals: {n_boot} bootstrap resamples of source data sets.", "",
        "## Decisions", "",
    ]
    prim = rules[rules.primary]
    dec = prim[["rule", "name", "intervention", "outcome", "rho", "rho_lo", "rho_hi", "cond1",
                "n_fire", "n_fire_sig_up", "cond2", "kept"]]
    lines += [_md(dec), ""]
    kept = prim[prim.kept]
    lines += ["Kept: " + (", ".join(f"{r.rule} -> {r.intervention} ({OUTCOME_LABEL[r.outcome]})"
                                    for r in kept.itertuples()) or "none"), ""]
    lines += ["## All tests (primary and secondary outcomes)", "", _md(rules.drop(
        columns=["name", "signal"])), ""]
    lines += ["## Interventions that help whatever the signal (WP 7.2 candidates)", "",
              f"`candidate`: significantly better than raw on at least {DEFAULT_SHARE:.0%} of "
              "the sets.", "", _md(defaults), ""]
    lines += ["## E5 check (damaged sets; reported, not decisive)", "", _md(e5), ""]
    keep_cols = ["dataset", "metric", "n", "d", "z_k_skewness", "z_antihub_ratio",
                 "z_distance_cv", "z_neighbor_consistency", "z_uniformity", "ID/TwoNN",
                 "Geometry/mean_vector_ratio", "Geometry/top_3_variance_share",
                 "Geometry/spectrum_alpha", "Geometry/top_1_coordinate_share"]
    dg = diag[[c for c in keep_cols if c in diag.columns]].sort_values(["metric", "dataset"])
    lines += ["## Signals per set", "", _md(dg, ".3g"), ""]
    piv = sel.pivot_table(index=["dataset"], columns=["metric", "intervention"], values="gain")
    lines += ["## Gains per set (validation-selected member, test gain over raw)", ""]
    for metric in OUTCOMES:
        if metric in piv.columns.get_level_values(0):
            lines += [f"### {OUTCOME_LABEL[metric]}", "",
                      _md(piv[metric].reset_index(), "+.3f"), ""]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default=str(HERE / "results"))
    p.add_argument("--out", default=None, help="default: <results>/e4")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    args = p.parse_args(argv)
    root = Path(args.results)
    out = Path(args.out or root / "e4")
    out.mkdir(parents=True, exist_ok=True)
    diag = e4_data.diagnostics_frame(root, "e4")
    gains = e4_data.gains_frame(root, "e4")
    if diag.empty or gains.empty:
        sys.exit(f"No E4 results under {root / 'e4'} (reports: {len(diag)}, gains: {len(gains)})")
    diag = diag[diag.dataset.isin(set(gains.dataset))]
    sel = selected_gains(gains)
    rules = evaluate_rules(diag, sel, n_boot=args.n_boot)
    defaults = candidate_defaults(sel)
    e5 = e5_check(root, rules)
    diag.to_csv(out / "summary_diagnostics.csv", index=False)
    sel.to_csv(out / "summary_gains.csv", index=False)
    rules.to_csv(out / "summary_rules.csv", index=False)
    defaults.to_csv(out / "summary_defaults.csv", index=False)
    if len(e5):
        e5.to_csv(out / "summary_e5_check.csv", index=False)
    (out / "REPORT.md").write_text(render(diag, sel, rules, defaults, e5, args.n_boot) + "\n")
    print(f"wrote {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
