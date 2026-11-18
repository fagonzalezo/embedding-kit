"""Data plumbing for experiment E4: diagnostics and intervention gains, one row per case.

    python benchmarks/e4_data.py [--results benchmarks/results]

Two tables, joined later on ``dataset`` (E4 itself, the regression of gains on diagnostics, is
not computed here):

* ``diagnostics_frame(root, experiment="e5")``: one row per (dataset, metric) from the
  detection reports (``<root>/<experiment>/_detect/*.json``, E5 or E4): every scalar of
  ``report.to_dataframe()`` (columns named ``Section/name``), every calibration z
  (``z_<stat>``), the damage descriptors and the git commit. Metric is the report's neighbour
  metric (cosine or euclidean). The pre-fix ``_detect_run1`` directory is ignored.
* ``gains_frame(root, experiment)`` for ``"e1"``, ``"e4"`` or ``"e5"``: one row per (dataset,
  method, task metric) with the test mean, the test mean of ``raw``, the gain (method minus
  raw), a paired bootstrap interval of the gain over test points
  (``embedkit.evaluation.compare``; NaN when the metric has no per-point outcomes), the method
  family and, for E1 and E4, the validation mean. No
  selection is done here: selecting a configuration within a family on validation is left to
  the analysis.

Gains are in the units of the metric and have no sign convention: for ``k_skewness`` a negative
gain is an improvement, for accuracies a positive one.

Running the module prints a coverage summary (datasets x families, detection reports, failed or
missing files).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from report import load_results  # noqa: E402

from embedkit.evaluation import compare  # noqa: E402

DAMAGE = re.compile(r"^dmg_([a-z]+)(\d+(?:\.\d+)?)$")
METRICS = ("cosine", "euclidean")


def split_dataset(name: str) -> tuple[str, str, str | None, float | None]:
    """``base__dmg_aniso100`` -> (base, "dmg_aniso100", "aniso", 100.0); a clean set -> "clean"."""
    base, _, variant = name.partition("__")
    m = DAMAGE.match(variant)
    if m:
        return base, variant, m.group(1), float(m.group(2))
    return base, variant or "clean", None, None


# ------------------------------------------------------------------------------- diagnostics
def _detect_records(root: Path, experiment: str = "e5"):
    d = Path(root) / experiment / "_detect"
    files = sorted(d.glob("*.json")) if d.exists() else []
    for f in files:
        yield f, json.loads(f.read_text())


def diagnostics_frame(root: str | Path, experiment: str = "e5") -> pd.DataFrame:
    """One row per (dataset, metric) report of ``<root>/<experiment>/_detect`` (E5 or E4);
    ``root`` is ``benchmarks/results``."""
    rows = []
    for f, r in _detect_records(Path(root), experiment):
        if "error" in r:
            continue
        base, variant, damage, level = split_dataset(r["dataset"])
        row = dict(dataset=r["dataset"], base=r["base"], metric=r["metric"], variant=variant,
                   damage=r.get("damage") or damage, damage_level=r.get("damage_level", level),
                   n=r["shape"][0], d=r["shape"][1],
                   git_commit=(r.get("git") or {}).get("commit"),
                   git_dirty=(r.get("git") or {}).get("dirty"),
                   embedkit_version=r.get("embedkit_version"), created=r.get("created"),
                   suggested_k=r.get("suggested_k"),
                   suggested_target_dim=r.get("suggested_target_dim"),
                   n_outside_null=r.get("n_outside_null"), file=f.name)
        row.update(r["scalars"])
        for k, c in r["calibration"].items():
            row[f"z_{k}"] = c["z"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------- gains
def _gains_one(dataset: str, stages: dict, meta: dict) -> pd.DataFrame:
    test = stages.get("test")
    if test is None or "raw" not in set(test["method"]):
        return pd.DataFrame()
    cmp = compare(test, reference="raw")
    cmp = cmp.rename(columns={"mean": "test_mean", "sd": "test_sd", "diff": "gain",
                              "diff_lo": "gain_lo", "diff_hi": "gain_hi"})
    raw = cmp[cmp.method == "raw"].set_index("metric")["test_mean"]
    cmp["raw_test_mean"] = cmp["metric"].map(raw)
    base, variant, damage, level = split_dataset(dataset)
    cmp.insert(0, "dataset", dataset)
    cmp.insert(1, "base", base)
    cmp.insert(2, "variant", variant)
    cmp.insert(3, "damage", damage)
    cmp.insert(4, "damage_level", level)
    cmp["family"] = cmp["method"].map(lambda m: meta[m]["family"])
    cmp["params"] = cmp["method"].map(lambda m: json.dumps(meta[m]["params"], sort_keys=True))
    val = stages.get("val")
    if val is not None:
        v = val.groupby(["method", "metric"])["value"].mean().rename("val_mean").reset_index()
        cmp = cmp.merge(v, on=["method", "metric"], how="left")
        rv = v[v.method == "raw"].set_index("metric")["val_mean"]
        cmp["raw_val_mean"] = cmp["metric"].map(rv)
    return cmp.rename(columns={"dim": "target_dim"})


def _result_dirs(root: Path, experiment: str) -> list[Path]:
    d = Path(root) / experiment
    return sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("_")) \
        if d.exists() else []


def gains_frame(root: str | Path, experiment: str) -> pd.DataFrame:
    """One row per (dataset, method, task metric) for ``experiment`` in {"e1", "e4", "e5"}."""
    if experiment not in ("e1", "e4", "e5"):
        raise ValueError("experiment must be 'e1', 'e4' or 'e5'")
    parts = []
    for d in _result_dirs(Path(root), experiment):
        stages, meta, _ = load_results(d.parent, d.name)
        if stages:
            parts.append(_gains_one(d.name, stages, meta))
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    for c in ("val_mean", "raw_val_mean"):
        if c in out and experiment == "e5":
            out = out.drop(columns=c)
    return out


# ------------------------------------------------------------------------------------ coverage
def failed_files(root: Path, experiment: str) -> list[str]:
    bad = []
    for d in _result_dirs(root, experiment):
        for f in sorted(d.glob("*.json")):
            r = json.loads(f.read_text())
            if "error" in r or r.get("errors"):
                bad.append(f"{experiment}/{d.name}/{f.name}: "
                           f"{str(r.get('error') or r.get('errors'))[:100]}")
    return bad


def coverage(root: str | Path) -> str:
    root = Path(root)
    lines: list[str] = []
    e1, e5 = gains_frame(root, "e1"), gains_frame(root, "e5")
    diag = diagnostics_frame(root)
    for name, g in (("E1", e1), ("E5", e5)):
        lines += [f"== {name}: {g.dataset.nunique()} datasets, {len(g)} rows, "
                  f"{g.method.nunique()} methods, {g.family.nunique()} families", ""]
        if g.empty:
            continue
        piv = g.drop_duplicates(["dataset", "method"]).groupby(["dataset", "family"]).size()
        lines.append(piv.unstack("family").fillna(0).astype(int).to_string())
        mt = g.groupby("dataset")["metric"].agg(lambda s: ", ".join(sorted(set(s))))
        sets = mt.groupby(mt).apply(lambda s: f"{len(s)} datasets, e.g. {s.index[0]}")
        lines += ["", "task metrics stored (distinct sets):"]
        lines += [f"  [{k}] {v}" for k, v in sets.items()] + [""]
        ci = g[g.method != "raw"].groupby(["metric", "ci_method"]).size()
        lines += ["rows by metric and interval kind (non-raw):", ci.to_string(), ""]
    lines += ["== Detection reports (e5/_detect)", ""]
    if diag.empty:
        lines.append("none")
    else:
        d = diag.copy()
        d["kind"] = np.where(d.variant == "clean", "clean",
                             np.where(d.variant.str.startswith("sub80"), "sub80", "damaged"))
        tab = d.groupby(["base", "metric", "kind"]).size().unstack(["metric", "kind"])
        lines.append(tab.fillna(0).astype(int).to_string())
    for name, g in (("E1", e1), ("E5", e5)):
        bases = set(g.base) if len(g) else set()
        have = set(diag.loc[diag.variant == "clean", "base"]) if len(diag) else set()
        lines += ["", f"{name} bases with a clean detection report: {sorted(bases & have)}",
                  f"{name} bases with NO detection report: {sorted(bases - have)}"]
        if len(g) and len(diag):
            diag_sets = set(diag.dataset)
            miss = sorted(set(g.dataset) - diag_sets)
            lines.append(f"{name} datasets without a diagnostics row: {len(miss)}"
                         + (f" ({', '.join(miss[:6])}{', ...' if len(miss) > 6 else ''})"
                            if miss else ""))
            both = set(diag.loc[diag.metric == "cosine", "dataset"]) & set(
                diag.loc[diag.metric == "euclidean", "dataset"])
            lines.append(f"{name} datasets with both metrics: "
                         f"{len(set(g.dataset) & both)} of {g.dataset.nunique()}")
    lines += ["", "== Detection datasets with a single metric only"]
    if len(diag):
        c = diag.groupby("dataset")["metric"].nunique()
        lines.append(", ".join(c[c < 2].index) or "none")
        lines.append("")
        lines.append("== E5 repair datasets without a detection report")
        if len(e5):
            lines.append(", ".join(sorted(set(e5.dataset) - set(diag.dataset))) or "none")
    errs = failed_files(root, "e1") + failed_files(root, "e5")
    errs += [f"e5/_detect/{f.name}: {r['error'][:100]}" for f, r in _detect_records(root)
             if "error" in r]
    lines += ["", f"== Failed files: {len(errs)}"] + errs
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default=str(HERE / "results"))
    p.add_argument("--out", default=None, help="Also write the coverage summary here")
    args = p.parse_args(argv)
    text = coverage(args.results)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
