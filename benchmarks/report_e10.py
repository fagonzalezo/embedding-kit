"""Experiment E10 analysis (WP 8.2 and WP 8.3): the default selection on the E4 sets

    python benchmarks/report_e10.py [--results benchmarks/results] [--out benchmarks/results/e10]

Reads ``<results>/e10/<set>/<method>.json`` (``run.py --experiment e10``, test stage) and applies
the criteria fixed in ``docs/IMPLEMENTATION_PLAN.md`` ("WP 8.2 design and criteria, fixed
before the runs"):

* WP 8.2, choice quality. ``select-cap<c>`` against the selection on all the rows (``select-all``
  where the default cap binds, otherwise ``select``) on test cosine 10-NN accuracy, a paired
  bootstrap over test points (``N_BOOT`` resamples, 95%). A (set, cap) pair fails when the
  interval lies entirely below 0. A cap passes when at most 1 of the sets it binds on fails, and
  that failure loses less than 1 point. The default cap (``select`` against ``select-all``)
  passes when none fails.
* WP 8.3, the default against the input. ``select`` against ``identity``, same test. Reported
  here; WP 8.3 sets its own criteria. Also the best candidate in hindsight (``identity``,
  ``center_l2``, ``pca_l2-k``, ``refiner``, each fitted on all the training rows) and how often
  each candidate is chosen.

Euclidean kNN and the linear probe are secondary outcomes (same comparisons, reported only).
Writes ``REPORT.md`` and ``summary_e10.csv`` to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run import decode_per_point  # noqa: E402

N_BOOT = 2000
PRIMARY = "knn_accuracy"  # cosine 10-NN, the selection's own criterion
SECONDARY = ("knn_accuracy_euclidean", "linear_probe_c1")
CAPS = (5_000, 10_000, 20_000)
CANDIDATES = ("identity", "center_l2", "pca_l2-64", "pca_l2-128", "pca_l2-256", "refiner")
MAX_FAILURES_PER_CAP = 1
MAX_FAILURE_LOSS = 0.01


def load(e10_dir: Path) -> dict:
    """``{set: {method: {"metrics": {metric: (value, per_point)}, "params": dict, ...}}}``."""
    out: dict = {}
    for path in sorted(e10_dir.glob("*/*.json")):
        rec = json.loads(path.read_text())
        if "error" in rec:
            out.setdefault(rec["dataset"], {})[rec["method"]] = dict(error=rec["error"])
            continue
        rows = rec["stages"].get("test", [])
        metrics = {}
        for r in rows:
            if r["seed"] != 0:
                continue
            metrics[r["metric"]] = (r["value"], decode_per_point(r["per_point"]))
        params = rows[0]["params"] if rows else {}
        out.setdefault(rec["dataset"], {})[rec["method"]] = dict(
            metrics=metrics, params=params, elapsed=rec.get("elapsed"),
            fit_time=rows[0].get("fit_time") if rows else None)
    return out


def paired(a: np.ndarray, b: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """Mean of ``a - b`` and its percentile interval over ``N_BOOT`` resamples of the points."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    boot = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def compare(runs: dict, a: str, b: str, metric: str):
    if a not in runs or b not in runs or "metrics" not in runs[a] or "metrics" not in runs[b]:
        return None
    va, pa = runs[a]["metrics"].get(metric, (None, None))
    vb, pb = runs[b]["metrics"].get(metric, (None, None))
    if pa is None or pb is None or len(pa) != len(pb):
        return None
    return paired(pa, pb)


def analyse(data: dict) -> pd.DataFrame:
    rows = []
    for name, runs in sorted(data.items()):
        full = "select-all" if "select-all" in runs else "select"
        pairs = [(f"select-cap{c}", full, f"cap{c}") for c in CAPS]
        pairs += [("select", "select-all", "default_cap"), ("select", "identity", "vs_identity")]
        for a, b, kind in pairs:
            for metric in (PRIMARY, *SECONDARY):
                res = compare(runs, a, b, metric)
                if res is None:
                    continue
                diff, lo, hi = res
                pa, pb = runs[a]["params"], runs[b]["params"]
                rows.append(dict(dataset=name, comparison=kind, method=a, reference=b,
                                 metric=metric, diff=diff, lo=lo, hi=hi, worse=hi < 0,
                                 better=lo > 0, selected=pa.get("selected"),
                                 reference_selected=pb.get("selected"),
                                 n_selection_rows=pa.get("n_selection_rows"),
                                 fit_time=runs[a].get("fit_time")))
    return pd.DataFrame(rows)


def oracle_table(data: dict) -> pd.DataFrame:
    rows = []
    for name, runs in sorted(data.items()):
        sel = runs.get("select", {})
        if "metrics" not in sel:
            continue
        cands = {c: runs[c]["metrics"][PRIMARY][0] for c in CANDIDATES
                 if c in runs and "metrics" in runs[c] and PRIMARY in runs[c]["metrics"]}
        if not cands:
            continue
        best = max(cands, key=cands.get)
        selected = sel["params"].get("selected")
        rows.append(dict(dataset=name, selected=selected, select=sel["metrics"][PRIMARY][0],
                         identity=cands.get("identity"), best=best, best_value=cands[best],
                         regret=cands[best] - sel["metrics"][PRIMARY][0],
                         select_fit_time=sel.get("fit_time"),
                         refiner_fit_time=runs.get("refiner", {}).get("fit_time")))
    return pd.DataFrame(rows)


def verdicts(df: pd.DataFrame) -> list[str]:
    out = []
    prim = df[df.metric == PRIMARY]
    for c in CAPS:
        sub = prim[prim.comparison == f"cap{c}"]
        if sub.empty:
            out.append(f"- cap {c:,}: binds on no set")
            continue
        fails = sub[sub.worse]
        ok = len(fails) <= MAX_FAILURES_PER_CAP and (fails["diff"] > -MAX_FAILURE_LOSS).all()
        same = int((sub.selected == sub.reference_selected).sum())
        listed = ", ".join(f"{r.dataset} {100 * r.diff:+.1f}" for r in fails.itertuples())
        out.append(f"- cap {c:,}: binds on {len(sub)} sets; significantly worse on {len(fails)}"
                   + (f" ({listed})" if len(fails) else "")
                   + f"; same choice on {same} of {len(sub)}. **{'pass' if ok else 'FAIL'}**")
    sub = prim[prim.comparison == "default_cap"]
    if sub.empty:
        out.append("- default cap 50,000: binds on no set (no E4 set has more than 50,000 "
                   "training rows)")
    else:
        fails = sub[sub.worse]
        same = int((sub.selected == sub.reference_selected).sum())
        out.append(f"- default cap 50,000: binds on {len(sub)} sets; significantly worse on "
                   f"{len(fails)}; same choice on {same} of {len(sub)}. "
                   f"**{'pass' if fails.empty else 'FAIL'}**")
    return out


def fmt_rows(df: pd.DataFrame, kind: str, metric: str) -> list[str]:
    sub = df[(df.comparison == kind) & (df.metric == metric)]
    if sub.empty:
        return ["(none)"]
    lines = ["| set | choice | reference choice | rows used | diff (points) | 95% interval |",
             "|---|---|---|---|---|---|"]
    for r in sub.itertuples():
        flag = " **worse**" if r.worse else (" better" if r.better else "")
        ref = r.reference_selected if isinstance(r.reference_selected, str) else r.reference
        lines.append(f"| {r.dataset} | {r.selected} | {ref} | "
                     f"{r.n_selection_rows} | {100 * r.diff:+.2f}{flag} | "
                     f"[{100 * r.lo:+.2f}, {100 * r.hi:+.2f}] |")
    return lines


def render(df: pd.DataFrame, oracle: pd.DataFrame, data: dict) -> str:
    errors = [(s, m, r["error"]) for s, runs in data.items() for m, r in runs.items()
              if "error" in r]
    out = ["# E10: the default selection on the E4 sets (WP 8.2, WP 8.3)", "",
           f"{len(data)} sets. Test stage, one seed. Primary outcome: test cosine 10-NN accuracy; "
           "intervals: paired bootstrap over test points (2,000 resamples, 95%). Criteria: "
           "`docs/IMPLEMENTATION_PLAN.md`, \"WP 8.2 design and criteria\".", "",
           "## WP 8.2 verdicts (primary outcome)", "", *verdicts(df), ""]
    for c in CAPS:
        out += [f"### Cap {c:,} against selection on all rows", "",
                *fmt_rows(df, f"cap{c}", PRIMARY), ""]
    out += ["### Default cap 50,000 against selection on all rows", "",
            *fmt_rows(df, "default_cap", PRIMARY), ""]
    out += ["## Secondary outcomes (count of sets significantly worse / better)", "",
            "| comparison | metric | sets | worse | better |", "|---|---|---|---|---|"]
    for (kind, metric), g in df.groupby(["comparison", "metric"]):
        if metric == PRIMARY:
            continue
        out.append(f"| {kind} | {metric} | {len(g)} | {int(g.worse.sum())} | "
                   f"{int(g.better.sum())} |")
    out += ["", "## WP 8.3 preview: the default against the input and the best candidate", ""]
    out += fmt_rows(df, "vs_identity", PRIMARY)
    if not oracle.empty:
        counts = Counter(oracle.selected)
        out += ["", "Chosen candidates: " + ", ".join(f"`{k}` {v}" for k, v in
                                                       counts.most_common()), "",
                "| set | chosen | select | identity | best in hindsight | best | regret "
                "(points) |", "|---|---|---|---|---|---|---|"]
        for r in oracle.itertuples():
            out.append(f"| {r.dataset} | {r.selected} | {r.select:.4f} | {r.identity:.4f} | "
                       f"{r.best} | {r.best_value:.4f} | {100 * r.regret:.2f} |")
    if errors:
        out += ["", "## Errors", ""] + [f"- {s} / {m}: {e[:200]}" for s, m, e in errors]
    return "\n".join(out) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default=str(HERE / "results"))
    p.add_argument("--out", default=None, help="default: <results>/e10")
    args = p.parse_args(argv)
    e10_dir = Path(args.results) / "e10"
    data = load(e10_dir)
    if not data:
        sys.exit(f"No E10 results under {e10_dir}")
    df, oracle = analyse(data), oracle_table(data)
    out = Path(args.out or e10_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary_e10.csv", index=False)
    oracle.to_csv(out / "summary_e10_oracle.csv", index=False)
    (out / "REPORT.md").write_text(render(df, oracle, data))
    print(f"wrote {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
