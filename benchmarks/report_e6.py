"""E6 report: does a refiner change beat the previous default? (phase 5 of the plan)

    python benchmarks/report_e6.py --reference e6_linear --out benchmarks/results/e6/REPORT.md

Every family (``e6_<arm>``) is reduced to its configuration best on validation (auto d or a
fixed d), as in E1. Each candidate family is then compared, per dataset, with:

- the **reference** family (the previous default): paired bootstrap over test points of the
  primary metric. The phase 5 rule accepts a change when it wins (difference above 0, interval
  excluding 0) on at least half the datasets.
- the **best training-free baseline at the same d** (PCA or PCA-whitening, best on validation):
  the phase 3 gate, repeated for the new arm.

Nested prefixes (WP 5.3), when ``matryoshka-d<k>`` methods exist: every prefix of the one nested
model against ``linear-d<k>`` trained at that d, at each fixed d with no selection. The WP 5.3 rule
is non-inferiority: accept unless the prefix loses at some d on more than one dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from report import load_results  # noqa: E402
from report_gate import _fmt, _stats  # noqa: E402

from embedkit.evaluation import compare  # noqa: E402

BASELINES = ("pca", "pca_whiten")


def paired(test, metric, method, reference):
    t = test[(test.metric == metric) & test.method.isin([method, reference])]
    r = compare(t, reference=reference).set_index("method").loc[method]
    lo, hi = r["diff_lo"], r["diff_hi"]
    verdict = ("win" if np.isfinite(lo) and lo > 0 else "loss" if np.isfinite(hi) and hi < 0
               else "tie" if np.isfinite(lo) else "n/a")
    ci = f" [{lo:+.3f}, {hi:+.3f}]" if np.isfinite(lo) else ""
    return verdict, f"{r['diff']:+.3f}{ci}"


def analyse(exp_dir: Path, dataset: str, reference: str):
    stages, meta, failed = load_results(exp_dir, dataset)
    val, test = stages["val"], stages["test"]
    primary = "knn_accuracy" if "knn_accuracy" in set(val.metric) else "sts_spearman"
    v = val[val.metric == primary].groupby("method")["value"].mean()
    dims = test.groupby("method")["dim"].first()
    fam = {m: meta[m]["family"] for m in v.index}
    families = sorted({f for f in fam.values() if f.startswith("e6_")})
    selected = {f: max((m for m in v.index if fam[m] == f), key=lambda m: v[m]) for f in families}
    rows = {}
    for f, m in selected.items():
        row = dict(method=m, d=int(dims[m]), value=_fmt(*_stats(test, m, primary)))
        if f != reference and reference in selected:
            row["vs_ref"] = paired(test, primary, m, selected[reference])
        base = [b for b in v.index if fam[b] in BASELINES and dims[b] == dims[m]]
        if base:
            b = max(base, key=lambda x: v[x])
            row["baseline"] = f"{b}: {_fmt(*_stats(test, b, primary))}"
            row["vs_base"] = paired(test, primary, m, b)
        rows[f] = row
    raw = _fmt(*_stats(test, "raw", primary)) if "raw" in v.index else "n/a"
    prefixes = []
    for m in sorted(v.index, key=lambda m: dims[m]):
        own = m.replace("matryoshka-", "linear-")
        if m.startswith("matryoshka-d") and own in v.index:
            prefixes.append(dict(d=int(dims[m]), prefix=_fmt(*_stats(test, m, primary)),
                                 own=_fmt(*_stats(test, own, primary)),
                                 vs=paired(test, primary, m, own)))
    return dict(primary=primary, rows=rows, raw=raw, failed=failed, prefixes=prefixes)


def render(results, reference) -> str:
    out = ["# Experiment e6", "", __doc__.split("\n\n", 2)[2].strip(), ""]
    families = sorted({f for r in results.values() for f in r["rows"]})
    out += [f"## Summary against `{reference}` (wins / ties / losses over datasets)", "",
            "| Family | vs reference | vs best baseline at the same d | accepted |",
            "|---|---|---|---|"]
    n = len(results)
    for f in families:
        def count(key):
            vs = [r["rows"][f][key][0] for r in results.values() if key in r["rows"].get(f, {})]
            return vs, f"{vs.count('win')} / {vs.count('tie')} / {vs.count('loss')}" + (
                f" ({vs.count('n/a')} n/a)" if "n/a" in vs else "")
        if f == reference:
            out.append(f"| {f} | (reference) | {count('vs_base')[1]} | |")
            continue
        wins, text = count("vs_ref")
        accepted = "yes" if wins.count("win") >= n / 2 else "no"
        out.append(f"| {f} | {text} | {count('vs_base')[1]} | {accepted} |")
    out += ["", f"A family is accepted when it wins against `{reference}` on at least half of the "
            f"{n} datasets.", ""]
    if any(r["prefixes"] for r in results.values()):
        out += ["## Nested prefixes against a model trained at each d (WP 5.3)", "",
                "| Dataset | d | prefix | trained at d | prefix - trained [95% CI] |",
                "|---|---|---|---|---|"]
        losing = set()
        for name, r in results.items():
            for x in r["prefixes"]:
                out.append(f"| {name} | {x['d']} | {x['prefix']} | {x['own']} | "
                           f"{x['vs'][1]} {x['vs'][0]} |")
                if x["vs"][0] == "loss":
                    losing.add(name)
        accepted = "yes" if len(losing) <= 1 else "no"
        out += ["", f"Datasets with a loss at some d: {len(losing)} "
                f"({', '.join(sorted(losing)) or 'none'}). Accepted (at most one): {accepted}.",
                ""]
    for name, r in results.items():
        out += [f"## {name} ({r['primary']}; raw {r['raw']})", "",
                "| Family | selected | d | test | vs reference [95% CI] | best baseline at d | "
                "vs baseline [95% CI] |", "|---|---|---|---|---|---|---|"]
        for f, row in r["rows"].items():
            vr = "reference" if f == reference else " ".join(reversed(row.get("vs_ref", ("", ""))))
            vb = " ".join(reversed(row["vs_base"])) if "vs_base" in row else ""
            out.append(f"| {f} | {row['method']} | {row['d']} | {row['value']} | {vr} | "
                       f"{row.get('baseline', 'none')} | {vb} |")
        if r["failed"]:
            out.append("")
            out.append("Failed: " + ", ".join(f"{m} ({e[:80]})" for m, e in r["failed"]))
        out.append("")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default=str(HERE / "results"))
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--reference", default="e6_linear", help="Family of the previous default")
    p.add_argument("--out", default=None, help="Write Markdown here instead of stdout")
    args = p.parse_args(argv)
    exp_dir = Path(args.results) / "e6"
    names = args.datasets or sorted(q.name for q in exp_dir.iterdir() if q.is_dir())
    doc = render({n: analyse(exp_dir, n, args.reference) for n in names}, args.reference)
    if args.out:
        Path(args.out).write_text(doc)
    else:
        sys.stdout.write(doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
