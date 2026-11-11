"""The phase 3 decision gate on E1 results (docs/IMPLEMENTATION_PLAN.md, "Decision gate").

    python benchmarks/report_gate.py --out benchmarks/results/e1/GATE.md

For each dataset:

- **Headline:** test value of the primary metric for raw, the best training-free method, and each
  ``EmbedKit`` family, every one selected on validation as in ``report.py``.
- **Gate:** the self-supervised ``EmbedKit`` configuration best on validation (default or fixed d)
  against the training-free baseline best on validation among those with the same output
  dimension. Paired bootstrap interval over test points (``compare``). A win is a difference above
  0 whose interval excludes 0; a loss is an interval below 0. For context, the best training-free
  baseline at a lower dimension, and trustworthiness against PCA at the same d (outcome B's
  condition).
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

from embedkit.evaluation import compare  # noqa: E402

EMBEDKIT_SELF = ("embedkit", "embedkit_fixed_d")
EMBEDKIT = (*EMBEDKIT_SELF, "embedkit_supervised")
# Training-free methods with a chosen output dimension: the gate's comparison set.
REDUCING = ("pca", "pca_whiten", "random_projection", "abtt_reduce")


def _stats(test, method, metric):
    v = test[(test.method == method) & (test.metric == metric)]["value"]
    if v.empty:
        return np.nan, np.nan
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else np.nan)


def _fmt(mean, sd=np.nan):
    if not np.isfinite(mean):
        return "n/a"
    return f"{mean:.3f}" if not np.isfinite(sd) else f"{mean:.3f} ± {sd:.3f}"


def analyse(exp_dir: Path, dataset: str) -> dict:
    stages, meta, _ = load_results(exp_dir, dataset)
    val, test = stages["val"], stages["test"]
    primary = "knn_accuracy" if "knn_accuracy" in set(val.metric) else "sts_spearman"
    v = val[val.metric == primary].groupby("method")["value"].mean()
    dims = test.groupby("method")["dim"].first()
    fam = {m: meta[m]["family"] for m in v.index}

    def best(cands):
        cands = [m for m in cands if m in v.index]
        return max(cands, key=lambda m: v[m]) if cands else None

    headline = {
        "raw": "raw",
        "training-free": best([m for m in v.index if fam[m] not in EMBEDKIT]),
        "embedkit (default)": best([m for m in v.index if fam[m] == "embedkit"]),
        "embedkit (fixed d)": best([m for m in v.index if fam[m] == "embedkit_fixed_d"]),
        "embedkit supervised": best([m for m in v.index if fam[m] == "embedkit_supervised"]),
    }
    ek = best([m for m in v.index if fam[m] in EMBEDKIT_SELF])
    d = int(dims[ek])
    base = best([m for m in v.index if fam[m] in REDUCING and dims[m] == d])
    lower = best([m for m in v.index if fam[m] in REDUCING and dims[m] < d])
    pca_d = next((m for m in v.index if fam[m] == "pca" and dims[m] == d), None)
    gate = dict(ek=ek, d=d, base=base, lower=lower, pca=pca_d)
    if base is not None:
        t = test[(test.metric == primary) & test.method.isin([ek, base])]
        r = compare(t, reference=base).set_index("method").loc[ek]
        gate.update(diff=r["diff"], lo=r["diff_lo"], hi=r["diff_hi"])
        gate["verdict"] = ("win" if np.isfinite(r["diff_lo"]) and r["diff_lo"] > 0 else
                           "loss" if np.isfinite(r["diff_hi"]) and r["diff_hi"] < 0 else
                           "tie" if np.isfinite(r["diff_lo"]) else "no interval")
    return dict(primary=primary, test=test, dims=dims, headline=headline, gate=gate)


def render(results: dict[str, dict]) -> str:
    out = ["# Phase 3 decision gate (E1)", "", __doc__.split("\n\n", 2)[2].strip(), "",
           "## Headline: test value of the primary metric, each family selected on validation",
           "", "| Dataset | metric | " + " | ".join(next(iter(results.values()))["headline"])
           + " |", "|---|---|" + "---|" * 5]
    for name, r in results.items():
        cells = []
        for m in r["headline"].values():
            if m is None:
                cells.append("n/a")
                continue
            cells.append(f"{_fmt(*_stats(r['test'], m, r['primary']))} ({m}, d = {r['dims'][m]})")
        out.append(f"| {name} | {r['primary']} | " + " | ".join(cells) + " |")
    out += ["",
            "## Gate: self-supervised EmbedKit vs the best training-free baseline at the same d",
            "", "| Dataset | EmbedKit (val-selected) | baseline at same d | difference [95% CI] | "
            "verdict | best baseline at lower d | trustworthiness, EmbedKit vs PCA same d |",
            "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        g, test, p = r["gate"], r["test"], r["primary"]
        ek = f"{g['ek']}: {_fmt(*_stats(test, g['ek'], p))}"
        if g["base"] is None:
            out.append(f"| {name} | {ek} | none | | | | |")
            continue
        base = f"{g['base']}: {_fmt(*_stats(test, g['base'], p))}"
        ci = f" [{g['lo']:+.3f}, {g['hi']:+.3f}]" if np.isfinite(g["lo"]) else ""
        lower = (f"{g['lower']}: {_fmt(*_stats(test, g['lower'], p))}" if g["lower"] else "none")
        tw = (f"{_fmt(_stats(test, g['ek'], 'trustworthiness')[0])} vs "
              f"{_fmt(_stats(test, g['pca'], 'trustworthiness')[0])}" if g["pca"] else "n/a")
        out.append(f"| {name} | {ek} | {base} | {g['diff']:+.3f}{ci} | {g['verdict']} | {lower} "
                   f"| {tw} |")
    verdicts = [r["gate"].get("verdict", "none") for r in results.values()]
    out += ["", "Verdicts: " + ", ".join(f"{verdicts.count(k)} {k}" for k in
                                         ("win", "tie", "loss", "no interval") if k in verdicts)
            + f" over {len(verdicts)} datasets.", ""]
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default=str(HERE / "results"))
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--out", default=None, help="Write Markdown here instead of stdout")
    args = p.parse_args(argv)
    exp_dir = Path(args.results) / "e1"
    names = args.datasets or sorted(q.name for q in exp_dir.iterdir() if q.is_dir())
    doc = render({n: analyse(exp_dir, n) for n in names})
    if args.out:
        Path(args.out).write_text(doc)
    else:
        sys.stdout.write(doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
