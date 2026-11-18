"""E7 report: learning curves and inductive against transductive fitting (phase 5 of the plan).

    python benchmarks/report_e7.py --out benchmarks/results/e7/REPORT.md

All methods work at one d per dataset (the d that the default refiner selected in E6), so no
selection on validation is involved. Learning curves: the transform is fitted on ``n`` training
rows (``-n<n>``) or on all of them (``-all``); the kNN gallery and the probe always use the whole
transformed training split. The refiner is compared with the better of PCA and PCA-whitening at
the same ``n`` (chosen on validation), with a paired bootstrap over test points of the primary
metric. Transduction: each transform fitted on train + test rows without labels
(``-transductive``) against the same transform fitted on train rows only (``-all``).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from report import load_results  # noqa: E402
from report_e6 import paired  # noqa: E402
from report_gate import _fmt, _stats  # noqa: E402

TRANSFORMS = ("embedkit", "pca", "pca_whiten")


def _size(method: str) -> int | None:
    m = re.fullmatch(r"[a-z_]+-n(\d+)", method)
    return int(m.group(1)) if m else None


def analyse(exp_dir: Path, dataset: str) -> dict:
    stages, meta, failed = load_results(exp_dir, dataset)
    val, test = stages["val"], stages["test"]
    primary = "knn_accuracy" if "knn_accuracy" in set(val.metric) else "sts_spearman"
    v = val[val.metric == primary].groupby("method")["value"].mean()
    methods = set(test.method)
    d = int(test[test.method.str.startswith("embedkit")]["dim"].iloc[0]) if any(
        m.startswith("embedkit") for m in methods) else None
    n_all = meta.get("embedkit-all", {}).get("params", {}).get("n_fit")
    sizes = sorted({s for m in methods if (s := _size(m)) is not None})
    curve = []
    for n in [*sizes, "all"]:
        suffix = f"n{n}" if n != "all" else "all"
        row = dict(n=n if n != "all" else f"all ({n_all})" if n_all else "all")
        for t in TRANSFORMS:
            row[t] = _fmt(*_stats(test, f"{t}-{suffix}", primary))
        ek = f"embedkit-{suffix}"
        base = [b for b in (f"pca-{suffix}", f"pca_whiten-{suffix}") if b in v.index]
        if ek in methods and base:
            b = max(base, key=lambda x: v[x])
            row["vs"] = (b.split("-")[0],) + paired(test, primary, ek, b)
        curve.append(row)
    trans = []
    for t in TRANSFORMS:
        ind, tra = f"{t}-all", f"{t}-transductive"
        if ind in methods and tra in methods:
            trans.append(dict(transform=t, inductive=_fmt(*_stats(test, ind, primary)),
                              transductive=_fmt(*_stats(test, tra, primary)),
                              diff=paired(test, primary, tra, ind)))
    raw = _fmt(*_stats(test, "raw", primary)) if "raw" in methods else "n/a"
    return dict(primary=primary, d=d, curve=curve, trans=trans, raw=raw, failed=failed)


def render(results) -> str:
    out = ["# Experiment e7", "", __doc__.split("\n\n", 2)[2].strip(), ""]
    out += ["## Summary", "",
            "| Dataset | d | refiner vs best PCA, smallest n | refiner vs best PCA, all rows | "
            "transductive - inductive (refiner) |", "|---|---|---|---|---|"]
    for name, r in results.items():
        with_vs = [row for row in r["curve"] if "vs" in row]
        def cell(vs):
            return f"{vs[2].split(' ')[0]} ({vs[1]})"

        first = f"n={with_vs[0]['n']}: {cell(with_vs[0]['vs'])}" if with_vs else "n/a"
        last = (cell(with_vs[-1]["vs"]) if with_vs and str(with_vs[-1]["n"]).startswith("all")
                else "n/a")
        tr = next((f"{x['diff'][1].split(' ')[0]} ({x['diff'][0]})" for x in r["trans"]
                   if x["transform"] == "embedkit"), "n/a")
        out.append(f"| {name} | {r['d']} | {first} | {last} | {tr} |")
    out.append("")
    for name, r in results.items():
        out += [f"## {name} ({r['primary']}, d = {r['d']}; raw {r['raw']})", "",
                "| n fitted | refiner | PCA | PCA-whitening | refiner - best baseline [95% CI] |",
                "|---|---|---|---|---|"]
        for row in r["curve"]:
            vs = row.get("vs")
            text = f"{vs[2]} vs {vs[0]} ({vs[1]})" if vs else ""
            out.append(f"| {row['n']} | {row['embedkit']} | {row['pca']} | {row['pca_whiten']} | "
                       f"{text} |")
        if r["trans"]:
            out += ["", "| Transform | inductive | transductive | difference [95% CI] |",
                    "|---|---|---|---|"]
            for x in r["trans"]:
                out.append(f"| {x['transform']} | {x['inductive']} | {x['transductive']} | "
                           f"{x['diff'][1]} ({x['diff'][0]}) |")
        if r["failed"]:
            out += ["", "Failed: " + ", ".join(f"{m} ({e[:80]})" for m, e in r["failed"])]
        out.append("")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default=str(HERE / "results"))
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--out", default=None, help="Write Markdown here instead of stdout")
    args = p.parse_args(argv)
    exp_dir = Path(args.results) / "e7"
    names = args.datasets or sorted(q.name for q in exp_dir.iterdir() if q.is_dir())
    doc = render({n: analyse(exp_dir, n) for n in names})
    if args.out:
        Path(args.out).write_text(doc)
    else:
        sys.stdout.write(doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
