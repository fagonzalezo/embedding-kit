"""E8 report: supervised and regression modes (phase 6 of the plan).

    python benchmarks/report_e8.py --out benchmarks/results/e8/REPORT.md

Classification sets are scored on test kNN accuracy (cosine, k = 5), regression sets on the test
R^2 of a ridge probe (``linear_probe``) and of 10-NN regression (``knn_r2``). A family with
several configurations (the spread weight) is reduced to the one best on validation. Each arm is
compared with the phase 6 default (``supcon``, ``rnc``) by a paired bootstrap over test points;
the phase rule accepts a change of default when it wins (difference above 0, 95% interval
excluding 0) on at least half the datasets. Baselines: raw and LDA (classification), raw and PCA
(regression). Label noise and superclass fitting change only the labels the transform is fitted
on; evaluation uses the clean fine labels. Label-efficiency sets (``<name>__lpc<k>``) keep k
labelled training rows per class for everything, gallery and probe included.
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

CLASS_ARMS = ("prev", "supcon_au", "supcon_mlp", "supcon_noes", "supcon_spread", "supcon_nested")
REG_ARMS = ("rnc_old", "rnc_noes", "rnc_noise", "rnc_mlp")


def _select(val, test, meta, family, primary):
    members = [m for m in meta if meta[m]["family"] == family and m in set(test.method)]
    if not members:
        return None
    if val is None or len(members) == 1:
        return members[0]
    v = val[val.metric == primary].groupby("method")["value"].mean()
    return max(members, key=lambda m: v.get(m, float("-inf")))


def _cell(test, metric, method, ref):
    if method is None or ref is None:
        return ("n/a", "")
    verdict, text = paired(test, metric, method, ref)
    return verdict, f"{text} {verdict}"


def analyse(exp_dir: Path, dataset: str) -> dict:
    stages, meta, failed = load_results(exp_dir, dataset)
    val, test = stages.get("val"), stages["test"]
    metrics = set(test.metric)
    task = "classification" if "knn_accuracy" in metrics else "regression"
    primary = "knn_accuracy" if task == "classification" else "linear_probe"
    second = "linear_probe" if task == "classification" else "knn_r2"
    default = "supcon" if task == "classification" else "rnc"
    sel = lambda fam: _select(val, test, meta, fam, primary)  # noqa: E731
    base = [b for b in ("raw", "lda", f"pca-{_dim(test, default)}") if b in set(test.method)]

    def row(method):
        out = dict(method=method, value=_fmt(*_stats(test, method, primary)),
                   second=_fmt(*_stats(test, method, second)))
        if "nc1" in metrics:
            out["nc1"] = _fmt(*_stats(test, method, "nc1"))
        out["vs_default"] = _cell(test, primary, method, default) if method != default else None
        out["vs_raw"] = _cell(test, primary, method, "raw") if method != "raw" else None
        out["vs_default2"] = (_cell(test, second, method, default) if method != default
                              else None)
        return out

    arms = {}
    for fam in (CLASS_ARMS if task == "classification" else REG_ARMS):
        m = sel(fam)
        if m is not None:
            arms[fam] = row(m)
    rows = dict(default=row(default) if default in set(test.method) else None,
                baselines=[row(b) for b in base], arms=arms)
    noise = {}
    for m in sorted(set(test.method)):
        if (hit := re.fullmatch(r"(supcon|lda)-noise(\d+)", m)):
            noise.setdefault(int(hit.group(2)), {})[hit.group(1)] = row(m)
    coarse = {m: row(m) for m in sorted(set(test.method)) if m.endswith("-coarse")}
    vs_lda = (_cell(test, primary, "supcon", "lda")[1]
              if {"supcon", "lda"} <= set(test.method) else "")
    return dict(task=task, primary=primary, second=second, rows=rows, noise=noise,
                coarse=coarse, supcon_vs_lda=vs_lda, failed=failed)


def _dim(test, method):
    d = test[test.method == method]["dim"]
    return int(d.iloc[0]) if len(d) else None


def render(results: dict) -> str:
    out = ["# Experiment e8", "", __doc__.split("\n\n", 2)[2].strip(), ""]
    main = {n: r for n, r in results.items() if "__lpc" not in n}
    lpc = {n: r for n, r in results.items() if "__lpc" in n}
    for task in ("classification", "regression"):
        sets = {n: r for n, r in main.items() if r["task"] == task}
        if not sets:
            continue
        arms = CLASS_ARMS if task == "classification" else REG_ARMS
        default = "supcon" if task == "classification" else "rnc"
        first = next(iter(sets.values()))
        out += [f"## Summary, {task}: each arm against `{default}` (wins / ties / losses)", "",
                f"| Arm | on {first['primary']} | on {first['second']} | accepted |",
                "|---|---|---|---|"]
        n = sum(1 for r in sets.values() if r["rows"]["default"] is not None)

        def count(fam, key):
            vs = [r["rows"]["arms"][fam][key][0] for r in sets.values()
                  if fam in r["rows"]["arms"]]
            return vs, f"{vs.count('win')} / {vs.count('tie')} / {vs.count('loss')}" + (
                f" ({vs.count('n/a')} n/a)" if "n/a" in vs else "")

        for fam in arms:
            vs, text = count(fam, "vs_default")
            if not vs:
                continue
            _, text2 = count(fam, "vs_default2")
            out.append(f"| {fam} | {text} | {text2} | "
                       f"{'yes' if vs.count('win') >= n / 2 else 'no'} |")
        out += ["", f"Accepted: wins on the first metric on at least half of the {n} datasets.",
                ""]
    for name, r in main.items():
        nc1 = "nc1" in (r["rows"]["default"] or {})
        out += [f"## {name} ({r['task']}; {r['primary']}, also {r['second']})", "",
                f"| Method | {r['primary']} | {r['second']} |" + (" NC1 |" if nc1 else "")
                + f" vs default, {r['primary']} [95% CI] | vs default, {r['second']} | "
                f"vs raw, {r['primary']} |",
                "|---|---|---|" + ("---|" if nc1 else "") + "---|---|---|"]
        entries = ([("default", r["rows"]["default"])] if r["rows"]["default"] else []) + [
            ("baseline", b) for b in r["rows"]["baselines"]] + list(r["rows"]["arms"].items())
        for label, x in entries:
            vd = x["vs_default"][1] if x["vs_default"] else "reference"
            vd2 = x["vs_default2"][1] if x["vs_default2"] else ""
            vr = x["vs_raw"][1] if x["vs_raw"] else ""
            out.append(f"| {x['method']} | {x['value']} | {x['second']} |"
                       + (f" {x.get('nc1', '')} |" if nc1 else "") + f" {vd} | {vd2} | {vr} |")
        if r["noise"]:
            out += ["", "Label noise (fitted on noisy labels, scored with clean ones):", "",
                    "| Noise | supcon | supcon vs raw | LDA | LDA vs raw |",
                    "|---|---|---|---|---|"]
            for rate, d in sorted(r["noise"].items()):
                s, lda = d.get("supcon"), d.get("lda")
                out.append(f"| {rate}% | {s['value'] if s else ''} | "
                           f"{s['vs_raw'][1] if s else ''} | {lda['value'] if lda else ''} | "
                           f"{lda['vs_raw'][1] if lda else ''} |")
        if r["coarse"]:
            out += ["", "Fitted on the 20 superclasses, scored with the 100 fine classes:", "",
                    "| Method | fine kNN | NC1 (fine) | vs supcon on fine labels | vs raw |",
                    "|---|---|---|---|---|"]
            for m, x in r["coarse"].items():
                out.append(f"| {m} | {x['value']} | {x.get('nc1', '')} | "
                           f"{x['vs_default'][1]} | {x['vs_raw'][1]} |")
        if r["failed"]:
            out += ["", "Failed: " + ", ".join(f"{m} ({e[:80]})" for m, e in r["failed"])]
        out.append("")
    if lpc:
        out += ["## Label efficiency (k labelled training rows per class)", "",
                "| Set | raw | LDA | supcon | prev | supcon_noes | supcon vs raw | "
                "supcon vs LDA |", "|---|---|---|---|---|---|---|---|"]
        for name in sorted(lpc, key=lambda s: (s.split("__")[0], int(s.split("lpc")[1]))):
            r = lpc[name]
            test_rows = {x["method"]: x for x in [r["rows"]["default"], *r["rows"]["baselines"],
                                                  *r["rows"]["arms"].values()] if x}
            g = lambda m: test_rows[m]["value"] if m in test_rows else ""  # noqa: E731
            vs_raw = test_rows.get("supcon", {}).get("vs_raw")
            out.append(f"| {name} | {g('raw')} | {g('lda')} | {g('supcon')} | {g('prev')} | "
                       f"{g('supcon_noes')} | {vs_raw[1] if vs_raw else ''} | "
                       f"{r['supcon_vs_lda']} |")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default=str(HERE / "results"))
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--out", default=None, help="Write Markdown here instead of stdout")
    args = p.parse_args(argv)
    exp_dir = Path(args.results) / "e8"
    names = args.datasets or sorted(q.name for q in exp_dir.iterdir() if q.is_dir())
    doc = render({n: analyse(exp_dir, n) for n in names})
    if args.out:
        Path(args.out).write_text(doc)
    else:
        sys.stdout.write(doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
