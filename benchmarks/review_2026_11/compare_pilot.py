"""Phase 2 exit check: do the harness's baseline rows reproduce the review pilot (section 3.7)?

    python benchmarks/review_2026_11/compare_pilot.py [--results benchmarks/results] [--tol 0.005]

Compares the test-stage values of ``benchmarks/run.py --experiment e1`` with the pilot's
``pilot/results_<dataset>.json`` for the training-free baselines, on the metrics both computed
the same way: kNN accuracy (k = 5, Euclidean and cosine, neighbours from the training split) and
the pilot's probe (logistic regression, C = 1, on standardised inputs). Trustworthiness and
continuity are listed for information: the pilot used a 2,000-point test subsample, the harness
up to 5,000. Exits with status 1 if any compared value differs by more than ``--tol``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = {"pets": "pets_resnet50", "digits": "digits", "synthetic": "synthetic_anisotropic"}
METRICS = {"knn_euc": "knn_accuracy_euclidean", "knn_cos": "knn_accuracy",
           "linprobe": "linear_probe_c1"}
INFO = {"trust": "trustworthiness", "contin": "continuity"}


def harness_name(pilot_name: str) -> str | None:
    fixed = {"raw": "raw", "centre+L2": "center_l2", "standardise": "standardize"}
    if pilot_name in fixed:
        return fixed[pilot_name]
    for pattern, fmt in ((r"PCA-whiten-(\d+)", "pca_whiten-{}"), (r"PCA-(\d+)$", "pca-{}"),
                         (r"ABTT-D(\d+)", "abtt-{}")):
        m = re.match(pattern, pilot_name)
        if m:
            return fmt.format(m.group(1))
    return None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=str(HERE.parent / "results"))
    p.add_argument("--tol", type=float, default=0.005)
    args = p.parse_args(argv)
    worst, rows = 0.0, []
    for pilot_ds, ds in DATASETS.items():
        pilot = json.loads((HERE / "pilot" / f"results_{pilot_ds}.json").read_text())["results"]
        for pr in pilot:
            name = harness_name(pr["method"])
            path = Path(args.results) / "e1" / ds / f"{name}.json"
            if name is None or not path.exists():
                continue
            rec = json.loads(path.read_text())
            test = {r["metric"]: r["value"] for r in rec["stages"]["test"]}
            for pk, hk in {**METRICS, **INFO}.items():
                if pk in pr and hk in test:
                    diff = test[hk] - pr[pk]
                    compared = pk in METRICS
                    if compared:
                        worst = max(worst, abs(diff))
                    rows.append((ds, name, hk, pr[pk], test[hk], diff, compared))
    print("| Dataset | Method | Metric | Pilot | Harness | Difference |")
    print("|---|---|---|---|---|---|")
    for ds, name, metric, a, b, diff, compared in rows:
        flag = " **over tolerance**" if compared and abs(diff) > args.tol else (
            "" if compared else " (info)")
        print(f"| {ds} | {name} | {metric} | {a:.3f} | {b:.3f} | {diff:+.4f}{flag} |")
    n = sum(r[6] for r in rows)
    print(f"\nCompared {n} values; largest absolute difference {worst:.4f} (tolerance {args.tol}).")
    return 1 if worst > args.tol else 0


if __name__ == "__main__":
    sys.exit(main())
