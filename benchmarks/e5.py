"""Experiment E5, detection half: does the report see a known damage? (review section 5)

    python benchmarks/e5.py --datasets pets_resnet50 --list
    python benchmarks/e5.py --datasets cifar100_dinov2 --data-root $DATA

For each base set and each of its damaged variants (``bench_datasets.damage``: a common
vector, nuisance directions, an anisotropic linear map, isotropic noise, padding with noise
coordinates, a second domain; two levels each), the default ``EmbedKitAnalyzer`` runs on the
training rows with the Euclidean and the cosine metric. One JSON file per (variant, metric)
under ``<out>/e5/_detect/``, with the same record as E3 (every scalar, the calibration, the
recommendations and properties). The noise floor of each reading comes from the clean report on
5 random subsets of 80% of the training rows (``<base>__sub80_s<i>``): a change counts as a
detection only when it exceeds 3 times their spread. The repair half runs through the shared
harness:

    python benchmarks/run.py --experiment e5 --stages test --datasets <name> <name>__dmg_...

``benchmarks/report_e5.py`` compares each variant with its clean base.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import bench_datasets as D  # noqa: E402
from e3 import _failed, report_record  # noqa: E402
from run import _json_default, git_state  # noqa: E402

import embedkit  # noqa: E402
from embedkit.analysis.report import EmbedKitAnalyzer  # noqa: E402

EXPERIMENT = "e5"
E5_SETS = ("cifar100_dinov2", "banking77_mpnet", "stl10_clip", "pets_resnet50")
METRICS = ("euclidean", "cosine")
N_FLOOR = 5  # clean reports on random 80% row subsets, for the noise floor of each reading


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--datasets", nargs="+", default=list(E5_SETS), help="Base sets")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default=str(HERE / "results"))
    p.add_argument("--list", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    out_dir = Path(args.out) / EXPERIMENT / "_detect"
    out_dir.mkdir(parents=True, exist_ok=True)
    state = git_state(extra=("benchmarks/e5.py", "benchmarks/e3.py"))
    failures = 0
    for base in args.datasets:
        variants = [base, *D.damage_names(base),
                    *(f"{base}__sub80_s{i}" for i in range(N_FLOOR))]
        if args.list:
            print(base, variants)
            continue
        clean = None
        for name in variants:
            todo = [m for m in METRICS if args.overwrite
                    or not (out_dir / f"{name}__{m}.json").exists()
                    or _failed(out_dir / f"{name}__{m}.json")]
            if not todo:
                continue
            if clean is None:
                clean = D.load(base, args.data_root)
            _, _, variant = name.partition("__")
            if variant.startswith("sub80_s"):
                rng = np.random.default_rng(int(variant[7:]))
                n = len(clean.X_train)
                keep = np.sort(rng.choice(n, int(0.8 * n), replace=False))
                ds = D.Dataset(name, clean.X_train[keep], None, clean.X_test, None,
                               dict(clean.meta))
            elif variant:
                spec = variant[4:]
                kind = spec.rstrip("0123456789.")
                ds = D.damage(clean, kind, float(spec[len(kind):]))
            else:
                ds = clean
            for metric in todo:
                path = out_dir / f"{name}__{metric}.json"
                record = dict(experiment=EXPERIMENT, part="detect", dataset=name, base=base,
                              metric=metric, damage=ds.meta.get("damage"),
                              damage_level=ds.meta.get("damage_level"),
                              shape=list(ds.X_train.shape), git=state,
                              embedkit_version=embedkit.__version__,
                              python=platform.python_version(), host=platform.node(),
                              created=dt.datetime.now(dt.timezone.utc).isoformat(
                                  timespec="seconds"))
                t0 = time.perf_counter()
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        report = EmbedKitAnalyzer(metric=metric, random_state=0).fit(ds.X_train)
                        record.update(report_record(report))
                    record["warnings"] = sorted(
                        {f"{w.category.__name__}: {str(w.message)[:200]}" for w in caught})
                except Exception as e:
                    record["error"] = f"{type(e).__name__}: {e}"
                    record["traceback"] = traceback.format_exc()
                    failures += 1
                record["elapsed"] = time.perf_counter() - t0
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(record, indent=1, default=_json_default))
                tmp.replace(path)
                print(f"{name} {metric}: {record.get('error', 'ok')[:100]} "
                      f"({record['elapsed']:.1f}s)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
