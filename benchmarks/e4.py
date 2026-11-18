"""Experiment E4, report half: what the default report reads on each natural set (WP 7.1)

    python benchmarks/e4.py --datasets pets_resnet50 --list
    python benchmarks/e4.py --datasets cifar100_dinov2 --data-root $DATA

For each data set, the default ``EmbedKitAnalyzer`` (calibrated, 20 draws) runs on the training
rows with the cosine and the Euclidean metric, as in E5's detection half. One JSON file per
(set, metric) under ``<out>/e4/_detect/``, with the same record as E3 and E5 (every scalar, the
calibration, the recommendations and properties). The gains half (training-free interventions
on the same sets) runs through the shared harness:

    python benchmarks/run.py --experiment e4 --datasets <name> --data-root $DATA

The default sets are ``bench_datasets.E4_SETS``. ``benchmarks/report_e4.py`` relates the two.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import bench_datasets as D  # noqa: E402
from e3 import _failed, report_record  # noqa: E402
from run import _json_default, git_state  # noqa: E402

import embedkit  # noqa: E402
from embedkit.analysis.report import EmbedKitAnalyzer  # noqa: E402

EXPERIMENT = "e4"
METRICS = ("cosine", "euclidean")


def main(argv=None):
    default_sets = getattr(D, "E4_SETS", None)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--datasets", nargs="+", default=list(default_sets) if default_sets else None,
                   help="Data sets (default: bench_datasets.E4_SETS)")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default=str(HERE / "results"))
    p.add_argument("--list", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    if not args.datasets:
        raise SystemExit("bench_datasets.E4_SETS does not exist (yet): pass --datasets explicitly")

    out_dir = Path(args.out) / EXPERIMENT / "_detect"
    out_dir.mkdir(parents=True, exist_ok=True)
    state = git_state(extra=("benchmarks/e4.py", "benchmarks/e3.py"))
    failures = 0
    for name in args.datasets:
        if args.list:
            print(name, [f"{name}__{m}" for m in METRICS])
            continue
        todo = [m for m in METRICS if args.overwrite
                or not (out_dir / f"{name}__{m}.json").exists()
                or _failed(out_dir / f"{name}__{m}.json")]
        if not todo:
            print(f"{name}: exists, skipped", flush=True)
            continue
        try:
            ds = D.load(name, args.data_root)
        except D.DatasetUnavailableError as e:
            print(f"[skip] {e}", flush=True)
            failures += 1
            continue
        for metric in todo:
            path = out_dir / f"{name}__{metric}.json"
            record = dict(experiment=EXPERIMENT, part="detect", dataset=name, base=name,
                          metric=metric, damage=None, damage_level=None,
                          shape=list(ds.X_train.shape), git=state,
                          embedkit_version=embedkit.__version__,
                          python=platform.python_version(), host=platform.node(),
                          created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
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
