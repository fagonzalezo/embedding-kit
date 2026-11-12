"""Run a benchmark experiment and write one JSON file per (dataset, method).

    python benchmarks/run.py --experiment e1 --datasets pets_resnet50 digits synthetic_anisotropic \
        --seeds 3 --device cpu

Each method is evaluated in two stages with :func:`embedkit.evaluation.evaluate_transform`:

- ``val``: fitted on the training split minus a stratified validation part
  (``--val-fraction``, seed 0) and scored on that validation part. Hyperparameters (the output
  dimension, ``n_remove``, ...) are selected on these numbers (review section 5, WP 3.1).
- ``test``: fitted on the whole training split and scored on the test split.

Results go to ``<out>/<experiment>/<dataset>/<method>.json``. A combination whose file exists is
skipped, so a killed run can be restarted. Files are written atomically when a method finishes.
Deterministic methods run with one seed; stochastic ones with seeds ``0 .. --seeds - 1``.

The same command runs unchanged on the laptop and on the cluster: ``--data-root`` (or
``EMBEDKIT_BENCH_DATA``) points to the datasets, ``--device`` to the accelerator.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import bench_datasets as D  # noqa: E402
from experiments import EXPERIMENTS  # noqa: E402

import embedkit  # noqa: E402
from embedkit.evaluation import evaluate_transform  # noqa: E402


def git_state(extra: tuple[str, ...] = ()) -> dict:
    """HEAD and whether the code behind a result (plus the ``extra`` paths) has local changes."""
    def run(*args):
        try:
            return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None

    dirty = run("status", "--porcelain", "--", "embedkit", "benchmarks/run.py",
                "benchmarks/experiments.py", "benchmarks/bench_datasets.py", *extra)
    return dict(commit=run("rev-parse", "HEAD"), dirty=bool(dirty) if dirty is not None else None)


def encode_per_point(pp):
    """0/1 outcomes as a compact string; anything else as a list of floats."""
    if pp is None:
        return None
    arr = np.asarray(pp)
    if np.isin(arr, (0.0, 1.0)).all():
        return "01:" + "".join("1" if v else "0" for v in arr.astype(bool))
    return [float(v) for v in arr]


def decode_per_point(obj):
    if obj is None:
        return None
    if isinstance(obj, str) and obj.startswith("01:"):
        return np.frombuffer(obj[3:].encode(), dtype=np.uint8).astype(np.float32) - ord("0")
    return np.asarray(obj, dtype=np.float32)


def validation_split(ds, fraction: float, seed: int = 0):
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(ds.X_train))
    stratify = ds.y_train if ds.meta.get("task") == "classification" else None
    if ds.meta.get("task") == "sts":
        return np.arange(len(ds.X_train)), np.arange(0)  # validation comes from meta["val_split"]
    tr, va = train_test_split(idx, test_size=fraction, random_state=seed, stratify=stratify)
    return np.sort(tr), np.sort(va)


def rows_to_records(df) -> list[dict]:
    records = []
    for r in df.to_dict("records"):
        r["per_point"] = encode_per_point(r["per_point"])
        records.append(r)
    return records


def run_method(method, ds, args, splits) -> dict:
    seeds = (0,) if method.deterministic else tuple(range(args.seeds))
    labelled = ds.y_train is not None
    out = dict(stages={}, errors={})
    if "val_split" in ds.meta:  # STS: a separate validation set with its own pairs
        X_val, val_pairs = ds.meta["val_split"]
        val_stage = (ds.X_train, X_val, None, None, val_pairs)
    else:
        val_stage = (ds.X_train[splits[0]], ds.X_train[splits[1]],
                     ds.y_train[splits[0]] if labelled else None,
                     ds.y_train[splits[1]] if labelled else None, None)
    stages = {
        "val": val_stage,
        "test": (ds.X_train, ds.X_test, ds.y_train, ds.y_test, ds.meta.get("test_pairs")),
    }
    for stage in args.stages:
        X_tr, X_te, y_tr, y_te, pairs = stages[stage]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = evaluate_transform(
                lambda s: method.make(s, ds, args.device), X_tr, X_te, y_tr, y_te,
                seeds=seeds, metrics=args.metrics or method.metrics,
                fit_with_labels=method.fit_with_labels, method=method.name,
                transductive=method.transductive,
                test_pairs=pairs,
                hubness_reduction=method.hubness_reduction,
            )
        out["stages"][stage] = rows_to_records(df)
        out.setdefault("warnings", {})[stage] = sorted(
            {f"{w.category.__name__}: {str(w.message)[:200]}" for w in caught}
        )
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--methods", nargs="*", default=None,
                   help="Only these method names (default: the whole grid)")
    p.add_argument("--seeds", type=int, default=5, help="Seeds for stochastic methods")
    p.add_argument("--device", default="cpu")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default=str(HERE / "results"))
    p.add_argument("--stages", nargs="+", default=["val", "test"], choices=["val", "test"])
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--metrics", nargs="*", default=None,
                   help="Metric names (default: the protocol's defaults for the labels given)")
    p.add_argument("--shard", default=None, metavar="I/N",
                   help="Run only methods i, i + N, i + 2N, ... of the grid (0-based), so that "
                        "N jobs can share one dataset")
    p.add_argument("--list", action="store_true", help="List the method grid and exit")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    out_root = Path(args.out) / args.experiment
    state = git_state()
    failures = 0
    for name in args.datasets:
        try:
            ds = D.load(name, args.data_root)
        except D.DatasetUnavailableError as e:
            print(f"[skip] {e}", flush=True)
            failures += 1
            continue
        methods = EXPERIMENTS[args.experiment](ds)
        if args.methods:
            unknown = set(args.methods) - {m.name for m in methods}
            if unknown:
                raise SystemExit(f"Unknown methods for {name}: {sorted(unknown)}")
            methods = [m for m in methods if m.name in args.methods]
        if args.shard:
            i, n_shards = (int(v) for v in args.shard.split("/"))
            if not 0 <= i < n_shards:
                raise SystemExit(f"--shard {args.shard}: need 0 <= I < N")
            methods = methods[i::n_shards]
        if args.list:
            print(name, [m.name for m in methods])
            continue
        splits = validation_split(ds, args.val_fraction)
        ds_dir = out_root / name
        ds_dir.mkdir(parents=True, exist_ok=True)
        print(f"== {name}: train {ds.X_train.shape}, test {ds.X_test.shape}, "
              f"{len(methods)} methods", flush=True)
        for method in methods:
            path = ds_dir / f"{method.name}.json"
            if path.exists() and not args.overwrite:
                print(f"   {method.name}: exists, skipped", flush=True)
                continue
            t0 = time.perf_counter()
            record = dict(
                experiment=args.experiment, dataset=name, method=method.name,
                family=method.family, params=method.params,
                deterministic=method.deterministic, fit_with_labels=method.fit_with_labels,
                seeds=1 if method.deterministic else args.seeds, device=args.device,
                val_fraction=args.val_fraction, git=state, embedkit_version=embedkit.__version__,
                python=platform.python_version(), host=platform.node(),
                created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                dataset_meta={k: v for k, v in ds.meta.items()
                              if isinstance(v, (str, int, float, bool))},
            )
            try:
                record.update(run_method(method, ds, args, splits))
            except Exception as e:  # recorded, so the grid continues; the file marks the failure
                record["error"] = f"{type(e).__name__}: {e}"
                record["traceback"] = traceback.format_exc()
                failures += 1
            record["elapsed"] = time.perf_counter() - t0
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(record, indent=1, default=_json_default))
            tmp.replace(path)
            summary = _one_line(record)
            print(f"   {method.name}: {summary} ({record['elapsed']:.1f}s)", flush=True)
    return 1 if failures else 0


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _one_line(record) -> str:
    if "error" in record:
        return "ERROR " + record["error"][:120]
    rows = record["stages"].get("test") or record["stages"].get("val") or []
    by_metric: dict[str, list[float]] = {}
    for r in rows:
        by_metric.setdefault(r["metric"], []).append(r["value"])
    keys = [k for k in ("knn_accuracy_euclidean", "knn_accuracy", "linear_probe",
                        "trustworthiness") if k in by_metric][:3]
    return ", ".join(f"{k} {np.mean(by_metric[k]):.3f}" for k in keys)


if __name__ == "__main__":
    sys.exit(main())
