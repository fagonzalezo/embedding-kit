"""Experiment E2: do label-free statistics track downstream quality? (review section 5, WP 3.2)

    python benchmarks/e2.py --datasets pets_resnet50 --device cpu
    python benchmarks/e2.py --datasets digits --configs c00 c01 --epochs 10      # smoke run
    python benchmarks/e2.py --datasets digits --list                             # show the grid

For each dataset, ~30 self-supervised refiner configurations (``e2_configs``) are trained for a
fixed epoch budget with no early stopping. Every ``--eval-every`` epochs the run logs:

- every monitor in :data:`embedkit.improvement.trainer.MONITORS`, computed as the trainer does
  during training: on its held-out split (``val_split=0.1`` of the training rows), plus the
  training loss of that epoch. These are the candidates for stopping and selection signals.
  ``val_knn_accuracy`` uses labels and is logged as the labelled reference.
- the downstream quality on the test split: kNN accuracy (cosine, k = 5, neighbours from the
  training split, as E1's ``knn_accuracy``) and a logistic-regression probe with C = 1 (as E1's
  ``linear_probe_c1``). To bound the cost, both use at most ``--max-pool`` training rows and
  ``--max-test`` test rows (stratified subsamples, seed 0; all rows on the smaller sets).

The losses are self-supervised and ignore labels; labels go to the trainer only for
``val_knn_accuracy``. Results: ``<out>/e2/<dataset>/<config>.json`` (one file per run, resumable)
and ``<out>/e2/<dataset>/_reference.json`` (raw-embedding values on the same subsamples).
``benchmarks/report_e2.py`` computes the correlations.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import bench_datasets as D  # noqa: E402
from run import _json_default, git_state  # noqa: E402

import embedkit  # noqa: E402
from embedkit.evaluation.metrics import knn_accuracy, linear_probe  # noqa: E402
from embedkit.improvement.augmentation import (  # noqa: E402
    EmbeddingMixup,
    FeatureDropout,
    GaussianNoise,
    KNNPairs,
)
from embedkit.improvement.losses import AlignUniformLoss, CombinedLoss, NTXentLoss  # noqa: E402
from embedkit.improvement.model import EmbeddingRefiner  # noqa: E402
from embedkit.improvement.trainer import MONITORS, Trainer  # noqa: E402

EXPERIMENT = "e2"
N_CONFIGS = 30
GRID_SEED = 0

# ------------------------------------------------------------------------------------- grid
LOSSES = ("ntxent_au", "ntxent", "align_uniform")
TEMPERATURES = (0.07, 0.2, 0.5)  # NT-Xent only
AUGMENTATIONS = ("knn5", "knn20", "noise", "dropout", "mixup")
TARGET_DIMS = (32, 64, 128)
LRS = (1e-4, 3e-4, 1e-3)
# EmbedKit's self-supervised recipe (api.py) at a fixed d.
DEFAULT_CONFIG = dict(loss="ntxent_au", temperature=0.07, augmentation="knn5", target_dim=64,
                      lr=3e-4)


def e2_configs(n: int = N_CONFIGS, seed: int = GRID_SEED) -> dict[str, dict]:
    """``c00`` is the default recipe; ``c01..`` are drawn without replacement from the grid.

    The draw is fixed by ``seed``; result files are keyed on the names, so changing the grid
    changes what a name means.
    """
    grid = []
    for loss, aug, d, lr in itertools.product(LOSSES, AUGMENTATIONS, TARGET_DIMS, LRS):
        for t in (TEMPERATURES if loss != "align_uniform" else (None,)):
            cfg = dict(loss=loss, temperature=t, augmentation=aug, target_dim=d, lr=lr)
            if cfg != DEFAULT_CONFIG:
                grid.append(cfg)
    picks = np.random.default_rng(seed).choice(len(grid), n - 1, replace=False)
    configs = [DEFAULT_CONFIG] + [grid[i] for i in picks]
    return {f"c{i:02d}": c for i, c in enumerate(configs)}


def _loss(cfg):
    if cfg["loss"] == "ntxent_au":
        return CombinedLoss([(NTXentLoss(temperature=cfg["temperature"]), 1.0),
                             (AlignUniformLoss(), 0.5)])
    if cfg["loss"] == "ntxent":
        return NTXentLoss(temperature=cfg["temperature"])
    return AlignUniformLoss()


def _augmentation(name):
    return {
        "knn5": lambda: KNNPairs(k=5),
        "knn20": lambda: KNNPairs(k=20),
        "noise": lambda: GaussianNoise(std=0.1),  # relative to each feature's std
        "dropout": lambda: FeatureDropout(p=0.2),
        "mixup": lambda: EmbeddingMixup(),
    }[name]()


# ---------------------------------------------------------------------------------- trainer
class TrackingTrainer(Trainer):
    """A Trainer that also scores the test split at every evaluation (never used for decisions:
    no early stopping, no plateau scheduler)."""

    def __init__(self, *args, downstream, **kwargs):
        super().__init__(*args, **kwargs)
        self._downstream = downstream  # (X_pool, y_pool, X_test, y_test)

    def _evaluate(self, X_np, y=None, X_ref=None, y_ref=None):
        t0 = time.perf_counter()
        out = super()._evaluate(X_np, y=y, X_ref=X_ref, y_ref=y_ref)
        out["eval_time_monitors"] = time.perf_counter() - t0
        out["train_loss"] = self.history["loss"][-1]
        t0 = time.perf_counter()
        Xp, yp, Xt, yt = self._downstream
        Zp, Zt = self.transform(Xp), self.transform(Xt)
        out["test_knn_accuracy"] = knn_accuracy(Zp, yp, Zt, yt, k=5, metric="cosine")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # L-BFGS convergence; counted nowhere, C = 1 as E1
            out["test_linear_probe"] = linear_probe(Zp, yp, Zt, yt, task="classification",
                                                    C=1.0, max_iter=3000)
        out["eval_time_downstream"] = time.perf_counter() - t0
        return out


def _stratified_subset(y, n_max, seed=0):
    if len(y) <= n_max:
        return np.arange(len(y))
    from sklearn.model_selection import train_test_split

    idx, _ = train_test_split(np.arange(len(y)), train_size=n_max, random_state=seed, stratify=y)
    return np.sort(idx)


def run_config(cfg, ds, downstream, args, seed=0) -> dict:
    d_in = ds.X_train.shape[1]
    target_dim = min(cfg["target_dim"], d_in)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        model = EmbeddingRefiner(input_dim=d_in, target_dim=target_dim, hidden_dim=256,
                                 n_layers=2, architecture="mlp")  # the model E2 measured
    trainer = TrackingTrainer(
        model=model, augmentation=_augmentation(cfg["augmentation"]), loss=_loss(cfg),
        epochs=args.epochs, batch_size=256, optimizer="adam", lr=cfg["lr"], weight_decay=1e-4,
        scheduler="cosine", eval_every=args.eval_every, eval_metrics=list(MONITORS),
        monitor="uniformity", early_stopping_patience=None, val_split=0.1,
        device=args.device, random_state=seed, downstream=downstream,
    )
    t0 = time.perf_counter()
    trainer.fit(ds.X_train, ds.y_train)
    fit_time = time.perf_counter() - t0
    h = trainer.history
    evals = {k: v for k, v in h.items() if k not in ("loss", "lr")}
    return dict(target_dim=target_dim, seed=seed, fit_time=fit_time,
                history=dict(evals=evals, loss=h["loss"], lr=h["lr"]),
                directions={k: MONITORS[k][1] for k in MONITORS} | {"train_loss": "min"})


def reference(downstream) -> dict:
    Xp, yp, Xt, yt = downstream
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probe = linear_probe(Xp, yp, Xt, yt, task="classification", C=1.0, max_iter=3000)
    return dict(raw_test_knn_accuracy=knn_accuracy(Xp, yp, Xt, yt, k=5, metric="cosine"),
                raw_test_linear_probe=probe, n_pool=len(yp), n_test=len(yt))


def _write(path: Path, record: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1, default=_json_default))
    tmp.replace(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--configs", nargs="*", default=None, help="Only these config names")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--max-pool", type=int, default=10_000)
    p.add_argument("--max-test", type=int, default=5_000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default=str(HERE / "results"))
    p.add_argument("--list", action="store_true", help="List the configurations and exit")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    configs = e2_configs()
    if args.configs:
        unknown = set(args.configs) - set(configs)
        if unknown:
            raise SystemExit(f"Unknown configs: {sorted(unknown)}")
        configs = {k: v for k, v in configs.items() if k in args.configs}
    if args.list:
        for name, cfg in configs.items():
            print(name, cfg)
        return 0

    state = git_state(extra=("benchmarks/e2.py",))
    failures = 0
    for name in args.datasets:
        ds = D.load(name, args.data_root)
        if ds.y_train is None or ds.meta.get("task") != "classification":
            raise SystemExit(f"E2 needs a classification set; {name} is not one")
        pool = _stratified_subset(ds.y_train, args.max_pool)
        test = _stratified_subset(ds.y_test, args.max_test)
        downstream = (ds.X_train[pool], ds.y_train[pool], ds.X_test[test], ds.y_test[test])
        ds_dir = Path(args.out) / EXPERIMENT / name
        ds_dir.mkdir(parents=True, exist_ok=True)
        common = dict(
            experiment=EXPERIMENT, dataset=name, epochs=args.epochs, eval_every=args.eval_every,
            max_pool=args.max_pool, max_test=args.max_test, device=args.device, git=state,
            embedkit_version=embedkit.__version__, python=platform.python_version(),
            host=platform.node(),
        )
        print(f"== {name}: train {ds.X_train.shape}, test {ds.X_test.shape}, downstream pool "
              f"{len(pool)}, test {len(test)}, {len(configs)} configs", flush=True)
        ref_path = ds_dir / "_reference.json"
        if not ref_path.exists() or args.overwrite:
            _write(ref_path, common | reference(downstream))
        for cname, cfg in configs.items():
            path = ds_dir / f"{cname}.json"
            if path.exists() and not args.overwrite:
                print(f"   {cname}: exists, skipped", flush=True)
                continue
            t0 = time.perf_counter()
            record = common | dict(
                config=cname, params=cfg,
                created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            )
            try:
                record.update(run_config(cfg, ds, downstream, args))
            except Exception as e:  # recorded, so the grid continues
                record["error"] = f"{type(e).__name__}: {e}"
                record["traceback"] = traceback.format_exc()
                failures += 1
            record["elapsed"] = time.perf_counter() - t0
            _write(path, record)
            if "error" in record:
                summary = "ERROR " + record["error"][:120]
            else:
                ev = record["history"]["evals"]
                summary = (f"final kNN {ev['test_knn_accuracy'][-1]:.3f} "
                           f"(max {max(ev['test_knn_accuracy']):.3f}), "
                           f"probe {ev['test_linear_probe'][-1]:.3f}")
            print(f"   {cname} {cfg}: {summary} ({record['elapsed']:.1f}s)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
