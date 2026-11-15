"""Experiment E3: calibration of the diagnostics on data with known answers (review section 5).

    python benchmarks/e3.py --part id --list                       # show the cells
    python benchmarks/e3.py --part id --shard 0/40                 # one array task
    python benchmarks/e3.py --part gauss --shard 3/20
    python benchmarks/e3.py --part subsample --datasets cifar100_dinov2 --data-root $DATA
    python benchmarks/e3.py --part id --cells sphere5__n5000_D256_noise0_linear_s0   # one cell

Three parts, one JSON file per cell under ``<out>/e3/<part>/<cell>.json`` (written atomically;
existing files are skipped, so a killed task can be resubmitted):

``id``
    Intrinsic dimension on manifolds of known dimension: hyperspheres S^m and hypercubes
    [0, 1]^m for m in 2, 5, 10, 20, and eight of the benchmark manifolds of Hein and Audibert
    (2005) and Campadelli et al. (2015) from ``skdim.datasets.BenchmarkManifolds``. Each is
    standardised and mapped into D dimensions, linearly (random orthonormal columns) or
    nonlinearly (``tanh`` of a random linear map, which keeps the dimension), and isotropic
    Gaussian noise is added whose norm is ``noise`` times the RMS norm of the centred data.
    The grid is one factor at a time around the centre cell (n = 5,000, D = 256, no noise,
    linear map): n from 500 to 50,000, D from 64 to 2,048, noise up to 20%, the nonlinear map.
    Every estimator runs with ``n_max = n``. TwoNN, MLE, lPCA and MOM run everywhere, with 20
    bootstrap resamples when n <= 5,000; ESS and DANCo (minutes per fit) only in the centre
    cell and the n sweep up to 5,000, without bootstrap. The TwoNN scale curve is recorded.
``gauss``
    The report on Gaussian data with a controlled covariance spectrum (flat; power law with
    exponent 0.5, 1 and 2; three spikes of 25 times the variance; a shared mean three times the
    RMS row norm), randomly rotated, over n in {500, 2,000, 10,000}, d in {32, 256, 1,024} and
    k in {5, 10, 30} with the Euclidean metric, plus the cosine metric at k = 10. Gaussian data
    are the null family of the calibration, so the z-scores should be within +-3; the raw
    statistics give the null tables. Two more families, ``flat_l2`` and ``power1_l2``, are
    the flat and ``i^-1`` Gaussians with every row L2-normalised (k = 10, both metrics): input
    that arrives on the sphere, the common case for cosine users, is not Gaussian, so these
    measure how often a report flags such data.
``subsample``
    The default report on random subsets of n in {1,000, 2,000, 5,000, 20,000} rows of the
    cached sets with at least 20,000 training rows, 5 subsets each: does any reading change
    with n (the phase 4 exit criterion)?

Results stay on the cluster (E8 pattern); ``benchmarks/report_e3.py`` writes the summary.
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

from run import _json_default, git_state  # noqa: E402

import embedkit  # noqa: E402
from embedkit.analysis.intrinsic_dim import IntrinsicDimensionEstimator  # noqa: E402
from embedkit.analysis.report import EmbedKitAnalyzer  # noqa: E402

EXPERIMENT = "e3"
SEEDS = 5

# ------------------------------------------------------------------------------------- part id
# name -> (intrinsic dimension, native number of variables); None for the generated families.
BENCHMARK_MANIFOLDS = {
    "M7_Roll": (2, 3), "M11_Moebius": (2, 3), "M3_Nonlinear_4to6": (4, 6),
    "M6_Nonlinear": (6, 36), "Mp3_Paraboloid": (9, 30), "Mbeta": (10, 40),
    "M8_Nonlinear": (12, 72), "Mn1_Nonlinear": (18, 72),
}
ID_FAMILIES = [f"{kind}{m}" for kind in ("sphere", "cube") for m in (2, 5, 10, 20)]
ID_MANIFOLDS = ID_FAMILIES + list(BENCHMARK_MANIFOLDS)
CENTRE = dict(n=5000, D=256, noise=0.0, map="linear")
N_SWEEP = (500, 1000, 2000, 10_000, 20_000, 50_000)
D_SWEEP = (64, 1024, 2048)
NOISE_SWEEP = (0.01, 0.05, 0.2)
FAST_METHODS = ("TwoNN", "MLE", "lPCA", "MOM")
SLOW_METHODS = ("ESS", "DANCo")
BOOTSTRAP_MAX_N = 5000


def true_id(manifold: str) -> int:
    if manifold in BENCHMARK_MANIFOLDS:
        return BENCHMARK_MANIFOLDS[manifold][0]
    return int(manifold.lstrip("spherecub"))


def native_dim(manifold: str) -> int:
    if manifold in BENCHMARK_MANIFOLDS:
        return BENCHMARK_MANIFOLDS[manifold][1]
    m = true_id(manifold)
    return m + 1 if manifold.startswith("sphere") else m


def id_cells() -> dict[str, dict]:
    """``{cell name: settings}``; the sweeps share the centre cell."""
    settings = [dict(CENTRE, sweep="centre")]
    settings += [dict(CENTRE, n=n, sweep="n") for n in N_SWEEP]
    settings += [dict(CENTRE, D=D, sweep="D") for D in D_SWEEP]
    settings += [dict(CENTRE, noise=e, sweep="noise") for e in NOISE_SWEEP]
    settings += [dict(CENTRE, map="tanh", sweep="map")]
    cells = {}
    for manifold in ID_MANIFOLDS:
        for s in settings:
            if s["D"] < native_dim(manifold):
                continue
            for seed in range(SEEDS):
                name = (f"{manifold}__n{s['n']}_D{s['D']}_noise{s['noise']:g}_{s['map']}"
                        f"_s{seed}")
                cells[name] = dict(s, manifold=manifold, seed=seed, true_id=true_id(manifold))
    return cells


def manifold_sample(manifold: str, n: int, seed: int) -> np.ndarray:
    """``n`` points of the manifold in its native coordinates."""
    import skdim

    if manifold.startswith("sphere"):
        return skdim.datasets.hyperSphere(n, true_id(manifold), random_state=seed)
    if manifold.startswith("cube"):
        return np.random.default_rng(seed).random((n, true_id(manifold)))
    m, dim = BENCHMARK_MANIFOLDS[manifold]
    return skdim.datasets.BenchmarkManifolds(random_state=seed).generate(
        manifold, n=n, dim=dim, d=m)


def embed(Z: np.ndarray, D: int, how: str, noise: float, seed: int) -> np.ndarray:
    """Standardise ``Z``, map it into ``D`` dimensions and add isotropic noise.

    The maps are drawn from ``seed + 10_000`` so that they do not depend on n.
    """
    Z = np.asarray(Z, dtype=np.float64)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    rng = np.random.default_rng(seed + 10_000)
    if how == "linear":
        Q, _ = np.linalg.qr(rng.standard_normal((D, Z.shape[1])))
        Y = Z @ Q.T
    elif how == "tanh":
        A = rng.standard_normal((Z.shape[1], D)) / np.sqrt(Z.shape[1])
        Y = np.tanh(Z @ A)
    else:
        raise ValueError(how)
    if noise > 0:
        Yc = Y - Y.mean(0)
        rms = float(np.sqrt(np.mean(np.sum(Yc * Yc, axis=1))))
        Y = Y + noise * rms / np.sqrt(D) * np.random.default_rng(seed + 20_000).standard_normal(
            Y.shape)
    return Y.astype(np.float32)


def run_id_cell(c: dict) -> dict:
    X = embed(manifold_sample(c["manifold"], c["n"], c["seed"]), c["D"], c["map"], c["noise"],
              c["seed"])
    slow = c["sweep"] in ("centre", "n") and c["n"] <= BOOTSTRAP_MAX_N
    methods = FAST_METHODS + (SLOW_METHODS if slow else ())
    out = dict(estimates={}, standard_errors={}, intervals={}, seconds={})
    for m in methods:
        t0 = time.perf_counter()
        r = IntrinsicDimensionEstimator(
            methods=[m], n_max=c["n"], scale_curve=(m == "TwoNN"),
            n_bootstrap=20 if (c["n"] <= BOOTSTRAP_MAX_N and m in FAST_METHODS) else 0,
            random_state=c["seed"],
        ).fit(X)
        out["seconds"][m] = time.perf_counter() - t0
        out["estimates"][m] = r.estimates[m]
        out["standard_errors"][m] = r.standard_errors[m]
        out["intervals"][m] = list(r.intervals[m])
        if m == "TwoNN":
            out["scale_curve"] = {str(k): v for k, v in r.scale_curve.items()}
            out["scale_plateau"] = r.scale_plateau
            out["n_duplicates_removed"] = r.n_duplicates_removed
    return out


# ---------------------------------------------------------------------------------- part gauss
GAUSS_FAMILIES = ("flat", "power0.5", "power1", "power2", "spiked", "mean")
GAUSS_UNIT_FAMILIES = ("flat_l2", "power1_l2")  # rows L2-normalised; k = 10, both metrics
GAUSS_N = (500, 2000, 10_000)
GAUSS_D = (32, 256, 1024)
GAUSS_K = (5, 10, 30)


def gauss_cells() -> dict[str, dict]:
    cells = {}
    for fam in GAUSS_FAMILIES + GAUSS_UNIT_FAMILIES:
        for n in GAUSS_N:
            for d in GAUSS_D:
                combos = ([(k, "euclidean") for k in GAUSS_K] if fam in GAUSS_FAMILIES
                          else [(10, "euclidean")]) + [(10, "cosine")]
                for k, metric in combos:
                    for seed in range(SEEDS):
                        name = f"{fam}__n{n}_d{d}_k{k}_{metric}_s{seed}"
                        cells[name] = dict(family=fam, n=n, d=d, k=k, metric=metric, seed=seed)
    return cells


def gaussian_family(family: str, n: int, d: int, seed: int) -> np.ndarray:
    """Rows of a randomly rotated Gaussian with the family's spectrum (and mean).

    ``<family>_l2`` is ``<family>`` with every row L2-normalised.
    """
    if family.endswith("_l2"):
        X = gaussian_family(family[:-3], n, d, seed).astype(np.float64)
        return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)
    rng = np.random.default_rng(seed)
    lam = np.ones(d)
    if family.startswith("power"):
        lam = np.arange(1, d + 1, dtype=float) ** -float(family[5:])
    elif family == "spiked":
        lam[:3] = 25.0
    Q, _ = np.linalg.qr(np.random.default_rng(seed + 10_000).standard_normal((d, d)))
    X = (rng.standard_normal((n, d)) * np.sqrt(lam)) @ Q.T
    if family == "mean":
        u = np.random.default_rng(seed + 20_000).standard_normal(d)
        X += 3.0 * np.sqrt(lam.sum()) * u / np.linalg.norm(u)
    return X.astype(np.float32)


def report_record(report) -> dict:
    """Every scalar of the report, the calibration, recommendations and properties."""
    df = report.to_dataframe()
    scalars = {f"{c}/{m}": v for c, m, v in df.itertuples(index=False)
               if isinstance(v, (int, float, np.floating, np.integer, bool, str, type(None)))}
    cal = {k: dict(observed=c.observed, null_mean=c.null_mean, null_std=c.null_std,
                   z=c.z_score)
           for k, c in (report.calibration or {}).items()}
    idr = report.intrinsic_dim
    return dict(
        scalars=scalars, calibration=cal, recommendations=report.recommendations,
        properties=report.properties, n_outside_null=report.n_outside_null,
        id=dict(estimates=idr.estimates, standard_errors=idr.standard_errors,
                intervals={m: list(v) for m, v in idr.intervals.items()},
                scale_curve={str(k): v for k, v in idr.scale_curve.items()},
                scale_plateau=idr.scale_plateau, reference_method=idr.reference_method),
        suggested_k=report.suggested_k, suggested_target_dim=report.suggested_target_dim,
    )


def run_gauss_cell(c: dict) -> dict:
    X = gaussian_family(c["family"], c["n"], c["d"], c["seed"])
    report = EmbedKitAnalyzer(k=c["k"], metric=c["metric"], id_methods=["TwoNN"],
                              id_bootstrap=0, id_scale_curve=False,
                              random_state=c["seed"]).fit(X)
    return report_record(report)


# ------------------------------------------------------------------------------ part subsample
SUBSAMPLE_SETS = ("cifar100_dinov2", "yelp_mpnet", "fluorescence_esm2", "glove100", "sift128",
                  "gist960")
SUBSAMPLE_N = (1000, 2000, 5000, 20_000)


def subsample_cells(datasets=SUBSAMPLE_SETS) -> dict[str, dict]:
    return {f"{ds}__n{n}_s{seed}": dict(dataset=ds, n=n, seed=seed)
            for ds in datasets for n in SUBSAMPLE_N for seed in range(SEEDS)}


_LOADED: dict[str, np.ndarray] = {}


def run_subsample_cell(c: dict, data_root) -> dict:
    import bench_datasets as D

    if c["dataset"] not in _LOADED:
        _LOADED.clear()  # one large set in memory at a time
        _LOADED[c["dataset"]] = D.load(c["dataset"], data_root).X_train
    X_all = _LOADED[c["dataset"]]
    rows = np.sort(np.random.default_rng(c["seed"]).choice(len(X_all), c["n"], replace=False))
    report = EmbedKitAnalyzer(random_state=c["seed"]).fit(X_all[rows])
    return report_record(report)


# ----------------------------------------------------------------------------------------- main
PARTS = {"id": id_cells, "gauss": gauss_cells, "subsample": subsample_cells}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--part", required=True, choices=sorted(PARTS))
    p.add_argument("--shard", default=None, metavar="I/N",
                   help="Run only cells i, i + N, i + 2N, ... (0-based) of the part's list")
    p.add_argument("--cells", nargs="*", default=None, help="Only these cell names")
    p.add_argument("--datasets", nargs="*", default=None, help="subsample: only these sets")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default=str(HERE / "results"))
    p.add_argument("--list", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    if args.part == "subsample" and args.datasets:
        cells = subsample_cells(args.datasets)
    else:
        cells = PARTS[args.part]()
    names = list(cells)
    if args.cells:
        unknown = set(args.cells) - set(cells)
        if unknown:
            raise SystemExit(f"Unknown cells: {sorted(unknown)[:5]}")
        names = [n for n in names if n in set(args.cells)]
    if args.shard:
        i, n_shards = (int(v) for v in args.shard.split("/"))
        if not 0 <= i < n_shards:
            raise SystemExit(f"--shard {args.shard}: need 0 <= I < N")
        names = names[i::n_shards]
    if args.list:
        print(f"{len(names)} cells")
        for n in names:
            print(" ", n)
        return 0

    out_dir = Path(args.out) / EXPERIMENT / args.part
    out_dir.mkdir(parents=True, exist_ok=True)
    state = git_state(extra=("benchmarks/e3.py",))
    failures = 0
    for name in names:
        path = out_dir / f"{name}.json"
        if path.exists() and not args.overwrite and not _failed(path):
            continue
        c = cells[name]
        record = dict(experiment=EXPERIMENT, part=args.part, cell=name, settings=c, git=state,
                      embedkit_version=embedkit.__version__, python=platform.python_version(),
                      host=platform.node(),
                      created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if args.part == "id":
                    record.update(run_id_cell(c))
                elif args.part == "gauss":
                    record.update(run_gauss_cell(c))
                else:
                    record.update(run_subsample_cell(c, args.data_root))
            record["warnings"] = sorted({f"{w.category.__name__}: {str(w.message)[:200]}"
                                         for w in caught})
        except Exception as e:  # recorded, so the grid continues
            record["error"] = f"{type(e).__name__}: {e}"
            record["traceback"] = traceback.format_exc()
            failures += 1
        record["elapsed"] = time.perf_counter() - t0
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=1, default=_json_default))
        tmp.replace(path)
        status = record.get("error", "ok")[:100]
        print(f"{name}: {status} ({record['elapsed']:.1f}s)", flush=True)
    return 1 if failures else 0


def _failed(path: Path) -> bool:
    """A result file that records an error: rerun it."""
    try:
        return "error" in json.loads(path.read_text())
    except (OSError, ValueError):
        return True


if __name__ == "__main__":
    sys.exit(main())
