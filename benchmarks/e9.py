"""Experiment E9: time and memory of ``EmbedKitAnalyzer.fit`` and ``EmbedKit.fit`` (WP 7.3)

    python benchmarks/e9.py --targets analyzer embedkit --n 1000 10000 --d 128 1024
    python benchmarks/e9.py --targets analyzer --n 1000000 --d 4096 --timeout 85500
    conda run -n pytorch python benchmarks/e9.py --rows 1000 --dims 64    # not --n/--d here

A cell is (target, n, d). Each cell runs in a fresh child process (this script re-invoked with
``--child``), so the peak resident memory is the cell's own. The data are synthetic float32,
generated in the child from a seed: ``N_CLASSES`` classes whose means lie along the leading
directions of a power-law spectrum (variance of coordinate ``i`` proportional to ``1 / i``), plus
noise with the same spectrum, so that labels and a realistic anisotropy exist. They are drawn
in chunks straight into the final array. Recorded per cell: wall time of ``fit`` alone, peak
RSS of the child (``ru_maxrss``: includes the interpreter, torch, the data and the fit), RSS
right after generating ``X`` (before ``fit``, libraries imported), the size of ``X``, CPUs and
threads, the torch device, version and commit, and a status: ``ok``, ``timeout`` (the child was
killed at ``--timeout`` seconds), ``oom_or_killed`` (SIGKILL, typically the OOM killer),
``oom`` (``MemoryError``) or ``error`` (with the message).

One JSON per cell under ``<results>/<experiment>/``, named ``<target>__n<n>__d<d>.json``;
finished cells are skipped unless ``--force`` (a recorded timeout or error counts as finished).
New targets are one line in ``TARGETS``. ``benchmarks/report_e9.py`` builds the tables.

``--experiment`` (default ``e9``) names the results directory. ``e9`` holds the 0.6.0
measurements (WP 7.3): there ``embedkit_labels`` selected on all the rows. ``e9b`` is the rerun
after WP 8.2, where the default ``EmbedKit`` selects on at most ``selection_max_rows`` (50,000)
rows; ``embedkit_labels_all`` pins ``selection_max_rows=None`` (the 0.6.0 selection) for the
GPU cells.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import resource
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

EXPERIMENT = "e9"
SEED = 0
N_CLASSES = 10
CHUNK_ROWS = 50_000
CLASS_SEPARATION = 1.5  # class means: this times sqrt(spectrum) times N(0, 1)
DEFAULT_TIMEOUT = 3 * 3600


# --------------------------------------------------------------------------------------------
# targets: name -> (function(X, y) -> dict of extra facts, whether it receives the labels)
# --------------------------------------------------------------------------------------------


def _analyzer(X, y):
    from embedkit import EmbedKitAnalyzer
    report = EmbedKitAnalyzer(random_state=0).fit(X)
    return dict(suggested_target_dim=report.suggested_target_dim)


def _analyzer_fast(X, y):
    from embedkit import EmbedKitAnalyzer
    report = EmbedKitAnalyzer(random_state=0, calibrate=False, id_bootstrap=0,
                              id_scale_curve=False).fit(X)
    return dict(suggested_target_dim=report.suggested_target_dim)


def _embedkit(X, y):
    from embedkit import EmbedKit
    model = EmbedKit(random_state=0).fit(X)
    return _embedkit_info(model)


def _embedkit_labels(X, y):
    from embedkit import EmbedKit
    model = EmbedKit(random_state=0).fit(X, y)
    return _embedkit_info(model)


def _embedkit_labels_all(X, y):
    from embedkit import EmbedKit
    model = EmbedKit(random_state=0, selection_max_rows=None).fit(X, y)
    return _embedkit_info(model)


def _embedkit_info(model):
    trainer = getattr(model, "_trainer", None)
    table = getattr(model, "selection_report_", None)
    attrs = dict(table.attrs) if table is not None else {}
    return dict(method=getattr(model, "method_", None),
                target_dim=getattr(model, "_config", {}).get("target_dim"),
                device=str(getattr(trainer, "device", None)),
                selected=getattr(model, "selected_", None),
                selection_max_rows=getattr(model, "selection_max_rows", None),
                n_selection_rows=attrs.get("n_selection_rows"))


TARGETS = {
    "analyzer": (_analyzer, False),
    "analyzer_fast": (_analyzer_fast, False),
    "embedkit": (_embedkit, False),
    "embedkit_labels": (_embedkit_labels, True),
    "embedkit_labels_all": (_embedkit_labels_all, True),
}


# --------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------


def make_data(n: int, d: int, seed: int = SEED):
    """``(X float32 (n, d), y int64 (n,))``: class means plus power-law noise, drawn in chunks."""
    spectrum = 1.0 / np.arange(1, d + 1, dtype=np.float64)
    scale = np.sqrt(spectrum).astype(np.float32)
    rng = np.random.default_rng([seed, 0])
    means = (CLASS_SEPARATION * scale * rng.standard_normal((N_CLASSES, d))).astype(np.float32)
    X = np.empty((n, d), dtype=np.float32)
    y = np.empty(n, dtype=np.int64)
    for c, a in enumerate(range(0, n, CHUNK_ROWS)):
        b = min(a + CHUNK_ROWS, n)
        rng = np.random.default_rng([seed, 1, c])
        y[a:b] = rng.integers(0, N_CLASSES, b - a)
        block = rng.standard_normal((b - a, d), dtype=np.float32)
        block *= scale
        block += means[y[a:b]]
        X[a:b] = block
    return X, y


# --------------------------------------------------------------------------------------------
# measurement helpers
# --------------------------------------------------------------------------------------------


def peak_rss_bytes() -> int:
    """Peak resident set size of this process (``ru_maxrss`` is kB on Linux, bytes on macOS)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def current_rss_bytes() -> int:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    try:  # Linux
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return peak_rss_bytes()  # an upper bound: better than nothing


def n_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None

    dirty = run("status", "--porcelain", "--", "embedkit", "benchmarks/e9.py")
    return dict(commit=run("rev-parse", "HEAD"), dirty=bool(dirty) if dirty is not None else None)


def machine_info() -> dict:
    env = os.environ
    return dict(
        hostname=socket.gethostname(), platform=platform.platform(),
        cpu_count=n_cpus(), os_cpu_count=os.cpu_count(),
        omp_num_threads=env.get("OMP_NUM_THREADS"),
        openblas_num_threads=env.get("OPENBLAS_NUM_THREADS"),
        slurm_job_id=env.get("SLURM_JOB_ID"), slurm_array_task_id=env.get("SLURM_ARRAY_TASK_ID"),
        slurm_partition=env.get("SLURM_JOB_PARTITION"),
        slurm_cpus_per_task=env.get("SLURM_CPUS_PER_TASK"),
        slurm_mem_per_node=env.get("SLURM_MEM_PER_NODE"),
        slurm_mem_per_cpu=env.get("SLURM_MEM_PER_CPU"),
    )


# --------------------------------------------------------------------------------------------
# child: one cell
# --------------------------------------------------------------------------------------------


def run_child(target: str, n: int, d: int, out: Path) -> None:
    """Generate the data, fit once, write the measurements to ``out`` (JSON)."""
    record: dict = {}

    def write(final: bool = True):
        if final:
            record["peak_rss_bytes"] = peak_rss_bytes()
        out.write_text(json.dumps(record, default=_json_default))

    try:
        import torch

        import embedkit  # noqa: F401  (imported before the baseline: library memory is in it)
        fn, uses_labels = TARGETS[target]
        record.update(embedkit_version=embedkit.__version__, torch_version=torch.__version__,
                      cuda_available=bool(torch.cuda.is_available()),
                      mps_available=bool(getattr(torch.backends, "mps", None)
                                         and torch.backends.mps.is_available()),
                      torch_threads=torch.get_num_threads())
        X, y = make_data(n, d)
        record.update(x_bytes=int(X.nbytes), rss_data_bytes=current_rss_bytes(),
                      peak_rss_after_data_bytes=peak_rss_bytes())
        write(final=False)  # kept if the child is killed during the fit
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                record["info"] = fn(X, y if uses_labels else None)
            record["status"] = "ok"
        except MemoryError as e:
            record.update(status="oom", error=f"MemoryError: {e}")
        except Exception as e:
            record.update(status="error", error=f"{type(e).__name__}: {e}",
                          traceback=traceback.format_exc()[-2000:])
        record["fit_seconds"] = time.perf_counter() - t0
    except Exception as e:  # setup failed (import, data generation)
        record.update(status="error", error=f"{type(e).__name__}: {e}",
                      traceback=traceback.format_exc()[-2000:])
    write()


# --------------------------------------------------------------------------------------------
# parent: spawn, watch, record
# --------------------------------------------------------------------------------------------


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def cell_path(out_dir: Path, target: str, n: int, d: int) -> Path:
    return out_dir / f"{target}__n{n}__d{d}.json"


def run_cell(target: str, n: int, d: int, out_dir: Path, timeout: float) -> dict:
    """Run one cell in a child process and return its record (also written to ``out_dir``)."""
    record = dict(experiment=EXPERIMENT, target=target, n=n, d=d, dtype="float32", seed=SEED,
                  n_classes=N_CLASSES, timeout_seconds=timeout, x_bytes=n * d * 4,
                  date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                  **git_state(), **machine_info())
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        child_out = Path(tmp) / "child.json"
        stderr_path = Path(tmp) / "stderr.txt"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "--targets", target,
               "--n", str(n), "--d", str(d), "--child-out", str(child_out)]
        t0 = time.perf_counter()
        with open(stderr_path, "w") as err:
            proc = subprocess.Popen(cmd, stderr=err, start_new_session=True)
            try:
                proc.wait(timeout=timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)  # the child and anything it spawned
                proc.wait()
                timed_out = True
        record["wall_seconds_child"] = time.perf_counter() - t0
        record["returncode"] = proc.returncode
        child = {}
        if child_out.exists():
            try:
                child = json.loads(child_out.read_text())
            except json.JSONDecodeError:
                child = {}
        stderr_tail = stderr_path.read_text(errors="replace")[-1500:]
    record.update(child)
    if timed_out:
        record.update(status="timeout", error=f"killed after {timeout:.0f} s")
        record.pop("fit_seconds", None)
    elif proc.returncode == -signal.SIGKILL:
        record.update(status="oom_or_killed", error="child received SIGKILL (OOM killer?)")
    elif "status" not in child:
        record.update(status="error",
                      error=f"child exited with code {proc.returncode} and no result",
                      stderr_tail=stderr_tail)
    elif child["status"] != "ok":
        record["stderr_tail"] = stderr_tail
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--targets", nargs="+", default=["analyzer"], choices=sorted(TARGETS))
    # --rows and --dims are aliases: `conda run` reads --n and --d as abbreviations of its own
    # options (--name, --dev), so under `conda run` use the long names.
    p.add_argument("--n", "--rows", dest="n", nargs="+", type=int, default=[1_000, 10_000])
    p.add_argument("--d", "--dims", dest="d", nargs="+", type=int, default=[128])
    p.add_argument("--results", default=str(HERE / "results"))
    p.add_argument("--experiment", default="e9",
                   help="results subdirectory and the record's experiment field (default: e9)")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                   help="seconds per cell before the child is killed (default: 3 h)")
    p.add_argument("--force", action="store_true", help="rerun cells that already have a file")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--child-out", default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    global EXPERIMENT
    EXPERIMENT = args.experiment

    if args.child:
        run_child(args.targets[0], args.n[0], args.d[0], Path(args.child_out))
        return

    out_dir = Path(args.results) / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)
    for target in args.targets:
        for n in args.n:
            for d in args.d:
                path = cell_path(out_dir, target, n, d)
                if path.exists() and not args.force:
                    print(f"{path.name}: exists, skipped", flush=True)
                    continue
                record = run_cell(target, n, d, out_dir, args.timeout)
                path.write_text(json.dumps(record, indent=1, default=_json_default) + "\n")
                fit = record.get("fit_seconds")
                peak = record.get("peak_rss_bytes")
                print(f"{path.name}: {record['status']}"
                      + (f", fit {fit:.1f} s" if fit is not None else "")
                      + (f", peak {peak / 1e9:.2f} GB" if peak else "")
                      + (f" ({record['error'][:100]})" if record["status"] != "ok" else ""),
                      flush=True)


if __name__ == "__main__":
    main()
