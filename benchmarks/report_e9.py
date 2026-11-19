"""Experiment E9 tables (WP 7.3): time and memory of ``fit`` over n and d

    python benchmarks/report_e9.py [--results benchmarks/results] [--experiment e9b]

Reads the cell files written by ``benchmarks/e9.py`` (``<results>/<experiment>/*.json``,
``e9`` by default) and writes
``REPORT.md`` (one table per target: rows n, columns d, each cell "wall time of fit / peak
resident memory of the process in GB (1 GB = 1e9 bytes)", or the status when the cell did not
finish) and ``summary_<experiment>.csv`` (one row per cell, every scalar field).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
STATUS_LABEL = {"timeout": "timeout", "oom_or_killed": "OOM / killed", "oom": "OOM",
                "error": "error"}


def load_cells(e9_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(e9_dir.glob("*__n*__d*.json")):
        try:
            rec = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"[skip] unreadable {path.name}", file=sys.stderr)
            continue
        info = rec.pop("info", None) or {}
        rec.update({f"info_{k}": v for k, v in info.items()})
        rec.pop("traceback", None)
        rec.pop("stderr_tail", None)
        rows.append(rec)
    return pd.DataFrame(rows)


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def fmt_bytes(b: float) -> str:
    return f"{b / 1e9:.2f} GB" if b >= 1e9 else f"{b / 1e6:.1f} MB"


def fmt_cell(row) -> str:
    status = row["status"]
    if status == "ok":
        return f"{fmt_time(row['fit_seconds'])} / {row['peak_rss_bytes'] / 1e9:.1f} GB"
    label = STATUS_LABEL.get(status, str(status))
    if status == "timeout" and pd.notna(row.get("timeout_seconds")):
        label += f" (> {fmt_time(row['timeout_seconds'])})"
    return label


def table(df: pd.DataFrame, target: str) -> str:
    sub = df[df.target == target]
    ns, ds = sorted(sub.n.unique()), sorted(sub.d.unique())
    lines = ["| n \\ d | " + " | ".join(f"{d:,}" for d in ds) + " |",
             "|---|" + "---|" * len(ds)]
    for n in ns:
        cells = []
        for d in ds:
            hit = sub[(sub.n == n) & (sub.d == d)]
            cells.append(fmt_cell(hit.iloc[0]) if len(hit) else "-")
        lines.append(f"| {n:,} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def distinct(df: pd.DataFrame, column: str) -> list:
    if column not in df:
        return []
    return sorted({str(v) for v in df[column].dropna().unique()})


def machine_note(df: pd.DataFrame) -> str:
    parts = []
    for label, col in (("host", "hostname"), ("platform", "platform"),
                       ("CPUs available to the process", "cpu_count"),
                       ("OMP_NUM_THREADS", "omp_num_threads"),
                       ("torch threads", "torch_threads"),
                       ("Slurm partition", "slurm_partition"),
                       ("Slurm CPUs per task", "slurm_cpus_per_task"),
                       ("Slurm memory per node (MB)", "slurm_mem_per_node"),
                       ("CUDA available", "cuda_available"), ("MPS available", "mps_available"),
                       ("embedkit", "embedkit_version"), ("commit", "commit"),
                       ("torch", "torch_version")):
        vals = distinct(df, col)
        if col == "commit":
            vals = [v[:8] for v in vals]
        if vals:
            parts.append(f"- {label}: {', '.join(vals)}")
    return "\n".join(parts)


TITLES = {"e9": "E9: time and memory of `fit` (WP 7.3)",
          "e9b": "E9b: time and memory of `fit` after WP 8.2 (selection on a subsample), CPU",
          "e9b_gpu": "E9b: time and memory of `fit` after WP 8.2, GPU"}


def render(df: pd.DataFrame, experiment: str = "e9") -> str:
    out = [f"# {TITLES.get(experiment, experiment)}", "",
           "Each cell is one `fit` in a fresh process on synthetic float32 data (10 classes, "
           "power-law spectrum, generated in chunks; `benchmarks/e9.py`). Entries are "
           "\"wall time of `fit` / peak resident memory of the process\" (GB = 1e9 bytes). The "
           "peak includes the interpreter and torch (a few hundred MB), the data array "
           "(n d x 4 bytes) and everything `fit` allocates. A cell that is not `ok` shows its "
           "status; the messages are listed below the tables.", ""]
    for target in sorted(df.target.unique()):
        out += [f"## {target}", "", table(df, target), ""]
    sizes = df.drop_duplicates(["n", "d"]).sort_values(["n", "d"])
    out += ["## Size of X (float32)", "",
            "| n \\ d | " + " | ".join(f"{d:,}" for d in sorted(df.d.unique())) + " |",
            "|---|" + "---|" * df.d.nunique()]
    for n in sorted(df.n.unique()):
        cells = []
        for d in sorted(df.d.unique()):
            hit = sizes[(sizes.n == n) & (sizes.d == d)]
            cells.append(fmt_bytes(hit.iloc[0]["x_bytes"]) if len(hit) else "-")
        out.append(f"| {n:,} | " + " | ".join(cells) + " |")
    out += ["", "## Machine", "", machine_note(df), ""]
    bad = df[df.status != "ok"]
    if len(bad):
        out += ["## Cells that did not finish", ""]
        for _, r in bad.sort_values(["target", "n", "d"]).iterrows():
            msg = str(r.get("error", "")).splitlines()[0][:200] if pd.notna(r.get("error")) else ""
            out.append(f"- `{r['target']}`, n = {r['n']:,}, d = {r['d']:,}: {r['status']}, {msg}")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default=str(HERE / "results"))
    p.add_argument("--experiment", default="e9", help="results subdirectory (e9, e9b)")
    p.add_argument("--out", default=None, help="default: <results>/<experiment>")
    args = p.parse_args(argv)
    e9_dir = Path(args.results) / args.experiment
    out = Path(args.out or e9_dir)
    df = load_cells(e9_dir)
    if df.empty:
        sys.exit(f"No E9 results under {e9_dir}")
    out.mkdir(parents=True, exist_ok=True)
    df.sort_values(["target", "n", "d"]).to_csv(out / f"summary_{args.experiment}.csv",
                                                index=False)
    (out / "REPORT.md").write_text(render(df, args.experiment) + "\n")
    print(f"wrote {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
