"""WP 8.4 acceptance: with the report's flags, does every calibrated statistic meet the E3 bar?

    python benchmarks/check_e3_flags.py [--results benchmarks/results] [--out FILE]

Reads E3's null-family cells (``e3/summary_gauss.csv``: data drawn from Gaussians with eight
spectra, n in {500, 2,000, 10,000}, d in {32, 256, 1,024}, both metrics, five seeds). For each
cell it asks ``embedkit.analysis.report.uncalibrated_statistics`` which statistics the report
would flag there (rows of the ``*_l2`` families have a common norm). The WP 4.5 bar: for each
(family, metric, statistic), the z-scores of the cells where the statistic is *not* flagged
are inside +-3 in at least 95% of cells. Prints, and writes to ``--out``, the misses before and
after the flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from embedkit.analysis.report import (  # noqa: E402
    CALIBRATED_STATISTICS,
    uncalibrated_statistics,
)

BAR = 0.95


def table(cells: pd.DataFrame, flagged: bool) -> pd.DataFrame:
    rows = []
    for (family, metric), g in cells.groupby(["family", "metric"]):
        normalised = family.endswith("_l2")
        for stat in CALIBRATED_STATISTICS:
            keep = g
            if flagged:
                keep = g[[stat not in uncalibrated_statistics(metric, int(d), normalised)
                          for d in g["d"]]]
            if keep.empty:
                rows.append(dict(family=family, metric=metric, statistic=stat, cells=0,
                                 inside=float("nan"), meets=True, flagged_cells=len(g)))
                continue
            inside = float((keep[f"z_{stat}"].abs() <= 3).mean())
            rows.append(dict(family=family, metric=metric, statistic=stat, cells=len(keep),
                             inside=inside, meets=inside >= BAR,
                             flagged_cells=len(g) - len(keep)))
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default=str(HERE / "results"))
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    cells = pd.read_csv(Path(args.results) / "e3" / "summary_gauss.csv")
    before, after = table(cells, False), table(cells, True)
    lines = ["# WP 8.4: E3 null cells with and without the report's flags", "",
             f"Bar: z inside +-3 in at least {BAR:.0%} of the cells of each (family, metric) "
             "where the statistic is not flagged. Source: `benchmarks/check_e3_flags.py`.", ""]
    for name, t in (("Without flags", before), ("With flags", after)):
        miss = t[~t.meets]
        lines += [f"## {name}: {len(miss)} misses", ""]
        if not miss.empty:
            lines += ["| family | metric | statistic | cells | inside +-3 |",
                      "|---|---|---|---|---|"]
            lines += [f"| {r.family} | {r.metric} | {r.statistic} | {r.cells} | "
                      f"{r.inside:.2f} |" for r in miss.itertuples()]
        lines.append("")
    flagged = after[after.flagged_cells > 0]
    lines += ["## Flagged (family, metric, statistic): cells flagged of all", "",
              "| family | metric | statistic | flagged | of |", "|---|---|---|---|---|"]
    lines += [f"| {r.family} | {r.metric} | {r.statistic} | {r.flagged_cells} | "
              f"{r.flagged_cells + r.cells} |" for r in flagged.itertuples()]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
