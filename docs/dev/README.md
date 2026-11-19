# Development records

These documents record how EmbedKit went from 0.1.2 to 0.7.0 (2026-11-10 to 2026-11-19). They are
kept for provenance and are not maintained. For the library as it is, read `docs/manual.md`,
`docs/MIGRATION.md` and `CHANGELOG.md`; for the evidence behind the defaults, `docs/benchmarks/`.

| Document | What it is |
|---|---|
| [`CRITICAL_REVIEW.md`](CRITICAL_REVIEW.md) | Review of 0.1.2: defects D1 to D30, the design critique and the experiments E1 to E9. The changelog and the benchmark reports use its D and E numbers. |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | The phased plan that followed from the review (phases 0 to 8, work packages "WP"). It holds the acceptance criteria of each experiment, written before the runs, and the outcome notes; `benchmarks/report_*.py` and `benchmarks/results/*/REPORT.md` cite it. |
| [`UNSUPERVISED_REFINEMENT_REPORT.md`](UNSUPERVISED_REFINEMENT_REPORT.md) | The pre-review pilot on self-supervised refinement. Superseded by E1 and E2 (`docs/benchmarks/BENCHMARK_E1_E2.md`). |

Paths inside these documents were updated when they moved here; nothing else was edited. Two things
they mention are not in the repository:

- `docs/BUG-rnc-ties.md` and `docs/BUG-supcon-nan.md` were private bug notes. Both bugs are fixed:
  the SupCon NaN in 0.1.2, the Rank-N-Contrast ties in 0.1.3 (D21; see `CHANGELOG.md`, with a
  regression test).
- The branch `wip/sphere-null` (a moment-matched sphere null for normalised rows) failed its bar and
  was not kept; the outcome is in `docs/benchmarks/BENCHMARK_DIAGNOSTICS.md`, Limitations.
