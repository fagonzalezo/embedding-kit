# Changelog

## 0.1.2 (2026-11-08)

### Fixed
- **`SupConLoss` returned NaN for every input** (both the full and the chunked paths, with and without labels).
  The −inf diagonal of `log_prob` was multiplied by the zero diagonal of the positive mask, and
  `0 * (−inf) = NaN`. Loss values from versions ≤ 0.1.1 were NaN, so `Trainer.history["loss"]` was NaN for any
  run that used `SupConLoss`, alone or inside `CombinedLoss` (including `EmbedKit(mode="supervised")` with
  `loss="auto"`). **The gradient was correct, so embeddings trained with earlier versions are unaffected.**
- **Chunked `SupConLoss` and `RankNContrastLoss` did not match the full computation** when `chunk_size` did not
  divide `2N`: they averaged the per-chunk means, which overweighted the shorter last chunk. Both now average
  over all rows, matching the unchunked loss.
- **`Trainer` trained nothing and logged NaN when `batch_size` exceeded the number of training samples**
  (default `batch_size=256`, so any dataset under ~256 rows). With `drop_last=True` the loader yielded zero
  batches per epoch. The batch size is now capped at the number of training samples, with a `UserWarning`;
  fewer than 2 training samples raises `ValueError`.
- **Running the analysis silenced all Python warnings in the user's process.** Importing `skdim` calls
  `warnings.filterwarnings("ignore")`; EmbedKit now imports it inside `warnings.catch_warnings()` so the global
  filters are restored.
- `embedkit.__version__` now matches the package version (it reported `0.1.0`).
