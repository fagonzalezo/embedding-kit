# WP 8.4: E3 null cells with and without the report's flags

Bar: z inside +-3 in at least 95% of the cells of each (family, metric) where the statistic is not flagged. Source: `benchmarks/check_e3_flags.py`.

## Without flags: 11 misses

| family | metric | statistic | cells | inside +-3 |
|---|---|---|---|---|
| mean | cosine | distance_cv | 45 | 0.93 |
| power0.5 | cosine | k_skewness | 45 | 0.76 |
| power0.5 | cosine | antihub_ratio | 45 | 0.89 |
| power1_l2 | cosine | antihub_ratio | 45 | 0.51 |
| power1_l2 | cosine | distance_cv | 45 | 0.00 |
| power1_l2 | cosine | uniformity | 45 | 0.00 |
| power1_l2 | euclidean | antihub_ratio | 45 | 0.51 |
| power1_l2 | euclidean | distance_cv | 45 | 0.00 |
| power1_l2 | euclidean | uniformity | 45 | 0.00 |
| spiked | cosine | antihub_ratio | 45 | 0.56 |
| spiked | euclidean | neighbor_consistency | 135 | 0.79 |

## With flags: 1 misses

| family | metric | statistic | cells | inside +-3 |
|---|---|---|---|---|
| mean | cosine | distance_cv | 45 | 0.93 |

## Flagged (family, metric, statistic): cells flagged of all

| family | metric | statistic | flagged | of |
|---|---|---|---|---|
| flat | cosine | k_skewness | 45 | 45 |
| flat | cosine | antihub_ratio | 45 | 45 |
| flat | euclidean | neighbor_consistency | 45 | 135 |
| flat_l2 | cosine | k_skewness | 45 | 45 |
| flat_l2 | cosine | antihub_ratio | 45 | 45 |
| flat_l2 | cosine | distance_cv | 45 | 45 |
| flat_l2 | cosine | uniformity | 45 | 45 |
| flat_l2 | euclidean | antihub_ratio | 45 | 45 |
| flat_l2 | euclidean | distance_cv | 45 | 45 |
| flat_l2 | euclidean | neighbor_consistency | 15 | 45 |
| flat_l2 | euclidean | uniformity | 45 | 45 |
| mean | cosine | k_skewness | 45 | 45 |
| mean | cosine | antihub_ratio | 45 | 45 |
| mean | euclidean | neighbor_consistency | 45 | 135 |
| power0.5 | cosine | k_skewness | 45 | 45 |
| power0.5 | cosine | antihub_ratio | 45 | 45 |
| power0.5 | euclidean | neighbor_consistency | 45 | 135 |
| power1 | cosine | k_skewness | 45 | 45 |
| power1 | cosine | antihub_ratio | 45 | 45 |
| power1 | euclidean | neighbor_consistency | 45 | 135 |
| power1_l2 | cosine | k_skewness | 45 | 45 |
| power1_l2 | cosine | antihub_ratio | 45 | 45 |
| power1_l2 | cosine | distance_cv | 45 | 45 |
| power1_l2 | cosine | uniformity | 45 | 45 |
| power1_l2 | euclidean | antihub_ratio | 45 | 45 |
| power1_l2 | euclidean | distance_cv | 45 | 45 |
| power1_l2 | euclidean | neighbor_consistency | 15 | 45 |
| power1_l2 | euclidean | uniformity | 45 | 45 |
| power2 | cosine | k_skewness | 45 | 45 |
| power2 | cosine | antihub_ratio | 45 | 45 |
| power2 | euclidean | neighbor_consistency | 45 | 135 |
| spiked | cosine | k_skewness | 45 | 45 |
| spiked | cosine | antihub_ratio | 45 | 45 |
| spiked | euclidean | neighbor_consistency | 45 | 135 |
