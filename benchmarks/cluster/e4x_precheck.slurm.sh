#!/bin/bash
#SBATCH --job-name=embedkit-e4x-precheck
#SBATCH --output=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e4x-precheck-%j.out
#SBATCH --error=/projects/iris/fabio.gonzalezosorio/logs/embedkit-e4x-precheck-%j.err
#SBATCH --account=iris
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=24CPUNodes
#
# WP 8.3 pre-check ("WP 8.3 design and criteria" in docs/dev/IMPLEMENTATION_PLAN.md): before the 25
# extra sets are analysed together with E4's 35, rerun E4's reports (e4.py, both metrics) for one
# original set at the current commit, into $TMPDIR, and compare every numeric scalar with the E4
# file (commit 15e9a08). The bar is |z_k_skewness difference| <= 1e-6 under the Euclidean metric.
#   sbatch e4x_precheck.slurm.sh            (SET=<name> to check another set)
# Submit from this directory.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
check_clean
SET=${SET:-agnews_minilm}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun singularity exec "$CONTAINER" "$PY" "$REPO/benchmarks/e4.py" --datasets "$SET" \
  --data-root "$DATA" --out "$TMPDIR/precheck"
cat > "$TMPDIR/compare.py" <<'EOF'
import json, sys
from pathlib import Path

new_dir, old_dir, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
ok = True
for metric in ("cosine", "euclidean"):
    new = json.loads((new_dir / f"{name}__{metric}.json").read_text())
    old = json.loads((old_dir / f"{name}__{metric}.json").read_text())
    a, b = new["scalars"], old["scalars"]
    diffs = {k: abs(a[k] - b[k]) for k in a.keys() & b.keys()
             if isinstance(a[k], (int, float)) and isinstance(b[k], (int, float))
             and a[k] == a[k] and b[k] == b[k]}
    worst = sorted(diffs.items(), key=lambda kv: -kv[1])[:5]
    print(f"{metric}: {len(diffs)} numeric scalars compared; only in new: "
          f"{sorted(a.keys() - b.keys())}; only in E4: {sorted(b.keys() - a.keys())}")
    print(f"  largest differences: {worst}")
    z = [k for k in diffs if k.endswith("k_skewness_z")]
    for k in z:
        print(f"  {k}: new {a[k]!r}, E4 {b[k]!r}, |diff| {diffs[k]:.3g}")
    if metric == "euclidean":
        ok = bool(z) and all(diffs[k] <= 1e-6 for k in z)
print("PRECHECK", "PASS" if ok else "FAIL")
EOF
singularity exec "$CONTAINER" "$PY" "$TMPDIR/compare.py" "$TMPDIR/precheck/e4/_detect" \
  "$REPO/benchmarks/results/e4/_detect" "$SET"
echo "embedkit-e4x-precheck $SET done"
exit 0
