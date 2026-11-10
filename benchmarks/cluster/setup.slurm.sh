#!/bin/bash
#SBATCH --job-name=embedkit-setup
#SBATCH --output=/projects/iris/fabio.gonzalezosorio/logs/embedkit-setup-%j.out
#SBATCH --error=/projects/iris/fabio.gonzalezosorio/logs/embedkit-setup-%j.err
#SBATCH --account=iris
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=24CPUNodes
#
# Builds the venv over the container's torch/numpy/scipy/sklearn and installs embedkit (editable)
# plus the feature-extraction packages. Idempotent. Submit from this directory:
#   sbatch setup.slurm.sh
set -uo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
[ -x "$PY" ] || singularity exec "$CONTAINER" virtualenv --system-site-packages "$VENV"
PIP="singularity exec $CONTAINER $VENV/bin/pip install --no-cache-dir"
failed=()
# embedkit's own dependencies, except torch (the container's build must not be replaced),
# one per call so one failure does not block the rest.
for pkg in "scikit-dimension>=0.3" "umap-learn>=0.5" "seaborn>=0.13" "pandas>=2.0" pytest \
           transformers sentence-transformers datasets h5py huggingface_hub; do
  echo "== pip install $pkg"; $PIP "$pkg" || failed+=("$pkg")
done
$PIP --no-deps -e "$REPO" || failed+=("embedkit")
echo "failed: ${failed[*]:-none}"
singularity exec "$CONTAINER" "$PY" - <<'PY'
import importlib
for m in ("torch", "torchvision", "numpy", "scipy", "sklearn", "pandas", "skdim", "umap",
          "transformers", "sentence_transformers", "datasets", "h5py", "embedkit"):
    try:
        mod = importlib.import_module(m)
        print(f"OK   {m:22s} {getattr(mod, '__version__', '')}  {mod.__file__}")
    except Exception as e:
        print(f"FAIL {m:22s} {type(e).__name__}: {e}")
PY
echo "embedkit-setup done"
exit 0
