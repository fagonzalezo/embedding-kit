# Shared settings for the embedding-kit benchmark jobs on OCCIDATA (sourced by every *.slurm.sh).
# The cluster rules these follow are summarised in benchmarks/README.md ("On the cluster").
REPO=$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)   # the clone the job was submitted from
BASE=/projects/iris/fabio.gonzalezosorio
CONTAINER=/apps/containerCollections/CUDA13/pytorch-NGC-26-02.sif
VENV=$BASE/envs/embedkit
PY=$VENV/bin/python
DATA=$BASE/data/embedkit-bench          # datasets and cached features (--data-root)
export EMBEDKIT_BENCH_DATA=$DATA
export HF_HOME=$BASE/hf_cache TORCH_HOME=$BASE/torch-home XDG_CACHE_HOME=$BASE/cache
export PIP_CACHE_DIR=$BASE/cache/pip
export PYTHONNOUSERSITE=1 TQDM_DISABLE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export TMPDIR=$BASE/tmp/embedkit_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$TMPDIR" "$DATA" && chmod 700 "$TMPDIR"

# Feature-extraction sets: GPU ones (images, text) and CPU ones (ANN-Benchmarks downloads).
GPU_SETS=(cifar100_dinov2 stl10_clip banking77_mpnet stsb_mpnet)
CPU_SETS=(glove100 sift128 gist960)
# Datasets of experiment E1 (one array task each).
E1_SETS=(pets_resnet50 digits synthetic_anisotropic cifar100_dinov2 stl10_clip banking77_mpnet stsb_mpnet)

check_clean() {
  # Results must match the commit recorded in each result file.
  if [ -n "$(git -C "$REPO" status --porcelain -- embedkit benchmarks pyproject.toml | grep -v 'benchmarks/results/')" ]; then
    echo "dirty tree in $REPO: results would not match the recorded commit"; git -C "$REPO" status --short; exit 1
  fi
  echo "repo $(git -C "$REPO" rev-parse --short HEAD) clean"
}
