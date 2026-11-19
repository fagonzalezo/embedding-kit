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
# OpenBLAS (numpy/scipy) with as many threads as the job has CPUs makes the many small
# logistic-regression fits of the linear probe 10-20x slower (E1 raw on synthetic, 8 CPUs:
# 276 s with 8 BLAS threads, 24 s with 1, 15 s with 4; same results). Use half the CPUs.
CPUS=${SLURM_CPUS_PER_TASK:-2}
export OPENBLAS_NUM_THREADS=$(( CPUS > 1 ? CPUS / 2 : 1 ))

# Feature-extraction sets: GPU ones (images, text) and CPU ones (ANN-Benchmarks downloads).
GPU_SETS=(cifar100_dinov2 stl10_clip banking77_mpnet stsb_mpnet yelp_mpnet fluorescence_esm2)
CPU_SETS=(glove100 sift128 gist960)
# Experiment E4 (WP 7.1): 35 natural classification sets `<data>_<encoder>` (same order as
# bench_datasets.E4_SETS, images first), and the 31 of them that extract.slurm.sh builds (the four
# that exist, pets_resnet50 being a local file, are left out). Extraction array over the 31:
#   SETS=E4_GPU_SETS sbatch --array=0-30%2 extract.slurm.sh
E4_GPU_SETS=(cifar10_resnet50 cifar10_dinov2 cifar10_clip cifar100_resnet50 cifar100_clip
             stl10_resnet50 stl10_dinov2 pets_dinov2 pets_clip flowers102_resnet50
             flowers102_dinov2 flowers102_clip banking77_minilm banking77_bge banking77_e5
             emotion_mpnet emotion_minilm emotion_bge emotion_e5 massive_mpnet massive_minilm
             massive_bge massive_e5 agnews_mpnet agnews_minilm agnews_bge agnews_e5
             newsgroups_mpnet newsgroups_minilm newsgroups_bge newsgroups_e5)
E4_SETS=(cifar10_resnet50 cifar10_dinov2 cifar10_clip cifar100_resnet50 cifar100_dinov2
         cifar100_clip stl10_resnet50 stl10_dinov2 stl10_clip pets_resnet50 pets_dinov2
         pets_clip flowers102_resnet50 flowers102_dinov2 flowers102_clip banking77_mpnet
         banking77_minilm banking77_bge banking77_e5 emotion_mpnet emotion_minilm emotion_bge
         emotion_e5 massive_mpnet massive_minilm massive_bge massive_e5 agnews_mpnet
         agnews_minilm agnews_bge agnews_e5 newsgroups_mpnet newsgroups_minilm newsgroups_bge
         newsgroups_e5)
# WP 8.3: 25 extra sets with five more encoders on the same sources (bench_datasets.E4X_SETS,
# same order), for the R1 retest (e4.slurm.sh) and E10 (e10.slurm.sh); and E8's four regression
# sets for E10. Extraction:  SETS=E4X_SETS sbatch --array=0-24%2 extract.slurm.sh
E4X_SETS=(cifar10_vit cifar10_convnext cifar100_vit cifar100_convnext stl10_vit stl10_convnext
          pets_vit pets_convnext flowers102_vit flowers102_convnext banking77_gte
          banking77_distilroberta banking77_glove emotion_gte emotion_distilroberta emotion_glove
          massive_gte massive_distilroberta massive_glove agnews_gte agnews_distilroberta
          agnews_glove newsgroups_gte newsgroups_distilroberta newsgroups_glove)
E10_REG_SETS=(synthetic_regression synthetic_regression_ties yelp_mpnet fluorescence_esm2)
# Datasets of experiment E1 (one array task each).
E1_SETS=(pets_resnet50 digits synthetic_anisotropic cifar100_dinov2 stl10_clip banking77_mpnet stsb_mpnet)
# Datasets of experiment E2 (one array task each; labelled sets only).
E2_SETS=(cifar100_dinov2 pets_resnet50 banking77_mpnet synthetic_anisotropic)
# Tasks of experiment E6: a dataset, or dataset:I/N for shard I of N of its grid. Longest first.
E6_TASKS=(cifar100_dinov2:0/3 cifar100_dinov2:1/3 cifar100_dinov2:2/3 banking77_mpnet stsb_mpnet
          pets_resnet50 synthetic_anisotropic stl10_clip digits)

# Tasks of experiment E8 (phase 6). Longest first; label-efficiency sets grouped with "+".
E8_TASKS=(cifar100_dinov2:0/4 cifar100_dinov2:1/4 cifar100_dinov2:2/4 cifar100_dinov2:3/4
          yelp_mpnet fluorescence_esm2 banking77_mpnet stl10_clip pets_resnet50
          synthetic_anisotropic synthetic_regression+synthetic_regression_ties+digits
          cifar100_dinov2__lpc100+cifar100_dinov2__lpc20
          cifar100_dinov2__lpc5+cifar100_dinov2__lpc1+banking77_mpnet__lpc1+banking77_mpnet__lpc5
          banking77_mpnet__lpc20+banking77_mpnet__lpc100+stl10_clip__lpc1+stl10_clip__lpc5
          stl10_clip__lpc20+stl10_clip__lpc100)

# Datasets of experiment E7 (one array task each). Longest first.
# E8 follow-up: the residual adapter (supcon_res, rnc_res) on the main sets, after the first E8
# results showed the plain MLP ahead in regression. "@" names the methods to run.
#   TASKS=E8R_TASKS sbatch --array=0-2%2 e8.slurm.sh
E8R_TASKS=(cifar100_dinov2@supcon_res
           yelp_mpnet+fluorescence_esm2+synthetic_regression+synthetic_regression_ties@rnc_res
           banking77_mpnet+stl10_clip+pets_resnet50+synthetic_anisotropic+digits@supcon_res)
# Second round: the MLP and residual arms with few labels, regression with few training rows
# (__n<k>, whole reduced grid), and the regression MLP combined with noise views / no early stop.
#   TASKS=E8C_TASKS sbatch --array=0-5%2 e8.slurm.sh
E8C_TASKS=(yelp_mpnet+fluorescence_esm2@rnc_mlp_noise,rnc_mlp_noes
           cifar100_dinov2__lpc100+cifar100_dinov2__lpc20+cifar100_dinov2__lpc5+cifar100_dinov2__lpc1@supcon_mlp,supcon_res
           banking77_mpnet__lpc100+banking77_mpnet__lpc20+banking77_mpnet__lpc5+banking77_mpnet__lpc1@supcon_mlp,supcon_res
           synthetic_regression+synthetic_regression_ties@rnc_mlp_noise,rnc_mlp_noes
           yelp_mpnet__n200+yelp_mpnet__n1000+fluorescence_esm2__n200+fluorescence_esm2__n1000+synthetic_regression__n200+synthetic_regression__n1000+synthetic_regression_ties__n200+synthetic_regression_ties__n1000
           stl10_clip__lpc100+stl10_clip__lpc20+stl10_clip__lpc5+stl10_clip__lpc1@supcon_mlp,supcon_res)

E7_SETS=(cifar100_dinov2 banking77_mpnet stsb_mpnet pets_resnet50 synthetic_anisotropic stl10_clip digits)

# Grid of experiment E9 (WP 7.3, time and memory of fit): array task i is
# n = E9_N[i / 3], d = E9_D[i % 3] (12 tasks, 0-11).
E9_N=(1000 10000 100000 1000000)
E9_D=(128 1024 4096)
# E9b (WP 8.2): the rerun after selection on a subsample; 50,000 is the default
# selection_max_rows. Chosen with E9_NS=E9B_N (9 tasks, 0-8) or E9_NS=E9B_GPU_N (3 tasks, 0-2).
E9B_N=(50000 100000 1000000)
E9B_GPU_N=(1000000)

check_clean() {
  # Results must match the commit recorded in each result file.
  if [ -n "$(git -C "$REPO" status --porcelain -- embedkit benchmarks pyproject.toml | grep -v 'benchmarks/results/')" ]; then
    echo "dirty tree in $REPO: results would not match the recorded commit"; git -C "$REPO" status --short; exit 1
  fi
  echo "repo $(git -C "$REPO" rev-parse --short HEAD) clean"
}
