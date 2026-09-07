#!/bin/bash
#SBATCH --job-name=ppcon_no_scalar_seeded
#SBATCH --output=slurm/logs/ppcon_no_scalar_seeded_%A_%a.out
#SBATCH --error=slurm/logs/ppcon_no_scalar_seeded_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sphill86@charlotte.edu
#SBATCH --array=0-14%10

# One array task = one (seed, target_var) pair for PPConNoScalar
# (manuscript name "PPCon-NoCoord"), 5 seeds x 3 targets = 15 tasks.
# Epoch counts match the existing single-run precedent for this model
# (slurm/train_eval_ppcon_no_scalar_{nitrate,chla}.slurm, and the BBP700
# epoch count read back from the max model_conv_*.pt snapshot already on
# disk at ~/ppcon_results_no_scalar/BBP700/model/): NITRATE=100, CHLA=150,
# BBP700=125. Same Adadelta/three-term-loss recipe as always -- this model
# was never on the Adam/warmup/cosine-decay recipe used by train.py's
# models, and we're not changing that here, just adding real seeding on
# top of it.
#
# ppcon_no_scalar/run.py's --seed now sets python/numpy/torch RNG and the
# train+val DataLoader shuffle generators (see seeding.py and the module
# docstring note in run.py: the old hardcoded random.seed(123) only ever
# seeded python's random module, never torch, so it never actually
# controlled weight init or batch order -- this is the real fix).
#
# Checkpoints land at
# ~/seeded_ablation_results/ppcon_no_scalar/seed{N}/{VAR}/model/model_conv_{epoch}.pt
# -- separate from ~/ppcon_results_no_scalar/ (the original unseeded run)
# and from everything else in ~/seeded_ablation_results/ (the other 6
# models' seeded sweep).

echo "======================================================"
echo "Start Time    : $(date)"
echo "Submit Dir    : $SLURM_SUBMIT_DIR"
echo "Job ID/Name   : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "Array Task ID : $SLURM_ARRAY_TASK_ID"
echo "Partition     : $SLURM_JOB_PARTITION"
echo "======================================================"

set -euo pipefail
cd $SLURM_SUBMIT_DIR
source .venv-ppcon/bin/activate

SEEDS=(0 1 2 3 4)
TARGETS=(NITRATE CHLA BBP700)
declare -A EPOCHS_FOR_VAR=( [NITRATE]=100 [CHLA]=150 [BBP700]=125 )

N_TARGETS=${#TARGETS[@]}

IDX=$SLURM_ARRAY_TASK_ID
TARGET_I=$(( IDX % N_TARGETS ))
SEED_I=$(( IDX / N_TARGETS ))

SEED=${SEEDS[$SEED_I]}
VAR=${TARGETS[$TARGET_I]}
EPOCHS=${EPOCHS_FOR_VAR[$VAR]}

echo "task $IDX -> ppcon_no_scalar seed=$SEED target=$VAR epochs=$EPOCHS"

SAVE_DIR=~/seeded_ablation_results/ppcon_no_scalar/seed${SEED}/${VAR}
mkdir -p "$SAVE_DIR"

python -u ppcon_no_scalar/run.py --variable $VAR --epochs $EPOCHS --seed $SEED \
    --save_dir "$SAVE_DIR"

python -u evaluate.py --model ppcon_no_scalar --target_var $VAR \
    --checkpoint_dir "$SAVE_DIR/model" --epoch $EPOCHS \
    > "$SAVE_DIR/eval_best.txt"

echo "======================================================"
echo "task $IDX done. End Time : $(date)"
echo "======================================================"
