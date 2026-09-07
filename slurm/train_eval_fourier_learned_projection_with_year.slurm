#!/bin/bash
#SBATCH --job-name=flp_with_year
#SBATCH --output=slurm/logs/flp_with_year_%A_%a.out
#SBATCH --error=slurm/logs/flp_with_year_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sphill86@charlotte.edu
#SBATCH --array=0-14%5

# One array task = one (seed, target_var) pair for
# fourier_learned_projection/model.py's FourierBGC (learned-projection,
# with-year) -- the model that isolates FourierBGC_Broadcast_NoYear's
# depth-constant Fourier fusion vs. PPCon's depth-varying fusion,
# independent of learned-vs-fixed, with year included (matching
# FourierBGCBroadcast, the locked comparison baseline in decisions.md/the
# paper). 5 seeds x 3 targets = 15 tasks.
# %5 caps 5 concurrent tasks (lowered from an initial %10 after that run
# hit GPU/node contention badly enough that most CHLA/BBP700 tasks got
# SIGKILLed at the 3-hour wall-time limit -- this model is tiny and should
# train in minutes on an unshared GPU, so >3h strongly suggests too many
# concurrent tasks fighting over the same node/GPU rather than genuine
# per-task cost. --time bumped 03:00:00 -> 08:00:00 as a safety margin on
# top of the concurrency fix, not a replacement for it. Raise/lower %N to
# match your partition's fair-share/QOS limits, same convention as
# train_eval_seeded_ablation.slurm.
#
# This mirrors train_eval_seeded_ablation.slurm's array structure but
# stays a separate script/job since it's a new, unlocked exploratory model,
# not part of that ablation spine's six models.
#
# Checkpoints land at ~/fourier_learned_projection_with_year_results/
# {VAR}/seed{N}/ on this cluster's home dir -- separate from
# fourier_learned_projection/results_with_year/ (this machine's local CHLA-
# only run) and from ~/fourierbgc_with_year_results or
# ~/seeded_ablation_results, which are different models entirely.

echo "======================================================"
echo "Start Time    : $(date)"
echo "Submit Dir    : $SLURM_SUBMIT_DIR"
echo "Job ID/Name   : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "Array Task ID : $SLURM_ARRAY_TASK_ID"
echo "Partition     : $SLURM_JOB_PARTITION"
echo "======================================================"

set -euo pipefail
cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

SEEDS=(0 1 2 3 4)
TARGETS=(NITRATE CHLA BBP700)

N_TARGETS=${#TARGETS[@]}
N_SEEDS=${#SEEDS[@]}

IDX=$SLURM_ARRAY_TASK_ID
TARGET_I=$(( IDX % N_TARGETS ))
SEED_I=$(( (IDX / N_TARGETS) % N_SEEDS ))

SEED=${SEEDS[$SEED_I]}
VAR=${TARGETS[$TARGET_I]}

echo "task $IDX -> seed=$SEED target=$VAR"

SAVE_DIR=~/fourier_learned_projection_with_year_results/${VAR}/seed${SEED}

python -u fourier_learned_projection/run_train_with_year.py \
    --target_var $VAR --epochs 100 --seed $SEED --save_dir "$SAVE_DIR"

python -u fourier_learned_projection/run_evaluate_with_year.py \
    --target_var $VAR --checkpoint "$SAVE_DIR/best.pt" \
    > "$SAVE_DIR/eval_best.txt"

echo "======================================================"
echo "task $IDX done. End Time : $(date)"
echo "======================================================"
