#!/bin/bash
#SBATCH --job-name=seeded_ablation
#SBATCH --output=slurm/logs/seeded_ablation_%A_%a.out
#SBATCH --error=slurm/logs/seeded_ablation_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sphill86@charlotte.edu
#SBATCH --array=0-89%10

# One array task = one (model, seed, target_var) triple, 6 models x 5 seeds x
# 3 targets = 90 tasks total. %10 caps 10 concurrent tasks; raise/lower to
# match your partition's fair-share/QOS limits.
#
# Models: cnn (RawCNNProbe), cnn_scalar (RawCNNScalarProbe), transformer
# (RawTransformerProbe), transformer_scalar (RawTransformerScalarProbe),
# fourierbgc (FourierBGC_Broadcast_NoYear), fourierbgc_with_year
# (FourierBGCBroadcast) -- the same --model names train.py/evaluate.py
# already use, no renaming.
#
# Seeds 0-4, same seeding.py infrastructure used for the earlier
# FourierBGCBroadcast-only sweep: --seed N sets python/numpy/torch RNG and
# the train DataLoader's shuffle generator.
#
# Checkpoints land at ~/seeded_ablation_results/{model}/seed{N}/{VAR}/model/
# on this cluster's home dir -- separate from results/ (unseeded baselines)
# and from any local-machine ~/seeded_ablation_results or
# ~/fourierbgc_with_year_results_seed* dirs, which live on a different
# machine and are not touched by this.

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

MODELS=(cnn cnn_scalar transformer transformer_scalar fourierbgc fourierbgc_with_year)
SEEDS=(0 1 2 3 4)
TARGETS=(NITRATE CHLA BBP700)

N_TARGETS=${#TARGETS[@]}
N_SEEDS=${#SEEDS[@]}

IDX=$SLURM_ARRAY_TASK_ID
TARGET_I=$(( IDX % N_TARGETS ))
SEED_I=$(( (IDX / N_TARGETS) % N_SEEDS ))
MODEL_I=$(( IDX / (N_TARGETS * N_SEEDS) ))

MODEL=${MODELS[$MODEL_I]}
SEED=${SEEDS[$SEED_I]}
VAR=${TARGETS[$TARGET_I]}

echo "task $IDX -> model=$MODEL seed=$SEED target=$VAR"

SAVE_DIR=~/seeded_ablation_results/${MODEL}/seed${SEED}/${VAR}/model
mkdir -p "$SAVE_DIR"

python -u train.py --model $MODEL --target_var $VAR --epochs 100 --seed $SEED \
    --save_dir "$SAVE_DIR"

python -u evaluate.py --model $MODEL --target_var $VAR \
    --checkpoint "$SAVE_DIR/best.pt" \
    > ~/seeded_ablation_results/${MODEL}/seed${SEED}/${VAR}/eval_best.txt

echo "======================================================"
echo "task $IDX done. End Time : $(date)"
echo "======================================================"
