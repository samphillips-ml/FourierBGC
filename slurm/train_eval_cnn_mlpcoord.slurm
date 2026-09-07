#!/bin/bash
#SBATCH --job-name=cnn_mlpcoord
#SBATCH --output=slurm/logs/cnn_mlpcoord_%A_%a.out
#SBATCH --error=slurm/logs/cnn_mlpcoord_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sphill86@charlotte.edu
#SBATCH --array=0-14%10

# CNN-MLPCoord: PPCon's four learned MLP encoders (1->80->140->200, SELU,
# 158,800 params, byte-identical in shape to ppcon_baseline's mlp.py) under
# the modern training recipe. This fills the one empty cell in the ablation
# grid -- every other modern-recipe model can be compared with the recipe
# held fixed, but PPCon's own encoding previously could not.
#
# 1 model x 5 seeds x 3 targets = 15 tasks. Same layout, epochs, seeds and
# save-dir convention as train_eval_seeded_ablation.slurm, so the existing
# aggregation scripts can read these results unchanged.
#
# Scalars are fed RAW and unnormalized, matching PPCon exactly (their code
# normalizes nothing, anywhere). If a run diverges, rerun that cell with
# --model cnn_mlpcoord_norm, which z-scores them first -- but report the
# divergence rather than silently substituting.

echo "======================================================"
echo "Start Time    : $(date)"
echo "Submit Dir    : $SLURM_SUBMIT_DIR"
echo "Job ID/Name   : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "Array Task ID : $SLURM_ARRAY_TASK_ID"
echo "Partition     : $SLURM_JOB_PARTITION"
echo "======================================================"

MODEL=cnn_mlpcoord
SEEDS=(0 1 2 3 4)
TARGETS=(NITRATE CHLA BBP700)

N_TARGETS=${#TARGETS[@]}

IDX=$SLURM_ARRAY_TASK_ID
TARGET_I=$(( IDX % N_TARGETS ))
SEED_I=$(( IDX / N_TARGETS ))

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
