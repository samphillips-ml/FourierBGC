#!/bin/bash
#SBATCH --job-name=final_ablation
#SBATCH --output=slurm/logs/final_ablation_%A_%a.out
#SBATCH --error=slurm/logs/final_ablation_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sphill86@charlotte.edu
#SBATCH --array=0-74%10

# Single-provenance rerun of every seeded model in the manuscript's ablation
# spine except CNN-MLPCoord, which was run cleanly under its own job and is
# not repeated here.
#
#   5 models x 5 seeds x 3 targets = 75 tasks.
#
# The point of running these together is that every number in the results
# table then comes from one submission, one job ID, and one code state.
# The previous results were assembled from four directories of different
# vintages, one of which had a near-identical duplicate differing only by a
# trailing " 2"; that folder held the complete run while its twin held a
# partial one, with nothing to distinguish them but the name.
#
# Three entry points are involved, so the script dispatches per model:
#   cnn, cnn_scalar, fourierbgc_with_year  -> train.py            (.venv)
#   fourier_learned_projection             -> run_train_with_year (.venv)
#   ppcon_no_scalar                        -> ppcon_no_scalar/run (.venv-ppcon)
#
# PPCon-NoCoord additionally uses PPCon's own per-variable epoch budget
# (100/150/125), matching its published configuration, rather than the 100
# epochs used by the common recipe.
#
# Everything lands under ~/final_ablation_results/{model}/seed{N}/{VAR}/.

echo "======================================================"
echo "Start Time    : $(date)"
echo "Submit Dir    : $SLURM_SUBMIT_DIR"
echo "Job ID/Name   : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "Array Task ID : $SLURM_ARRAY_TASK_ID"
echo "Partition     : $SLURM_JOB_PARTITION"
echo "Git commit    : $(cd $SLURM_SUBMIT_DIR && git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
echo "======================================================"

set -euo pipefail
cd $SLURM_SUBMIT_DIR

MODELS=(cnn cnn_scalar fourierbgc_with_year fourier_learned_projection ppcon_no_scalar)
SEEDS=(0 1 2 3 4)
TARGETS=(NITRATE CHLA BBP700)
declare -A PPCON_EPOCHS=( [NITRATE]=100 [CHLA]=150 [BBP700]=125 )

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

ROOT=~/final_ablation_results/${MODEL}/seed${SEED}/${VAR}
SAVE_DIR=${ROOT}/model
mkdir -p "$SAVE_DIR"

if [ "$MODEL" = "ppcon_no_scalar" ]; then
    source .venv-ppcon/bin/activate
    EPOCHS=${PPCON_EPOCHS[$VAR]}
    echo "  (PPCon recipe, $EPOCHS epochs)"
    python -u ppcon_no_scalar/run.py --variable $VAR --epochs $EPOCHS --seed $SEED \
        --save_dir "$ROOT"
    python -u evaluate.py --model ppcon_no_scalar --target_var $VAR \
        --checkpoint_dir "$ROOT/model" --epoch $EPOCHS \
        > "$ROOT/eval_best.txt"

elif [ "$MODEL" = "fourier_learned_projection" ]; then
    source .venv/bin/activate
    python -u fourier_learned_projection/run_train_with_year.py \
        --target_var $VAR --epochs 100 --seed $SEED --save_dir "$SAVE_DIR"
    python -u fourier_learned_projection/run_evaluate_with_year.py \
        --target_var $VAR --checkpoint "$SAVE_DIR/best.pt" \
        > "$ROOT/eval_best.txt"

else
    source .venv/bin/activate
    python -u train.py --model $MODEL --target_var $VAR --epochs 100 --seed $SEED \
        --save_dir "$SAVE_DIR"
    python -u evaluate.py --model $MODEL --target_var $VAR \
        --checkpoint "$SAVE_DIR/best.pt" \
        > "$ROOT/eval_best.txt"
fi

echo "======================================================"
echo "task $IDX done. End Time : $(date)"
echo "======================================================"
