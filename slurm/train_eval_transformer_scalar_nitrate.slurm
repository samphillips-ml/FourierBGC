#!/bin/bash
#SBATCH --job-name=argo_former_transformer_scalar_nitrate
#SBATCH --output=slurm/logs/transformer_scalar_nitrate_%j.out
#SBATCH --error=slurm/logs/transformer_scalar_nitrate_%j.err
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sphill86@charlotte.edu

echo "======================================================"
echo "Start Time  : $(date)"
echo "Submit Dir  : $SLURM_SUBMIT_DIR"
echo "Job ID/Name : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "Partition   : $SLURM_JOB_PARTITION"
echo "======================================================"

set -euo pipefail
cd $SLURM_SUBMIT_DIR

mkdir -p results/NITRATE/transformer_scalar
source .venv/bin/activate

python -u train.py --model transformer_scalar --target_var NITRATE --epochs 100

python -u evaluate.py --model transformer_scalar --target_var NITRATE \
    --checkpoint results/NITRATE/transformer_scalar/best.pt

echo "======================================================"
echo "End Time : $(date)"
echo "======================================================"
