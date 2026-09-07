#!/bin/bash
#SBATCH --job-name=ppcon_no_scalar_chla
#SBATCH --output=slurm/logs/ppcon_no_scalar_chla_%j.out
#SBATCH --error=slurm/logs/ppcon_no_scalar_chla_%j.err
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

source .venv-ppcon/bin/activate

python -u ppcon_no_scalar/run.py --variable CHLA --epochs 150

python -u evaluate.py --model ppcon_no_scalar --target_var CHLA \
    --checkpoint_dir $HOME/ppcon_results_no_scalar/CHLA/model --epoch 150

echo "======================================================"
echo "End Time : $(date)"
echo "======================================================"
