#!/bin/bash
#SBATCH --account=def-spadon
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-02:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --job-name=era5_baseline

# Compute ERA5 SST bilinear baseline PSD ratio over 2022-2023 test set.
# Output: results_era5_sst_baseline.json in project root.

module load python/3.10 scipy-stack cuda/12.2
source /home/pdoshi/eddyflow-venv/bin/activate
cd /scratch/pdoshi/my_project/eddyflow

python scripts/eval_era5_sst_baseline.py \
    --output results_era5_sst_baseline.json

echo "Done. Output: results_era5_sst_baseline.json"
