#!/bin/bash
#SBATCH --account=def-spadon
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-03:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --job-name=wenhai_corr

# WenHai evaluation with ERA5-SST persistence denominator (shortcut 1).
# Evaluates GSL, BOF, GOM from a single global inference pass.
# IC: sample_GLORYS_23lev.nc (2019-01-01), target: 2019-01-02.
#
# Outputs:
#   results_wenhai_corrected_gsl.json
#   results_wenhai_corrected_bof.json
#   results_wenhai_corrected_gom.json

module load python/3.10 scipy-stack cuda/12.2
source /home/pdoshi/eddyflow-venv/bin/activate
cd /scratch/pdoshi/my_project/eddyflow

python scripts/eval_wenhai_corrected.py \
    --onnx          scripts/WenHai.onnx \
    --sample_glorys scripts/sample_GLORYS_23lev.nc \
    --output_prefix results_wenhai_corrected

echo "Done."
