#!/bin/bash
#SBATCH --account=def-spadon          
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-12:00
#SBATCH --output=slurm_logs/%x-%j.out


# Load modules
module load python/3.10 scipy-stack cuda/12.2

# Activate the venv created in /home (not /scratch)
source /home/pdoshi/eddyflow-venv/bin/activate

# Change to your project directory on /scratch
cd /scratch/pdoshi/my_project/eddyflow/src

# Run your training script
python evaluate_v2.py --stage 4v2 --ckpt ../outputs/checkpoints/stage4v2_try2_merged.pt --psd_curves --save_examples