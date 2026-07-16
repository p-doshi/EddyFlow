#!/bin/bash
#SBATCH --account=def-spadon
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --job-name=sst_anim
#SBATCH --output=slurm_logs/sst_anim_%j.out

module load StdEnv/2023 python/3.10 scipy-stack
source /home/pdoshi/eddyflow-venv/bin/activate
cd /scratch/pdoshi/my_project/eddyflow

python scripts/make_sst_animation.py \
    --domain gsl \
    --start 2021-06-01 \
    --end 2022-11-30 \
    --months 6,7,8,9,10,11 \
    --every 4 \
    --fps 12 \
    --stride 3 \
    --dpi 90 \
    --out outputs/figures/paper/sst_animation.gif
