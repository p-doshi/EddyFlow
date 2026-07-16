#!/bin/bash
#SBATCH --account=def-spadon
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --job-name=qual_fig
#SBATCH --output=slurm_logs/qual_fig_%j.out

module load StdEnv/2023 gcc/12.3 proj
module load python/3.10 scipy-stack cuda/12.2
export PROJ_DATA=$EBROOTPROJ/share/proj
export PROJ_LIB=$PROJ_DATA
source /home/pdoshi/eddyflow-venv/bin/activate
cd /scratch/pdoshi/my_project/eddyflow

python scripts/make_qualitative_figure.py \
    --ckpt outputs/checkpoints/stage4_sweep_4_123_lr0_sw0_merged.pt \
    --date 2022-09-01 \
    --baseline_only
