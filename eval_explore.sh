#!/bin/bash
#SBATCH --account=def-spadon
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=1:00:00
#SBATCH --job-name=explore
#SBATCH --output=slurm_logs/explore_%j.out

module load StdEnv/2023 gcc/12.3 proj
module load python/3.10 scipy-stack cuda/12.2
export PROJ_DATA=$EBROOTPROJ/share/proj
export PROJ_LIB=$PROJ_DATA
source /home/pdoshi/eddyflow-venv/bin/activate
cd /scratch/pdoshi/my_project/eddyflow

echo "========================================"
echo "STEP 1: Bar chart variants (no GPU needed)"
echo "========================================"
python scripts/explore_bar_charts.py

echo ""
echo "========================================"
echo "STEP 2: Qualitative figure variants"
echo "        (4 dates × 3 layouts = 12 figs)"
echo "========================================"
python scripts/explore_qualitative.py \
    --ckpt outputs/checkpoints/stage4_sweep_4_123_lr0_sw0_merged.pt

echo ""
echo "========================================"
echo "All explore figures saved to:"
echo "  outputs/figures/explore/bars/"
echo "  outputs/figures/explore/qualitative/"
echo "========================================"
