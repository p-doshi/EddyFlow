#!/bin/bash
#SBATCH --account=def-spadon
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-12:00
#SBATCH --output=slurm_logs/eval_gom_stage5a-%j.out
#SBATCH --job-name=ef_gom_s5a

module load python/3.10 scipy-stack cuda/12.2
source /home/pdoshi/eddyflow-venv/bin/activate

CKPT=/scratch/pdoshi/my_project/eddyflow/outputs/checkpoints
EVAL=/scratch/pdoshi/my_project/eddyflow/src/eval_fixed

cd /scratch/pdoshi/my_project/eddyflow/src

echo "========================================================"
echo "GOM zero-shot evaluation — Stage 5a"
echo "Start: $(date)"
echo "========================================================"

if [ ! -f "$EVAL/eval_gom_stage5a_sw0lr0.json" ]; then
    python eval_gom.py \
        --ckpt   "$CKPT/stage5a_sweep_5a_0_lr0_sw0_merged.pt" \
        --stage  5a \
        --output "$EVAL/eval_gom_stage5a_sw0lr0.json" \
        --psd_curves --save_examples
else
    echo "SKIP: eval_gom_stage5a_sw0lr0.json already exists"
fi

echo "Done: $(date)"
