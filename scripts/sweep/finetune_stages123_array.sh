#!/bin/bash
#SBATCH --account=def-spadon
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-12:00
#SBATCH --array=0-8
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --job-name=eft_s123

# Spec-weight sweep for Stages 1, 2, 3 (phase-2 finetune only).
# Produces comparable results to the Stage 4/4v2/5a 18-run sweep.
# Array mapping:
#   0-2 : Stage 1 x sw=[2.5, 5.0, 7.5]
#   3-5 : Stage 2 x sw=[2.5, 5.0, 7.5]
#   6-8 : Stage 3 x sw=[2.5, 5.0, 7.5]

module load python/3.10 scipy-stack cuda/12.2
source /home/pdoshi/eddyflow-venv/bin/activate
cd /scratch/pdoshi/my_project/eddyflow/src

STAGES=(1 1 1 2 2 2 3 3 3)
SW_VALS=(2.5 5.0 7.5 2.5 5.0 7.5 2.5 5.0 7.5)
SW_IDX=(0 1 2 0 1 2 0 1 2)

BASE_CKPTS=(
    "../outputs/checkpoints/stage1_new3_T_28_best.pt"
    "../outputs/checkpoints/stage1_new3_T_28_best.pt"
    "../outputs/checkpoints/stage1_new3_T_28_best.pt"
    "../outputs/checkpoints/stage2_new3_T_28_best.pt"
    "../outputs/checkpoints/stage2_new3_T_28_best.pt"
    "../outputs/checkpoints/stage2_new3_T_28_best.pt"
    "../outputs/checkpoints/stage3_new4_T_28_best.pt"
    "../outputs/checkpoints/stage3_new4_T_28_best.pt"
    "../outputs/checkpoints/stage3_new4_T_28_best.pt"
)

I=${SLURM_ARRAY_TASK_ID}
STAGE=${STAGES[$I]}
SW=${SW_VALS[$I]}
SIDX=${SW_IDX[$I]}
CKPT=${BASE_CKPTS[$I]}
RUN_TAG="sw_s${STAGE}_sw${SIDX}"

echo "=== Finetune Stage ${STAGE}, spec_weight=${SW}, run_tag=${RUN_TAG} ==="
echo "  Base checkpoint: ${CKPT}"

python finetune_diffusion.py \
    --stage ${STAGE} \
    --ckpt  ${CKPT} \
    --run_tag ${RUN_TAG} \
    --spec_weight ${SW} \
    --epochs 40
