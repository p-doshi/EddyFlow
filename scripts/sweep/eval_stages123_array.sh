#!/bin/bash
#SBATCH --account=def-spadon
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-08:00
#SBATCH --array=0-26
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --job-name=eval_s123

# Evaluate Stage 1/2/3 spec-weight sweep checkpoints on GSL + BOF + GOM.
# Depends on finetune_stages123_array completing (use --dependency=afterok:<jobid>).
#
# Array layout (27 = 9 checkpoints x 3 domains):
#   0-8   : GSL eval  (checkpoint 0..8)
#   9-17  : BOF eval  (checkpoint 0..8)
#   18-26 : GOM eval  (checkpoint 0..8)

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

# Determine which checkpoint and domain this task handles
CKPT_IDX=$(( I % 9 ))
DOMAIN_IDX=$(( I / 9 ))   # 0=GSL, 1=BOF, 2=GOM

STAGE=${STAGES[$CKPT_IDX]}
SW=${SW_VALS[$CKPT_IDX]}
SIDX=${SW_IDX[$CKPT_IDX]}
BASE=${BASE_CKPTS[$CKPT_IDX]}
RUN_TAG="sw_s${STAGE}_sw${SIDX}"
PHASE2_CKPT="../outputs/checkpoints/stage${STAGE}_phase2_${RUN_TAG}_best.pt"
MERGED="../outputs/checkpoints/stage${STAGE}_phase2_${RUN_TAG}_merged.pt"

echo "=== Eval Stage ${STAGE}, spec_weight=${SW}, domain=${DOMAIN_IDX} ==="

# Merge phase1 + phase2 if not already done
if [ ! -f "${MERGED}" ]; then
    echo "  Merging checkpoints -> ${MERGED}"
    python finetune_diffusion.py \
        --merge \
        --stage ${STAGE} \
        --ckpt  ${BASE} \
        --phase2_ckpt ${PHASE2_CKPT} \
        --out ${MERGED}
fi

if [ ${DOMAIN_IDX} -eq 0 ]; then
    # GSL evaluation
    OUT="../src/eval_fixed/eval_gsl_stage${STAGE}_${RUN_TAG}.json"
    echo "  GSL eval -> ${OUT}"
    python evaluate.py \
        --checkpoint ${MERGED} \
        --stage ${STAGE} \
        --output ${OUT} \
        --psd_curves

elif [ ${DOMAIN_IDX} -eq 1 ]; then
    # BOF evaluation
    OUT="../src/eval_fixed/eval_bof_stage${STAGE}_${RUN_TAG}.json"
    echo "  BOF eval -> ${OUT}"
    python eval_bof.py \
        --ckpt  ${MERGED} \
        --stage ${STAGE} \
        --output ${OUT}

elif [ ${DOMAIN_IDX} -eq 2 ]; then
    # GOM evaluation
    OUT="../src/eval_fixed/eval_gom_stage${STAGE}_${RUN_TAG}.json"
    echo "  GOM eval -> ${OUT}"
    python eval_gom.py \
        --ckpt  ${MERGED} \
        --stage ${STAGE} \
        --output ${OUT}
fi

echo "Done."
