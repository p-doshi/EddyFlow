#!/bin/bash
#SBATCH --account=def-spadon
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0-04:00
#SBATCH --array=0-107
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --job-name=fshot_ft

# Few-shot domain adaptation sweep.
# Best checkpoint per stage × 2 domains × 3 n_shots × 3 seeds = 108 jobs.
# Uses 40GB MIG slice so 3 tasks can run in parallel on one H100 node.
#
# Array index layout (108 = 6 stages × 2 domains × 3 n_shots × 3 seeds):
#   outer: stage (0-5)
#   inner: domain × n_shots × seed
#
# Stages (index 0-5):
#   0=stage1  1=stage2  2=stage3  3=stage4  4=stage4v2  5=stage5a
#
# Best checkpoints selected from completed sweeps.
# NOTE: Stages 1-3 use pre-sweep best runs; will be updated once
#       the spec-weight sweep (jobs 47942289/90) completes.

module load python/3.10 scipy-stack cuda/12.2
source /home/pdoshi/eddyflow-venv/bin/activate
cd /scratch/pdoshi/my_project/eddyflow/src

# ── Config arrays ─────────────────────────────────────────────────────────────
STAGE_NAMES=(1 2 3 4 4v2 5a)
BEST_CKPTS=(
  "../outputs/checkpoints/stage1_phase2_sw_s1_sw1_merged.pt"
  "../outputs/checkpoints/stage2_phase2_sw_s2_sw0_merged.pt"
  "../outputs/checkpoints/stage3_phase2_sw_s3_sw0_merged.pt"
  "../outputs/checkpoints/stage4_sweep_4_123_lr0_sw2_merged.pt"
  "../outputs/checkpoints/stage4v2_sweep_4v2_0_lr1_sw1_merged.pt"
  "../outputs/checkpoints/stage5a_sweep_5a_0_lr0_sw0_merged.pt"
)
DOMAINS=(bof gom)
N_SHOTS=(1 7 30)
SEEDS=(0 42 123)

# ── Decode task index ─────────────────────────────────────────────────────────
# 108 = 6 stages × 2 domains × 3 n_shots × 3 seeds
I=${SLURM_ARRAY_TASK_ID}

STAGE_IDX=$(( I / (2*3*3) ))   # 0-5
REM=$(( I % (2*3*3) ))
DOM_IDX=$(( REM / (3*3) ))     # 0-1
REM=$(( REM % (3*3) ))
SHOT_IDX=$(( REM / 3 ))        # 0-2
SEED_IDX=$(( REM % 3 ))        # 0-2

STAGE=${STAGE_NAMES[$STAGE_IDX]}
CKPT=${BEST_CKPTS[$STAGE_IDX]}
DOMAIN=${DOMAINS[$DOM_IDX]}
N=${N_SHOTS[$SHOT_IDX]}
SEED=${SEEDS[$SEED_IDX]}
RUN_TAG="n${N}_s${SEED}"

echo "=== Few-shot finetune: stage=${STAGE}  domain=${DOMAIN}  n_shots=${N}  seed=${SEED} ==="
echo "  Checkpoint: ${CKPT}"
echo "  run_tag: ${RUN_TAG}"

python finetune_fewshot.py \
    --stage      ${STAGE} \
    --ckpt       ${CKPT} \
    --domain     ${DOMAIN} \
    --n_shots    ${N} \
    --seed       ${SEED} \
    --lr         5e-6 \
    --epochs     20 \
    --spec_weight 5.0 \
    --run_tag    ${RUN_TAG}
