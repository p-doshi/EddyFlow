#!/bin/bash
#SBATCH --account=def-spadon
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0-06:00
#SBATCH --array=0-107
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --job-name=fshot_ev

# Evaluate few-shot fine-tuned checkpoints on the target domain test set.
# One eval job per finetune job (same array index → same config).
# Must be run after fewshot_finetune_array.sh (use --dependency=afterok).

module load python/3.10 scipy-stack cuda/12.2
source /home/pdoshi/eddyflow-venv/bin/activate
cd /scratch/pdoshi/my_project/eddyflow/src

# ── Same config arrays as finetune script ────────────────────────────────────
STAGE_NAMES=(1 2 3 4 4v2 5a)
DOMAINS=(bof gom)
N_SHOTS=(1 7 30)
SEEDS=(0 42 123)

I=${SLURM_ARRAY_TASK_ID}
STAGE_IDX=$(( I / (2*3*3) ))
REM=$(( I % (2*3*3) ))
DOM_IDX=$(( REM / (3*3) ))
REM=$(( REM % (3*3) ))
SHOT_IDX=$(( REM / 3 ))
SEED_IDX=$(( REM % 3 ))

STAGE=${STAGE_NAMES[$STAGE_IDX]}
DOMAIN=${DOMAINS[$DOM_IDX]}
N=${N_SHOTS[$SHOT_IDX]}
SEED=${SEEDS[$SEED_IDX]}
RUN_TAG="n${N}_s${SEED}"

FT_CKPT="../outputs/checkpoints/stage${STAGE}_fewshot_${DOMAIN}_${RUN_TAG}_best.pt"
OUT_JSON="../src/eval_fixed/eval_${DOMAIN}_fewshot_stage${STAGE}_${RUN_TAG}.json"

echo "=== Few-shot eval: stage=${STAGE}  domain=${DOMAIN}  n_shots=${N}  seed=${SEED} ==="
echo "  Checkpoint: ${FT_CKPT}"
echo "  Output:     ${OUT_JSON}"

if [ ! -f "${FT_CKPT}" ]; then
    echo "ERROR: Checkpoint not found: ${FT_CKPT}"
    exit 1
fi

if [ "${DOMAIN}" = "bof" ]; then
    python eval_bof.py \
        --ckpt   ${FT_CKPT} \
        --stage  ${STAGE} \
        --output ${OUT_JSON}
elif [ "${DOMAIN}" = "gom" ]; then
    python eval_gom.py \
        --ckpt   ${FT_CKPT} \
        --stage  ${STAGE} \
        --output ${OUT_JSON}
fi

echo "Done."
