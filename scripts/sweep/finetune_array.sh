#!/bin/bash
#SBATCH --account=def-spadon
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-16:00
#SBATCH --output=slurm_logs/finetune_array-%A_%a.out
#SBATCH --job-name=ef_finetune

module load python/3.10 scipy-stack cuda/12.2
source /home/pdoshi/eddyflow-venv/bin/activate

EXP_JSON=/scratch/pdoshi/my_project/eddyflow/scripts/sweep/experiments.json
CKPT_DIR=/scratch/pdoshi/my_project/eddyflow/outputs/checkpoints
T=28

read STAGE SPEC_WEIGHT LR SEED RUN_TAG PT_RUN_TAG EPOCHS <<< "$(python3 -c "
import json, sys, os
with open('$EXP_JSON') as f:
    exps = json.load(f)['finetune']
idx = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
if idx >= len(exps):
    sys.exit(0)
e = exps[idx]
print(e['stage'], e['spec_weight'], e['lr'], e['seed'], e['run_tag'], e['pretrain_run_tag'], e['epochs'])
" 2>/dev/null)"

if [ -z "$STAGE" ]; then
    echo "No experiment for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID -- exiting."
    exit 0
fi

echo "========================================================"
echo "Finetune: stage=$STAGE  sw=$SPEC_WEIGHT  lr=$LR  seed=$SEED  run_tag=$RUN_TAG"
echo "Array task $SLURM_ARRAY_TASK_ID / job $SLURM_JOB_ID"
echo "========================================================"

cd /scratch/pdoshi/my_project/eddyflow/src

PHASE1="${CKPT_DIR}/stage${STAGE}_T_${T}_${PT_RUN_TAG}_best.pt"
if [ ! -f "$PHASE1" ]; then
    echo "ERROR: Phase-1 checkpoint not found: $PHASE1"
    exit 1
fi

if [ "$STAGE" = "4" ]; then
    python finetune_diffusion.py --stage 4 \
        --ckpt "$PHASE1" \
        --run_tag "$RUN_TAG" \
        --spec_weight "$SPEC_WEIGHT" \
        --lr "$LR" \
        --seed "$SEED" \
        --epochs "$EPOCHS"
else
    python finetune_sweep_v2.py --stage "$STAGE" \
        --ckpt "$PHASE1" \
        --run_tag "$RUN_TAG" \
        --spec_weight "$SPEC_WEIGHT" \
        --lr "$LR" \
        --seed "$SEED" \
        --epochs "$EPOCHS"
fi
