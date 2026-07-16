#!/bin/bash
# pretrain_array.sh -- SLURM array for phase-1 pretraining.
# Submitted by launch_sweep.sh with --array=0-N.
#SBATCH --account=def-spadon
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-150:00
#SBATCH --output=slurm_logs/pretrain_array-%A_%a.out
#SBATCH --job-name=ef_pretrain

module load python/3.10 scipy-stack cuda/12.2
source /home/pdoshi/eddyflow-venv/bin/activate

EXP_JSON=/scratch/pdoshi/my_project/eddyflow/scripts/sweep/experiments.json

# Read this task's config from experiments.json
read STAGE LR SEED RUN_TAG EPOCHS <<< "$(python3 -c "
import json, sys, os
with open('$EXP_JSON') as f:
    exps = json.load(f)['pretrain']
idx = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
if idx >= len(exps):
    sys.exit(0)
e = exps[idx]
print(e['stage'], e['lr'], e['seed'], e['run_tag'], e['epochs'])
" 2>/dev/null)"

if [ -z "$STAGE" ]; then
    echo "No experiment for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID — exiting."
    exit 0
fi

echo "========================================================"
echo "Pretrain: stage=$STAGE  lr=$LR  seed=$SEED  run_tag=$RUN_TAG  epochs=$EPOCHS"
echo "Array task $SLURM_ARRAY_TASK_ID / job $SLURM_JOB_ID"
echo "========================================================"

cd /scratch/pdoshi/my_project/eddyflow/src

if [ "$STAGE" = "4" ]; then
    python train.py --stage 4 \
        --lr "$LR" --seed "$SEED" --run_tag "$RUN_TAG" --epochs "$EPOCHS"
else
    python trainv2.py --stage "$STAGE" \
        --lr "$LR" --seed "$SEED" --run_tag "$RUN_TAG" --epochs "$EPOCHS"
fi
