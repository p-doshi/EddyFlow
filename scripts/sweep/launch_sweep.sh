#!/bin/bash
# launch_sweep.sh -- Master launcher for the EddyFlow hyperparameter sweep.
#
# Run from project root:
#   bash scripts/sweep/launch_sweep.sh
#
# What it does:
#   1. Generates scripts/sweep/experiments.json from generate_experiments.py
#   2. Submits pretrain SLURM array (one GPU per experiment)
#   3. Submits finetune array (depends on ALL pretrain jobs completing)
#   4. Submits merge+eval array (depends on ALL finetune jobs completing)
#
# After everything completes, plots are in outputs/figures/sweep/.
# Eval JSONs are in src/eval_gsl_*.json and src/eval_bof_*.json.

set -e
cd /scratch/pdoshi/my_project/eddyflow

echo "=== EddyFlow Hyperparameter Sweep ==="
echo "Started: $(date)"
echo ""

# ---- Step 1: Generate experiments.json ----------------------------------
echo "Generating experiment configs..."
python3 scripts/sweep/generate_experiments.py
echo ""

# ---- Step 2: Compute array bounds ---------------------------------------
N_PRETRAIN=$(python3 -c "
import json
d = json.load(open('scripts/sweep/experiments.json'))
print(len(d['pretrain']) - 1)
")
N_FINETUNE=$(python3 -c "
import json
d = json.load(open('scripts/sweep/experiments.json'))
print(len(d['finetune']) - 1)
")
echo "Pretrain experiments : $((N_PRETRAIN+1))"
echo "Finetune experiments : $((N_FINETUNE+1))"
echo ""

# ---- Step 3: Submit pretrain array --------------------------------------
echo "Submitting phase-1 pretrain array (0-${N_PRETRAIN})..."
PRETRAIN_JID=$(sbatch --parsable \
    --array=0-${N_PRETRAIN} \
    scripts/sweep/pretrain_array.sh)
echo "  -> Job ${PRETRAIN_JID} (array of $((N_PRETRAIN+1)) tasks)"

# ---- Step 4: Submit finetune array (after ALL pretrain tasks done) ------
echo "Submitting phase-2 finetune array (depends on ${PRETRAIN_JID})..."
FINETUNE_JID=$(sbatch --parsable \
    --array=0-${N_FINETUNE} \
    --dependency=afterok:${PRETRAIN_JID} \
    scripts/sweep/finetune_array.sh)
echo "  -> Job ${FINETUNE_JID} (array of $((N_FINETUNE+1)) tasks)"

# ---- Step 5: Submit merge+eval array (after ALL finetune tasks done) ----
echo "Submitting merge+eval array (depends on ${FINETUNE_JID})..."
EVAL_JID=$(sbatch --parsable \
    --array=0-${N_FINETUNE} \
    --dependency=afterok:${FINETUNE_JID} \
    scripts/sweep/merge_eval_array.sh)
echo "  -> Job ${EVAL_JID} (array of $((N_FINETUNE+1)) tasks)"

echo ""
echo "=== All phases submitted ==="
echo "  Phase 1 pretrain  : ${PRETRAIN_JID}   ($((N_PRETRAIN+1)) tasks)"
echo "  Phase 2 finetune  : ${FINETUNE_JID}   ($((N_FINETUNE+1)) tasks)"
echo "  Merge + eval      : ${EVAL_JID}        ($((N_FINETUNE+1)) tasks)"
echo ""
echo "Monitor:  squeue -u $USER"
echo "Cancel all: scancel ${PRETRAIN_JID} ${FINETUNE_JID} ${EVAL_JID}"
echo ""
echo "Outputs when done:"
echo "  Training logs : outputs/logs/stage*_sweep_*_log.json"
echo "  Checkpoints   : outputs/checkpoints/stage*_sweep_*_merged.pt"
echo "  Eval GSL      : src/eval_gsl_*.json"
echo "  Eval BOF      : src/eval_bof_*.json"
echo "  Plots         : outputs/figures/sweep/*.png"
