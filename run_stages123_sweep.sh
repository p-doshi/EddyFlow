#!/bin/bash
# Launch spec-weight sweep for Stages 1/2/3 and chain eval.
# Usage: bash run_stages123_sweep.sh
#
# Submits:
#   1. Finetune array  (9 jobs: stages 1/2/3 x sw=[2.5, 5.0, 7.5])  -- full H100, 12h
#   2. Eval array     (27 jobs: above x 3 domains)                   -- full H100, 8h
#      chained afterok on the finetune array

set -e
cd "$(dirname "$0")"

echo "=== Stage 1/2/3 spec-weight sweep ==="
echo ""

FT_JID=$(sbatch --parsable scripts/sweep/finetune_stages123_array.sh)
echo "Finetune array submitted: job ${FT_JID}  (9 tasks x 12h, full H100)"

EVAL_JID=$(sbatch --parsable \
    --dependency=afterok:${FT_JID} \
    scripts/sweep/eval_stages123_array.sh)
echo "Eval array submitted:     job ${EVAL_JID}  (27 tasks x 8h, full H100, afterok:${FT_JID})"

echo ""
echo "Monitor:  squeue -u pdoshi"
echo "Results:  src/eval_fixed/eval_{gsl,bof,gom}_stage{1,2,3}_sw_s*_sw*.json"
echo ""
echo "Once complete, run:  python scripts/compile_ablation.py  (if updated)"
echo "  or manually pick best-sw per stage and update ablation_final.txt."
