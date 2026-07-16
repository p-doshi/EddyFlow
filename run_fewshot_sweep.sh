#!/bin/bash
# Launch the few-shot domain adaptation sweep.
# Usage: bash run_fewshot_sweep.sh
#
# 108 finetune jobs (6 stages × 2 domains × 3 n_shots × 3 seeds)
# 108 eval jobs (chained afterok)
# All on 3g.40gb MIG slices — 3 jobs share each H100, maximising utilisation.

set -e
cd "$(dirname "$0")"

echo "=== Few-shot domain adaptation sweep ==="
echo "  108 finetune + 108 eval jobs"
echo "  GPU: 40GB MIG slices (3 per H100 node)"
echo ""

FT_JID=$(sbatch --parsable scripts/sweep/fewshot_finetune_array.sh)
echo "Finetune array: job ${FT_JID}  (108 tasks × 4h, 3g.40gb)"

EV_JID=$(sbatch --parsable \
    --dependency=afterok:${FT_JID} \
    scripts/sweep/fewshot_eval_array.sh)
echo "Eval array:     job ${EV_JID}  (108 tasks × 6h, 3g.40gb, afterok:${FT_JID})"

echo ""
echo "Monitor: squeue -u pdoshi"
echo "Results: src/eval_fixed/eval_{bof,gom}_fewshot_stage*_n*_s*.json"
