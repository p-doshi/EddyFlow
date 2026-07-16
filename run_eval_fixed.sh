#!/bin/bash
# Run all 54 sweep evaluations with the fixed ERA5 persistence baseline.
# Saves results to src/eval_fixed/ without touching the original src/eval_gsl_*.json files.
#
# Usage (interactive GPU session or salloc):
#   cd /scratch/pdoshi/my_project/eddyflow
#   bash run_eval_fixed.sh
#
# Or as a single SLURM job:
#   sbatch --account=def-spadon \
#          --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1 \
#          --cpus-per-task=4 --mem=32G --time=2-00:00 \
#          run_eval_fixed.sh

set -e
cd /scratch/pdoshi/my_project/eddyflow

module load python/3.10 scipy-stack cuda/12.2
source /home/pdoshi/eddyflow-venv/bin/activate

CKPT_DIR=outputs/checkpoints
EVAL_DIR=src/eval_fixed
FIG_DIR=outputs/figures/sweep_fixed
mkdir -p "$EVAL_DIR" "$FIG_DIR"

EXP_JSON=scripts/sweep/experiments.json
T=28

echo "============================================"
echo "EddyFlow sweep eval — fixed persistence"
echo "$(date)"
echo "============================================"
echo "Results → $EVAL_DIR"
echo ""

cd src

TOTAL=$(python3 -c "import json; print(len(json.load(open('../$EXP_JSON'))['finetune']))")
IDX=0

python3 -c "
import json
with open('../$EXP_JSON') as f:
    exps = json.load(f)['finetune']
for e in exps:
    print(e['stage'], e['run_tag'])
" | while read STAGE RUN_TAG; do

    IDX=$((IDX + 1))
    MERGED="../${CKPT_DIR}/stage${STAGE}_${RUN_TAG}_merged.pt"
    GSL_OUT="../${EVAL_DIR}/eval_gsl_${RUN_TAG}.json"
    BOF_OUT="../${EVAL_DIR}/eval_bof_${RUN_TAG}.json"
    FIG_RUN="../${FIG_DIR}/${RUN_TAG}"
    mkdir -p "${FIG_RUN}/gsl" "${FIG_RUN}/bof"

    if [ ! -f "$MERGED" ]; then
        echo "[${IDX}/${TOTAL}] SKIP (no merged ckpt): $RUN_TAG"
        continue
    fi

    echo ""
    echo "[${IDX}/${TOTAL}] $STAGE  $RUN_TAG"
    echo "  ckpt: $MERGED"

    # ---- GSL eval -------------------------------------------------------
    if [ -f "$GSL_OUT" ]; then
        echo "  GSL: already done, skipping"
    else
        if [ "$STAGE" = "4" ]; then
            python evaluate.py --stage 4 \
                --checkpoint "$MERGED" \
                --psd_curves \
                --output "$GSL_OUT"
        else
            python evaluate_v2.py --stage "$STAGE" \
                --ckpt "$MERGED" \
                --psd_curves \
                --output "$GSL_OUT" \
                --save_examples \
                --example_dir "${FIG_RUN}/gsl"
        fi
        SK=$(python3 -c "import json; d=json.load(open('$GSL_OUT')); print(f\"{d.get('skill_score','?'):.4f}\")" 2>/dev/null || echo "?")
        RM=$(python3 -c "import json; d=json.load(open('$GSL_OUT')); print(f\"{d.get('rmse_model_K','?'):.4f}\")" 2>/dev/null || echo "?")
        echo "  GSL skill=$SK  rmse=${RM} C"
    fi

    # ---- BOF eval -------------------------------------------------------
    if [ -f "$BOF_OUT" ]; then
        echo "  BOF: already done, skipping"
    else
        python eval_bof.py --stage "$STAGE" \
            --ckpt "$MERGED" \
            --psd_curves \
            --output "$BOF_OUT" \
            --save_examples \
            --example_dir "${FIG_RUN}/bof"
        SK=$(python3 -c "import json; d=json.load(open('$BOF_OUT')); print(f\"{d.get('skill_score','?'):.4f}\")" 2>/dev/null || echo "?")
        RM=$(python3 -c "import json; d=json.load(open('$BOF_OUT')); print(f\"{d.get('rmse_model_K','?'):.4f}\")" 2>/dev/null || echo "?")
        echo "  BOF skill=$SK  rmse=${RM} C"
    fi

done

echo ""
echo "============================================"
echo "All evals done. Generating plots..."
echo "============================================"

python plot_sweep.py \
    --eval_dir "../${EVAL_DIR}" \
    --out_dir  "../${FIG_DIR}"

echo ""
echo "Done: $(date)"
echo "Plots → ../${FIG_DIR}"
echo "JSONs → ../${EVAL_DIR}"
