#!/bin/bash
#SBATCH --account=def-spadon
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-08:00
#SBATCH --output=slurm_logs/eval_best_bias-%j.out
#SBATCH --job-name=ef_best_bias

# Re-evaluate Stage 4 / 4v2 / 5a best sweep runs to get seasonal_bias_K.
# Also covers Stage 3 try4 BOF and Stage 3 try5 GSL+BOF as fallbacks in case
# eval_stages123 (47570573) times out before completing those evals.
# All steps guarded by skip-if-exists so the job is restartable.

module load python/3.10 scipy-stack cuda/12.2
source /home/pdoshi/eddyflow-venv/bin/activate

CKPT=/scratch/pdoshi/my_project/eddyflow/outputs/checkpoints
EVAL=/scratch/pdoshi/my_project/eddyflow/src/eval_fixed
cd /scratch/pdoshi/my_project/eddyflow/src

echo "========================================================"
echo "Stage 3 try4/try5 + Stage 4 / 4v2 / 5a BEST-RUN BIAS RE-EVAL"
echo "Start: $(date)"
echo "========================================================"

# ── Stage 3 try4 BOF fallback ─────────────────────────────────────────────────
echo ""
echo "── Stage 3 try4 BOF fallback (skip if already done) ──"
if [ ! -f "$EVAL/eval_bof_stage3_try4.json" ]; then
    echo "  BOF eval — stage3 try4..."
    python eval_bof.py --stage 3 \
        --ckpt "$CKPT/stage3_phase2_try4_merged.pt" \
        --psd_curves --output "$EVAL/eval_bof_stage3_try4.json"
else
    echo "  BOF eval: SKIP (file exists)"
fi

# ── Stage 3 try5 GSL+BOF fallback ────────────────────────────────────────────
echo ""
echo "── Stage 3 try5 GSL+BOF fallback (skip if already done) ──"
if [ ! -f "$EVAL/eval_gsl_stage3_try5.json" ]; then
    echo "  GSL eval — stage3 try5..."
    python evaluate.py --stage 3 \
        --checkpoint "$CKPT/stage3_phase2_try5_merged.pt" \
        --psd_curves --output "$EVAL/eval_gsl_stage3_try5.json"
else
    echo "  GSL eval: SKIP (file exists)"
fi

if [ ! -f "$EVAL/eval_bof_stage3_try5.json" ]; then
    echo "  BOF eval — stage3 try5..."
    python eval_bof.py --stage 3 \
        --ckpt "$CKPT/stage3_phase2_try5_merged.pt" \
        --psd_curves --output "$EVAL/eval_bof_stage3_try5.json"
else
    echo "  BOF eval: SKIP (file exists)"
fi

# ── Stage 4 best: seed=123, lr0, sw0 ──────────────────────────────────────────
echo ""
echo "── Stage 4 best (seed=123, lr0, sw0) ──"
if [ ! -f "$EVAL/eval_gsl_stage4_best_bias.json" ]; then
    echo "  GSL eval..."
    python evaluate.py --stage 4 \
        --checkpoint "$CKPT/stage4_sweep_4_123_lr0_sw0_merged.pt" \
        --psd_curves --output "$EVAL/eval_gsl_stage4_best_bias.json"
else
    echo "  GSL eval: SKIP (file exists)"
fi

if [ ! -f "$EVAL/eval_bof_stage4_best_bias.json" ]; then
    echo "  BOF eval..."
    python eval_bof.py --stage 4 \
        --ckpt "$CKPT/stage4_sweep_4_123_lr0_sw0_merged.pt" \
        --psd_curves --output "$EVAL/eval_bof_stage4_best_bias.json"
else
    echo "  BOF eval: SKIP (file exists)"
fi

# ── Stage 4v2 best: seed=0, lr0, sw0 ──────────────────────────────────────────
echo ""
echo "── Stage 4v2 best (seed=0, lr0, sw0) ──"
if [ ! -f "$EVAL/eval_gsl_stage4v2_best_bias.json" ]; then
    echo "  GSL eval..."
    python evaluate_v2.py --stage 4v2 \
        --ckpt "$CKPT/stage4v2_sweep_4v2_0_lr0_sw0_merged.pt" \
        --psd_curves --output "$EVAL/eval_gsl_stage4v2_best_bias.json"
else
    echo "  GSL eval: SKIP (file exists)"
fi

if [ ! -f "$EVAL/eval_bof_stage4v2_best_bias.json" ]; then
    echo "  BOF eval..."
    python eval_bof.py --stage 4v2 \
        --ckpt "$CKPT/stage4v2_sweep_4v2_0_lr0_sw0_merged.pt" \
        --psd_curves --output "$EVAL/eval_bof_stage4v2_best_bias.json"
else
    echo "  BOF eval: SKIP (file exists)"
fi

# ── Stage 5a best: seed=0, lr0, sw0 ──────────────────────────────────────────
echo ""
echo "── Stage 5a best (seed=0, lr0, sw0) ──"
if [ ! -f "$EVAL/eval_gsl_stage5a_best_bias.json" ]; then
    echo "  GSL eval..."
    python evaluate_v2.py --stage 5a \
        --ckpt "$CKPT/stage5a_sweep_5a_0_lr0_sw0_merged.pt" \
        --psd_curves --output "$EVAL/eval_gsl_stage5a_best_bias.json"
else
    echo "  GSL eval: SKIP (file exists)"
fi

if [ ! -f "$EVAL/eval_bof_stage5a_best_bias.json" ]; then
    echo "  BOF eval..."
    python eval_bof.py --stage 5a \
        --ckpt "$CKPT/stage5a_sweep_5a_0_lr0_sw0_merged.pt" \
        --psd_curves --output "$EVAL/eval_bof_stage5a_best_bias.json"
else
    echo "  BOF eval: SKIP (file exists)"
fi

echo ""
echo "========================================================"
echo "ALL BEST-BIAS EVALS COMPLETE: $(date)"
echo "========================================================"

# Quick summary
python3 - <<'PYEOF'
import json, os

EVAL = '/scratch/pdoshi/my_project/eddyflow/src/eval_fixed'
runs = [
    ('Stage 4 best',   'stage4_best_bias'),
    ('Stage 4v2 best', 'stage4v2_best_bias'),
    ('Stage 5a best',  'stage5a_best_bias'),
]
print(f"\n{'Run':20s}  {'GSL RMSE':>10}  {'GSL Skill':>10}  {'BOF RMSE':>10}  {'BOF Skill':>10}")
print("-"*70)
for label, tag in runs:
    gf = f'{EVAL}/eval_gsl_{tag}.json'
    bf = f'{EVAL}/eval_bof_{tag}.json'
    try:
        g = json.load(open(gf)); b = json.load(open(bf))
        print(f"{label:20s}  {g['rmse_model_K']:10.4f}  {g['skill_score']:10.4f}  "
              f"{b['rmse_model_K']:10.4f}  {b['skill_score']:10.4f}")
        for domain, d in [('GSL', g), ('BOF', b)]:
            if 'seasonal_bias_K' in d:
                sb = d['seasonal_bias_K']
                print(f"  {domain} bias: DJF={sb['DJF']:+.3f} MAM={sb['MAM']:+.3f} JJA={sb['JJA']:+.3f} SON={sb['SON']:+.3f}")
    except Exception as e:
        print(f"{label:20s}  ERROR: {e}")
PYEOF
