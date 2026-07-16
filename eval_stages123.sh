#!/bin/bash
#SBATCH --account=def-spadon
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-06:00
#SBATCH --output=slurm_logs/eval_stages123-%j.out
#SBATCH --job-name=ef_eval123

module load python/3.10 scipy-stack cuda/12.2
source /home/pdoshi/eddyflow-venv/bin/activate

CKPT=/scratch/pdoshi/my_project/eddyflow/outputs/checkpoints
EVAL=/scratch/pdoshi/my_project/eddyflow/src/eval_fixed
cd /scratch/pdoshi/my_project/eddyflow/src

# ─────────────────────────────────────────────────────────────────
# STAGE 1 — merge first, then eval
# ─────────────────────────────────────────────────────────────────
echo "========================================================"
echo "STAGE 1 — merge + GSL + BOF"
echo "========================================================"

if [ ! -f "$CKPT/stage1_phase2_try2_merged.pt" ]; then
    echo "Merging stage1..."
    python finetune_diffusion.py --merge --stage 1 \
        --ckpt       "$CKPT/stage1_new3_T_28_best.pt" \
        --phase2_ckpt "$CKPT/stage1_phase2_try2_best.pt" \
        --out        "$CKPT/stage1_phase2_try2_merged.pt"
fi

if [ ! -f "$EVAL/eval_gsl_stage1_try2.json" ]; then
    echo "GSL eval — stage1 try2..."
    python evaluate.py --stage 1 \
        --checkpoint "$CKPT/stage1_phase2_try2_merged.pt" \
        --psd_curves --output "$EVAL/eval_gsl_stage1_try2.json"
fi

if [ ! -f "$EVAL/eval_bof_stage1_try2.json" ]; then
    echo "BOF eval — stage1 try2..."
    python eval_bof.py --stage 1 \
        --ckpt "$CKPT/stage1_phase2_try2_merged.pt" \
        --psd_curves --output "$EVAL/eval_bof_stage1_try2.json"
fi

# ─────────────────────────────────────────────────────────────────
# STAGE 2 — try3 (best RMSE) and try4 (best PSD)
# ─────────────────────────────────────────────────────────────────
echo "========================================================"
echo "STAGE 2 try3 — GSL + BOF"
echo "========================================================"

if [ ! -f "$EVAL/eval_gsl_stage2_try3.json" ]; then
    python evaluate.py --stage 2 \
        --checkpoint "$CKPT/stage2_phase2_try3_merged.pt" \
        --psd_curves --output "$EVAL/eval_gsl_stage2_try3.json"
fi

if [ ! -f "$EVAL/eval_bof_stage2_try3.json" ]; then
    python eval_bof.py --stage 2 \
        --ckpt "$CKPT/stage2_phase2_try3_merged.pt" \
        --psd_curves --output "$EVAL/eval_bof_stage2_try3.json"
fi

echo "========================================================"
echo "STAGE 2 try4 — GSL + BOF"
echo "========================================================"

if [ ! -f "$EVAL/eval_gsl_stage2_try4.json" ]; then
    python evaluate.py --stage 2 \
        --checkpoint "$CKPT/stage2_phase2_try4_merged.pt" \
        --psd_curves --output "$EVAL/eval_gsl_stage2_try4.json"
fi

if [ ! -f "$EVAL/eval_bof_stage2_try4.json" ]; then
    python eval_bof.py --stage 2 \
        --ckpt "$CKPT/stage2_phase2_try4_merged.pt" \
        --psd_curves --output "$EVAL/eval_bof_stage2_try4.json"
fi

# ─────────────────────────────────────────────────────────────────
# STAGE 3 — try4 and try5 (best)
# ─────────────────────────────────────────────────────────────────
echo "========================================================"
echo "STAGE 3 try4 — GSL + BOF"
echo "========================================================"

if [ ! -f "$EVAL/eval_gsl_stage3_try4.json" ]; then
    python evaluate.py --stage 3 \
        --checkpoint "$CKPT/stage3_phase2_try4_merged.pt" \
        --psd_curves --output "$EVAL/eval_gsl_stage3_try4.json"
fi

if [ ! -f "$EVAL/eval_bof_stage3_try4.json" ]; then
    python eval_bof.py --stage 3 \
        --ckpt "$CKPT/stage3_phase2_try4_merged.pt" \
        --psd_curves --output "$EVAL/eval_bof_stage3_try4.json"
fi

echo "========================================================"
echo "STAGE 3 try5 — GSL + BOF"
echo "========================================================"

if [ ! -f "$EVAL/eval_gsl_stage3_try5.json" ]; then
    python evaluate.py --stage 3 \
        --checkpoint "$CKPT/stage3_phase2_try5_merged.pt" \
        --psd_curves --output "$EVAL/eval_gsl_stage3_try5.json"
fi

if [ ! -f "$EVAL/eval_bof_stage3_try5.json" ]; then
    python eval_bof.py --stage 3 \
        --ckpt "$CKPT/stage3_phase2_try5_merged.pt" \
        --psd_curves --output "$EVAL/eval_bof_stage3_try5.json"
fi

# ─────────────────────────────────────────────────────────────────
# ALSO re-eval the sweep Stage-4 BEST to get seasonal bias
# (existing JSONs lack seasonal_bias_K since evaluator was just updated)
# ─────────────────────────────────────────────────────────────────
echo "========================================================"
echo "STAGE 4 best — re-eval for seasonal bias (seed=123,lr0,sw0)"
echo "========================================================"
python evaluate.py --stage 4 \
    --checkpoint "$CKPT/stage4_sweep_4_123_lr0_sw0_merged.pt" \
    --psd_curves --output "$EVAL/eval_gsl_stage4_best_bias.json"

python eval_bof.py --stage 4 \
    --ckpt "$CKPT/stage4_sweep_4_123_lr0_sw0_merged.pt" \
    --psd_curves --output "$EVAL/eval_bof_stage4_best_bias.json"

echo "========================================================"
echo "STAGE 4v2 best — re-eval for seasonal bias (seed=0,lr0,sw0)"
echo "========================================================"
python evaluate_v2.py --stage 4v2 \
    --ckpt "$CKPT/stage4v2_sweep_4v2_0_lr0_sw0_merged.pt" \
    --psd_curves --output "$EVAL/eval_gsl_stage4v2_best_bias.json"

python eval_bof.py --stage 4v2 \
    --ckpt "$CKPT/stage4v2_sweep_4v2_0_lr0_sw0_merged.pt" \
    --psd_curves --output "$EVAL/eval_bof_stage4v2_best_bias.json"

echo "========================================================"
echo "STAGE 5a best — re-eval for seasonal bias (seed=0,lr0,sw0)"
echo "========================================================"
python evaluate_v2.py --stage 5a \
    --ckpt "$CKPT/stage5a_sweep_5a_0_lr0_sw0_merged.pt" \
    --psd_curves --output "$EVAL/eval_gsl_stage5a_best_bias.json"

python eval_bof.py --stage 5a \
    --ckpt "$CKPT/stage5a_sweep_5a_0_lr0_sw0_merged.pt" \
    --psd_curves --output "$EVAL/eval_bof_stage5a_best_bias.json"

echo "========================================================"
echo "ALL EVALUATIONS COMPLETE"
echo "========================================================"

# Print summary
python3 - <<'PYEOF'
import json, glob, os

EVAL = '/scratch/pdoshi/my_project/eddyflow/src/eval_fixed'
SKIP = {'psd_curves','psd_log10_by_k','psd_ratio_by_k'}
CURRENT_PERSIST_GSL = 2.0229
CURRENT_PERSIST_BOF = 4.6300

runs = [
    ('Stage 1 try2',   'stage1_try2'),
    ('Stage 2 try3',   'stage2_try3'),
    ('Stage 2 try4',   'stage2_try4'),
    ('Stage 3 try4',   'stage3_try4'),
    ('Stage 3 try5',   'stage3_try5'),
    ('Stage 4 best',   'stage4_best_bias'),
    ('Stage 4v2 best', 'stage4v2_best_bias'),
    ('Stage 5a best',  'stage5a_best_bias'),
]

print(f"\n{'Run':20s}  {'GSL RMSE':>10}  {'GSL Skill':>10}  {'GSL PSD':>8}  {'BOF RMSE':>10}  {'BOF Skill':>10}  {'BOF PSD':>8}")
print("-"*90)
for label, tag in runs:
    gf = f'{EVAL}/eval_gsl_{tag}.json'
    bf = f'{EVAL}/eval_bof_{tag}.json'
    try:
        g = json.load(open(gf)); b = json.load(open(bf))
        print(f"{label:20s}  {g['rmse_model_K']:10.4f}  {g['skill_score']:10.4f}  "
              f"{g['psd_ratio']:8.4f}  {b['rmse_model_K']:10.4f}  "
              f"{b['skill_score']:10.4f}  {b['psd_ratio']:8.4f}")
        if 'seasonal_bias_K' in g:
            sb = g['seasonal_bias_K']
            print(f"  {'':18s}  GSL seasonal bias: DJF={sb['DJF']:+.3f}  MAM={sb['MAM']:+.3f}  JJA={sb['JJA']:+.3f}  SON={sb['SON']:+.3f}  °C")
        if 'seasonal_bias_K' in b:
            sb = b['seasonal_bias_K']
            print(f"  {'':18s}  BOF seasonal bias: DJF={sb['DJF']:+.3f}  MAM={sb['MAM']:+.3f}  JJA={sb['JJA']:+.3f}  SON={sb['SON']:+.3f}  °C")
    except Exception as e:
        print(f"{label:20s}  ERROR: {e}")
PYEOF
