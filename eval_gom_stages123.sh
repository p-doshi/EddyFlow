#!/bin/bash
#SBATCH --account=def-spadon
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-04:00
#SBATCH --output=slurm_logs/eval_gom_stages123-%j.out
#SBATCH --job-name=ef_gom_123

# GOM zero-shot eval for ablation stages 1-3 (T_oce=0, no ocean history).
# These stages use ERA5-only input so all 729 test samples are available.

module load python/3.10 scipy-stack cuda/12.2
source /home/pdoshi/eddyflow-venv/bin/activate

CKPT=/scratch/pdoshi/my_project/eddyflow/outputs/checkpoints
EVAL=/scratch/pdoshi/my_project/eddyflow/src/eval_fixed

cd /scratch/pdoshi/my_project/eddyflow/src

echo "========================================================"
echo "GOM zero-shot evaluation — Stages 1, 2, 3"
echo "Start: $(date)"
echo "========================================================"

# Verify GEBCO is present
GEBCO=/scratch/pdoshi/my_project/eddyflow/data/domains/gom/gebco_gom.nc
if [ ! -f "$GEBCO" ]; then
    echo "ERROR: GEBCO bathymetry missing: $GEBCO"
    exit 1
fi

# Verify test MUR data exists
python3 -c "
import numpy as np, sys
t = np.load('../data/domains/gom/mur/mur_gom_times.npy', allow_pickle=False)
test = t[(t >= np.datetime64('2022-01-01')) & (t <= np.datetime64('2023-12-31'))]
if len(test) == 0:
    print('ERROR: No 2022-2023 MUR data. Run data_gom.sh first.', file=sys.stderr)
    sys.exit(1)
print(f'MUR test data: {len(test)} days (2022-2023)')
"
if [ $? -ne 0 ]; then exit 1; fi

echo ""
echo "── Stage 1 best (try2) ─────────────────────────────────"
if [ ! -f "$EVAL/eval_gom_stage1_try2.json" ]; then
    python eval_gom.py \
        --ckpt   "$CKPT/stage1_phase2_try2_merged.pt" \
        --stage  1 \
        --output "$EVAL/eval_gom_stage1_try2.json" \
        --psd_curves --save_examples
else
    echo "  SKIP (already exists)"
fi

echo ""
echo "── Stage 2 best (try3) ─────────────────────────────────"
if [ ! -f "$EVAL/eval_gom_stage2_try3.json" ]; then
    python eval_gom.py \
        --ckpt   "$CKPT/stage2_phase2_try3_merged.pt" \
        --stage  2 \
        --output "$EVAL/eval_gom_stage2_try3.json" \
        --psd_curves --save_examples
else
    echo "  SKIP (already exists)"
fi

echo ""
echo "── Stage 3 best (try5) ─────────────────────────────────"
if [ ! -f "$EVAL/eval_gom_stage3_try5.json" ]; then
    python eval_gom.py \
        --ckpt   "$CKPT/stage3_phase2_try5_merged.pt" \
        --stage  3 \
        --output "$EVAL/eval_gom_stage3_try5.json" \
        --psd_curves --save_examples
else
    echo "  SKIP (already exists)"
fi

echo ""
echo "========================================================"
echo "GOM evaluation complete: $(date)"
echo "========================================================"

python3 -c "
import json, glob, os

EVAL = '/scratch/pdoshi/my_project/eddyflow/src/eval_fixed'
files = sorted(glob.glob(f'{EVAL}/eval_gom_*.json'))
print(f'\n{\"Stage\":25s}  {\"RMSE\":>8}  {\"Persist\":>8}  {\"Skill\":>8}  {\"PSD\":>8}  DJF   MAM   JJA   SON')
print('-'*90)
for f in files:
    d = json.load(open(f))
    tag = os.path.basename(f).replace('eval_gom_','').replace('.json','')
    sb = d.get('seasonal_bias_K', {})
    djf = sb.get('DJF', float('nan'))
    mam = sb.get('MAM', float('nan'))
    jja = sb.get('JJA', float('nan'))
    son = sb.get('SON', float('nan'))
    print(f\"{tag:25s}  {d['rmse_model_K']:8.4f}  {d['rmse_persist_K']:8.4f}  {d['skill_score']:8.4f}  {d['psd_ratio']:8.4f}  {djf:+.2f}  {mam:+.2f}  {jja:+.2f}  {son:+.2f}\")
"
