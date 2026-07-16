#!/bin/bash
#SBATCH --account=def-spadon
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-06:00
#SBATCH --output=slurm_logs/eval_gom-%j.out
#SBATCH --job-name=ef_eval_gom

# GOM is 1301×1801 pixels (≈4× GSL), so 64 GB RAM requested.

module load python/3.10 scipy-stack cuda/12.2
source /home/pdoshi/eddyflow-venv/bin/activate

CKPT=/scratch/pdoshi/my_project/eddyflow/outputs/checkpoints
EVAL=/scratch/pdoshi/my_project/eddyflow/src/eval_fixed

cd /scratch/pdoshi/my_project/eddyflow/src

echo "========================================================"
echo "GOM zero-shot evaluation"
echo "Start: $(date)"
echo "========================================================"

# Verify GEBCO is present — bail early if not
GEBCO=/scratch/pdoshi/my_project/eddyflow/data/domains/gom/gebco_gom.nc
if [ ! -f "$GEBCO" ]; then
    echo "ERROR: GEBCO bathymetry missing: $GEBCO"
    echo "  Download from https://download.gebco.net  (lat 18-31N, lon 98-80W, NetCDF)"
    echo "  Then scp/sftp to cluster and resubmit this job."
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
echo "── Stage 4v2 best (sweep seed=0, lr0, sw0) ────────────────"
if [ ! -f "$EVAL/eval_gom_stage4v2_sw0lr0.json" ]; then
    python eval_gom.py \
        --ckpt   "$CKPT/stage4v2_sweep_4v2_0_lr0_sw0_merged.pt" \
        --stage  4v2 \
        --output "$EVAL/eval_gom_stage4v2_sw0lr0.json" \
        --psd_curves --save_examples
else
    echo "  SKIP (already exists)"
fi

echo ""
echo "── Stage 5a best (sweep seed=0, lr0, sw0) ─────────────────"
if [ ! -f "$EVAL/eval_gom_stage5a_sw0lr0.json" ]; then
    python eval_gom.py \
        --ckpt   "$CKPT/stage5a_sweep_5a_0_lr0_sw0_merged.pt" \
        --stage  5a \
        --output "$EVAL/eval_gom_stage5a_sw0lr0.json" \
        --psd_curves --save_examples
else
    echo "  SKIP (already exists)"
fi

echo ""
echo "── Stage 4 best (sweep seed=123, lr0, sw0) ────────────────"
if [ ! -f "$EVAL/eval_gom_stage4_sw0lr0.json" ]; then
    python eval_gom.py \
        --ckpt   "$CKPT/stage4_sweep_4_123_lr0_sw0_merged.pt" \
        --stage  4 \
        --output "$EVAL/eval_gom_stage4_sw0lr0.json" \
        --psd_curves --save_examples
else
    echo "  SKIP (already exists)"
fi

echo ""
echo "========================================================"
echo "GOM evaluation complete: $(date)"
echo "========================================================"

# Summary table
python3 -c "
import json, glob, os

EVAL = '/scratch/pdoshi/my_project/eddyflow/src/eval_fixed'
files = sorted(glob.glob(f'{EVAL}/eval_gom_*.json'))
print(f'\n{\"Stage\":20s}  {\"RMSE\":>8}  {\"Persist\":>8}  {\"Skill\":>8}  {\"PSD\":>8}  DJF  MAM  JJA  SON')
print('-'*80)
for f in files:
    d = json.load(open(f))
    tag = os.path.basename(f).replace('eval_gom_','').replace('.json','')
    sb = d.get('seasonal_bias_K', {})
    djf = sb.get('DJF', float('nan'))
    mam = sb.get('MAM', float('nan'))
    jja = sb.get('JJA', float('nan'))
    son = sb.get('SON', float('nan'))
    print(f\"{tag:20s}  {d['rmse_model_K']:8.4f}  {d['rmse_persist_K']:8.4f}  {d['skill_score']:8.4f}  {d['psd_ratio']:8.4f}  {djf:+.2f} {mam:+.2f} {jja:+.2f} {son:+.2f}\")
"
