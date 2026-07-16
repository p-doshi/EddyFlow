#!/bin/bash
#SBATCH --account=def-spadon
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:30
#SBATCH --output=slurm_logs/eval_bof_stage1-%j.out
#SBATCH --job-name=ef_bof1

module load python/3.10 scipy-stack cuda/12.2
source /home/pdoshi/eddyflow-venv/bin/activate

CKPT=/scratch/pdoshi/my_project/eddyflow/outputs/checkpoints
EVAL=/scratch/pdoshi/my_project/eddyflow/src/eval_fixed
cd /scratch/pdoshi/my_project/eddyflow/src

echo "========================================================"
echo "Stage 1 BOF re-eval (mur_seq bug was fixed)"
echo "Start: $(date)"
echo "========================================================"

python eval_bof.py --stage 1 \
    --ckpt "$CKPT/stage1_phase2_try2_merged.pt" \
    --psd_curves --output "$EVAL/eval_bof_stage1_try2.json"

echo ""
echo "Done: $(date)"
python3 -c "
import json
d = json.load(open('$EVAL/eval_bof_stage1_try2.json'))
print(f'Stage 1 BOF: rmse={d[\"rmse_model_K\"]:.4f}  skill={d[\"skill_score\"]:.4f}  psd={d[\"psd_ratio\"]:.4f}  n={d[\"n_samples\"]}')
sb = d.get('seasonal_bias_K',{})
print(f'  Seasonal bias: DJF={sb.get(\"DJF\",0):+.3f}  MAM={sb.get(\"MAM\",0):+.3f}  JJA={sb.get(\"JJA\",0):+.3f}  SON={sb.get(\"SON\",0):+.3f}')
"
