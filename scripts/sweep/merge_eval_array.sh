#!/bin/bash
#SBATCH --account=def-spadon
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-12:00
#SBATCH --output=slurm_logs/merge_eval_array-%A_%a.out
#SBATCH --job-name=ef_eval

module load python/3.10 scipy-stack cuda/12.2
source /home/pdoshi/eddyflow-venv/bin/activate

EXP_JSON=/scratch/pdoshi/my_project/eddyflow/scripts/sweep/experiments.json
CKPT_DIR=/scratch/pdoshi/my_project/eddyflow/outputs/checkpoints
T=28

eval_field() {
    python3 -c "import json,sys; d=json.load(open('$1')); print(f'{d.get(\"$2\",float(\"nan\")):.3f}')" 2>/dev/null || echo "N/A"
}

# Read config for this array task
TASK_CONFIG=$(python3 -c "
import json, sys, os
with open('${EXP_JSON}') as f:
    exps = json.load(f)['finetune']
idx = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
if idx >= len(exps):
    sys.exit(0)
e = exps[idx]
print(e['stage'], e['spec_weight'], e['lr'], e['seed'], e['run_tag'], e['pretrain_run_tag'])
" 2>/dev/null)

if [ -z "$TASK_CONFIG" ]; then
    echo "No experiment for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID -- exiting."
    exit 0
fi

read -r STAGE SPEC_WEIGHT LR SEED RUN_TAG PT_RUN_TAG <<< "$TASK_CONFIG"

echo "========================================================"
echo "Merge+Eval: stage=$STAGE  run_tag=$RUN_TAG  seed=$SEED"
echo "Array task $SLURM_ARRAY_TASK_ID / job $SLURM_JOB_ID"
echo "========================================================"

cd /scratch/pdoshi/my_project/eddyflow/src

PHASE1="${CKPT_DIR}/stage${STAGE}_T_${T}_${PT_RUN_TAG}_best.pt"
PHASE2="${CKPT_DIR}/stage${STAGE}_phase2_${RUN_TAG}_best.pt"
MERGED="${CKPT_DIR}/stage${STAGE}_${RUN_TAG}_merged.pt"

# ---- MERGE (skip if already done) ----------------------------------------
if [ ! -f "$PHASE2" ]; then
    echo "ERROR: phase-2 checkpoint not found: $PHASE2"
    exit 1
fi

if [ -f "$MERGED" ]; then
    echo "Merged checkpoint already exists, skipping merge."
elif [ "$STAGE" = "4" ]; then
    python finetune_diffusion.py --merge --stage 4 \
        --ckpt "$PHASE1" --phase2_ckpt "$PHASE2" --out "$MERGED"
else
    python finetune_sweep_v2.py --merge --stage "$STAGE" \
        --ckpt "$PHASE1" --phase2_ckpt "$PHASE2" --out "$MERGED" --run_tag "$RUN_TAG"
fi

if [ ! -f "$MERGED" ]; then
    echo "ERROR: merge failed — $MERGED not found"
    exit 1
fi

EVAL_DIR=/scratch/pdoshi/my_project/eddyflow/src/eval_fixed
FIG_DIR=/scratch/pdoshi/my_project/eddyflow/outputs/figures/sweep_fixed/${RUN_TAG}
mkdir -p "${EVAL_DIR}" "${FIG_DIR}/gsl" "${FIG_DIR}/bof"

# ---- GSL EVAL (skip if already done) ------------------------------------
GSL_OUT=${EVAL_DIR}/eval_gsl_${RUN_TAG}.json
if [ -f "$GSL_OUT" ]; then
    echo "GSL eval already done, skipping."
    echo "GSL skill=$(eval_field $GSL_OUT skill_score)  rmse=$(eval_field $GSL_OUT rmse_model_K) C"
elif [ "$STAGE" = "4" ]; then
    python evaluate.py --stage 4 --checkpoint "$MERGED" \
        --psd_curves --output "$GSL_OUT"
    echo "GSL skill=$(eval_field $GSL_OUT skill_score)  rmse=$(eval_field $GSL_OUT rmse_model_K) C"
else
    python evaluate_v2.py --stage "$STAGE" --ckpt "$MERGED" \
        --psd_curves --output "$GSL_OUT" \
        --save_examples --example_dir "${FIG_DIR}/gsl"
    echo "GSL skill=$(eval_field $GSL_OUT skill_score)  rmse=$(eval_field $GSL_OUT rmse_model_K) C"
fi

# ---- BOF EVAL (skip if already done) ------------------------------------
BOF_OUT=${EVAL_DIR}/eval_bof_${RUN_TAG}.json
if [ -f "$BOF_OUT" ]; then
    echo "BOF eval already done, skipping."
    echo "BOF skill=$(eval_field $BOF_OUT skill_score)  rmse=$(eval_field $BOF_OUT rmse_model_K) C"
else
    python eval_bof.py --stage "$STAGE" --ckpt "$MERGED" \
        --psd_curves --output "$BOF_OUT" \
        --save_examples --example_dir "${FIG_DIR}/bof"
    echo "BOF skill=$(eval_field $BOF_OUT skill_score)  rmse=$(eval_field $BOF_OUT rmse_model_K) C"
fi

# ---- PLOT (last task only) -----------------------------------------------
N_LAST=$(python3 -c "import json; print(len(json.load(open('${EXP_JSON}'))['finetune'])-1)")
if [ "$SLURM_ARRAY_TASK_ID" = "$N_LAST" ]; then
    echo "Generating sweep plots..."
    python /scratch/pdoshi/my_project/eddyflow/src/plot_sweep.py \
        --eval_dir "${EVAL_DIR}" \
        --out_dir /scratch/pdoshi/my_project/eddyflow/outputs/figures/sweep_fixed
fi
