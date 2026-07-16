#!/bin/bash
# run_seasonal_gifs.sh — Submit 6 seasonal SST animation jobs.
# Usage: bash scripts/run_seasonal_gifs.sh

set -e
cd /scratch/pdoshi/my_project/eddyflow
mkdir -p outputs/figures/explore/gifs slurm_logs

SCRIPT="scripts/make_sst_animation.py"
OUTDIR="outputs/figures/explore/gifs"

# Shared SLURM options
COMMON="--account=def-spadon --cpus-per-task=4 --mem=16G --time=0:40:00"

submit() {
    local name=$1 start=$2 end=$3 months=$4
    local out="${OUTDIR}/sst_${name}.gif"
    local log="slurm_logs/gif_${name}_%j.out"
    sbatch $COMMON \
        --job-name="gif_${name}" \
        --output="$log" \
        --wrap="
module load StdEnv/2023 python/3.10 scipy-stack
source /home/pdoshi/eddyflow-venv/bin/activate
cd /scratch/pdoshi/my_project/eddyflow
python ${SCRIPT} \
    --domain gsl \
    --start ${start} --end ${end} \
    --months ${months} \
    --every 2 --fps 14 --stride 3 --dpi 90 \
    --out ${out}
"
    echo "Submitted: ${name}  (${start} → ${end}, months=${months})"
}

# ── Summer: Jun–Oct each year ──────────────────────────────────────────────────
submit "2021_summer"  "2021-06-01"  "2021-10-31"  "6,7,8,9,10"
submit "2022_summer"  "2022-06-01"  "2022-10-31"  "6,7,8,9,10"
submit "2023_summer"  "2023-06-01"  "2023-10-31"  "6,7,8,9,10"

# ── Winter: Dec–Mar (labelled by the calendar year that starts the winter) ────
submit "2021_winter"  "2020-12-01"  "2021-03-31"  "12,1,2,3"
submit "2022_winter"  "2021-12-01"  "2022-03-31"  "12,1,2,3"
submit "2023_winter"  "2022-12-01"  "2023-03-31"  "12,1,2,3"

echo ""
echo "All 6 GIF jobs submitted → ${OUTDIR}/"
echo "Track with: squeue -u $USER"
