#!/bin/bash
#SBATCH --account=def-spadon
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0-00:10
#SBATCH --output=slurm_logs/compile_ablation-%j.out
#SBATCH --job-name=ef_ablation

# Lightweight: reads all eval JSONs and writes ablation_final.txt
# Typically submitted with --dependency=afterok:<eval_job_id>

module load python/3.10 scipy-stack
source /home/pdoshi/eddyflow-venv/bin/activate

cd /scratch/pdoshi/my_project/eddyflow

echo "========================================================"
echo "Compiling ablation results"
echo "Start: $(date)"
echo "========================================================"

python scripts/compile_ablation.py

echo ""
echo "========================================================"
echo "Done: $(date)"
echo "========================================================"

echo ""
echo "--- Contents of ablation_final.txt ---"
cat ablation_final.txt
