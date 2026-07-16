#!/bin/bash
#SBATCH --account=def-spadon          
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-72:00
#SBATCH --output=slurm_logs/%x-%j.out


# Load modules
module load python/3.11 scipy-stack cuda/12.2 mpich/4.0.0 mpi4py/4.0.0 h5py

# Activate the venv created in /home (not /scratch)
source /home/pdoshi/eddyflow-venv/bin/activate

# Change to your project directory on /scratch
cd /scratch/pdoshi/my_project/eddyflow/scripts

# Run your training script
python download_domains.py --domain lab