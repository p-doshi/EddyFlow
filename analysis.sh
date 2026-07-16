#!/bin/bash
#SBATCH --account=def-spadon          
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-72:00
#SBATCH --output=slurm_logs/%x-%j.out


# Load modules
module --force purge
module load StdEnv/2023
module load python/3.11
module load proj/9.2.0
module load geos/3.12.0
module load hdf5/1.14.2
module load netcdf/4.9.2
module load gdal/3.7.2

unset PYTHONPATH
export PYTHONNOUSERSITE=1

source ~/eddyflow-venv/bin/activate

cd ~/scratch/my_project/eddyflow/scripts
python data_analysis.py --domain bof --root ../data/domains