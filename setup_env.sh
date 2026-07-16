#!/bin/bash
module --force purge
module load StdEnv/2023
module load python/3.11
module load proj/9.2.0
module load geos/3.12.0
module load hdf5/1.14.2
module load netcdf/4.9.2
module load gdal/3.7.2

unset PYTHONPATH           # prevent CVMFS site-packages from leaking in
export PYTHONNOUSERSITE=1  # ignore ~/.local/lib/... user site

source ~/eddyflow-venv/bin/activate