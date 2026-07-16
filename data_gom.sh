#!/bin/bash
#SBATCH --account=def-spadon
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-72:00
#SBATCH --output=slurm_logs/data_gom-%j.out
#SBATCH --job-name=ef_data_gom

# Load modules for Python 3.11 with GIS stack (required by download_domains.py)
module load StdEnv/2023 python/3.11 proj/9.2.0 geos/3.12.0 hdf5/1.14.2 netcdf/4.9.2 gdal/3.7.2

source /home/pdoshi/eddyflow-venv/bin/activate

cd /scratch/pdoshi/my_project/eddyflow/scripts

echo "========================================================"
echo "GOM data download — MUR + ERA5"
echo "Start: $(date)"
echo "========================================================"

# ── MUR: resume from 2020-11-05 → download 2020-2023 ──────────────────────────
# Existing data covers 2013-01-01..2020-11-05 (2867 granules).
# This will skip already-downloaded granules automatically.
echo ""
echo "── MUR GOM (years 2020-2023) ──────────────────────────────"
python download_domains.py --domain gom --years 2020 2023 \
    --skip_era5 --skip_gebco

echo ""
echo "MUR download done: $(date)"
echo "========================================================"

# ── ERA5: download 2022-2023 only ────────────────────────────────────────────
# The T_ATM=28 lookback for Jan 2022 would need Dec 2021 ERA5, but downloading
# 2021 adds ~11h of CDS API time and risks hitting the 72h SLURM limit.
# build_sample_index() drops samples with missing lookback automatically;
# at most the first 28 test days (2022-01-01..28) are excluded — ~4% of samples.
echo ""
echo "── ERA5 GOM (years 2022-2023) ──────────────────────────────"
python download_domains.py --domain gom --years 2022 2023 \
    --skip_mur --skip_gebco

echo ""
echo "ERA5 download done: $(date)"
echo "========================================================"

# ── Bathymetry: ETOPO 2022 via NOAA ERDDAP (replaces broken GEBCO API) ────────
# GEBCO API returns 404. Downloaded ETOPO 2022 (~1 arcmin, variable=altitude)
# from coastwatch.pfeg.noaa.gov/erddap during session on 2026-07-07.
# File already saved to data/domains/gom/gebco_gom.nc  (1.7 MB, classic NetCDF)
GEBCO=/scratch/pdoshi/my_project/eddyflow/data/domains/gom/gebco_gom.nc
if [ -f "$GEBCO" ]; then
    echo "GEBCO/ETOPO: $GEBCO  ✓  ($(du -h $GEBCO | cut -f1))"
else
    echo "GEBCO/ETOPO: NOT FOUND at $GEBCO"
    echo "  Re-download with:"
    echo "  curl -o $GEBCO 'https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc?altitude%5B(18.0):(31.0)%5D%5B(-98.0):(-80.0)%5D'"
fi

echo ""
echo "========================================================"
echo "GOM data download complete: $(date)"
echo "========================================================"

# Quick inventory
python3 -c "
import numpy as np
from pathlib import Path

gom = Path('../data/domains/gom')
times = gom / 'mur/mur_gom_times.npy'
if times.exists():
    t = np.load(times, allow_pickle=False)
    print(f'MUR GOM: {len(t)} granules  ({t.min()} → {t.max()})')
    import numpy as np2
    test = t[(t >= np.datetime64('2022-01-01')) & (t <= np.datetime64('2023-12-31'))]
    print(f'  Test 2022-2023: {len(test)} granules')
else:
    print('MUR GOM: times file not found')

import os
era5_dir = gom / 'era5'
if era5_dir.exists():
    files = sorted(era5_dir.glob('era5_gom_*.nc'))
    print(f'ERA5 GOM: {len(files)} year files — {[f.stem for f in files]}')
else:
    print('ERA5 GOM: no data')

gebco = gom / 'gebco_gom.nc'
print(f'GEBCO GOM: {\"found\" if gebco.exists() else \"MISSING\"}')
"
