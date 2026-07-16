# scripts/03_download_glorys.py
"""
Downloads CMEMS GLORYS12 ocean reanalysis for eddy validation.
This is NOT a training target — used only to verify eddy detection
after training by comparing predicted SST anomalies with SSH rings.

Resolution: 0.083° (~9 km)
Variables:  zos (SSH), uo/vo (currents), thetao (temperature), mlotst (MLD)
Period:     2010–2021  (GLORYS ends at 2021; use CMEMS NRT after)
Output:     data/raw/glorys/glorys_2010_2021.nc

Expected runtime: ~45 minutes
Expected size:    ~4 GB
"""

import copernicusmarine
import os

os.makedirs('data/raw/glorys', exist_ok=True)

copernicusmarine.subset(
    dataset_id = 'cmems_mod_glo_phy_my_0.083deg_P1D-m',
    variables  = ['zos', 'uo', 'vo', 'thetao', 'mlotst'],
    minimum_longitude =  -68.0,
    maximum_longitude =  -56.0,
    minimum_latitude  =   47.0,
    maximum_latitude  =   52.0,
    minimum_depth     =    0.5,
    maximum_depth     =    1.5,   # surface layer only
    start_datetime    = '2010-01-01T00:00:00',
    end_datetime      = '2021-12-31T00:00:00',
    output_filename   = 'glorys_2010_2021.nc',
    output_directory  = 'data/raw/glorys/',
    force_download    = False,
)

print('GLORYS download complete.')