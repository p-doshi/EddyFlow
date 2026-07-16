# scripts/04_download_gebco.py
"""
Downloads GEBCO 2023 bathymetry cropped to GSL domain.
One-time download — this never changes.

Resolution: 15 arc-seconds (~450 m)
Output:     data/raw/gebco/gebco_gsl.nc

Runtime: ~5 minutes
Size:    ~120 MB
"""

import urllib.request
import xarray as xr
import os

os.makedirs('data/raw/gebco', exist_ok=True)

# GEBCO provides a tile-based subset API
# This URL crops directly to your bounding box
url = (
    "https://download.gebco.net/api/tile?"
    "format=nc"
    "&north=52&south=47&west=-68&east=-56"
    "&resolution=15"   # 15 arc-seconds
)

out_raw = 'data/raw/gebco/gebco_gsl_raw.nc'
if not os.path.exists(out_raw):
    print('Downloading GEBCO...')
    urllib.request.urlretrieve(url, out_raw)

# Load, clean, save
bathy = xr.open_dataset(out_raw)
print(bathy)

# The elevation variable is 'elevation' (negative = ocean depth)
# Convert to positive depth, clip land to 0
depth = (-bathy['elevation']).clip(min=0)
depth.name = 'depth'
depth.attrs = {'units': 'meters', 'long_name': 'ocean depth'}

ds = depth.to_dataset()
ds.to_netcdf('data/raw/gebco/gebco_gsl.nc')
print(f'GEBCO saved. Grid: {dict(ds.dims)}')
# Expect: lat=1200, lon=2880 at 15 arc-sec over this domain