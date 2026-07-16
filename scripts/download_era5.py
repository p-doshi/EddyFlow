# scripts/01_download_era5.py
"""
Downloads ERA5 surface + pressure level fields for the
Gulf of St. Lawrence pilot domain.

Domain:  47°N–52°N, 68°W–56°W
Period:  2010–2023
Output:  data/raw/era5/era5_surface_YYYY.nc
         data/raw/era5/era5_pressure_YYYY.nc

Expected runtime: 2–6 hours (CDS queue dependent)
Expected size:    ~18 GB total
"""

import cdsapi
import os

os.makedirs('../data/raw/era5', exist_ok=True)

c = cdsapi.Client()

YEARS  = [str(y) for y in range(2010, 2024)]
MONTHS = [f'{m:02d}' for m in range(1, 13)]
DAYS   = [f'{d:02d}' for d in range(1, 32)]
TIMES  = ['00:00', '06:00', '12:00', '18:00']
AREA   = [52, -68, 45, -56]   # N W S E

# ── Surface fields ────────────────────────────────────────────────────────────
for year in YEARS:
    for month in MONTHS:

        out = f'data/raw/era5/era5_surface_{year}_{month}.nc'
        if os.path.exists(out):
            print(f'Skipping {out} (exists)')
            continue

        print(f'Requesting ERA5 surface {year}-{month}...')

        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    '10m_u_component_of_wind',
                    '10m_v_component_of_wind',
                    'mean_sea_level_pressure',
                    'sea_surface_temperature',
                    '2m_temperature',
                    'sea_ice_cover',
                ],
                'year': year,
                'month': month,
                'day': DAYS,
                'time': TIMES,
                'area': AREA,
                'data_format': 'netcdf',
                'grid': [0.25, 0.25],
            },
            out,
        )

# ── Pressure level fields ─────────────────────────────────────────────────────
for year in YEARS:
    for month in MONTHS:

        out = f'data/raw/era5/era5_pressure_{year}_{month}.nc'
        if os.path.exists(out):
            print(f'Skipping {out} (exists)')
            continue

        print(f'Requesting ERA5 pressure levels {year}-{month}...')

        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    'u_component_of_wind',
                    'v_component_of_wind',
                    'temperature',
                    'geopotential',
                    'specific_humidity',
                ],
                'pressure_level': ['850', '700', '500'],
                'year': year,
                'month': month,
                'day': DAYS,
                'time': TIMES,
                'area': AREA,
                'data_format': 'netcdf',
                'grid': [0.25, 0.25],
            },
            out,
        )

print('ERA5 download complete.')