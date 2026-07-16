"""
scripts/05_preprocess.py
Aligns ERA5, MUR SST, GLORYS, and GEBCO onto consistent grids,
applies ice masking, and writes Zarr stores ready for training.

Outputs (all under data/processed/):
  era5.zarr        — [time, 21, LAT_C, LON_C]   float32  raw physical units
  mur.zarr         — [time, LAT_F, LON_F]        float32  SST in °C
  mur_error.zarr   — [time, LAT_F, LON_F]        float32  analysis error
  ice.zarr         — [time, LAT_F, LON_F]        float32  SIC 0–1
  bathy.npy        — [LAT_F, LON_F]              float32  depth in m (≥0)
  glorys.zarr      — [time, 5, LAT_F, LON_F]    float32
  era5_times.npy   — [time]                      datetime64
  mur_times.npy    — [time]                      datetime64
  glorys_times.npy — [time]                      datetime64

Channel order (21 total):
  0  u10      1  v10      2  msl      3  sst      4  t2m      5  siconc
  6  u850     7  v850     8  t850     9  z850    10  q850
  11 u700    12  v700    13  t700    14  z700    15  q700
  16 u500    17  v500    18  t500    19  z500    20  q500

Notes:
  - ERA5 is stored RAW (no normalisation). Normalisation happens in dataset.py.
  - ch3 (sst) and ch5 (siconc) are NaN over land — this is expected.
    Do NOT add them to NAN_CHANNELS in config.py; nan_to_num in dataset.py
    handles them before z-scoring.
  - Run 06_compute_stats.py after this script.

Runtime: ~2 hours (MUR regrid is the slow step).
"""

import os
import zipfile

import numpy as np
import xarray as xr
import zarr
from scipy.interpolate import RegularGridInterpolator

# ── Output directories ────────────────────────────────────────────────────────
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/stats',     exist_ok=True)

# ── Grid definitions ──────────────────────────────────────────────────────────
# Coarse grid (ERA5): 0.25°   20 × 48 points
LAT_C, LON_C = 20, 48
lat_c = np.linspace(47.0, 51.75, LAT_C)    # 47.00 … 51.75 °N
lon_c = np.linspace(-68.0, -56.25, LON_C)  # -68.00 … -56.25 °E

# Fine grid (MUR): 0.01°   500 × 1200 points
LAT_F, LON_F = 500, 1200
lat_f = np.linspace(47.0, 51.99, LAT_F)    # 47.00 … 51.99 °N
lon_f = np.linspace(-68.0, -56.01, LON_F)  # -68.00 … -56.01 °E

print(f'Coarse grid: {LAT_C} × {LON_C}')
print(f'Fine   grid: {LAT_F} × {LON_F}')

# ── ERA5 channel definitions ──────────────────────────────────────────────────
SURFACE_VARS = ['u10', 'v10', 'msl', 'sst', 't2m', 'siconc']   # 6 channels
PRESSURE_LEVELS = ['850', '700', '500']
PRESSURE_VARS   = ['u', 'v', 't', 'z', 'q']                     # 5 × 3 = 15 channels
N_CH = len(SURFACE_VARS) + len(PRESSURE_LEVELS) * len(PRESSURE_VARS)  # 21
print(f'ERA5 channels: {N_CH}')


def unzip_if_needed(path: str) -> str:
    try:
        with zipfile.ZipFile(path, 'r') as z:
            names = z.namelist()
            z.extractall(os.path.dirname(path))
            os.remove(path)
            return os.path.join(os.path.dirname(path), names[0])
    except zipfile.BadZipFile:
        return path


# ═══════════════════════════════════════════════════════════════════════════════
# [1/5]  ERA5
# ═══════════════════════════════════════════════════════════════════════════════
print('\n[1/5] Processing ERA5...')

years  = list(range(2010, 2024))
months = [f'{m:02d}' for m in range(1, 13)]

all_times: list = []
all_data:  list = []

for year in years:
    for month in months:
        surf_path = f'data/raw/era5/era5_surface_{year}_{month}.nc'
        pres_path = f'data/raw/era5/era5_pressure_{year}_{month}.nc'

        if not os.path.exists(surf_path) or not os.path.exists(pres_path):
            print(f'  Skipping {year}-{month} (missing file)')
            continue

        surf_path = unzip_if_needed(surf_path)
        pres_path = unzip_if_needed(pres_path)

        # Open once each — no duplicate open
        ds_s = xr.open_dataset(surf_path)
        ds_p = xr.open_dataset(pres_path)

        times    = ds_s.valid_time.values
        channels = []

        # Surface channels (6)
        for v in SURFACE_VARS:
            arr = ds_s[v].sel(
                latitude=lat_c, longitude=lon_c, method='nearest'
            ).values.astype(np.float32)          # [T, LAT_C, LON_C]
            channels.append(arr)

        # Pressure channels (15)
        for level in PRESSURE_LEVELS:
            for v in PRESSURE_VARS:
                arr = ds_p[v].sel(
                    latitude=lat_c, longitude=lon_c,
                    pressure_level=int(level), method='nearest'
                ).values.astype(np.float32)      # [T, LAT_C, LON_C]
                channels.append(arr)

        ds_s.close()
        ds_p.close()

        data = np.stack(channels, axis=1)        # [T, N_CH, LAT_C, LON_C]
        assert data.shape[1] == N_CH, \
            f'Expected {N_CH} channels, got {data.shape[1]}'

        all_times.append(times)
        all_data.append(data)

all_times_arr = np.concatenate(all_times)
all_data_arr  = np.concatenate(all_data, axis=0)   # [T_total, 21, LAT_C, LON_C]
print(f'  ERA5 shape: {all_data_arr.shape}')

store = zarr.open(
    'data/processed/era5.zarr', mode='w',
    shape  = all_data_arr.shape,
    chunks = (16, N_CH, LAT_C, LON_C),
    dtype  = 'float32',
)
store[:] = all_data_arr
np.save('data/processed/era5_times.npy', all_times_arr)
print('  ERA5 Zarr written.')


# ═══════════════════════════════════════════════════════════════════════════════
# [2/5]  MUR SST
# ═══════════════════════════════════════════════════════════════════════════════
print('\n[2/5] Processing MUR SST...')

lat_da = xr.DataArray(lat_f, dims='lat')
lon_da = xr.DataArray(lon_f, dims='lon')

mur_sst_list:   list = []
mur_err_list:   list = []
mur_ice_list:   list = []
mur_times_list: list = []

for year in years:
    fpath = f'data/raw/mur_sst/mur_{year}.nc'
    if not os.path.exists(fpath):
        print(f'  Skipping MUR {year} (missing)')
        continue
    print(f'  MUR {year}...')
    ds = xr.open_dataset(fpath)
    ds_crop = ds.sel(lat=lat_da, lon=lon_da, method='nearest')

    mur_sst_list.append(ds_crop['analysed_sst'].values.astype(np.float32))
    mur_err_list.append(ds_crop['analysis_error'].values.astype(np.float32))
    mur_ice_list.append(
        np.nan_to_num(
            ds_crop['sea_ice_fraction'].values.astype(np.float32),
            nan=0.0   # NaN in MUR ice = open ocean
        )
    )
    mur_times_list.append(ds_crop.time.values)
    ds.close()

mur_sst   = np.concatenate(mur_sst_list,   axis=0)
mur_err   = np.concatenate(mur_err_list,   axis=0)
mur_ice   = np.concatenate(mur_ice_list,   axis=0)
mur_times = np.concatenate(mur_times_list)
print(f'  MUR SST shape: {mur_sst.shape}')

for name, arr in [('mur', mur_sst), ('mur_error', mur_err), ('ice', mur_ice)]:
    s = zarr.open(
        f'data/processed/{name}.zarr', mode='w',
        shape  = arr.shape,
        chunks = (8, LAT_F, LON_F),
        dtype  = 'float32',
    )
    s[:] = arr

np.save('data/processed/mur_times.npy', mur_times)
print('  MUR Zarr written.')


# ═══════════════════════════════════════════════════════════════════════════════
# [3/5]  GEBCO bathymetry  (integrated — no separate script needed)
# ═══════════════════════════════════════════════════════════════════════════════
print('\n[3/5] Processing GEBCO bathymetry...')

bathy_ds  = xr.open_dataset(
    'data/raw/gebco/GEBCO_25_Mar_2026_eaa5140f3fc4/'
    'gebco_2025_n52.0_s43.0_w-68.0_e-56.0.nc'
)
bathy_lat   = bathy_ds.lat.values
bathy_lon   = bathy_ds.lon.values
bathy_depth = bathy_ds['elevation'].values.astype(np.float32)
bathy_ds.close()

print(f'  GEBCO raw:  {bathy_depth.shape}  '
      f'lat {bathy_lat.min():.4f}→{bathy_lat.max():.4f}  '
      f'lon {bathy_lon.min():.4f}→{bathy_lon.max():.4f}')

# Query grid — clamp strictly inside GEBCO bounds
lat_lo = max(lat_f.min(), float(bathy_lat.min()) + 1e-4)
lat_hi = min(lat_f.max(), float(bathy_lat.max()) - 1e-4)
lon_lo = max(lon_f.min(), float(bathy_lon.min()) + 1e-4)
lon_hi = min(lon_f.max(), float(bathy_lon.max()) - 1e-4)

lat_q = np.linspace(lat_lo, lat_hi, LAT_F)
lon_q = np.linspace(lon_lo, lon_hi, LON_F)
print(f'  Query:      lat {lat_q[0]:.4f}→{lat_q[-1]:.4f}  '
      f'lon {lon_q[0]:.4f}→{lon_q[-1]:.4f}')

interp = RegularGridInterpolator(
    (bathy_lat, bathy_lon), bathy_depth,
    method='linear', bounds_error=True,
)
grid_lat, grid_lon = np.meshgrid(lat_q, lon_q, indexing='ij')
bathy_fine = interp(
    np.stack([grid_lat.ravel(), grid_lon.ravel()], axis=-1)
).reshape(LAT_F, LON_F).astype(np.float32)
bathy_fine = (-bathy_fine).clip(min=0.0)   # elevation→depth, clip land to 0

np.save('data/processed/bathy.npy', bathy_fine)
print(f'  Saved bathy.npy  shape={bathy_fine.shape}  '
      f'depth range: {bathy_fine.min():.0f}–{bathy_fine.max():.0f} m')

# Bathy normalisation stats (log-depth)
bathy_log  = np.log1p(bathy_fine)
bathy_mean = np.float32(bathy_log.mean())
bathy_std  = np.float32(bathy_log.std())
np.save('data/stats/bathy_mean.npy', bathy_mean)
np.save('data/stats/bathy_std.npy',  bathy_std)
print(f'  Bathy log stats: mean={bathy_mean:.3f}  std={bathy_std:.3f}')


# ═══════════════════════════════════════════════════════════════════════════════
# [4/5]  GLORYS
# ═══════════════════════════════════════════════════════════════════════════════
print('\n[4/5] Processing GLORYS...')

GLORYS_VARS = ['zos', 'uo', 'vo', 'thetao', 'mlotst']
glorys_path = 'data/raw/glorys/glorys_2010_2021.nc'

if os.path.exists(glorys_path):
    ds_g = xr.open_dataset(glorys_path)
    print(f'  dims:  {dict(ds_g.dims)}')
    print(f'  vars:  {list(ds_g.data_vars)}')

    g_arrays: list = []
    for v in GLORYS_VARS:
        da = ds_g[v]
        if 'depth' in da.dims and da.sizes['depth'] == 1:
            da = da.squeeze('depth')
        if 'latitude' in da.dims:
            da = da.rename({'latitude': 'lat', 'longitude': 'lon'})

        arr = da.interp(
            lat=xr.DataArray(lat_f, dims='lat'),
            lon=xr.DataArray(lon_f, dims='lon'),
            method='linear',
        ).values.astype(np.float32)
        print(f'  {v}: {arr.shape}')
        g_arrays.append(arr)

    glorys_data  = np.stack(g_arrays, axis=1)   # [T, 5, LAT_F, LON_F]
    glorys_times = ds_g.time.values
    ds_g.close()

    s = zarr.open(
        'data/processed/glorys.zarr', mode='w',
        shape  = glorys_data.shape,
        chunks = (16, 5, LAT_F, LON_F),
        dtype  = 'float32',
    )
    s[:] = glorys_data
    np.save('data/processed/glorys_times.npy', glorys_times)
    print(f'  GLORYS Zarr written. Shape: {glorys_data.shape}')
else:
    print(f'  Skipping GLORYS (not found at {glorys_path})')


# ═══════════════════════════════════════════════════════════════════════════════
# [5/5]  Verify
# ═══════════════════════════════════════════════════════════════════════════════
print('\n[5/5] Verification...')

e = zarr.open('data/processed/era5.zarr',  'r')
m = zarr.open('data/processed/mur.zarr',   'r')
b = np.load('data/processed/bathy.npy')

print(f'  ERA5  zarr : {e.shape}   expect (T, {N_CH}, {LAT_C}, {LON_C})')
print(f'  MUR   zarr : {m.shape}   expect (T, {LAT_F}, {LON_F})')
print(f'  Bathy npy  : {b.shape}   expect ({LAT_F}, {LON_F})')

assert e.shape[1] == N_CH,  f'ERA5 channel mismatch: {e.shape[1]} ≠ {N_CH}'
assert e.shape[2] == LAT_C, f'ERA5 lat mismatch: {e.shape[2]} ≠ {LAT_C}'
assert e.shape[3] == LON_C, f'ERA5 lon mismatch: {e.shape[3]} ≠ {LON_C}'
assert m.shape[1] == LAT_F, f'MUR lat mismatch: {m.shape[1]} ≠ {LAT_F}'
assert m.shape[2] == LON_F, f'MUR lon mismatch: {m.shape[2]} ≠ {LON_F}'
assert b.shape   == (LAT_F, LON_F), f'Bathy shape mismatch: {b.shape}'

print('\nAll assertions passed.')
print('Next step: python scripts/06_compute_stats.py')