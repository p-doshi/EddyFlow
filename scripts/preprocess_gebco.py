# scripts/preprocess_gebco.py
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import zarr
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Confirm target size from actual MUR zarr ──────────────────────────────────
mur = zarr.open('scripts/data/processed/mur.zarr', 'r')
_, LAT_F, LON_F = mur.shape
print(f'MUR zarr grid: {LAT_F} × {LON_F}')

# ── Load GEBCO ────────────────────────────────────────────────────────────────
bathy_raw = xr.open_dataset(
    'scripts/data/raw/gebco/GEBCO_25_Mar_2026_eaa5140f3fc4/'
    'gebco_2025_n52.0_s43.0_w-68.0_e-56.0.nc'
)
bathy_lat   = bathy_raw.lat.values
bathy_lon   = bathy_raw.lon.values
bathy_depth = bathy_raw['elevation'].values.astype(np.float32)

print(f'GEBCO grid:    {bathy_depth.shape}')
print(f'GEBCO lat:     {bathy_lat.min():.6f} → {bathy_lat.max():.6f}')
print(f'GEBCO lon:     {bathy_lon.min():.6f} → {bathy_lon.max():.6f}')

# ── Build query grid — our DOMAIN is 47°N–52°N, not 43°N–52°N ────────────────
# GEBCO file is larger than our domain — we must crop to 47–52°N explicitly
DOMAIN_LAT = (47.0, 52.0)
DOMAIN_LON = (-68.0, -56.0)

# Stay strictly inside both GEBCO bounds AND our domain bounds
lat_lo = max(DOMAIN_LAT[0], float(bathy_lat.min()) + 1e-4)
lat_hi = min(DOMAIN_LAT[1], float(bathy_lat.max()) - 1e-4)
lon_lo = max(DOMAIN_LON[0], float(bathy_lon.min()) + 1e-4)
lon_hi = min(DOMAIN_LON[1], float(bathy_lon.max()) - 1e-4)

lat_q = np.linspace(lat_lo, lat_hi, LAT_F)    # exactly 501 pts over 47–52°N
lon_q = np.linspace(lon_lo, lon_hi, LON_F)    # exactly 1201 pts over 68–56°W

print(f'Query lat:     {lat_q[0]:.6f} → {lat_q[-1]:.6f}  ({len(lat_q)} pts)')
print(f'Query lon:     {lon_q[0]:.6f} → {lon_q[-1]:.6f}  ({len(lon_q)} pts)')
assert len(lat_q) == LAT_F and len(lon_q) == LON_F

# ── Interpolate ───────────────────────────────────────────────────────────────
interp = RegularGridInterpolator(
    (bathy_lat, bathy_lon),
    bathy_depth,
    method       = 'linear',
    bounds_error = True,
)

grid_lat, grid_lon = np.meshgrid(lat_q, lon_q, indexing='ij')
qq = np.stack([grid_lat.ravel(), grid_lon.ravel()], axis=-1)

print(f'Query points:  {qq.shape[0]}  (expected {LAT_F * LON_F})')
assert qq.shape[0] == LAT_F * LON_F

bathy_fine = interp(qq).reshape(LAT_F, LON_F).astype(np.float32)
bathy_fine = (-bathy_fine).clip(min=0.0)   # flip sign, clip land to 0

# ── Save ──────────────────────────────────────────────────────────────────────
np.save('scripts/data/processed/bathy.npy', bathy_fine)
print(f'\nSaved scripts/data/processed/bathy.npy')
print(f'Shape:       {bathy_fine.shape}')
print(f'Depth range: {bathy_fine.min():.0f} – {bathy_fine.max():.0f} m')
print(f'Expected:    0 – ~500 m  (Laurentian Channel)')

# ── Recompute bathy stats ─────────────────────────────────────────────────────
bathy_log  = np.log1p(bathy_fine)
bathy_mean = np.float32(bathy_log.mean())
bathy_std  = np.float32(bathy_log.std())
np.save('scripts/data/stats/bathy_mean.npy', bathy_mean)
np.save('scripts/data/stats/bathy_std.npy',  bathy_std)
print(f'Bathy log:   mean={bathy_mean:.3f}  std={bathy_std:.3f}')

# ── Verify match against MUR ──────────────────────────────────────────────────
b = np.load('scripts/data/processed/bathy.npy')
print(f'\nVerify: bathy={b.shape}  mur={mur.shape[1:]}')
assert b.shape == mur.shape[1:], f"MISMATCH: {b.shape} vs {mur.shape[1:]}"
print('Shapes match. Ready for training.')