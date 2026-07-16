"""
scripts/06_compute_stats.py
Computes normalisation statistics over the training split (2010–2020).

Outputs (all under data/stats/):
  era5_mean.npy          — [21]   per-channel mean  (raw physical units)
  era5_std.npy           — [21]   per-channel std   (raw physical units)
  era5_channel_names.npy — [21]   string labels
  sst_mean.npy           — scalar  °C   (ocean pixels only)
  sst_std.npy            — scalar  °C
  bathy_mean.npy         — scalar  log(1 + depth_m)
  bathy_std.npy          — scalar

IMPORTANT — NAN_CHANNELS:
  ERA5 ch3 (sst) and ch5 (siconc) are NaN over land — ~52% of pixels.
  Their stats are computed here using only finite pixels (nanmean/nanstd).
  They must NOT be in NAN_CHANNELS in config.py; dataset.py's nan_to_num
  replaces NaN with 0 BEFORE z-scoring so the stats computed here are correct.
"""

import numpy as np
import zarr
import pandas as pd
import os

os.makedirs('data/stats', exist_ok=True)

# ── Load zarr stores ──────────────────────────────────────────────────────────
era5 = zarr.open('../data/domains/gsl/era5.zarr', 'r')
mur  = zarr.open('../data/domains/gsl/mur.zarr',  'r')

T_ERA5, N_CH, LAT_C, LON_C = era5.shape
T_MUR,  LAT_F, LON_F        = mur.shape

print(f'ERA5 zarr : {era5.shape}  ({N_CH} channels)')
print(f'MUR  zarr : {mur.shape}')

# ── Channel names ─────────────────────────────────────────────────────────────
CHANNEL_NAMES = [
    'u10', 'v10', 'msl', 'sst', 't2m', 'siconc',
    'u850', 'v850', 't850', 'z850', 'q850',
    'u700', 'v700', 't700', 'z700', 'q700',
    'u500', 'v500', 't500', 'z500', 'q500',
]
assert len(CHANNEL_NAMES) == N_CH, \
    f'Name list length {len(CHANNEL_NAMES)} ≠ zarr channels {N_CH}'

# ── Training time indices ─────────────────────────────────────────────────────
TRAIN_YEARS = (2010, 2020)

era5_times = np.load('../data/domains/gsl/era5_times.npy', allow_pickle=True)
try:
    era5_pd = pd.DatetimeIndex(era5_times)
except Exception:
    era5_pd = pd.DatetimeIndex([str(t) for t in era5_times])

tr_idx = np.where(
    (era5_pd.year >= TRAIN_YEARS[0]) & (era5_pd.year <= TRAIN_YEARS[1])
)[0]
print(f'\nERA5 training timesteps : {len(tr_idx)}  '
      f'({era5_pd[tr_idx[0]].date()} → {era5_pd[tr_idx[-1]].date()})')

# ═══════════════════════════════════════════════════════════════════════════════
# ERA5 stats — Welford-style online algorithm, NaN-safe per channel
# ═══════════════════════════════════════════════════════════════════════════════
print('\nComputing ERA5 stats (NaN-safe, training split)...')

# Accumulators: sum, sum-of-squares, count — all per channel
ch_sum   = np.zeros(N_CH, dtype=np.float64)
ch_sum2  = np.zeros(N_CH, dtype=np.float64)
ch_count = np.zeros(N_CH, dtype=np.int64)

CHUNK = 256

for start in range(0, len(tr_idx), CHUNK):
    idx   = tr_idx[start : start + CHUNK]
    batch = era5[idx].astype(np.float64)   # [chunk, N_CH, LAT_C, LON_C]

    for c in range(N_CH):
        vals = batch[:, c, :, :]           # [chunk, LAT_C, LON_C]
        fin  = vals[np.isfinite(vals)]     # flatten + mask NaN (land in sst/siconc)
        ch_sum[c]   += fin.sum()
        ch_sum2[c]  += (fin ** 2).sum()
        ch_count[c] += fin.size

    if (start // CHUNK) % 10 == 0:
        done = min(start + CHUNK, len(tr_idx))
        print(f'  {done}/{len(tr_idx)} timesteps...')

era5_mean = (ch_sum   / ch_count).astype(np.float32)
era5_var  = np.maximum(
    ch_sum2 / ch_count - (ch_sum / ch_count) ** 2,
    1e-6
)
era5_std  = era5_var.astype(np.float32) ** 0.5

print('\nERA5 channel stats:')
for i, name in enumerate(CHANNEL_NAMES):
    print(f'  ch{i:2d}  {name:8s}  mean={era5_mean[i]:12.4f}  '
          f'std={era5_std[i]:10.4f}  '
          f'n_finite={ch_count[i]:,}')

np.save('data/stats/era5_mean.npy',          era5_mean)
np.save('data/stats/era5_std.npy',           era5_std)
np.save('data/stats/era5_channel_names.npy', np.array(CHANNEL_NAMES))
print('ERA5 stats saved.')

# ═══════════════════════════════════════════════════════════════════════════════
# MUR SST stats — ocean pixels only, training split
# ═══════════════════════════════════════════════════════════════════════════════
print('\nComputing MUR SST stats...')

mur_times = np.load('../data/domains/gsl/mur_times.npy', allow_pickle=True)
try:
    mur_pd = pd.DatetimeIndex(mur_times)
except Exception:
    mur_pd = pd.DatetimeIndex([str(t) for t in mur_times])

mur_tr = np.where(
    (mur_pd.year >= TRAIN_YEARS[0]) & (mur_pd.year <= TRAIN_YEARS[1])
)[0]
print(f'MUR training days : {len(mur_tr)}  '
      f'({mur_pd[mur_tr[0]].date()} → {mur_pd[mur_tr[-1]].date()})')

# Sample every 14 days — enough for stable stats, fast to compute
sst_sum   = np.float64(0)
sst_sum2  = np.float64(0)
sst_count = np.int64(0)

for i in mur_tr[::14]:
    slab = mur[i].astype(np.float64)                     # [LAT_F, LON_F]
    # Ocean pixels only: finite, plausible SST range for GSL (-2 to 30 °C)
    valid = slab[np.isfinite(slab) & (slab > -2.5) & (slab < 35.0)]
    if valid.size > 0:
        sst_sum   += valid.sum()
        sst_sum2  += (valid ** 2).sum()
        sst_count += valid.size

sst_mean = np.float32(sst_sum  / sst_count)
sst_std  = np.float32(
    np.sqrt(max(sst_sum2 / sst_count - (sst_sum / sst_count) ** 2, 1e-6))
)
print(f'SST : mean={sst_mean:.4f} °C   std={sst_std:.4f} °C   '
      f'n={sst_count:,}')

np.save('data/stats/sst_mean.npy', sst_mean)
np.save('data/stats/sst_std.npy',  sst_std)
print('SST stats saved.')

# ═══════════════════════════════════════════════════════════════════════════════
# Bathymetry stats
# ═══════════════════════════════════════════════════════════════════════════════
print('\nComputing bathymetry stats...')

bathy     = np.load('../data/domains/gsl/bathy.npy')
bathy_log = np.log1p(bathy)
bathy_mean = np.float32(bathy_log.mean())
bathy_std  = np.float32(bathy_log.std())

print(f'Bathy raw : min={bathy.min():.0f} m   max={bathy.max():.0f} m')
print(f'Bathy log : mean={bathy_mean:.4f}   std={bathy_std:.4f}')

np.save('data/stats/bathy_mean.npy', bathy_mean)
np.save('data/stats/bathy_std.npy',  bathy_std)
print('Bathy stats saved.')

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print('\n' + '─' * 60)
print('All stats written to data/stats/:')
for fname in [
    'era5_mean', 'era5_std', 'era5_channel_names',
    'sst_mean', 'sst_std',
    'bathy_mean', 'bathy_std',
]:
    arr  = np.load(f'data/stats/{fname}.npy', allow_pickle=True)
    print(f'  {fname}.npy   shape={arr.shape}')

print(f'\nGrid  : coarse {LAT_C}×{LON_C}   fine {LAT_F}×{LON_F}')
print(f'N_CH  : {N_CH}')
print('\nIMPORTANT: set NAN_CHANNELS = [] in config.py before retraining.')
print('ch3 (sst) and ch5 (siconc) have land NaNs but are handled by')
print('nan_to_num in dataset.py — they do not need to be in NAN_CHANNELS.')