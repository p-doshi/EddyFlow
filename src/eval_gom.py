"""
eval_gom.py — Evaluate TemporalDownscaler (stage 4/4v2/5a) on Gulf of Mexico.

Zero-shot domain transfer: model was trained on GSL, evaluated on GOM.

Domain: lat 18–31°N, lon 98–80°W (Loop Current eddies, shelf dynamics)
MUR fine grid: 1301×1801 at 0.01°
Test period:   2022–2023

Data required (see data_gom.sh to download):
  ERA5 : data/domains/gom/era5/era5_gom_{2021..2023}.nc
  MUR  : data/domains/gom/mur/mur_gom.zarr  (through 2023-12-31)
  GEBCO: data/domains/gom/gebco_gom.nc  (manual upload — GEBCO API broken)

Usage:
  python eval_gom.py \\
      --ckpt   ../outputs/checkpoints/stage4v2_sweep_4v2_0_lr0_sw0_merged.pt \\
      --stage  4v2 \\
      --output ../src/eval_fixed/eval_gom_stage4v2_try1.json \\
      --psd_curves --save_examples
"""

import os
import argparse
import gc
import json
import math
from pathlib import Path
from collections import defaultdict

# ── PROJ patch — must run before cartopy loads ────────────────────────────────
_proj_candidates = [
    os.environ.get('PROJ_DATA', ''),
    os.environ.get('PROJ_LIB',  ''),
    '/home/pdoshi/eddyflow-venv/share/proj',
    '/usr/share/proj',
]
for _p in _proj_candidates:
    if _p and os.path.isfile(os.path.join(_p, 'proj.db')):
        os.environ['PROJ_DATA'] = _p
        os.environ['PROJ_LIB']  = _p
        print(f'[proj] using PROJ data dir: {_p}')
        break
else:
    import sys, unittest.mock as _mock
    for _mod in ('cartopy', 'cartopy.crs', 'cartopy.feature'):
        sys.modules.setdefault(_mod, _mock.MagicMock())
    print('[proj] WARNING: PROJ data dir not found — cartopy mocked, '
          'map plots disabled. Metrics and JSON output unaffected.')
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
import zarr
from tqdm import tqdm

from config import cfg
from model  import TemporalDownscaler, TemporalDownscalerV2, TemporalDownscalerV5

STAGE_CHOICES = ('1', '2', '3', '4', '4v2', '5a', '5b')

# ── GOM domain constants ──────────────────────────────────────────────────────
GOM_EXTENT   = [-98.0, -80.0, 18.0, 31.0]   # [lon_min, lon_max, lat_min, lat_max]
GOM_DATA_DIR = Path('../data/domains/gom')
TEST_YEARS   = (2022, 2023)
SEASON_MONTHS = [3, 6, 9, 12]
ERA5_VARS_SHORT = ['t2m', 'msl', 'u10', 'v10', 'sst_era5', 'u850', 'v850', 't850']
ERA5_CHANNELS_FULL = [
    'u10', 'v10', 'msl', 'sst_era5', 't2m', 'siconc',
    'u850', 'v850', 't850', 'z850', 'q850',
    'u700', 'v700', 't700', 'z700', 'q700',
    'u500', 'v500', 't500', 'z500', 'q500',
]
ERA5_SST_CH = ERA5_CHANNELS_FULL.index('sst_era5')  # 3


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_stats():
    """Load SST and bathy stats from GSL training."""
    stats_dir = Path(cfg.STATS_DIR)
    sst_mean   = float(np.load(stats_dir / 'sst_mean.npy'))
    sst_std    = float(np.load(stats_dir / 'sst_std.npy'))
    bathy_mean = float(np.load(stats_dir / 'bathy_mean.npy'))
    bathy_std  = float(np.load(stats_dir / 'bathy_std.npy'))
    print(f'  sst_mean={sst_mean:.4f}  sst_std={sst_std:.4f}')
    return dict(sst_mean=sst_mean, sst_std=sst_std,
                bathy_mean=bathy_mean, bathy_std=bathy_std)


def compute_era5_stats_gom(era5_ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel ERA5 normalisation stats from the GOM test data."""
    means, stds = [], []
    print(f'  ERA5 GOM stats ({era5_ds.sizes["time"]} timesteps):')
    for v in ERA5_CHANNELS_FULL:
        if v in era5_ds:
            data = era5_ds[v].values.astype(np.float32)
            valid = data[np.isfinite(data)]
            mean = float(valid.mean()) if valid.size else 0.0
            std  = float(valid.std())  if valid.size else 1.0
            std  = std if std > 1e-6 else 1.0
            print(f'    {v:8s} mean={mean:10.3f}  std={std:8.4f}')
        else:
            mean, std = 0.0, 1.0
            print(f'    {v:8s} mean={mean:10.3f}  std={std:8.4f}  [missing->zero]')
        means.append(mean)
        stds.append(std)
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


def load_bathy_gom(stats: dict) -> torch.Tensor:
    """Load and normalise GOM GEBCO bathymetry."""
    bathy_nc = GOM_DATA_DIR / 'gebco_gom.nc'
    if not bathy_nc.exists():
        raise FileNotFoundError(
            f'Missing GOM bathy: {bathy_nc}\n'
            f'  Re-download with:\n'
            f'  curl -o {bathy_nc} "https://coastwatch.pfeg.noaa.gov/erddap/griddap/'
            f'etopo180.nc?altitude%5B(18.0):(31.0)%5D%5B(-98.0):(-80.0)%5D"'
        )

    ds = None
    for engine in ('h5netcdf', 'scipy', 'netcdf4', None):
        try:
            ds = xr.open_dataset(bathy_nc, engine=engine) if engine else xr.open_dataset(bathy_nc)
            break
        except Exception:
            continue

    if ds is None:
        raise RuntimeError(f'Could not open bathy file: {bathy_nc}')

    elev_var = 'elevation' if 'elevation' in ds.data_vars else next(iter(ds.data_vars))
    elev = ds[elev_var].values.astype(np.float32)

    depth = np.where(elev < 0, -elev, 0.0)
    blog  = np.log1p(depth)
    bnorm = (blog - stats['bathy_mean']) / (stats['bathy_std'] + 1e-6)
    ds.close()
    return torch.from_numpy(bnorm).unsqueeze(0)  # [1, H_bathy, W_bathy]


def open_era5_gom(years) -> xr.Dataset:
    """Load GOM ERA5 NetCDF files for the given year range."""
    files = []
    for yr in range(years[0], years[1] + 1):
        p = GOM_DATA_DIR / 'era5' / f'era5_gom_{yr}.nc'
        if not p.exists():
            raise FileNotFoundError(f'Missing ERA5 file: {p}  (run data_gom.sh first)')
        files.append(str(p))
    ds = xr.open_mfdataset(files, combine='by_coords',
                            decode_timedelta=False, engine='h5netcdf')
    if 'valid_time' in ds and 'time' not in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    return ds


def open_mur_gom():
    """Open GOM MUR zarr and times array."""
    zarr_path = GOM_DATA_DIR / 'mur' / 'mur_gom.zarr'
    times_npy = GOM_DATA_DIR / 'mur' / 'mur_gom_times.npy'
    if not zarr_path.exists():
        raise FileNotFoundError(f'Missing MUR zarr: {zarr_path}')
    store = zarr.open(str(zarr_path), 'r')
    times = np.load(str(times_npy), allow_pickle=True)
    return store, times


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE INDEX
# ═══════════════════════════════════════════════════════════════════════════════

def build_sample_index(era5_ds: xr.Dataset, mur_times_raw, T: int, T_oce: int = 0):
    """
    T      — ERA5 lookback days (cfg.T_ATM for stage 4+)
    T_oce  — MUR ocean history frames (cfg.T_OCE for stage 4+)
    """
    try:
        era5_pd = pd.DatetimeIndex(era5_ds['time'].values)
    except Exception:
        era5_pd = pd.DatetimeIndex([str(t) for t in era5_ds['time'].values])

    try:
        mur_pd = pd.DatetimeIndex(mur_times_raw)
    except Exception:
        mur_pd = pd.DatetimeIndex([str(t) for t in mur_times_raw])

    era5_date_to_idx = {t.date(): i for i, t in enumerate(era5_pd)}

    samples = []
    for mur_i, mur_ts in enumerate(mur_pd):
        mur_date = mur_ts.date()

        era5_frames, valid = [], True
        for days_back in range(T - 1, -1, -1):
            target = (pd.Timestamp(mur_date) - pd.Timedelta(days=days_back)).date()
            if target not in era5_date_to_idx:
                valid = False
                break
            era5_frames.append(era5_date_to_idx[target])

        if not valid or len(era5_frames) != T:
            continue

        if T_oce > 0 and mur_i < T_oce:
            continue

        mur_prev_idx = max(mur_i - 1, 0)
        samples.append((mur_i, era5_frames, mur_prev_idx,
                        str(mur_date), int(mur_ts.month)))

    print(f'  Built {len(samples)} samples '
          f'(dropped {len(mur_pd) - len(samples)} — missing lookback)')
    return samples, era5_pd, mur_pd


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _load_mur_norm(idx, mur_store, sst_mean, sst_std):
    """Load one MUR frame, convert K→°C (GOM MUR is in Kelvin), normalise."""
    raw  = mur_store['analysed_sst'][idx].astype(np.float32)
    raw  = np.where(np.isfinite(raw), raw - 273.15, raw)   # K → °C
    norm = (raw - sst_mean) / (sst_std + 1e-6)
    norm = np.where(np.isfinite(norm), norm, 0.0)
    return torch.from_numpy(norm).unsqueeze(0)   # [1, H_f, W_f]


def build_batch(sample, era5_ds, mur_store, stats, bathy,
                T, LAT_C, LON_C, T_oce=0):
    mur_i, era5_frames, mur_prev_i, date_str, month = sample
    sst_mean, sst_std = stats['sst_mean'], stats['sst_std']
    era5_mean = stats['era5_mean']
    era5_std  = stats['era5_std']
    n_era5_ch = len(ERA5_CHANNELS_FULL)

    # ── ERA5 window ──────────────────────────────────────────────────────────
    frames = []
    for ei in era5_frames:
        chans = []
        for ci, v in enumerate(ERA5_CHANNELS_FULL):
            if v in era5_ds:
                arr = era5_ds[v].isel(time=ei).values.astype(np.float32)
                if arr.ndim == 3:
                    arr = arr[0]
            else:
                ref = era5_ds[list(era5_ds.data_vars)[0]].isel(time=ei).values.astype(np.float32)
                if ref.ndim == 3:
                    ref = ref[0]
                arr = np.zeros_like(ref, dtype=np.float32)

            fill_value = cfg.NAN_FILL_CHANNELS.get(ci, None)
            if fill_value is not None:
                arr = np.where(np.isfinite(arr), arr, fill_value)

            arr = (arr - era5_mean[ci]) / (era5_std[ci] + 1e-6)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            chans.append(arr)

        frames.append(np.stack(chans, axis=0))  # [21, H_e, W_e]

    era5_window = np.stack(frames, axis=0)   # [T, 21, H_e, W_e]
    era5_t = torch.from_numpy(era5_window)
    era5_t = F.interpolate(
        era5_t.reshape(T * n_era5_ch, 1, era5_t.shape[-2], era5_t.shape[-1]),
        size=(LAT_C, LON_C), mode='bilinear', align_corners=False
    ).reshape(T, n_era5_ch, LAT_C, LON_C)

    ice_coarse    = torch.zeros(1, LAT_C, LON_C, dtype=torch.float32)
    bathy_coarse  = F.adaptive_avg_pool2d(bathy.unsqueeze(0), (LAT_C, LON_C)).squeeze(0)
    static        = torch.cat([bathy_coarse, ice_coarse], dim=0)
    static_seq    = static.unsqueeze(0).expand(T, -1, -1, -1).contiguous()
    era5_input    = torch.cat([era5_t, static_seq], dim=1)  # [T, 23, LAT_C, LON_C]
    era5_input    = torch.nan_to_num(era5_input, nan=0.0)

    # ── MUR target ───────────────────────────────────────────────────────────
    sst_abs  = _load_mur_norm(mur_i,      mur_store, sst_mean, sst_std)
    sst_prev = _load_mur_norm(mur_prev_i, mur_store, sst_mean, sst_std)
    delta    = sst_abs - sst_prev

    try:
        ice_fine = torch.from_numpy(
            mur_store['sea_ice_fraction'][mur_i].astype(np.float32)).unsqueeze(0)
        ice_mask = (ice_fine > 0.15)
    except Exception:
        ice_mask = torch.zeros_like(sst_abs, dtype=torch.bool)

    sst_raw_valid = torch.isfinite(torch.from_numpy(
        mur_store['analysed_sst'][mur_i].astype(np.float32)).unsqueeze(0))
    weight = (~ice_mask & sst_raw_valid).float()

    # ── Resize bathy to MUR fine grid ────────────────────────────────────────
    fine_size = sst_abs.shape[-2:]
    if tuple(bathy.shape[-2:]) != tuple(fine_size):
        bathy = F.interpolate(bathy.unsqueeze(0), size=fine_size,
                              mode='bilinear', align_corners=False).squeeze(0)

    # ── ERA5 SST persistence baseline (fixes NAN_FILL_CHANNELS bug) ──────────
    ei_last = era5_frames[-1]
    arr_sst_raw = era5_ds['sst_era5'].isel(time=ei_last).values.astype(np.float32)
    if arr_sst_raw.ndim == 3:
        arr_sst_raw = arr_sst_raw[0]
    arr_sst_raw = np.where(
        np.isfinite(arr_sst_raw),
        arr_sst_raw,
        float(era5_mean[ERA5_SST_CH]),   # fill land with ERA5 SST mean (K)
    )
    era5_sst_raw  = torch.from_numpy(arr_sst_raw).unsqueeze(0)
    era5_sst_c    = era5_sst_raw - 273.15
    era5_sst_fine = F.interpolate(
        era5_sst_c.unsqueeze(0), size=fine_size, mode='bilinear', align_corners=False
    ).squeeze(0)
    era5_sst_fine = (era5_sst_fine - sst_mean) / (sst_std + 1e-6)

    out = {
        'era5':          era5_input.unsqueeze(0),
        'sst':           delta.unsqueeze(0),
        'sst_abs':       sst_abs.unsqueeze(0),
        'sst_prev':      sst_prev.unsqueeze(0),
        'weight':        weight.unsqueeze(0),
        'bathy':         bathy.unsqueeze(0),
        'era5_sst_fine': era5_sst_fine.unsqueeze(0),
        'date_str':      [date_str],
        'time_str':      ['09:00:00'],
        'month':         torch.tensor([month]),
    }

    if T_oce > 0:
        hist = []
        for days_back in range(T_oce, 0, -1):
            hi = mur_i - days_back
            hist.append(_load_mur_norm(hi, mur_store, sst_mean, sst_std))
        out['mur_seq'] = torch.stack(hist, dim=0).unsqueeze(0)

    if T_oce > 0 and 'mur_seq' not in out:
        raise RuntimeError('Stage 4/4v2 requires mur_seq.')

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# PSD / VIZ UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def azimuthal_psd(field, dx_km):
    H, W   = field.shape
    f2d    = np.fft.rfft2(field)
    p2d    = (np.abs(f2d) ** 2) / (H * W)
    ky     = np.fft.fftfreq(H,  d=dx_km)
    kx     = np.fft.rfftfreq(W, d=dx_km)
    KX, KY = np.meshgrid(kx, ky)
    K2D    = np.sqrt(KX**2 + KY**2)
    k_max  = min(kx.max(), np.abs(ky).max())
    n_bins = max(H, W) // 2
    k_edges = np.linspace(0.0, k_max, n_bins + 1)
    k_cents = 0.5 * (k_edges[:-1] + k_edges[1:])
    psd = np.array([
        p2d[(K2D >= k_edges[i]) & (K2D < k_edges[i+1])].mean()
        if ((K2D >= k_edges[i]) & (K2D < k_edges[i+1])).sum() > 0 else 0.0
        for i in range(n_bins)
    ])
    return k_cents, psd


def band_mean_psd(field, dx_km, lam_lo=5.0, lam_hi=50.0):
    k, psd = azimuthal_psd(field, dx_km)
    sel = (k >= 1/lam_hi) & (k <= 1/lam_lo)
    return float(psd[sel].mean()) if sel.sum() > 0 else float('nan')


def safe_slug(s):
    return str(s).replace(':', '-').replace(' ', '_').replace('/', '-')


def month_name(m):
    return ['','Jan','Feb','Mar','Apr','May','Jun',
            'Jul','Aug','Sep','Oct','Nov','Dec'][int(m)]


def compute_field_metrics(pred, target):
    mask = np.isfinite(pred) & np.isfinite(target)
    if not mask.any():
        return dict(n_valid=0, rmse=None, mae=None, bias=None,
                    min_err=None, max_err=None)
    err = pred[mask] - target[mask]
    return dict(n_valid=int(mask.sum()),
                rmse=float(np.sqrt(np.mean(err**2))),
                mae=float(np.mean(np.abs(err))),
                bias=float(np.mean(err)),
                min_err=float(err.min()), max_err=float(err.max()))


def save_map_png(out_path, field, title, extent=GOM_EXTENT,
                 cmap='viridis', vmin=None, vmax=None):
    vis = np.ma.masked_invalid(gaussian_filter(
        np.where(np.isfinite(field), field, 0.0), sigma=0.4))
    vis = np.ma.masked_where(~np.isfinite(field), vis)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='#b0b0b0')
    fig = plt.figure(figsize=(14, 7), dpi=150)
    try:
        ax  = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    except (ValueError, TypeError, AttributeError):
        # cartopy mocked (PROJ not loaded) — save plain imshow instead
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
        lon0, lon1, lat0, lat1 = extent
        im = ax.imshow(vis, origin='lower', extent=[lon0, lon1, lat0, lat1],
                       aspect='auto', cmap=cmap_obj, vmin=vmin, vmax=vmax)
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03).set_label('Normalised SST')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return
    ax.coastlines(resolution='10m', linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor='#d9d3c3', zorder=10)
    gl  = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False
    im  = ax.imshow(vis, origin='lower', extent=extent,
                    transform=ccrs.PlateCarree(), cmap=cmap_obj,
                    vmin=vmin, vmax=vmax, interpolation='bicubic', zorder=1)
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03).set_label('Normalised SST')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_comparison_set(out_root, date_str, time_str, tag,
                        pred, era5_vis, mur, extent=GOM_EXTENT):
    folder = out_root / f'{safe_slug(date_str)}_{safe_slug(time_str)}_{tag}'
    folder.mkdir(parents=True, exist_ok=True)
    mur_vals = mur[np.isfinite(mur)]
    vmin = float(np.nanpercentile(mur_vals, 1))
    vmax = float(np.nanpercentile(mur_vals, 99))
    if vmin == vmax:
        vmax = vmin + 1e-6

    save_map_png(folder/'me.png',   pred,     f'Prediction — {date_str}',
                 extent=extent, vmin=vmin, vmax=vmax)
    save_map_png(folder/'mur.png',  mur,      f'MUR target — {date_str}',
                 extent=extent, vmin=vmin, vmax=vmax)
    save_map_png(folder/'era5.png', era5_vis, f'ERA5 SST — {date_str}',
                 extent=extent, vmin=vmin, vmax=vmax)

    m1 = compute_field_metrics(pred, mur)
    m2 = compute_field_metrics(era5_vis, mur)

    def fmt(v): return f'{v:.6f}' if v is not None else 'N/A'
    stats_html = f"""
    <div class="meta-card"><h3>Sample details</h3><table>
      <tr><td>Date</td><td>{date_str}</td></tr>
      <tr><td>Domain</td><td>Gulf of Mexico</td></tr></table></div>
    <div class="meta-card"><h3>Model vs MUR</h3><table>
      <tr><td>Valid pixels</td><td>{m1['n_valid']}</td></tr>
      <tr><td>RMSE</td><td>{fmt(m1['rmse'])}</td></tr>
      <tr><td>MAE</td><td>{fmt(m1['mae'])}</td></tr>
      <tr><td>Bias</td><td>{fmt(m1['bias'])}</td></tr></table></div>
    <div class="meta-card"><h3>ERA5 vs MUR</h3><table>
      <tr><td>Valid pixels</td><td>{m2['n_valid']}</td></tr>
      <tr><td>RMSE</td><td>{fmt(m2['rmse'])}</td></tr>
      <tr><td>MAE</td><td>{fmt(m2['mae'])}</td></tr>
      <tr><td>Bias</td><td>{fmt(m2['bias'])}</td></tr></table></div>"""

    # Minimal slider HTML
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>GOM {date_str}</title>
<style>
body{{margin:0;font-family:sans-serif;background:#0f1115;color:#e8edf2}}
.wrap{{max-width:1320px;margin:0 auto;padding:20px}}
.compare{{position:relative;width:100%;overflow:hidden;cursor:ew-resize;user-select:none}}
.compare>img{{display:block;width:100%}}
.img-top{{position:absolute;inset:0;will-change:clip-path}}
.img-top img{{width:100%;height:100%;object-fit:cover}}
.divider{{position:absolute;top:0;bottom:0;width:2px;background:#fff;transform:translateX(-50%);z-index:10;will-change:left}}
.knob{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:36px;height:36px;border-radius:50%;background:#fff;display:flex;align-items:center;justify-content:center}}
.lbl{{position:absolute;top:8px;padding:2px 8px;border-radius:4px;font-size:.7rem;font-weight:700;background:rgba(0,0,0,.6);color:#fff;z-index:11}}
.lbl-l{{left:8px}}.lbl-r{{right:8px}}
</style></head><body><div class="wrap">
<h2>GOM — {date_str}</h2>
<div class="compare" id="c">
  <img src="me.png" alt="Prediction">
  <div class="img-top" id="it" style="clip-path:inset(0 50% 0 0)">
    <img src="mur.png" alt="MUR">
  </div>
  <div class="divider" id="div" style="left:50%">
    <div class="knob">⟺</div>
  </div>
  <span class="lbl lbl-l">MUR truth</span>
  <span class="lbl lbl-r">Model pred</span>
</div>
<div style="background:#171a21;border-radius:12px;padding:16px;margin-top:16px">
{stats_html}
</div></div>
<script>
const w=document.getElementById('c'),top=document.getElementById('it'),dv=document.getElementById('div');
let drag=false,raf=null,px=0;
function apply(){{const r=w.getBoundingClientRect(),p=Math.max(0,Math.min(100,(px-r.left)/r.width*100));
  top.style.clipPath=`inset(0 ${{100-p}}% 0 0)`;dv.style.left=p+'%';raf=null;}}
function sched(x){{px=x;if(!raf)raf=requestAnimationFrame(apply);}}
w.addEventListener('mousedown',e=>{{drag=true;sched(e.clientX);e.preventDefault()}});
window.addEventListener('mousemove',e=>{{if(drag)sched(e.clientX)}});
window.addEventListener('mouseup',()=>drag=false);
</script></body></html>"""
    (folder / 'compare.html').write_text(html, encoding='utf-8')
    return folder


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint: str, stage: str, device: torch.device,
               n_diff_steps: int | None = None):
    if stage in ('1', '2', '3', '4'):
        model = TemporalDownscaler(stage=int(stage))
    elif stage == '4v2':
        model = TemporalDownscalerV2()
    elif stage in ('5a', '5b'):
        model = TemporalDownscalerV5()
    else:
        raise ValueError(f'Unknown stage: {stage}')

    ckpt = torch.load(checkpoint, map_location=device)
    sd   = ckpt.get('model',
           ckpt.get('model_state_dict',
           ckpt.get('state_dict', None)))
    if sd is None:
        raise KeyError(f'Cannot find model weights in checkpoint. Keys: {list(ckpt.keys())}')

    model.load_state_dict(sd, strict=True)
    if n_diff_steps is not None:
        cfg.DIFF_SAMPLE_STEPS = n_diff_steps
    return model.to(device).eval()


def resolve_window(stage: str) -> tuple[int, int]:
    if stage in ('4', '4v2', '5a', '5b'):
        return cfg.T_ATM, cfg.T_OCE
    else:
        return cfg.T, 0


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device)
    stats  = load_stats()

    # ERA5 GOM — 2021-2023 if available (2021 provides T_ATM=28 day lookback for early Jan 2022),
    # falls back to 2022-2023 if 2021 was not downloaded (data_gom.sh downloads 2022-2023 only;
    # build_sample_index automatically drops the first ~28 Jan 2022 samples with missing lookback).
    print('\nLoading GOM ERA5...')
    try:
        era5_ds = open_era5_gom((2021, 2023))
    except FileNotFoundError:
        print('  2021 ERA5 not found — using 2022-2023 only (~28 Jan 2022 samples dropped)')
        era5_ds = open_era5_gom((2022, 2023))
    era5_mean_gom, era5_std_gom = compute_era5_stats_gom(era5_ds)
    stats['era5_mean'] = era5_mean_gom
    stats['era5_std']  = era5_std_gom

    sst_std = args.sst_std if args.sst_std else stats['sst_std']

    print(f'\n{"═"*60}')
    print(f'  GOM Evaluation — Stage {args.stage}')
    print(f'{"═"*60}')
    print(f'  Checkpoint : {args.ckpt}')
    print(f'  Test years : {TEST_YEARS}')
    print(f'  Device     : {device}')

    model    = load_model(args.ckpt, args.stage, device, args.n_diff_steps)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters : {n_params / 1e6:.2f} M')

    bathy                = load_bathy_gom(stats)
    mur_store, mur_times = open_mur_gom()

    T, T_oce = resolve_window(args.stage)
    LAT_C = cfg.LAT_C
    LON_C = cfg.LON_C

    # Filter MUR to test years
    mur_pd    = pd.DatetimeIndex(mur_times)
    year_mask = (mur_pd.year >= TEST_YEARS[0]) & (mur_pd.year <= TEST_YEARS[1])

    mur_times_test     = mur_times[year_mask]
    mur_global_offsets = np.where(year_mask)[0]

    if len(mur_times_test) == 0:
        raise RuntimeError(
            f'No MUR data for test years {TEST_YEARS}. '
            f'Available range: {mur_pd.min()} → {mur_pd.max()}. '
            f'Run data_gom.sh to download 2022-2023 data first.'
        )

    samples, era5_pd, _ = build_sample_index(era5_ds, mur_times_test, T, T_oce)
    print(f'  Samples    : {len(samples)}\n')

    sq_model = sq_persist = sq_murprev = 0.0
    n_valid  = 0
    psd_pred_band, psd_targ_band = [], []
    n_samples = 0
    bias_sum   = {m: 0.0 for m in range(1, 13)}
    bias_count = {m: 0   for m in range(1, 13)}

    example_dir    = Path(args.example_dir)
    month_to_saved = {m: 0 for m in SEASON_MONTHS}

    for sample in tqdm(samples, desc='GOM eval'):
        local_mur_i     = sample[0]
        global_mur_i    = int(mur_global_offsets[local_mur_i])
        global_prev_i   = max(global_mur_i - 1, 0)
        sample_global   = (global_mur_i, sample[1], global_prev_i,
                           sample[3], sample[4])

        try:
            batch = build_batch(sample_global, era5_ds, mur_store, stats,
                                bathy, T, LAT_C, LON_C, T_oce=T_oce)
        except Exception as e:
            print(f'  Skipping {sample[3]}: {e}')
            continue

        dev_batch = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred = model.sample(dev_batch)

        pred         = (pred + dev_batch['sst_prev']).float()
        sst          = dev_batch['sst_abs'].float()
        weight       = dev_batch['weight'].float()
        persist      = dev_batch['sst_prev'].float()
        era5_persist = dev_batch['era5_sst_fine'].float()

        valid = (weight > 0) & torch.isfinite(sst) & torch.isfinite(pred)
        vf    = valid.float()

        sq_model   += ((pred         - sst)**2 * vf).sum().item()
        sq_persist += ((era5_persist - sst)**2 * vf).sum().item()
        sq_murprev += ((persist      - sst)**2 * vf).sum().item()
        n_valid    += valid.sum().item()

        # Seasonal mean bias (physical °C)
        err_phys = ((pred - sst) * vf * sst_std)
        m_idx = int(sample[4])
        bias_sum[m_idx]   += err_phys.sum().item()
        bias_count[m_idx] += valid.sum().item()

        pred_np = (pred * vf).squeeze().cpu().numpy()
        targ_np = (sst  * vf).squeeze().cpu().numpy()
        psd_pred_band.append(band_mean_psd(pred_np, args.dx_km, args.lam_lo_km, args.lam_hi_km))
        psd_targ_band.append(band_mean_psd(targ_np, args.dx_km, args.lam_lo_km, args.lam_hi_km))
        n_samples += 1

        if args.save_examples:
            m_idx2 = int(sample[4])
            if m_idx2 in SEASON_MONTHS and month_to_saved.get(m_idx2, 0) < 1:
                date_str = sample[3]
                pred_vis   = (pred * vf).squeeze().cpu().numpy()
                sst_vis    = (sst  * vf).squeeze().cpu().numpy()
                era5_vis_n = (era5_persist * vf).squeeze().cpu().numpy()
                save_comparison_set(
                    example_dir, date_str, '09-00-00',
                    f'{month_name(m_idx2).lower()}_00',
                    pred_vis, era5_vis_n, sst_vis
                )
                month_to_saved[m_idx2] = 1

        gc.collect()

    # ── Compute final metrics ─────────────────────────────────────────────────
    if n_valid == 0:
        raise RuntimeError('No valid pixels — check data and mask.')

    rmse_model   = math.sqrt(sq_model   / n_valid)
    rmse_persist = math.sqrt(sq_persist / n_valid)
    rmse_murprev = math.sqrt(sq_murprev / n_valid)
    skill_score  = 1.0 - rmse_model / rmse_persist

    # Seasonal bias
    SEASON_MAP = {
        'DJF': [12, 1, 2], 'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],  'SON': [9, 10, 11],
    }
    seasonal_bias, monthly_bias = {}, {}
    for season, months in SEASON_MAP.items():
        s_sum = sum(bias_sum[m]   for m in months)
        s_cnt = sum(bias_count[m] for m in months)
        seasonal_bias[season] = float(s_sum / s_cnt) if s_cnt > 0 else float('nan')
    for m_i in range(1, 13):
        monthly_bias[str(m_i)] = (float(bias_sum[m_i] / bias_count[m_i])
                                  if bias_count[m_i] > 0 else float('nan'))

    # PSD ratio
    valid_psd = [(p, t) for p, t in zip(psd_pred_band, psd_targ_band)
                 if math.isfinite(p) and math.isfinite(t) and t > 0]
    if valid_psd:
        ratios    = [p / t for p, t in valid_psd]
        psd_ratio = float(np.mean(ratios))
        psd_log10 = float(np.mean(np.log10(ratios)))
    else:
        psd_ratio = float('nan')
        psd_log10 = float('nan')

    print(f'\n{"═"*60}')
    print(f'  GOM Results — Stage {args.stage}')
    print(f'  n_samples    : {n_samples}')
    print(f'  n_valid px   : {n_valid:,.0f}')
    print(f'  rmse_model   : {rmse_model:.4f}°C (normalised) | '
          f'{rmse_model * sst_std:.4f}°C physical')
    print(f'  rmse_persist : {rmse_persist:.4f} (ERA5 coarse)')
    print(f'  rmse_murprev : {rmse_murprev:.4f} (oracle: MUR yesterday)')
    print(f'  skill_score  : {skill_score:.4f}')
    print(f'  psd_ratio    : {psd_ratio:.4f}')
    print(f'  seasonal bias (°C): DJF={seasonal_bias["DJF"]:+.3f}  '
          f'MAM={seasonal_bias["MAM"]:+.3f}  '
          f'JJA={seasonal_bias["JJA"]:+.3f}  '
          f'SON={seasonal_bias["SON"]:+.3f}')

    results = {
        'domain':          'gom',
        'stage':           args.stage,
        'checkpoint':      args.ckpt,
        'test_years':      list(TEST_YEARS),
        'n_samples':       n_samples,
        'n_valid_pixels':  n_valid,
        'rmse_model_K':    rmse_model,
        'rmse_persist_K':  rmse_persist,
        'rmse_murprev_K':  rmse_murprev,
        'skill_score':     skill_score,
        'psd_ratio':       psd_ratio,
        'psd_log10_bias':  psd_log10,
        'seasonal_bias_K': seasonal_bias,
        'monthly_bias_K':  monthly_bias,
        'dx_km':           args.dx_km,
        'lam_lo_km':       args.lam_lo_km,
        'lam_hi_km':       args.lam_hi_km,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\n  Results saved → {args.output}')

    return results


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate EddyFlow on GOM (zero-shot)')
    p.add_argument('--ckpt',          required=True,  help='Path to merged checkpoint')
    p.add_argument('--stage',         default='4v2',  choices=STAGE_CHOICES)
    p.add_argument('--output',        default='../src/eval_fixed/eval_gom_stage4v2.json')
    p.add_argument('--device',        default='cuda')
    p.add_argument('--n_diff_steps',  type=int, default=None)
    p.add_argument('--sst_std',       type=float, default=None,
                   help='Override sst_std for physical RMSE conversion')
    p.add_argument('--dx_km',         type=float, default=1.11,
                   help='Pixel size in km for PSD (0.01° ≈ 1.11 km, varies with lat)')
    p.add_argument('--lam_lo_km',     type=float, default=5.0)
    p.add_argument('--lam_hi_km',     type=float, default=50.0)
    p.add_argument('--psd_curves',    action='store_true')
    p.add_argument('--save_examples', action='store_true')
    p.add_argument('--example_dir',   default='../outputs/figures/gom')
    return p.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
