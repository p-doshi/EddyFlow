"""
evaluate_bof.py — Evaluate TemporalDownscaler on Bay of Fundy domain data.

Loads:
  - ERA5: data/domains/bof/era5/era5_bof_{year}.nc  (2022-2023 for test)
  - MUR:  data/domains/bof/mur/mur_bof.zarr
  - Stats: computed on-the-fly from BOF ERA5 (no GSL stats required)

Usage:
  python eval_bof.py \
      --checkpoint ../outputs/checkpoints/stage4_phase2_new2try1_merged.pt \
      --stage      4 \
      --output     results_bof_stage4.json \
      --save_examples \
      --example_dir ../outputs/figures/bof
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

# ── BOF domain constants ──────────────────────────────────────────────────────
BOF_EXTENT   = [-67.0, -63.0, 44.0, 47.0]
BOF_DATA_DIR = Path('../data/domains/bof')
TEST_YEARS   = (2022, 2023)
SEASON_MONTHS = [3, 6, 9, 12]
ERA5_VARS_SHORT = ['t2m', 'msl', 'u10', 'v10', 'sst_era5', 'u850', 'v850', 't850']
ERA5_CHANNELS_FULL = [
    'u10', 'v10', 'msl', 'sst_era5', 't2m', 'siconc',
    'u850', 'v850', 't850', 'z850', 'q850',
    'u700', 'v700', 't700', 'z700', 'q700',
    'u500', 'v500', 't500', 'z500', 'q500',
]
ERA5_SST_CH = ERA5_CHANNELS_FULL.index('sst_era5')  # index 3 in the full channel list
print(len(ERA5_CHANNELS_FULL), ERA5_CHANNELS_FULL)
# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_stats():
    """Load SST and bathy stats from GSL training. ERA5 stats computed separately."""
    stats_dir = Path(cfg.STATS_DIR)
    sst_mean   = float(np.load(stats_dir / 'sst_mean.npy'))
    sst_std    = float(np.load(stats_dir / 'sst_std.npy'))
    bathy_mean = float(np.load(stats_dir / 'bathy_mean.npy'))
    bathy_std  = float(np.load(stats_dir / 'bathy_std.npy'))
    print(f'  sst_mean={sst_mean:.4f}  sst_std={sst_std:.4f}')
    return dict(sst_mean=sst_mean, sst_std=sst_std,
                bathy_mean=bathy_mean, bathy_std=bathy_std)


def compute_era5_stats_bof(era5_ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    means, stds = [], []
    print(f'  ERA5 BOF stats ({era5_ds.sizes["time"]} timesteps):')
    for v in ERA5_CHANNELS_FULL:
        if v in era5_ds:
            data = era5_ds[v].values.astype(np.float32)
            valid = data[np.isfinite(data)]
            mean = float(valid.mean()) if valid.size else 0.0
            std  = float(valid.std()) if valid.size else 1.0
            std  = std if std > 1e-6 else 1.0
            print(f'    {v:8s} mean={mean:10.3f}  std={std:8.4f}')
        else:
            mean, std = 0.0, 1.0
            print(f'    {v:8s} mean={mean:10.3f}  std={std:8.4f}  [missing->zero]')
        means.append(mean)
        stds.append(std)
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


def load_bathy_bof(stats: dict) -> torch.Tensor:
    bathy_nc = BOF_DATA_DIR / 'gebco_bof.nc'
    if not bathy_nc.exists():
        raise FileNotFoundError(f'Missing BOF bathy: {bathy_nc}')

    ds = None
    for engine in ('scipy', 'h5netcdf', 'netcdf4', None):
        try:
            ds = xr.open_dataset(bathy_nc, engine=engine) if engine else xr.open_dataset(bathy_nc)
            break
        except Exception:
            continue

    if ds is None:
        raise RuntimeError(f'Could not open bathy file with any backend: {bathy_nc}')

    elev_var = 'elevation' if 'elevation' in ds.data_vars else next(iter(ds.data_vars))
    elev = ds[elev_var].values.astype(np.float32)

    depth = np.where(elev < 0, -elev, 0.0)
    blog  = np.log1p(depth)
    bnorm = (blog - stats['bathy_mean']) / (stats['bathy_std'] + 1e-6)
    ds.close()
    return torch.from_numpy(bnorm).unsqueeze(0)

def open_era5_bof(years) -> xr.Dataset:
    files = []
    for yr in range(years[0], years[1] + 1):
        p = BOF_DATA_DIR / 'era5' / f'era5_bof_{yr}.nc'
        if not p.exists():
            raise FileNotFoundError(f'Missing ERA5 file: {p}')
        files.append(str(p))
    ds = xr.open_mfdataset(files, combine='by_coords',
                           decode_timedelta=False, engine='h5netcdf')
    if 'valid_time' in ds and 'time' not in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    return ds


def open_mur_bof():
    zarr_path = BOF_DATA_DIR / 'mur' / 'mur_bof.zarr'
    times_npy = BOF_DATA_DIR / 'mur' / 'mur_bof_times.npy'
    if not zarr_path.exists():
        raise FileNotFoundError(f'Missing MUR zarr: {zarr_path}')
    store = zarr.open(str(zarr_path), 'r')
    times = np.load(str(times_npy), allow_pickle=True)
    return store, times


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE INDEX BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_sample_index(era5_ds: xr.Dataset, mur_times_raw, T: int,
                       T_oce: int = 0):
    """
    T      — ERA5 lookback days (cfg.T_ATM for stage 4, cfg.T otherwise)
    T_oce  — MUR ocean history days (cfg.T_OCE for stage 4, 0 otherwise)
    """
    try:
        era5_pd = pd.DatetimeIndex(era5_ds['time'].values)
    except Exception:
        era5_pd = pd.DatetimeIndex([str(t) for t in era5_ds['time'].values])

    try:
        mur_pd = pd.DatetimeIndex(mur_times_raw)
    except Exception:
        mur_pd = pd.DatetimeIndex([str(t) for t in mur_times_raw])

    # BOF ERA5 is daily at 12:00 — key lookup by date only
    era5_date_to_idx = {t.date(): i for i, t in enumerate(era5_pd)}

    samples = []
    for mur_i, mur_ts in enumerate(mur_pd):
        mur_date = mur_ts.date()

        # T daily ERA5 frames ending on mur_date
        era5_frames, valid = [], True
        for days_back in range(T - 1, -1, -1):
            target = (pd.Timestamp(mur_date) - pd.Timedelta(days=days_back)).date()
            if target not in era5_date_to_idx:
                valid = False
                break
            era5_frames.append(era5_date_to_idx[target])

        if not valid or len(era5_frames) != T:
            continue

        # T_oce MUR history frames — need mur_i - T_oce >= 0 within the slice
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
    raw  = mur_store['analysed_sst'][idx].astype(np.float32)
    # BOF MUR zarr stores SST in Kelvin; GSL stats are in Celsius → convert
    raw  = np.where(np.isfinite(raw), raw - 273.15, raw)
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

    frames = []
    for ei in era5_frames:
        chans = []
        for ci, v in enumerate(ERA5_CHANNELS_FULL):
            if v in era5_ds:
                arr = era5_ds[v].isel(time=ei).values.astype(np.float32)
                if arr.ndim == 3:
                    arr = arr[0]
            else:
                sample_ref = era5_ds[list(era5_ds.data_vars)[0]].isel(time=ei).values.astype(np.float32)
                if sample_ref.ndim == 3:
                    sample_ref = sample_ref[0]
                arr = np.zeros_like(sample_ref, dtype=np.float32)

            fill_value = cfg.NAN_FILL_CHANNELS.get(ci, None)
            if fill_value is not None:
                arr = np.where(np.isfinite(arr), arr, fill_value)

            arr = (arr - era5_mean[ci]) / (era5_std[ci] + 1e-6)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            chans.append(arr)

        frame = np.stack(chans, axis=0)  # [21, H_e, W_e]
        frames.append(frame)

    era5_window = np.stack(frames, axis=0)  # [T, 21, H_e, W_e]

    era5_t = torch.from_numpy(era5_window)
    era5_t = F.interpolate(
        era5_t.reshape(T * n_era5_ch, 1, era5_t.shape[-2], era5_t.shape[-1]),
        size=(LAT_C, LON_C), mode='bilinear', align_corners=False
    ).reshape(T, n_era5_ch, LAT_C, LON_C)

    ice_coarse = torch.zeros(1, LAT_C, LON_C, dtype=torch.float32)
    bathy_coarse = F.adaptive_avg_pool2d(bathy.unsqueeze(0), (LAT_C, LON_C)).squeeze(0)
    static = torch.cat([bathy_coarse, ice_coarse], dim=0)
    static_seq = static.unsqueeze(0).expand(T, -1, -1, -1).contiguous()

    era5_input = torch.cat([era5_t, static_seq], dim=1)  # [T, 23, LAT_C, LON_C]
    era5_input = torch.nan_to_num(era5_input, nan=0.0)

    sst_abs  = _load_mur_norm(mur_i, mur_store, sst_mean, sst_std)
    sst_prev = _load_mur_norm(mur_prev_i, mur_store, sst_mean, sst_std)
    delta    = sst_abs - sst_prev

    try:
        ice_fine = torch.from_numpy(mur_store['sea_ice_fraction'][mur_i].astype(np.float32)).unsqueeze(0)
        ice_mask = (ice_fine > 0.15)
    except Exception:
        ice_mask = torch.zeros_like(sst_abs, dtype=torch.bool)

    sst_raw_valid = torch.isfinite(torch.from_numpy(mur_store['analysed_sst'][mur_i].astype(np.float32)).unsqueeze(0))
    weight = (~ice_mask & sst_raw_valid).float()

    # Resize bathy to match MUR fine grid — GEBCO may be at native resolution
    # (e.g. 720×960 for 15 arc-sec) while MUR is 0.01° (301×401 for BOF)
    fine_size = sst_abs.shape[-2:]
    if tuple(bathy.shape[-2:]) != tuple(fine_size):
        bathy = F.interpolate(bathy.unsqueeze(0), size=fine_size,
                              mode='bilinear', align_corners=False).squeeze(0)

    # ERA5 SST for persistence baseline: load raw Kelvin values at the last frame,
    # bypassing cfg.NAN_FILL_CHANNELS[3]=0.0 which maps land NaN→0K (=−273°C after
    # K→C), producing z-scores of −42 in MUR space that corrupt the RMSE metric.
    # Replace NaN land pixels with the ERA5 mean (→ z≈0, a neutral fill) instead.
    ei_last = era5_frames[-1]
    arr_sst_raw = era5_ds['sst_era5'].isel(time=ei_last).values.astype(np.float32)
    if arr_sst_raw.ndim == 3:
        arr_sst_raw = arr_sst_raw[0]
    arr_sst_raw = np.where(
        np.isfinite(arr_sst_raw),
        arr_sst_raw,
        float(era5_mean[ERA5_SST_CH]),   # fill land with ERA5 SST mean (K)
    )
    era5_sst_raw = torch.from_numpy(arr_sst_raw).unsqueeze(0)   # [1, H_e, W_e] Kelvin
    era5_sst_c   = era5_sst_raw - 273.15                         # → Celsius
    era5_sst_fine = (
        F.interpolate(era5_sst_c.unsqueeze(0),
                      size=sst_abs.shape[-2:], mode='bilinear', align_corners=False)
        .squeeze(0)
    )
    era5_sst_fine = (era5_sst_fine - sst_mean) / (sst_std + 1e-6)   # normalise like MUR

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
        raise RuntimeError('Stage 4/4v2 evaluation requires mur_seq.')

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


def save_map_png(out_path, field, title, extent=BOF_EXTENT,
                 cmap='viridis', vmin=None, vmax=None):
    # Use a masked array so NaN pixels are transparent / grey rather than
    # appearing as vmin colour (dark purple), which would be mistaken for data.
    vis = np.ma.masked_invalid(gaussian_filter(
        np.where(np.isfinite(field), field, 0.0), sigma=0.4))
    vis = np.ma.masked_where(~np.isfinite(field), vis)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='#b0b0b0')  # light grey for masked/NaN pixels
    fig = plt.figure(figsize=(10, 7), dpi=200)
    ax  = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m', linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor='#d9d3c3', zorder=10)
    gl  = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False
    im  = ax.imshow(vis, origin='lower', extent=extent,
                    transform=ccrs.PlateCarree(), cmap=cmap_obj,
                    vmin=vmin, vmax=vmax, interpolation='bicubic', zorder=1)
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.03).set_label('Normalised SST')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def write_slider_html(out_path, title, left_label, right_label, stats_html):
    html = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><title>{title}</title>
<style>
:root{{--bg:#0f1115;--panel:#171a21;--panel2:#1e2430;--text:#e8edf2;--muted:#9aa6b2;--border:#2a3443}}
*{{box-sizing:border-box}}body{{margin:0;font-family:Inter,Arial,sans-serif;background:var(--bg);color:var(--text)}}
.wrap{{max-width:1320px;margin:0 auto;padding:20px}}h1{{margin:0 0 8px;font-size:1.25rem}}
p{{margin:0 0 14px;color:var(--muted)}}.grid{{display:grid;grid-template-columns:1.4fr .9fr;gap:20px}}
.panel{{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:16px}}
.compare{{position:relative;width:100%;overflow:hidden;border-radius:12px;cursor:ew-resize;
  user-select:none;background:#0b0d11;touch-action:none}}
.compare>img{{display:block;width:100%;height:auto;pointer-events:none}}
.img-top{{position:absolute;inset:0;will-change:clip-path}}
.img-top img{{width:100%;height:100%;object-fit:cover;pointer-events:none}}
.divider{{position:absolute;top:0;bottom:0;width:2px;background:#fff;transform:translateX(-50%);
  pointer-events:none;z-index:10;will-change:left}}
.knob{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:40px;height:40px;
  border-radius:50%;background:#fff;display:flex;align-items:center;justify-content:center;
  box-shadow:0 2px 10px rgba(0,0,0,.5)}}
.lbl{{position:absolute;top:10px;padding:3px 9px;border-radius:5px;font-size:.75rem;font-weight:700;
  background:rgba(0,0,0,.6);color:#fff;pointer-events:none;z-index:11}}
.lbl-l{{left:10px}}.lbl-r{{right:10px}}
.meta{{display:grid;gap:12px}}
.meta-card{{background:var(--panel2);border-radius:12px;padding:12px 14px;border:1px solid var(--border)}}
.meta-card h3{{margin:0 0 8px;font-size:.95rem}}
.meta-card table{{width:100%;border-collapse:collapse;font-size:.92rem}}
.meta-card td{{padding:4px 0;vertical-align:top}}
.meta-card td:first-child{{color:var(--muted);width:44%}}
</style></head><body>
<div class="wrap"><h1>{title}</h1><p>Drag the divider to compare maps over the Bay of Fundy.</p>
<div class="grid"><div class="panel">
<div class="compare" id="c">
<img id="ib" src="me.png" alt="{right_label}">
<div class="img-top" id="it" style="clip-path:inset(0 50% 0 0)">
  <img id="it2" src="mur.png" alt="{left_label}">
</div>
<div class="divider" id="div" style="left:50%"><div class="knob">
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#333"
       stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
    <polyline points="15 18 9 12 15 6"/><polyline points="9 18 3 12 9 6" transform="translate(12,0)"/>
  </svg></div></div>
<span class="lbl lbl-l">{left_label}</span><span class="lbl lbl-r">{right_label}</span>
</div></div>
<div class="meta">{stats_html}</div></div></div>
<script>
const w=document.getElementById('c'),top=document.getElementById('it'),
      dv=document.getElementById('div');
let drag=false,raf=null,px=0;
function apply(){{const r=w.getBoundingClientRect(),p=Math.max(0,Math.min(100,(px-r.left)/r.width*100));
  top.style.clipPath=`inset(0 ${{100-p}}% 0 0)`;dv.style.left=p+'%';raf=null;}}
function sched(x){{px=x;if(!raf)raf=requestAnimationFrame(apply);}}
w.addEventListener('mousedown',e=>{{drag=true;sched(e.clientX);e.preventDefault()}});
window.addEventListener('mousemove',e=>{{if(drag)sched(e.clientX)}});
window.addEventListener('mouseup',()=>drag=false);
w.addEventListener('touchstart',e=>{{drag=true;sched(e.touches[0].clientX)}},{{passive:true}});
window.addEventListener('touchmove',e=>{{if(drag)sched(e.touches[0].clientX)}},{{passive:true}});
window.addEventListener('touchend',()=>drag=false);
</script></body></html>"""
    out_path.write_text(html, encoding='utf-8')


def save_comparison_set(out_root, date_str, time_str, tag,
                        pred, era5_vis, mur, extent=BOF_EXTENT):
    folder = out_root / f'{safe_slug(date_str)}_{safe_slug(time_str)}_{tag}'
    folder.mkdir(parents=True, exist_ok=True)
    # Anchor the colour scale to MUR ground truth only — prediction outliers
    # (from spatial non-invariance) must not distort the shared colourbar.
    mur_vals = mur[np.isfinite(mur)]
    vmin = float(np.nanpercentile(mur_vals, 1))
    vmax = float(np.nanpercentile(mur_vals, 99))
    if vmin == vmax: vmax = vmin + 1e-6

    save_map_png(folder/'me.png',   pred,     f'Prediction — {date_str}',
                 extent=extent, vmin=vmin, vmax=vmax)
    save_map_png(folder/'mur.png',  mur,      f'MUR target — {date_str}',
                 extent=extent, vmin=vmin, vmax=vmax)
    save_map_png(folder/'era5.png', era5_vis, f'ERA5 SST — {date_str}',
                 extent=extent, vmin=vmin, vmax=vmax)

    m1 = compute_field_metrics(pred, mur)
    m2 = compute_field_metrics(era5_vis, mur)
    m3 = compute_field_metrics(pred, era5_vis)

    def fmt(v): return f'{v:.6f}' if v is not None else 'N/A'
    stats_html = f"""
    <div class="meta-card"><h3>Sample details</h3><table>
      <tr><td>Date</td><td>{date_str}</td></tr>
      <tr><td>Time</td><td>{time_str}</td></tr></table></div>
    <div class="meta-card"><h3>Me vs MUR</h3><table>
      <tr><td>Valid pixels</td><td>{m1['n_valid']}</td></tr>
      <tr><td>RMSE</td><td>{fmt(m1['rmse'])}</td></tr>
      <tr><td>MAE</td><td>{fmt(m1['mae'])}</td></tr>
      <tr><td>Bias</td><td>{fmt(m1['bias'])}</td></tr>
      <tr><td>Min error</td><td>{fmt(m1['min_err'])}</td></tr>
      <tr><td>Max error</td><td>{fmt(m1['max_err'])}</td></tr></table></div>
    <div class="meta-card"><h3>ERA5 vs MUR</h3><table>
      <tr><td>Valid pixels</td><td>{m2['n_valid']}</td></tr>
      <tr><td>RMSE</td><td>{fmt(m2['rmse'])}</td></tr>
      <tr><td>MAE</td><td>{fmt(m2['mae'])}</td></tr>
      <tr><td>Bias</td><td>{fmt(m2['bias'])}</td></tr></table></div>
    <div class="meta-card"><h3>Me vs ERA5</h3><table>
      <tr><td>Valid pixels</td><td>{m3['n_valid']}</td></tr>
      <tr><td>RMSE</td><td>{fmt(m3['rmse'])}</td></tr>
      <tr><td>MAE</td><td>{fmt(m3['mae'])}</td></tr>
      <tr><td>Bias</td><td>{fmt(m3['bias'])}</td></tr></table></div>"""

    write_slider_html(folder/'me_vs_mur.html',
                      f'{date_str} — Me vs MUR', 'MUR', 'Me', stats_html)
    write_slider_html(folder/'me_vs_era5.html',
                      f'{date_str} — Me vs ERA5', 'ERA5', 'Me', stats_html)
    return folder


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION LOOP
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
        raise KeyError(
            f'Cannot find model weights in checkpoint. '
            f'Keys found: {list(ckpt.keys())}'
        )

    if 'diffusion' in ckpt and 'model' not in ckpt:
        print('  Detected Phase 2 checkpoint — loading diffusion weights only.')
        print('  WARNING: encoder+baseline weights will be random unless you')
        print('  pass a merged checkpoint. Use merge_checkpoints() first.')
        model.diffusion.load_state_dict(ckpt['diffusion'])
    else:
        model.load_state_dict(sd, strict=True)

    if n_diff_steps is not None:
        cfg.DIFF_SAMPLE_STEPS = n_diff_steps

    return model.to(device).eval()


def resolve_window(stage: str) -> tuple[int, int]:
    """Returns (T, T_oce). Stages 4/4v2/5x use the stage-4 pipeline."""
    if stage in ('4', '4v2', '5a', '5b'):
        return cfg.T_ATM, cfg.T_OCE
    else:
        return cfg.T, 0

@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device)
    stats  = load_stats()   # SST + bathy stats only

    # ERA5 for BOF — compute normalisation stats on the fly
    era5_ds = open_era5_bof(TEST_YEARS)
    era5_mean_bof, era5_std_bof = compute_era5_stats_bof(era5_ds)
    stats['era5_mean'] = era5_mean_bof   # [8]
    stats['era5_std']  = era5_std_bof    # [8]

    sst_std = args.sst_std if args.sst_std else stats['sst_std']

    print(f'\n{"═"*60}')
    print(f'  BOF Evaluation — Stage {args.stage}')
    print(f'{"═"*60}')
    print(f'  Checkpoint : {args.ckpt}')
    print(f'  Test years : {TEST_YEARS}')
    print(f'  Device     : {device}')
    print(f'  cfg.N_INPUT     : {cfg.N_INPUT}')
    # Load model
    model    = load_model(args.ckpt, args.stage, device, args.n_diff_steps)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters : {n_params / 1e6:.2f} M\n')
    print(f'  encoder in_ch  : {model.encoder.patch_embed.proj.in_channels}')

    # Data
    bathy              = load_bathy_bof(stats)
    mur_store, mur_times = open_mur_bof()

    T, T_oce = resolve_window(args.stage)
    LAT_C = cfg.LAT_C
    LON_C = cfg.LON_C

    # Filter MUR to test years
    mur_pd    = pd.DatetimeIndex(mur_times)
    year_mask = (mur_pd.year >= TEST_YEARS[0]) & (mur_pd.year <= TEST_YEARS[1])

    # Keep global indices so T_oce lookback can reach into pre-test history
    mur_times_test      = mur_times[year_mask]
    mur_global_offsets  = np.where(year_mask)[0]  # maps local idx → global idx

    samples, era5_pd, _ = build_sample_index(era5_ds, mur_times_test, T, T_oce)
    print(f'  Samples    : {len(samples)}\n')

    # Accumulators
    sq_model = sq_persist = sq_murprev = 0.0
    n_valid  = 0
    psd_pred_band, psd_targ_band = [], []
    n_samples = 0
    bias_sum   = {m: 0.0 for m in range(1, 13)}
    bias_count = {m: 0   for m in range(1, 13)}

    example_dir    = Path(args.example_dir)
    month_to_saved = {m: 0 for m in SEASON_MONTHS}

    for sample in tqdm(samples, desc='BOF eval'):
        # Remap local mur_i → global mur_i so T_oce lookback is correct
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
        # print('  era5 shape     :', tuple(dev_batch['era5'].shape))
        # if 'mur_seq' in dev_batch:
        #     print('  mur_seq shape  :', tuple(dev_batch['mur_seq'].shape))
        # print('  mur_seq shape  :', tuple(dev_batch['mur_seq'].shape))
        # # Infer dynamic patch grid from the pe buffer if it exists
        if hasattr(model.encoder, 'oce_spatial_pe'):
            pe = model.encoder.oce_spatial_pe.pe
            print(f'  oce pe shape   : {tuple(pe.shape)}')
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred = model.sample(dev_batch)
        
        pred    = (pred + dev_batch['sst_prev']).float()
        sst     = dev_batch['sst_abs'].float()
        weight  = dev_batch['weight'].float()
        persist      = dev_batch['sst_prev'].float()         # oracle: MUR yesterday
        era5_persist = dev_batch['era5_sst_fine'].float()   # ERA5 coarse baseline (~3.28 °C)

        valid = (weight > 0) & torch.isfinite(sst) & torch.isfinite(pred)
        vf    = valid.float()

        sq_model   += ((pred         - sst)**2 * vf).sum().item()
        sq_persist += ((era5_persist - sst)**2 * vf).sum().item()  # ERA5 baseline
        sq_murprev += ((persist      - sst)**2 * vf).sum().item()  # oracle
        n_valid    += valid.sum().item()

        # Seasonal mean bias (physical °C)
        err_phys = ((pred - sst) * vf * sst_std)   # [1, 1, H, W] in °C
        month = int(sample[4])
        bias_sum[month]   += err_phys.sum().item()
        bias_count[month] += valid.sum().item()

        pred_np = (pred * vf).squeeze().cpu().numpy()
        targ_np = (sst  * vf).squeeze().cpu().numpy()
        psd_pred_band.append(band_mean_psd(pred_np, args.dx_km, args.lam_lo_km, args.lam_hi_km))
        psd_targ_band.append(band_mean_psd(targ_np, args.dx_km, args.lam_lo_km, args.lam_hi_km))
        n_samples += 1

        if args.save_examples:
            month = int(sample[4])
            if month in SEASON_MONTHS and month_to_saved.get(month, 0) < 1:
                date_str = sample[3]
                time_str = '09-00-00'
                tag      = f'{month_name(month).lower()}_00'

                valid_np = valid.squeeze().cpu().numpy()
                weight_np = weight.squeeze().cpu().numpy()
                # pred/mur: mask by full valid (includes isfinite(pred))
                pred_vis = np.where(valid_np, pred.squeeze().cpu().numpy(), np.nan)
                mur_vis  = np.where(valid_np, sst.squeeze().cpu().numpy(),  np.nan)
                # ERA5: use independent mask — show all ice-free, finite-SST pixels
                # regardless of whether the model produced a valid prediction there.
                era5_sst_np = dev_batch['era5_sst_fine'].squeeze().cpu().numpy()
                era5_mask = (weight_np > 0) & np.isfinite(era5_sst_np)
                era5_vis = np.where(era5_mask, era5_sst_np, np.nan)
                try:
                    folder = save_comparison_set(
                        example_dir, date_str, time_str, tag,
                        pred_vis, era5_vis, mur_vis
                    )
                    month_to_saved[month] += 1
                    print(f'  [viz] {month_name(month)} → {folder}')
                except Exception as e:
                    print(f'  [viz] failed {date_str}: {e}')

        gc.collect()

    # ── Metrics ───────────────────────────────────────────────────────────────
    n            = max(n_valid, 1)
    rmse_model   = math.sqrt(sq_model   / n)
    rmse_persist = math.sqrt(sq_persist / n)
    rmse_murprev = math.sqrt(sq_murprev / n)
    ss           = 1.0 - rmse_model / (rmse_persist + 1e-12)
    psd_pred     = float(np.nanmean(psd_pred_band))
    psd_targ     = float(np.nanmean(psd_targ_band))
    psd_ratio    = psd_pred / (psd_targ + 1e-30)

    results = {
        'domain':            'bof',
        'test_years':        list(TEST_YEARS),
        'n_samples':         n_samples,
        'n_valid_pixels':    n_valid,
        'rmse_model_norm':   round(rmse_model,             6),
        'rmse_persist_norm': round(rmse_persist,           6),
        'rmse_model_K':      round(rmse_model   * sst_std, 4),
        'rmse_persist_K':    round(rmse_persist * sst_std, 4),
        'skill_score':       round(ss,                     6),
        'psd_pred_band':     round(psd_pred,               6),
        'psd_targ_band':     round(psd_targ,               6),
        'psd_ratio':         round(psd_ratio,              6),
        'psd_log10_bias':    round(float(np.log10(psd_ratio + 1e-30)), 4),
        'rmse_murprev_K':    round(rmse_murprev * sst_std, 4),
    }

    # Seasonal mean bias
    season_map = {'DJF': [12,1,2], 'MAM': [3,4,5], 'JJA': [6,7,8], 'SON': [9,10,11]}
    results['seasonal_bias_K'] = {
        s: round(sum(bias_sum[m] for m in ms) / max(sum(bias_count[m] for m in ms), 1), 6)
        for s, ms in season_map.items()
    }
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    results['monthly_bias_K'] = {
        month_names[m-1]: round(bias_sum[m] / max(bias_count[m], 1), 6)
        for m in range(1, 13)
    }

    print(f'\n{"═"*60}')
    print(f'  BOF RESULTS')
    print(f'{"═"*60}')
    for k, v in results.items():
        print(f'  {k:<24}: {v}')
    print(f'{"═"*60}\n')

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f'Results → {args.output}')
    era5_ds.close()


def parse_args():
    p = argparse.ArgumentParser(
        description='BOF evaluation — all stages (1, 2, 3, 4, 4v2, 5a, 5b)'
    )
    p.add_argument('--stage',         required=True, choices=STAGE_CHOICES)
    p.add_argument('--ckpt',          required=True,
                   help='Path to merged checkpoint (Phase1+2) or Phase 1 only')
    p.add_argument('--n_diff_steps',  type=int,   default=None)
    p.add_argument('--dx_km',         type=float, default=1.0)
    p.add_argument('--lam_lo_km',     type=float, default=5.0)
    p.add_argument('--lam_hi_km',     type=float, default=50.0)
    p.add_argument('--sst_std',       type=float, default=None)
    p.add_argument('--psd_curves',    action='store_true')
    p.add_argument('--save_examples', action='store_true')
    p.add_argument('--example_dir',   default='../outputs/figures/bof')
    p.add_argument('--max_examples',  type=int,   default=8)
    p.add_argument('--output',        default=None,
                   help='Default: eval_results_bof_<stage>.json')
    p.add_argument('--device',
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())