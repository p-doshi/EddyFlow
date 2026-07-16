"""
make_qualitative_figure.py — Paper figure: ERA5 coarse vs Stage4 model vs MUR truth
for GSL (training domain), BOF (OOD), and GOM (OOD) on the same date (2022-09-01).

Layout: 3 rows (domains) × 3 cols (ERA5 upsampled | Stage4 | MUR truth)
Output: outputs/figures/paper/fig_qualitative.{pdf,png}

Usage:
    cd /scratch/pdoshi/my_project/eddyflow
    python scripts/make_qualitative_figure.py [--ckpt PATH] [--date 2022-09-01]
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import zarr
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from config import cfg
from model  import TemporalDownscaler

STATS_DIR  = ROOT / 'scripts' / 'data' / 'stats'
CKPT_DEF   = ROOT / 'outputs' / 'checkpoints' / 'stage4_sweep_4_123_lr0_sw0_merged.pt'
OUT_DIR    = ROOT / 'outputs' / 'figures' / 'paper'
TARGET_DATE = '2022-09-01'

ERA5_CHANNELS = [
    'u10','v10','msl','sst_era5','t2m','siconc',
    'u850','v850','t850','z850','q850',
    'u700','v700','t700','z700','q700',
    'u500','v500','t500','z500','q500',
]
ERA5_SST_CH = ERA5_CHANNELS.index('sst_era5')   # 3
NAN_FILL    = {3: 0.0, 5: 0.0}                  # sst_era5, siconc → 0 over land
T_ATM       = cfg.T_ATM    # 28


class ZarrERA5:
    """Minimal xr.Dataset-like adapter for the GSL era5.zarr array."""
    class _Ch:
        def __init__(self, z, ci): self._z = z; self._ci = ci
        def isel(self, time):
            class _V:
                def __init__(self, a): self.values = a; self.ndim = a.ndim
            return _V(self._z[time, self._ci].astype(np.float32))

    class _Time:
        def __init__(self, t): self.values = t.to_numpy()

    def __init__(self, z, times):
        self._z  = z
        self._pd = pd.DatetimeIndex(times)
        self.sizes = {'time': z.shape[0], 'latitude': z.shape[2], 'longitude': z.shape[3]}
        self.time  = self._Time(self._pd)

    def __contains__(self, k): return k in ERA5_CHANNELS or k == 'time'
    def __getitem__(self, v):
        if v == 'time': return self.time
        return self._Ch(self._z, ERA5_CHANNELS.index(v))
T_OCE       = cfg.T_OCE    # 60
LAT_C       = cfg.LAT_C    # 20
LON_C       = cfg.LON_C    # 48


# ─── normalisation stats ──────────────────────────────────────────────────────
def load_stats():
    return dict(
        era5_mean  = np.load(STATS_DIR / 'era5_mean.npy'),
        era5_std   = np.load(STATS_DIR / 'era5_std.npy'),
        sst_mean   = float(np.load(STATS_DIR / 'sst_mean.npy')),
        sst_std    = float(np.load(STATS_DIR / 'sst_std.npy')),
        bathy_mean = float(np.load(STATS_DIR / 'bathy_mean.npy')),
        bathy_std  = float(np.load(STATS_DIR / 'bathy_std.npy')),
    )


# ─── ERA5 frame builder ───────────────────────────────────────────────────────
def era5_window_tensor(era5_ds, frame_indices, stats, domain_stats=None):
    """
    Build [T, 21, LAT_C, LON_C] ERA5 tensor for the given day-indices.
    domain_stats: (means, stds) arrays if per-domain normalization is needed.
    """
    em = stats['era5_mean'] if domain_stats is None else domain_stats[0]
    es = stats['era5_std']  if domain_stats is None else domain_stats[1]

    frames = []
    for ei in frame_indices:
        chans = []
        for ci, v in enumerate(ERA5_CHANNELS):
            if v in era5_ds:
                arr = era5_ds[v].isel(time=ei).values.astype(np.float32)
                if arr.ndim == 3:
                    arr = arr[0]
            else:
                arr = np.zeros(
                    (era5_ds.sizes.get('latitude', LAT_C),
                     era5_ds.sizes.get('longitude', LON_C)),
                    dtype=np.float32)
            fill = NAN_FILL.get(ci, None)
            if fill is not None:
                arr = np.where(np.isfinite(arr), arr, fill)
            arr = (arr - em[ci]) / (es[ci] + 1e-6)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            chans.append(arr)
        frames.append(np.stack(chans, axis=0))          # [21, H_e, W_e]

    win  = np.stack(frames, axis=0)                     # [T, 21, H_e, W_e]
    t    = torch.from_numpy(win)
    T, C, H, W = t.shape
    t    = F.interpolate(
        t.reshape(T * C, 1, H, W),
        size=(LAT_C, LON_C), mode='bilinear', align_corners=False
    ).reshape(T, C, LAT_C, LON_C)
    return t.unsqueeze(0)                               # [1, T, 21, LAT_C, LON_C]


def era5_sst_physical(era5_ds, ei):
    """Return raw ERA5 SST for frame ei in °C.
    Tries 'sst_era5' (GSL zarr adapter) then 'sst' (CDS NetCDF naming).
    """
    for var in ('sst_era5', 'sst'):
        try:
            if hasattr(era5_ds, 'data_vars') and var not in era5_ds.data_vars:
                continue
            arr = era5_ds[var].isel(time=ei).values.astype(np.float32)
            if arr.ndim == 3:
                arr = arr[0]
            fin = arr[np.isfinite(arr)]
            if fin.size and fin.mean() > 100:
                arr = arr - 273.15
            return arr
        except (KeyError, AttributeError):
            continue
    return np.full((LAT_C, LON_C), np.nan, dtype=np.float32)


# ─── MUR helpers ─────────────────────────────────────────────────────────────
def mur_physical_gsl(mur_z, idx):
    """GSL MUR zarr stores °C directly as a flat array."""
    raw = mur_z[idx].astype(np.float32)
    return raw


def mur_physical_grp(mur_z, idx):
    """BOF/GOM MUR zarr is a Group with 'analysed_sst' in Kelvin."""
    raw = mur_z['analysed_sst'][idx].astype(np.float32)
    raw = np.where(np.isfinite(raw), raw - 273.15, np.nan)
    return raw


def mur_norm_tensor(mur_phys, sst_mean, sst_std):
    n = (mur_phys - sst_mean) / (sst_std + 1e-6)
    n = np.where(np.isfinite(n), n, 0.0)
    return torch.from_numpy(n).unsqueeze(0)             # [1, H_f, W_f]


# ─── bathy helpers ───────────────────────────────────────────────────────────
def load_bathy_norm(path, stats, fine_h, fine_w):
    """Load bathy from .npy or .nc, normalise, return [1, fine_h, fine_w] tensor."""
    p = Path(path)
    if p.suffix == '.npy':
        depth = np.load(str(p)).astype(np.float32)
    else:
        ds = None
        for eng in ('h5netcdf', 'scipy', 'netcdf4', None):
            try:
                ds = xr.open_dataset(str(p), engine=eng) if eng else xr.open_dataset(str(p))
                break
            except Exception:
                pass
        elev_var = 'elevation' if 'elevation' in ds.data_vars else next(iter(ds.data_vars))
        elev  = ds[elev_var].values.astype(np.float32)
        depth = np.where(elev < 0, -elev, 0.0)
        ds.close()
    blog  = np.log1p(depth)
    bnorm = (blog - stats['bathy_mean']) / (stats['bathy_std'] + 1e-6)
    t = torch.from_numpy(bnorm).unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
    return F.interpolate(t, size=(fine_h, fine_w),
                         mode='bilinear', align_corners=False).squeeze(0)  # [1,H,W]


# ─── per-domain inference ─────────────────────────────────────────────────────
def find_era5_frames(era5_ds, target_date, T):
    """
    Return list of T ERA5 day-indices (daily, 1 per day) ending on target_date.
    era5_ds may have multiple frames per day (6-hourly); take the last one.
    """
    try:
        era5_pd = pd.DatetimeIndex(era5_ds['time'].values)
    except Exception:
        era5_pd = pd.DatetimeIndex([str(t) for t in era5_ds['time'].values])

    date_to_last_idx = {}
    for i, t in enumerate(era5_pd):
        date_to_last_idx[t.date()] = i

    tgt = pd.Timestamp(target_date)
    frames = []
    for days_back in range(T - 1, -1, -1):
        d = (tgt - pd.Timedelta(days=days_back)).date()
        if d not in date_to_last_idx:
            raise KeyError(f'ERA5 missing date {d}')
        frames.append(date_to_last_idx[d])
    return frames


def find_mur_idx(mur_times, target_date):
    try:
        pd_times = pd.DatetimeIndex(mur_times)
    except Exception:
        pd_times = pd.DatetimeIndex([str(t) for t in mur_times])
    tgt  = pd.Timestamp(target_date)
    delt = np.abs((pd_times - tgt).total_seconds())
    idx  = int(delt.argmin())
    assert delt[idx] / 86400 < 2, f'No MUR frame close to {target_date}'
    return idx, max(idx - 1, 0)


def compute_domain_era5_stats(era5_ds):
    """Compute per-channel stats from an xarray Dataset."""
    means, stds = [], []
    for v in ERA5_CHANNELS:
        if v in era5_ds:
            data  = era5_ds[v].values.astype(np.float32).ravel()
            valid = data[np.isfinite(data)]
            m = float(valid.mean()) if valid.size else 0.0
            s = float(valid.std())  if valid.size else 1.0
            s = s if s > 1e-6 else 1.0
        else:
            m, s = 0.0, 1.0
        means.append(m); stds.append(s)
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


@torch.no_grad()
def run_inference(model, device, stats, domain_cfg, target_date):
    """
    Returns dict:
        era5_sst   : [H_e, W_e] physical °C, coarse
        pred_phys  : [H_f, W_f] physical °C, model prediction
        mur_phys   : [H_f, W_f] physical °C, MUR truth
        valid_mask : [H_f, W_f] bool
    """
    dc = domain_cfg
    era5_ds   = dc['era5_ds']
    mur_z     = dc['mur_z']
    mur_times = dc['mur_times']
    fine_h    = dc['fine_h']
    fine_w    = dc['fine_w']
    bathy     = dc['bathy']          # [1, fine_h, fine_w] tensor

    # ERA5 frames
    era5_frames = find_era5_frames(era5_ds, target_date, T_ATM)
    ei_today    = era5_frames[-1]

    # MUR frames
    mur_i, mur_prev_i = find_mur_idx(mur_times, target_date)

    # Per-domain ERA5 stats; may be pre-supplied for zarr-backed GSL
    if 'era5_stats' in dc:
        dom_stats = dc['era5_stats']
    else:
        dom_stats = compute_domain_era5_stats(era5_ds)

    # Tensors
    era5_t = era5_window_tensor(era5_ds, era5_frames, stats, dom_stats)

    load_mur = dc['load_mur']
    mur_phys      = load_mur(mur_z, mur_i)
    mur_prev_phys = load_mur(mur_z, mur_prev_i)
    mur_norm      = mur_norm_tensor(mur_phys, stats['sst_mean'], stats['sst_std'])
    mur_prev_norm = mur_norm_tensor(mur_prev_phys, stats['sst_mean'], stats['sst_std'])

    # Ocean history: T_OCE days of MUR ending on mur_i
    oce_frames = []
    for k in range(T_OCE - 1, -1, -1):
        oi = max(mur_i - k, 0)
        oce_frames.append(mur_norm_tensor(load_mur(mur_z, oi),
                                          stats['sst_mean'], stats['sst_std']))
    oce_t = torch.stack(oce_frames, dim=0).unsqueeze(0)   # [1, T_OCE, 1, H_f, W_f]

    # Coarsen bathy and ice to atmospheric grid, then append to ERA5 as channels 22-23
    bathy_c = F.interpolate(bathy.unsqueeze(0), size=(LAT_C, LON_C),
                            mode='bilinear', align_corners=False)   # [1, 1, LAT_C, LON_C]
    ice_c   = torch.zeros(1, 1, LAT_C, LON_C)                      # ice-free in September
    static  = torch.cat([bathy_c, ice_c], dim=1)                   # [1, 2, LAT_C, LON_C]
    # Append bathy+ice to every ERA5 frame: [1, T, 21, C, W] + [1, 1, 2, C, W]
    B_T = era5_t.shape[1]                                           # T
    static_seq = static.unsqueeze(1).expand(-1, B_T, -1, -1, -1)   # [1, T, 2, LAT_C, LON_C]
    era5_23 = torch.cat([era5_t, static_seq], dim=2)                # [1, T, 23, LAT_C, LON_C]

    # Fine-grid bathy for baseline decoder
    bathy_fine = bathy.unsqueeze(0)                                  # [1, 1, fine_h, fine_w]

    batch = dict(
        era5      = era5_23.to(device),                         # [1, T, 23, LAT_C, LON_C]
        bathy     = bathy_fine.to(device),                      # [1, 1, fine_h, fine_w]
        sst_prev  = mur_prev_norm.unsqueeze(0).to(device),      # [1, 1, H_f, W_f]
        mur_seq   = oce_t.to(device),                           # [1, T_OCE, 1, H_f, W_f]
        sst       = mur_norm.unsqueeze(0).to(device),           # [1, 1, H_f, W_f]
        weight    = torch.ones(1, 1, fine_h, fine_w, device=device),
    )

    # Forward pass
    if dc.get('baseline_only', False):
        with torch.no_grad():
            latent    = model._encode(batch)
            pred_norm = model.baseline(latent, batch['bathy']).squeeze().cpu().numpy()
    else:
        pred_norm = model.sample(batch, n_steps=dc.get('n_diff_steps', 20)).squeeze().cpu().numpy()

    # Reconstruct absolute SST: pred is delta, add sst_prev
    sst_prev_norm = mur_prev_norm.squeeze().numpy()
    pred_abs_norm = pred_norm + sst_prev_norm
    pred_phys     = pred_abs_norm * stats['sst_std'] + stats['sst_mean']

    # ERA5 SST upsampled to fine grid
    # Fill NaN (land) before bilinear interpolation so coastal ocean cells
    # are not contaminated; restore NaN mask afterwards.
    era5_sst_coarse = era5_sst_physical(era5_ds, ei_today)
    coarse_valid = np.isfinite(era5_sst_coarse)
    fill_val     = float(np.nanmean(era5_sst_coarse)) if np.any(coarse_valid) else 0.0
    era5_fill    = np.where(coarse_valid, era5_sst_coarse, fill_val)
    era5_t_fine  = torch.from_numpy(era5_fill).unsqueeze(0).unsqueeze(0)
    era5_fine    = F.interpolate(era5_t_fine, size=(fine_h, fine_w),
                                 mode='bilinear', align_corners=False
                                 ).squeeze().numpy()
    cv_t    = torch.from_numpy(coarse_valid.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    cv_fine = F.interpolate(cv_t, size=(fine_h, fine_w),
                            mode='bilinear', align_corners=False).squeeze().numpy()
    era5_fine = np.where(cv_fine > 0.1, era5_fine, np.nan)

    valid = np.isfinite(mur_phys) & (mur_phys > -3)

    # lat / lon arrays for pcolormesh
    lats = dc.get('lats')
    lons = dc.get('lons')

    return dict(
        era5_sst  = era5_fine,
        pred_phys = pred_phys,
        mur_phys  = mur_phys,
        valid     = valid,
        lats      = lats,
        lons      = lons,
    )


# ─── plotting ─────────────────────────────────────────────────────────────────
# PROJ patch — must run before cartopy import
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
        break

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

try:
    import cmocean
    CMAP = cmocean.cm.thermal
except ImportError:
    CMAP = 'RdYlBu_r'

# Domain extents [lon_min, lon_max, lat_min, lat_max]
EXTENTS = {
    'GSL': [-69.5, -56.0,  44.0, 52.8],
    'BOF': [-67.0, -63.0,  44.0, 47.0],
    'GOM': [-98.0, -80.0,  18.0, 31.0],
}


def _add_map_features(ax, extent):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    # Override cartopy's geographic aspect ratio so the panel fills its subplot
    ax.set_aspect('auto')
    land = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                         facecolor='#d4cfc9', edgecolor='#888', linewidth=0.4)
    ax.add_feature(land, zorder=3)
    gl = ax.gridlines(draw_labels=False, linewidth=0.3, color='#aaa',
                      linestyle='--', zorder=2)
    gl.xlocator = ticker.MaxNLocator(3)
    gl.ylocator = ticker.MaxNLocator(3)


def plot_panel_map(ax, lons, lats, data, valid, vmin, vmax,
                   extent, cmap=CMAP, show_cbar=False, cbar_ax=None):
    masked = np.where(valid, data, np.nan)
    if HAS_CARTOPY:
        im = ax.pcolormesh(lons, lats, masked,
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(),
                           shading='auto', rasterized=True, zorder=1)
        _add_map_features(ax, extent)
    else:
        # Fallback: plain imshow with south-up fix (lat goes S→N → origin='lower')
        im = ax.imshow(masked, origin='lower', aspect='auto',
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='bilinear')
        ax.set_xticks([]); ax.set_yticks([])

    if show_cbar and cbar_ax is not None:
        cb = plt.colorbar(im, cax=cbar_ax, orientation='vertical')
        cb.ax.tick_params(labelsize=6.5)
        cb.set_label('SST (°C)', fontsize=7)
        cb.locator = ticker.MaxNLocator(nbins=5)
        cb.update_ticks()
    return im


def make_figure(results, target_date):
    domains    = ['GSL', 'BOF', 'GOM']
    col_titles = ['ERA5 SST  (0.25° input, upsampled)',
                  'Stage 4   (0.01° prediction)',
                  'MUR SST  (0.01° reference)']

    row_labels = {
        'GSL': 'Gulf of St. Lawrence\n(training domain)',
        'BOF': 'Bay of Fundy\n(OOD transfer)',
        'GOM': 'Gulf of Mexico\n(OOD transfer)',
    }

    # Aspect ratios vary by domain — compute from lat/lon spans
    aspects = {}
    for dom in domains:
        e = EXTENTS[dom]
        dlon = e[1] - e[0]; dlat = e[3] - e[2]
        aspects[dom] = dlon / dlat   # wider domains get more horizontal space

    nrows, ncols = 3, 3
    # Width ratios: scale each data column by domain widest aspect (GOM is widest)
    # Use a fixed ratio since subplots share column widths
    fig = plt.figure(figsize=(7.2, 6.8))

    gs = gridspec.GridSpec(
        nrows, ncols + 1,
        figure=fig,
        width_ratios=[1, 1, 1, 0.04],
        hspace=0.06,
        wspace=0.04,
        left=0.10, right=0.92, top=0.92, bottom=0.03,
    )

    proj = ccrs.PlateCarree() if HAS_CARTOPY else None

    for ri, dom in enumerate(domains):
        r      = results[dom]
        lats   = r['lats']
        lons   = r['lons']
        mur    = r['mur_phys']
        valid  = r['valid']
        extent = EXTENTS[dom]

        ocean = mur[valid]
        vmin  = float(np.nanpercentile(ocean, 2))
        vmax  = float(np.nanpercentile(ocean, 98))

        cbar_ax = fig.add_subplot(gs[ri, ncols])

        for ci, (key, ctitle) in enumerate(zip(
                ['era5_sst', 'pred_phys', 'mur_phys'], col_titles)):

            subplot_kw = {'projection': proj} if HAS_CARTOPY else {}
            ax = fig.add_subplot(gs[ri, ci], **subplot_kw)

            show_cbar = (ci == ncols - 1)
            data = r[key]

            # ERA5 panel uses its own finite-value mask (coarse grid coverage);
            # MUR panels use the fine-resolution ocean mask.
            v = np.isfinite(r['era5_sst']) if key == 'era5_sst' else valid
            plot_panel_map(ax, lons, lats, data, v,
                           vmin, vmax, extent,
                           show_cbar=show_cbar,
                           cbar_ax=cbar_ax if show_cbar else None)

            if ri == 0:
                ax.set_title(ctitle, fontsize=7.5, pad=4, fontweight='bold')

            if ci == 0:
                if HAS_CARTOPY:
                    ax.text(-0.18, 0.5, row_labels[dom],
                            transform=ax.transAxes,
                            fontsize=7.5, va='center', ha='right',
                            rotation=0, multialignment='center')
                else:
                    ax.set_ylabel(row_labels[dom], fontsize=7.5, labelpad=4)

    fig.text(0.50, 0.96, f'{target_date}  ·  Stage 4 zero-shot evaluation',
             ha='center', va='top', fontsize=8, color='#555', style='italic')

    return fig


# ─── main ────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    stats = load_stats()
    print('Loaded normalisation stats.')

    # Load model
    print(f'Loading checkpoint: {args.ckpt}')
    model = TemporalDownscaler(stage=4)
    ckpt  = torch.load(args.ckpt, map_location=device)
    sd    = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    print('Model loaded.')

    # ── GSL domain ────────────────────────────────────────────────────────────
    print('\n=== GSL ===')
    gsl_dir = ROOT / 'data' / 'domains' / 'gsl'
    era5_gsl = zarr.open(str(gsl_dir / 'era5.zarr'), 'r')
    era5_times_gsl = np.load(str(gsl_dir / 'era5_times.npy'), allow_pickle=True)

    era5_ds_gsl = ZarrERA5(era5_gsl, era5_times_gsl)

    mur_z_gsl   = zarr.open(str(gsl_dir / 'mur.zarr'), 'r')
    mur_t_gsl   = np.load(str(gsl_dir / 'mur_times.npy'), allow_pickle=True)
    bathy_gsl   = load_bathy_norm(gsl_dir / 'bathy.npy', stats,
                                  fine_h=500, fine_w=1200)

    _mode = dict(baseline_only=args.baseline_only, n_diff_steps=args.n_diff_steps)

    _gsl_lats = np.linspace(44.0, 52.8, 500)
    _gsl_lons = np.linspace(-69.5, -56.0, 1200)
    gsl_cfg = dict(
        era5_ds=era5_ds_gsl, mur_z=mur_z_gsl, mur_times=mur_t_gsl,
        fine_h=500, fine_w=1200, bathy=bathy_gsl,
        load_mur=mur_physical_gsl,
        era5_stats=(stats['era5_mean'], stats['era5_std']),
        lats=_gsl_lats, lons=_gsl_lons,
        **_mode,
    )
    gsl_res = run_inference(model, device, stats, gsl_cfg, args.date)
    print(f'  ERA5 SST range: [{np.nanmin(gsl_res["era5_sst"]):.1f}, {np.nanmax(gsl_res["era5_sst"]):.1f}]°C')
    print(f'  Pred range: [{np.nanmin(gsl_res["pred_phys"]):.1f}, {np.nanmax(gsl_res["pred_phys"]):.1f}]°C')
    print(f'  MUR  range: [{np.nanmin(gsl_res["mur_phys"]):.1f}, {np.nanmax(gsl_res["mur_phys"]):.1f}]°C')

    # ── BOF domain ────────────────────────────────────────────────────────────
    print('\n=== BOF ===')
    bof_dir  = ROOT / 'data' / 'domains' / 'bof'
    era5_bof = xr.open_mfdataset(
        sorted((bof_dir / 'era5').glob('era5_bof_202*.nc')),
        combine='by_coords', decode_timedelta=False, engine='h5netcdf')
    if 'valid_time' in era5_bof and 'time' not in era5_bof.dims:
        era5_bof = era5_bof.rename({'valid_time': 'time'})
    mur_z_bof  = zarr.open(str(bof_dir / 'mur' / 'mur_bof.zarr'), 'r')
    mur_t_bof  = np.load(str(bof_dir / 'mur' / 'mur_bof_times.npy'), allow_pickle=True)
    bathy_bof  = load_bathy_norm(bof_dir / 'gebco_bof.nc', stats,
                                 fine_h=301, fine_w=401)

    _bof_lats = zarr.open(str(bof_dir / 'mur' / 'mur_bof.zarr'), 'r')['lat'][:]
    _bof_lons = zarr.open(str(bof_dir / 'mur' / 'mur_bof.zarr'), 'r')['lon'][:]
    bof_cfg = dict(
        era5_ds=era5_bof, mur_z=mur_z_bof, mur_times=mur_t_bof,
        fine_h=301, fine_w=401, bathy=bathy_bof,
        load_mur=mur_physical_grp,
        lats=_bof_lats, lons=_bof_lons,
        **_mode,
    )
    bof_res = run_inference(model, device, stats, bof_cfg, args.date)
    print(f'  ERA5 SST range: [{np.nanmin(bof_res["era5_sst"]):.1f}, {np.nanmax(bof_res["era5_sst"]):.1f}]°C')
    print(f'  Pred range: [{np.nanmin(bof_res["pred_phys"]):.1f}, {np.nanmax(bof_res["pred_phys"]):.1f}]°C')
    print(f'  MUR  range: [{np.nanmin(bof_res["mur_phys"]):.1f}, {np.nanmax(bof_res["mur_phys"]):.1f}]°C')

    # ── GOM domain ────────────────────────────────────────────────────────────
    print('\n=== GOM ===')
    gom_dir  = ROOT / 'data' / 'domains' / 'gom'
    era5_gom = xr.open_mfdataset(
        sorted((gom_dir / 'era5').glob('era5_gom_202*.nc')),
        combine='by_coords', decode_timedelta=False, engine='h5netcdf')
    if 'valid_time' in era5_gom and 'time' not in era5_gom.dims:
        era5_gom = era5_gom.rename({'valid_time': 'time'})
    mur_z_gom  = zarr.open(str(gom_dir / 'mur' / 'mur_gom.zarr'), 'r')
    mur_t_gom  = np.load(str(gom_dir / 'mur' / 'mur_gom_times.npy'), allow_pickle=True)
    bathy_gom  = load_bathy_norm(gom_dir / 'gebco_gom.nc', stats,
                                 fine_h=1301, fine_w=1801)

    _gom_lats = zarr.open(str(gom_dir / 'mur' / 'mur_gom.zarr'), 'r')['lat'][:]
    _gom_lons = zarr.open(str(gom_dir / 'mur' / 'mur_gom.zarr'), 'r')['lon'][:]
    gom_cfg = dict(
        era5_ds=era5_gom, mur_z=mur_z_gom, mur_times=mur_t_gom,
        fine_h=1301, fine_w=1801, bathy=bathy_gom,
        load_mur=mur_physical_grp,
        lats=_gom_lats, lons=_gom_lons,
        **_mode,
    )
    gom_res = run_inference(model, device, stats, gom_cfg, args.date)
    print(f'  ERA5 SST range: [{np.nanmin(gom_res["era5_sst"]):.1f}, {np.nanmax(gom_res["era5_sst"]):.1f}]°C')
    print(f'  Pred range: [{np.nanmin(gom_res["pred_phys"]):.1f}, {np.nanmax(gom_res["pred_phys"]):.1f}]°C')
    print(f'  MUR  range: [{np.nanmin(gom_res["mur_phys"]):.1f}, {np.nanmax(gom_res["mur_phys"]):.1f}]°C')

    # ── Plot ──────────────────────────────────────────────────────────────────
    results = {'GSL': gsl_res, 'BOF': bof_res, 'GOM': gom_res}
    fig = make_figure(results, args.date)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        path = OUT_DIR / f'fig_qualitative.{ext}'
        fig.savefig(str(path), dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=str(CKPT_DEF))
    p.add_argument('--date', default=TARGET_DATE)
    p.add_argument('--n_diff_steps', type=int, default=20)
    p.add_argument('--baseline_only', action='store_true',
                   help='Skip DDIM; use baseline decoder only (fast, CPU-friendly)')
    main(p.parse_args())
