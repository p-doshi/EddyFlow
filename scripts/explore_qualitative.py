"""
explore_qualitative.py — Multiple dates × multiple layout variants.

Dates: Sep, Dec, Mar, Jun (2022 / 2022-2023 season).
Layouts: 3 variants (natural aspect, equal rows wide, equal rows very wide).

Output: outputs/figures/explore/qualitative/

Usage:
    python scripts/explore_qualitative.py \
        --ckpt outputs/checkpoints/stage4_sweep_4_123_lr0_sw0_merged.pt
"""

import argparse, sys, os
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
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from config import cfg
from model  import TemporalDownscaler

# Import shared helpers from make_qualitative_figure (ZarrERA5 now at module level)
import make_qualitative_figure as mqf

OUT  = ROOT / 'outputs' / 'figures' / 'explore' / 'qualitative'
OUT.mkdir(parents=True, exist_ok=True)

CKPT_DEF = ROOT / 'outputs' / 'checkpoints' / 'stage4_sweep_4_123_lr0_sw0_merged.pt'

# 4 representative dates
DATES = ['2022-06-01', '2022-09-01', '2022-12-01', '2023-03-01']
DATE_LABELS = {
    '2022-06-01':  'June 2022',
    '2022-09-01':  'September 2022',
    '2022-12-01':  'December 2022',
    '2023-03-01':  'March 2023',
}

# Domain extent: [lon_min, lon_max, lat_min, lat_max]
EXTENTS = mqf.EXTENTS
DOMAINS = ['GSL', 'BOF', 'GOM']

# Geographic aspect (lat_span / lon_span) for natural height_ratios
_asp = {
    'GSL': (52.8 - 44.0) / (69.5 - 56.0),   # 8.8 / 13.5 = 0.652
    'BOF': (47.0 - 44.0) / (67.0 - 63.0),   # 3.0 /  4.0 = 0.750
    'GOM': (31.0 - 18.0) / (98.0 - 80.0),   # 13.0 / 18.0 = 0.722
}
NAT_RATIOS = [_asp['GSL'], _asp['BOF'], _asp['GOM']]  # [0.652, 0.750, 0.722]

ROW_LABELS = {
    'GSL': 'Gulf of St. Lawrence\n(training domain)',
    'BOF': 'Bay of Fundy\n(OOD transfer)',
    'GOM': 'Gulf of Mexico\n(OOD transfer)',
}
COL_TITLES = ['ERA5 SST  (0.25° input)',
              'Stage 4   (0.01° prediction)',
              'MUR SST   (0.01° reference)']

try:
    import cmocean; CMAP = cmocean.cm.thermal
except ImportError: CMAP = 'RdYlBu_r'

HAS_CARTOPY = mqf.HAS_CARTOPY
if HAS_CARTOPY:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature


# ─── Map feature helpers ──────────────────────────────────────────────────────
def _add_features(ax, extent, natural_aspect=False):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    if not natural_aspect:
        ax.set_aspect('auto')
    land = cfeature.NaturalEarthFeature(
        'physical', 'land', '10m',
        facecolor='#d4cfc9', edgecolor='#888', linewidth=0.4)
    ax.add_feature(land, zorder=3)
    gl = ax.gridlines(draw_labels=False, linewidth=0.3, color='#aaa',
                      linestyle='--', zorder=2)
    gl.xlocator = ticker.MaxNLocator(3)
    gl.ylocator = ticker.MaxNLocator(3)


def plot_panel(ax, lons, lats, data, valid, vmin, vmax, extent,
               cmap=None, cbar_ax=None, natural_aspect=False):
    cmap = cmap or CMAP
    masked = np.where(valid, data, np.nan)
    if HAS_CARTOPY:
        im = ax.pcolormesh(lons, lats, masked,
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs.PlateCarree(),
                           shading='auto', rasterized=True, zorder=1)
        _add_features(ax, extent, natural_aspect=natural_aspect)
    else:
        im = ax.imshow(masked, origin='lower', aspect='auto',
                       cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
        ax.set_xticks([]); ax.set_yticks([])
    if cbar_ax is not None:
        cb = plt.colorbar(im, cax=cbar_ax, orientation='vertical')
        cb.ax.tick_params(labelsize=6.5)
        cb.set_label('SST (°C)', fontsize=7)
        cb.locator = ticker.MaxNLocator(nbins=5); cb.update_ticks()
    return im


# ─── Figure builder ──────────────────────────────────────────────────────────
def make_figure(results, date_label, figsize, height_ratios, natural_aspect,
                hspace=0.06, wspace=0.04):
    nrows, ncols = 3, 3
    proj = ccrs.PlateCarree() if HAS_CARTOPY else None

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(
        nrows, ncols + 1,
        figure=fig,
        height_ratios=height_ratios,
        width_ratios=[1, 1, 1, 0.04],
        hspace=hspace, wspace=wspace,
        left=0.11, right=0.93, top=0.91, bottom=0.03,
    )

    for ri, dom in enumerate(DOMAINS):
        r      = results[dom]
        lats, lons = r['lats'], r['lons']
        mur    = r['mur_phys']
        valid  = r['valid']
        extent = EXTENTS[dom]
        ocean  = mur[valid]
        vmin   = float(np.nanpercentile(ocean, 2))
        vmax   = float(np.nanpercentile(ocean, 98))
        cbar_ax = fig.add_subplot(gs[ri, ncols])

        for ci, (key, ctitle) in enumerate(zip(
                ['era5_sst', 'pred_phys', 'mur_phys'], COL_TITLES)):
            subplot_kw = {'projection': proj} if HAS_CARTOPY else {}
            ax = fig.add_subplot(gs[ri, ci], **subplot_kw)
            v  = np.isfinite(r['era5_sst']) if key == 'era5_sst' else valid
            plot_panel(ax, lons, lats, r[key], v,
                       vmin, vmax, extent,
                       cbar_ax=cbar_ax if ci == ncols - 1 else None,
                       natural_aspect=natural_aspect)
            if ri == 0:
                ax.set_title(ctitle, fontsize=7, pad=4, fontweight='bold')
            if ci == 0:
                kw = dict(transform=ax.transAxes) if HAS_CARTOPY else {}
                ax.text(-0.20, 0.5, ROW_LABELS[dom], fontsize=7,
                        va='center', ha='right', multialignment='center', **kw)

    fig.text(0.50, 0.955, f'{date_label}  ·  Stage 4  ·  baseline decoder',
             ha='center', va='top', fontsize=8, color='#555', style='italic')
    return fig


# ─── Layouts ─────────────────────────────────────────────────────────────────
LAYOUTS = {
    # Natural geo aspect ratios (rows sized to lat/lon) — no stretch
    'vA_natural': dict(
        figsize=(9.0, 7.5),
        height_ratios=NAT_RATIOS,
        natural_aspect=True,
        hspace=0.10, wspace=0.05,
    ),
    # Equal rows, fill panels (set_aspect auto), wider canvas
    'vB_equal_wide': dict(
        figsize=(9.5, 7.0),
        height_ratios=[1, 1, 1],
        natural_aspect=False,
        hspace=0.06, wspace=0.04,
    ),
    # Very wide — more horizontal breathing room for maps
    'vC_very_wide': dict(
        figsize=(11.5, 6.5),
        height_ratios=[1, 1, 1],
        natural_aspect=False,
        hspace=0.06, wspace=0.06,
    ),
}


# ─── Domain setup (called once) ───────────────────────────────────────────────
def build_domain_cfgs(stats, baseline_only=True):
    """Returns dict of domain_cfg for GSL, BOF, GOM."""
    _mode = dict(baseline_only=baseline_only, n_diff_steps=20)

    # GSL
    gsl_dir = ROOT / 'data/domains/gsl'
    era5_z  = zarr.open(str(gsl_dir / 'era5.zarr'), 'r')
    era5_t  = np.load(str(gsl_dir / 'era5_times.npy'), allow_pickle=True)
    mur_z   = zarr.open(str(gsl_dir / 'mur.zarr'), 'r')
    mur_t   = np.load(str(gsl_dir / 'mur_times.npy'), allow_pickle=True)
    bathy   = mqf.load_bathy_norm(gsl_dir / 'bathy.npy', stats, 500, 1200)
    gsl_cfg = dict(
        era5_ds=mqf.ZarrERA5(era5_z, era5_t),
        mur_z=mur_z, mur_times=mur_t,
        fine_h=500, fine_w=1200, bathy=bathy,
        load_mur=mqf.mur_physical_gsl,
        era5_stats=(stats['era5_mean'], stats['era5_std']),
        lats=np.linspace(44.0, 52.8, 500),
        lons=np.linspace(-69.5, -56.0, 1200),
        **_mode,
    )

    # BOF
    bof_dir = ROOT / 'data/domains/bof'
    era5_bof = xr.open_mfdataset(
        sorted((bof_dir / 'era5').glob('era5_bof_202*.nc')),
        combine='by_coords', decode_timedelta=False, engine='h5netcdf')
    if 'valid_time' in era5_bof and 'time' not in era5_bof.dims:
        era5_bof = era5_bof.rename({'valid_time': 'time'})
    mur_z_bof = zarr.open(str(bof_dir / 'mur/mur_bof.zarr'), 'r')
    mur_t_bof = np.load(str(bof_dir / 'mur/mur_bof_times.npy'), allow_pickle=True)
    bathy_bof = mqf.load_bathy_norm(bof_dir / 'gebco_bof.nc', stats, 301, 401)
    bof_cfg = dict(
        era5_ds=era5_bof, mur_z=mur_z_bof, mur_times=mur_t_bof,
        fine_h=301, fine_w=401, bathy=bathy_bof,
        load_mur=mqf.mur_physical_grp,
        lats=mur_z_bof['lat'][:], lons=mur_z_bof['lon'][:],
        **_mode,
    )

    # GOM
    gom_dir = ROOT / 'data/domains/gom'
    era5_gom = xr.open_mfdataset(
        sorted((gom_dir / 'era5').glob('era5_gom_202*.nc')),
        combine='by_coords', decode_timedelta=False, engine='h5netcdf')
    if 'valid_time' in era5_gom and 'time' not in era5_gom.dims:
        era5_gom = era5_gom.rename({'valid_time': 'time'})
    mur_z_gom = zarr.open(str(gom_dir / 'mur/mur_gom.zarr'), 'r')
    mur_t_gom = np.load(str(gom_dir / 'mur/mur_gom_times.npy'), allow_pickle=True)
    bathy_gom = mqf.load_bathy_norm(gom_dir / 'gebco_gom.nc', stats, 1301, 1801)
    gom_cfg = dict(
        era5_ds=era5_gom, mur_z=mur_z_gom, mur_times=mur_t_gom,
        fine_h=1301, fine_w=1801, bathy=bathy_gom,
        load_mur=mqf.mur_physical_grp,
        lats=mur_z_gom['lat'][:], lons=mur_z_gom['lon'][:],
        **_mode,
    )

    return {'GSL': gsl_cfg, 'BOF': bof_cfg, 'GOM': gom_cfg}


# ─── Main ─────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    stats = mqf.load_stats()
    print('Stats loaded.')

    model = TemporalDownscaler(stage=4)
    ckpt  = torch.load(args.ckpt, map_location=device)
    sd    = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    print(f'Model loaded: {Path(args.ckpt).name}')

    domain_cfgs = build_domain_cfgs(stats, baseline_only=True)

    total = len(DATES) * len(LAYOUTS)
    done  = 0
    for date in DATES:
        label = DATE_LABELS[date]
        print(f'\n── {label} ──')

        # Run inference for all 3 domains for this date (model only loaded once)
        results = {}
        for dom, dcfg in domain_cfgs.items():
            print(f'  {dom} ...', end=' ', flush=True)
            try:
                r = mqf.run_inference(model, device, stats, dcfg, date)
                results[dom] = r
                print(f'pred [{np.nanmin(r["pred_phys"]):.1f}, {np.nanmax(r["pred_phys"]):.1f}]°C')
            except Exception as e:
                print(f'ERROR: {e}')
                results[dom] = None

        # Skip if any domain failed
        if any(v is None for v in results.values()):
            print(f'  Skipping {date} (inference error)')
            continue

        # Generate all layout variants
        date_slug = date.replace('-', '')
        for vname, lkw in LAYOUTS.items():
            fig = make_figure(results, label, **lkw)
            stem = f'qual_{date_slug}_{vname}'
            for ext in ('pdf', 'png'):
                path = OUT / f'{stem}.{ext}'
                fig.savefig(str(path), dpi=300, bbox_inches='tight')
            plt.close(fig)
            done += 1
            print(f'  [{done}/{total}] Saved {stem}')

    print(f'\nDone. {done} figures in {OUT}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=str(CKPT_DEF))
    main(p.parse_args())
