"""
make_sst_animation.py — MUR SST time-lapse animation.

Generates a GIF (default) or MP4 showing daily SST flowing across time
for a chosen domain.  No GPU or model inference needed.

Usage:
    python scripts/make_sst_animation.py                        # GSL, Jun-Nov 2021-2022
    python scripts/make_sst_animation.py --domain bof
    python scripts/make_sst_animation.py --out outputs/figures/paper/sst.mp4

Defaults (GSL, Jun–Nov across 18 months, every 4 days → ~90 frames at 12fps ≈ 7.5s):
    --domain gsl  --start 2021-06-01  --end 2022-11-30  --months 6,7,8,9,10,11
    --every 4  --fps 12  --stride 3  --dpi 90
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ROOT = Path(__file__).resolve().parent.parent

try:
    import cmocean
    CMAP_BASE = cmocean.cm.thermal
except ImportError:
    CMAP_BASE = plt.cm.get_cmap('inferno')

BG   = '#080d1a'   # deep night-ocean background
LAND = '#b8b0a6'   # warm sandy grey — clearly distinct from all SST colours


DOMAIN_CFG = {
    'gsl': {
        'zarr':   ROOT / 'data/domains/gsl/mur.zarr',
        'times':  ROOT / 'data/domains/gsl/mur_times.npy',
        'mode':   'flat',   # plain [T, H, W] zarr, °C
        'kelvin': False,
        'lat0': 44.0, 'lat1': 52.8,
        'lon0': -69.5, 'lon1': -56.0,
        'title': 'Gulf of St. Lawrence  ·  MUR SST',
    },
    'bof': {
        'zarr':   ROOT / 'data/domains/bof/mur/mur_bof.zarr',
        'times':  ROOT / 'data/domains/bof/mur/mur_bof_times.npy',
        'mode':   'group',  # zarr Group with 'analysed_sst', 'lat', 'lon'
        'kelvin': True,
        'title': 'Bay of Fundy  ·  MUR SST',
    },
    'gom': {
        'zarr':   ROOT / 'data/domains/gom/mur/mur_gom.zarr',
        'times':  ROOT / 'data/domains/gom/mur/mur_gom_times.npy',
        'mode':   'group',
        'kelvin': True,
        'title': 'Gulf of Mexico  ·  MUR SST',
    },
}


def read_frame(z, mode, kelvin, idx, stride):
    if mode == 'flat':
        arr = np.array(z[idx][::stride, ::stride], dtype=np.float32)
    else:
        arr = np.array(z['analysed_sst'][idx][::stride, ::stride], dtype=np.float32)
        if kelvin:
            arr = np.where(np.isfinite(arr), arr - 273.15, np.nan)
    return arr


def main(args):
    dcfg   = DOMAIN_CFG[args.domain]
    z      = zarr.open(str(dcfg['zarr']),  'r')
    times  = np.load(str(dcfg['times']), allow_pickle=True)
    pd_t   = pd.DatetimeIndex(times)
    mode   = dcfg['mode']
    kelvin = dcfg['kelvin']
    stride = args.stride

    # ── Select frame indices (date range + optional month filter) ────────────
    t0 = pd.Timestamp(args.start)
    t1 = pd.Timestamp(args.end)
    months_set = set(int(m) for m in args.months.split(',')) if args.months else None

    date_mask = (pd_t >= t0) & (pd_t <= t1)
    if months_set:
        date_mask &= np.array([t.month in months_set for t in pd_t])
    idx_all = np.where(date_mask)[0]
    if len(idx_all) == 0:
        raise ValueError(f'No MUR frames found for selected date range / months')
    indices = idx_all[::args.every]
    nf  = len(indices)
    dur = nf / args.fps
    print(f'Domain  : {args.domain.upper()}')
    print(f'Period  : {pd_t[indices[0]].date()} → {pd_t[indices[-1]].date()}')
    if months_set:
        print(f'Months  : {sorted(months_set)}  (seasonal filter)')
    print(f'Frames  : {nf}  (every {args.every} day(s))  →  {dur:.1f}s at {args.fps}fps')

    # ── Lat/lon arrays ────────────────────────────────────────────────────────
    if mode == 'flat':
        H_full, W_full = z.shape[1], z.shape[2]
        lats = np.linspace(dcfg['lat0'], dcfg['lat1'], H_full)[::stride]
        lons = np.linspace(dcfg['lon0'], dcfg['lon1'], W_full)[::stride]
    else:
        lats = np.array(z['lat'][:])[::stride]
        lons = np.array(z['lon'][:])[::stride]

    # ── Robust colour range from the filtered frames only ────────────────────
    # Sampling from the seasonal subset keeps winter cold out of the scale.
    print('Computing colour range from seasonal frames...')
    samp = indices[::max(1, nf // 30)]
    vals = []
    for si in samp:
        f = read_frame(z, mode, kelvin, int(si), stride)
        v = f[np.isfinite(f)]
        if v.size:
            vals.append(v)
    all_vals = np.concatenate(vals)
    vmin = float(np.percentile(all_vals, 1))
    vmax = float(np.percentile(all_vals, 99))
    print(f'SST range: {vmin:.1f} – {vmax:.1f} °C  (seasonal subset)')

    # ── Figure ────────────────────────────────────────────────────────────────
    cmap = CMAP_BASE.copy()
    cmap.set_bad(LAND)   # land = sandy grey, clearly distinct from cold water

    lat_span = abs(lats[-1] - lats[0])
    lon_span = abs(lons[-1] - lons[0])
    fig_h = 3.6
    fig_w = max(4.0, min(9.0, fig_h * (lon_span / lat_span) * 0.90))

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)
    ax  = fig.add_axes([0.03, 0.07, 0.87, 0.83], facecolor=BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    LON2, LAT2 = np.meshgrid(lons, lats)
    frame0  = read_frame(z, mode, kelvin, int(indices[0]), stride)
    masked0 = np.ma.masked_invalid(frame0)

    pcm = ax.pcolormesh(LON2, LAT2, masked0,
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        shading='auto', rasterized=True)

    # Colorbar
    cax = fig.add_axes([0.92, 0.10, 0.025, 0.75])
    cb  = plt.colorbar(pcm, cax=cax)
    cb.set_label('SST (°C)', color='white', fontsize=8.5)
    cb.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=7.5)
    cb.outline.set_edgecolor('#445566')
    cb.outline.set_linewidth(0.6)

    # Date stamp (top-left, monospace)
    date_text = ax.text(
        0.02, 0.97, '', transform=ax.transAxes,
        color='white', fontsize=10, va='top', ha='left',
        fontfamily='monospace', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=BG,
                  alpha=0.65, edgecolor='none'))

    # Domain subtitle
    ax.set_title(dcfg['title'], color='#9ab4d0', fontsize=9, pad=5)

    # ── Animation loop ────────────────────────────────────────────────────────
    def update(fi):
        if fi % 20 == 0:
            print(f'  {fi:4d} / {nf}', end='\r', flush=True)
        i   = int(indices[fi])
        arr = read_frame(z, mode, kelvin, i, stride)
        pcm.set_array(np.ma.masked_invalid(arr).ravel())
        dt = pd_t[i]
        date_text.set_text(dt.strftime('%Y  %b  %d'))
        return pcm, date_text

    ani = animation.FuncAnimation(
        fig, update, frames=nf,
        interval=1000 / args.fps, blit=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ext = out.suffix.lower()
    if ext == '.gif':
        print(f'\nSaving GIF → {out}  (dpi={args.dpi}) ...')
        ani.save(str(out), writer='pillow', fps=args.fps, dpi=args.dpi)
    elif ext in ('.mp4', '.mov'):
        print(f'\nSaving MP4 → {out}  (dpi={args.dpi}) ...')
        ani.save(str(out), writer='ffmpeg', fps=args.fps, dpi=args.dpi,
                 extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18'])
    else:
        raise ValueError(f'Unknown output format: {ext}. Use .gif or .mp4')

    plt.close(fig)
    mb = out.stat().st_size / 1e6
    print(f'Done: {out}  ({mb:.1f} MB, {dur:.1f}s)')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='MUR SST time-lapse animation')
    p.add_argument('--domain',  default='gsl', choices=list(DOMAIN_CFG),
                   help='Domain to animate (default: gsl)')
    p.add_argument('--start',   default='2021-06-01',
                   help='Start date (default: 2021-06-01)')
    p.add_argument('--end',     default='2022-11-30',
                   help='End date (default: 2022-11-30)')
    p.add_argument('--months',  default='6,7,8,9,10,11',
                   help='Comma-separated months to include (default: 6-11 = Jun-Nov). '
                        'Set empty string to disable filter.')
    p.add_argument('--every',   type=int, default=4,
                   help='Sample every N days (default: 4)')
    p.add_argument('--fps',     type=int, default=12,
                   help='Frames per second (default: 12 — gives more time per frame)')
    p.add_argument('--stride',  type=int, default=3,
                   help='Spatial downsampling factor (default: 3)')
    p.add_argument('--dpi',     type=int, default=90,
                   help='Output DPI (default: 90)')
    p.add_argument('--out',  default='outputs/figures/paper/sst_animation.gif',
                   help='Output path (.gif or .mp4)')
    main(p.parse_args())
