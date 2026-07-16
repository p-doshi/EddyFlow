"""
eval_wenhai.py — WenHai ONNX single-step evaluation over the Gulf of St. Lawrence.

Workflow:
  1. Load IC  from sample_GLORYS_23lev.nc  → [1, 93, 2041, 4320]  (Jan 1, 2019 12Z)
  2. Load forcing from sample_ERA5_d0.083.nc → [1, 8, 2041, 4320]
     (zero-flux fallback if file absent)
  3. Single WenHai ONNX inference → predicted Jan 2 state
  4. Crop GSL native slice [native_h × native_w] from output
  5. PSD computed at native 1/12° resolution (avoids upsample spectral smear)
  6. Upsample to zarr resolution [500 × 1200] for RMSE vs MUR Jan 2
  7. Persistence baseline = IC thetao sfc held constant (not ΔSST=0 vs 0)

Metrics:
  - RMSE (normalised + physical °C) vs MUR SST Jan 2 2019
  - Skill Score SS = 1 - RMSE(WenHai) / RMSE(IC persistence)
  - PSD ratio at native 1/12° resolution in configurable wavelength band
  - Optional: full PSD curves saved to JSON

Usage:
  python scripts/eval_wenhai.py \
      --onnx          scripts/WenHai.onnx \
      --sample_glorys scripts/sample_GLORYS_23lev.nc \
      [--sample_era5  scripts/sample_ERA5_d0.083.nc] \
      --output        results_wenhai.json \
      [--dx_km 9.0] [--lam_lo_km 20.0] [--lam_hi_km 200.0] [--psd_curves]

Notes:
  - IC date is 2019-01-01T12Z → target date is 2019-01-02
  - dx_km should match native GLORYS 1/12° resolution (~9 km at GSL latitudes)
  - min/max_GLORYS.npy must be 93-channel (from WenHai repo, not your prepare script)
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
import xarray as xr
from scipy.ndimage import zoom as nd_zoom

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import cfg

GLOBAL_H = 2041
GLOBAL_W = 4320

# WenHai input.1 channel layout (93 total):
#   thetao × 23  →  ch 0–22    (surface = ch 0, depth level 0)
#   so     × 23  →  ch 23–45
#   uo     × 23  →  ch 46–68
#   vo     × 23  →  ch 69–91
#   zos    × 1   →  ch 92
THETAO_SFC_CH = 0


# ─────────────────────────────────────────────────────────────────────────────
# CFG PATH FIX
# ─────────────────────────────────────────────────────────────────────────────

def fix_cfg_paths():
    root         = Path(__file__).resolve().parents[1]
    scripts_data = root / 'scripts' / 'data'
    cfg.MUR_ZARR   = str(scripts_data / 'processed' / 'mur.zarr')
    cfg.MUR_TIMES  = str(scripts_data / 'processed' / 'mur_times.npy')
    cfg.SST_MEAN   = str(scripts_data / 'stats' / 'sst_mean.npy')
    cfg.SST_STD    = str(scripts_data / 'stats' / 'sst_std.npy')
    cfg.STATS_DIR  = str(scripts_data / 'stats')


# ─────────────────────────────────────────────────────────────────────────────
# LOAD WENHAI SAMPLE FILES
# ─────────────────────────────────────────────────────────────────────────────

def load_wenhai_inputs(sample_glorys_nc: str,
                       sample_era5_nc,
                       min_GLORYS: np.ndarray,
                       max_GLORYS: np.ndarray,
                       min_flux: np.ndarray,
                       max_flux: np.ndarray):
    """
    Returns:
      init_global : [1, 93, 2041, 4320] float16   normalised GLORYS state
      bulk_global : [1,  8, 2041, 4320] float16   normalised bulk flux
      ic_date     : pd.Timestamp of the IC (to auto-select MUR target)
    """
    # ── Initial condition ────────────────────────────────────────────────────
    print(f'  Loading GLORYS init  : {sample_glorys_nc}')
    ds = xr.open_dataset(sample_glorys_nc)

    # Extract IC date from time coordinate
    if 'time' in ds.coords:
        ic_date = pd.Timestamp(ds.time.values[0] if ds.time.values.ndim > 0
                               else ds.time.values)
    else:
        print('  ⚠  No time coord in GLORYS NC — assuming 2019-01-01')
        ic_date = pd.Timestamp('2019-01-01')
    print(f'  IC date              : {ic_date}')

    def get_var(name, fallbacks=None):
        for cand in [name] + (fallbacks or []):
            if cand in ds:
                return ds[cand].values.astype(np.float32)
        raise KeyError(f'{name!r} not in {sample_glorys_nc}. '
                       f'Available: {list(ds.data_vars)}')

    thetao = get_var('thetao')
    so     = get_var('so')
    uo     = get_var('uo', ['u'])
    vo     = get_var('vo', ['v'])
    zos    = get_var('zos', ['ssh', 'adt'])
    ds.close()

    # Ensure all arrays are [1, levels, H, W]
    def ensure_4d_batch(x):
        if x.ndim == 2: x = x[np.newaxis, np.newaxis]   # [H,W] → [1,1,H,W]
        if x.ndim == 3: x = x[np.newaxis]               # [C,H,W] or [1,H,W] → [1,C,H,W]
        return x

    thetao = ensure_4d_batch(thetao)   # [1, 23, H, W]
    so     = ensure_4d_batch(so)
    uo     = ensure_4d_batch(uo)
    vo     = ensure_4d_batch(vo)
    zos    = ensure_4d_batch(zos)      # [1, 1, H, W]

    print(f'    thetao {thetao.shape}  so {so.shape}  '
          f'uo {uo.shape}  vo {vo.shape}  zos {zos.shape}')

    # Stack → [1, 93, H, W]
    init = np.concatenate([thetao, so, uo, vo, zos], axis=1).astype(np.float32)
    assert init.shape[1] == 93, (
        f'Expected 93 channels, got {init.shape[1]}. '
        f'thetao={thetao.shape[1]}, so={so.shape[1]}, '
        f'uo={uo.shape[1]}, vo={vo.shape[1]}, zos={zos.shape[1]}.'
    )

    init = np.nan_to_num(init)
    init = (init - min_GLORYS) / (max_GLORYS - min_GLORYS + 1e-30)
    init = np.clip(init, 0, 1).astype(np.float16)
    print(f'    init_global : {init.shape}  '
          f'range=[{float(init.min()):.3f}, {float(init.max()):.3f}]')

    # ── Bulk flux ────────────────────────────────────────────────────────────
    bulk = None
    if sample_era5_nc and Path(sample_era5_nc).exists():
        print(f'  Loading ERA5 bulk    : {sample_era5_nc}')
        ds2 = xr.open_dataset(sample_era5_nc)
        print(f'    ERA5 vars: {list(ds2.data_vars)}')

        # Try single stacked array
        for cand in ['bulk', 'flux', 'forcing', 'input3']:
            if cand in ds2:
                bulk = ds2[cand].values.astype(np.float32)
                break

        # Fall back: stack individual flux vars in WenHai order
        if bulk is None:
            flux_vars = ['strd', 'ssr', 't2m', 'd2m', 'u10', 'v10', 'msl', 'mtpr']
            found = [v for v in flux_vars if v in ds2]
            if len(found) == 8:
                def squeeze_time(arr):
                    # drop any leading time/batch dims, keep [H, W]
                    while arr.ndim > 2:
                        arr = arr[0]
                    return arr
                bulk = np.stack([squeeze_time(ds2[v].values.astype(np.float32))
                                for v in flux_vars], axis=0)   # [8, H, W]
            else:
                print(f'    ⚠  Could not match flux vars (found: {found}). '
                      f'Using zero flux.')
        ds2.close()

    if bulk is not None:
        bulk = ensure_4d_batch(bulk) if bulk.ndim < 4 else bulk
        if bulk.ndim == 3:
            bulk = bulk[np.newaxis]
        bulk = np.nan_to_num(bulk)
        bulk = (bulk - min_flux[np.newaxis]) / (max_flux[np.newaxis] - min_flux[np.newaxis] + 1e-30)
        bulk = np.clip(bulk, 0, 1).astype(np.float16)
        print(f'    bulk_global : {bulk.shape}  '
              f'range=[{float(bulk.min()):.3f}, {float(bulk.max()):.3f}]')
    else:
        print('  ⚠  No ERA5 bulk flux — using zeros (model forced with no atmosphere).')
        bulk = np.zeros((1, 8, GLOBAL_H, GLOBAL_W), dtype=np.float16)

    return init, bulk, ic_date


# ─────────────────────────────────────────────────────────────────────────────
# GSL SLICE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_gsl_slice(sample_glorys_nc, gsl_lat, gsl_lon):
    if sample_glorys_nc and Path(sample_glorys_nc).exists():
        ds      = xr.open_dataset(sample_glorys_nc)
        glb_lat = ds.latitude.values
        glb_lon = ds.longitude.values
        ds.close()
        print(f'  Global grid from sample : '
              f'lat [{glb_lat[0]:.3f}→{glb_lat[-1]:.3f}], '
              f'lon [{glb_lon[0]:.3f}→{glb_lon[-1]:.3f}]')
    else:
        glb_lat = np.linspace(-80.0,  90.0,  GLOBAL_H)
        glb_lon = np.linspace(-180.0, 180.0, GLOBAL_W)
        print(f'  Global grid (analytical): '
              f'lat [{glb_lat[0]:.3f}→{glb_lat[-1]:.3f}], '
              f'lon [{glb_lon[0]:.3f}→{glb_lon[-1]:.3f}]')

    lat_lo, lat_hi = gsl_lat.min() - 0.1, gsl_lat.max() + 0.1
    lon_lo, lon_hi = gsl_lon.min() - 0.1, gsl_lon.max() + 0.1

    lat_idx = np.where((glb_lat >= lat_lo) & (glb_lat <= lat_hi))[0]
    lon_idx = np.where((glb_lon >= lon_lo) & (glb_lon <= lon_hi))[0]

    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError(
            f'GSL domain not found in global grid. '
            f'lat [{lat_lo:.2f}→{lat_hi:.2f}], lon [{lon_lo:.2f}→{lon_hi:.2f}].'
        )

    lat_sl = slice(lat_idx[0], lat_idx[-1] + 1)
    lon_sl = slice(lon_idx[0], lon_idx[-1] + 1)
    print(f'  GSL lat slice : [{lat_sl.start}:{lat_sl.stop}]  '
          f'({lat_sl.stop - lat_sl.start} rows at 1/12°)')
    print(f'  GSL lon slice : [{lon_sl.start}:{lon_sl.stop}]  '
          f'({lon_sl.stop - lon_sl.start} cols at 1/12°)')
    return lat_sl, lon_sl, glb_lat, glb_lon


# ─────────────────────────────────────────────────────────────────────────────
# RESAMPLE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def upsample_from_native(field: np.ndarray,
                         target_h: int, target_w: int) -> np.ndarray:
    """[1, C, nh, nw] → [1, C, target_h, target_w] bilinear."""
    B, C, H, W = field.shape
    out = np.zeros((B, C, target_h, target_w), dtype=field.dtype)
    zh, zw = target_h / H, target_w / W
    for c in range(C):
        out[0, c] = nd_zoom(
            field[0, c].astype(np.float32), (zh, zw), order=1
        ).astype(field.dtype)
    return out

def downsample_2d(field: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """[H, W] → [target_h, target_w] bilinear."""
    return nd_zoom(field.astype(np.float32),
                   (target_h / field.shape[0], target_w / field.shape[1]),
                   order=1)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL GRID CROP
# ─────────────────────────────────────────────────────────────────────────────

def crop_from_global(field_global: np.ndarray,
                     lat_sl: slice, lon_sl: slice) -> np.ndarray:
    """[1, C, 2041, 4320] → [1, C, native_h, native_w]"""
    return field_global[:, :, lat_sl, lon_sl]


# ─────────────────────────────────────────────────────────────────────────────
# PSD UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def azimuthal_psd(field: np.ndarray, dx_km: float):
    H, W    = field.shape
    f2d     = np.fft.rfft2(field)
    p2d     = (np.abs(f2d) ** 2) / (H * W)
    ky      = np.fft.fftfreq(H, d=dx_km)
    kx      = np.fft.rfftfreq(W, d=dx_km)
    KX, KY  = np.meshgrid(kx, ky)
    K2D     = np.sqrt(KX ** 2 + KY ** 2)
    k_max   = min(kx.max(), np.abs(ky).max())
    n_bins  = max(H, W) // 2
    k_edges = np.linspace(0.0, k_max, n_bins + 1)
    k_cents = 0.5 * (k_edges[:-1] + k_edges[1:])
    psd = np.zeros(n_bins)
    for i in range(n_bins):
        m = (K2D >= k_edges[i]) & (K2D < k_edges[i + 1])
        if m.sum() > 0:
            psd[i] = p2d[m].mean()
    return k_cents, psd

def band_mean_psd(field: np.ndarray, dx_km: float,
                  lam_lo_km: float, lam_hi_km: float) -> float:
    k_cents, psd = azimuthal_psd(field, dx_km)
    lo  = 1.0 / lam_hi_km
    hi  = 1.0 / lam_lo_km
    sel = (k_cents >= lo) & (k_cents <= hi)
    return float(psd[sel].mean()) if sel.sum() > 0 else float('nan')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    import onnxruntime as ort

    stats_dir = Path(cfg.STATS_DIR)

    # ── Stats (must be 93-channel from WenHai repo) ───────────────────────────
    min_GLORYS = np.load(Path(args.onnx).parent / 'min_GLORYS.npy')
    max_GLORYS = np.load(Path(args.onnx).parent / 'max_GLORYS.npy')
    min_flux   = np.load(Path(args.onnx).parent / 'min_flux.npy')
    max_flux   = np.load(Path(args.onnx).parent / 'max_flux.npy')

    n_glorys_ch = min_GLORYS.size
    print(f'  Stats: GLORYS {n_glorys_ch} ch, flux {min_flux.size} ch')
    assert n_glorys_ch == 93, (
        f'min_GLORYS.npy has {n_glorys_ch} channels — expected 93.\n'
        f'Use the stats files shipped with the WenHai repo:\n'
        f'  cp /path/to/WenHai/min_GLORYS.npy scripts/\n'
        f'  cp /path/to/WenHai/max_GLORYS.npy scripts/\n'
        f'  cp /path/to/WenHai/min_flux.npy   scripts/\n'
        f'  cp /path/to/WenHai/max_flux.npy   scripts/'
    )

    min_GLORYS = min_GLORYS.reshape(1, -1, 1, 1)   # [1, 93, 1, 1]
    max_GLORYS = max_GLORYS.reshape(1, -1, 1, 1)
    min_flux   = min_flux.reshape(-1, 1, 1)         # [8, 1, 1]
    max_flux   = max_flux.reshape(-1, 1, 1)

    glor_mask = xr.open_dataset(stats_dir / 'mask_GLORYS.nc').mask.values[0, 0]
    sst_mean  = float(np.load(cfg.SST_MEAN))
    sst_std   = float(np.load(cfg.SST_STD))
    H, W      = glor_mask.shape   # 500 × 1200

    # ── MUR ground truth ──────────────────────────────────────────────────────
    mur_z  = zarr.open(cfg.MUR_ZARR, mode='r')
    mur_pd = pd.DatetimeIndex(np.load(cfg.MUR_TIMES, allow_pickle=True))

    # ── GSL grid coordinates ──────────────────────────────────────────────────
    grid_lat = np.linspace(47.0, 51.99, H)
    grid_lon = np.linspace(-68.0, -56.01, W)

    # ── GSL slice in global 1/12° grid ───────────────────────────────────────
    lat_sl, lon_sl, glb_lat, glb_lon = detect_gsl_slice(
        args.sample_glorys, grid_lat, grid_lon
    )
    native_h = lat_sl.stop - lat_sl.start
    native_w = lon_sl.stop - lon_sl.start
    # native dx at ~49°N: 1/12° ≈ 9.26 km (longitude), 9.26 km (latitude)
    native_dx_km = args.dx_km
    print(f'  Zarr res  : {H} × {W}  |  Native slice : {native_h} × {native_w}  '
          f'|  dx_km : {native_dx_km}')

    # ── Load WenHai sample inputs ─────────────────────────────────────────────
    init_global, bulk_global, ic_date = load_wenhai_inputs(
        args.sample_glorys, args.sample_era5,
        min_GLORYS, max_GLORYS, min_flux, max_flux
    )

    # ── Target date = IC + 1 day ──────────────────────────────────────────────
    target_date = ic_date + pd.Timedelta(days=1)
    target_ts = target_date.value  # nanoseconds since epoch
    mur_i = int(np.argmin(np.abs(mur_pd.asi8 - target_ts)))    
    mur_i = int(np.clip(mur_i, 0, len(mur_pd) - 1))
    print(f'  IC date   : {ic_date.date()}  →  target MUR: {mur_pd[mur_i].date()}')
    print(f'  sst_mean  : {sst_mean:.4f}   sst_std: {sst_std:.4f}\n')

    # ── ONNX session ──────────────────────────────────────────────────────────
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session   = ort.InferenceSession(args.onnx, providers=providers)
    inp_info  = session.get_inputs()
    name1, name2 = inp_info[0].name, inp_info[1].name
    print(f'  ONNX inputs : {name1} {list(inp_info[0].shape)}, '
          f'{name2} {list(inp_info[1].shape)}')
    print(f'  Provider    : {session.get_providers()[0]}\n')

    # ─────────────────────────────────────────────────────────────────────────
    # SINGLE-STEP INFERENCE  (IC = Jan 1, target = Jan 2)
    # ─────────────────────────────────────────────────────────────────────────
    print('  Running WenHai inference ...')
    out_global = session.run(
        None, {name1: init_global, name2: bulk_global}
    )[0]   # [1, 93, 2041, 4320]

    # Residual update: state_t+1 = state_t + Δ, clipped to [0,1]
    out_global = np.clip(
        out_global.astype(np.float32) + init_global.astype(np.float32),
        0.0, 1.0
    ).astype(np.float16)
    print(f'  out_global : {out_global.shape}  '
          f'range=[{float(out_global.min()):.3f}, {float(out_global.max()):.3f}]')

    mn = float(min_GLORYS[0, THETAO_SFC_CH, 0, 0])
    mx = float(max_GLORYS[0, THETAO_SFC_CH, 0, 0])

    # ── Crop thetao surface at native 1/12° resolution ───────────────────────
    out_thetao_native  = crop_from_global(
        out_global[:, THETAO_SFC_CH:THETAO_SFC_CH+1], lat_sl, lon_sl
    )  # [1, 1, native_h, native_w]
    init_thetao_native = crop_from_global(
        init_global[:, THETAO_SFC_CH:THETAO_SFC_CH+1], lat_sl, lon_sl
    )  # [1, 1, native_h, native_w]  — persistence IC

    # Denorm to physical °C at native res
    pred_native_phys = out_thetao_native[0, 0].astype(np.float32) * (mx - mn) + mn
    pers_native_phys = init_thetao_native[0, 0].astype(np.float32) * (mx - mn) + mn

    # Normalise into MUR space at native res (for PSD)
    pred_native_norm = (pred_native_phys - sst_mean) / (sst_std + 1e-6)
    pers_native_norm = (pers_native_phys - sst_mean) / (sst_std + 1e-6)

    # ── Upsample native → zarr resolution for pixel-level RMSE ──────────────
    out_zarr  = upsample_from_native(out_thetao_native,  H, W)  # [1,1,H,W]
    pers_zarr = upsample_from_native(init_thetao_native, H, W)  # [1,1,H,W]
    out_zarr  = out_zarr  * glor_mask[None, None]
    pers_zarr = pers_zarr * glor_mask[None, None]

    pred_phys = out_zarr[0, 0].astype(np.float32)  * (mx - mn) + mn
    pers_phys = pers_zarr[0, 0].astype(np.float32) * (mx - mn) + mn
    pred_norm = (pred_phys - sst_mean) / (sst_std + 1e-6)
    pers_norm = (pers_phys - sst_mean) / (sst_std + 1e-6)

    # ── MUR ground truth ──────────────────────────────────────────────────────
    targ_raw  = mur_z[mur_i].astype(np.float32)
    targ_norm = (targ_raw - sst_mean) / (sst_std + 1e-6)
    targ_norm = np.where(np.isfinite(targ_norm), targ_norm, 0.0)

    # ── Valid ocean mask (zarr resolution) ───────────────────────────────────
    valid = (glor_mask > 0) & np.isfinite(pred_norm) & np.isfinite(targ_norm)
    n_valid = int(valid.sum())

    sq_model   = float(((pred_norm[valid] - targ_norm[valid]) ** 2).sum())
    sq_persist = float(((pers_norm[valid] - targ_norm[valid]) ** 2).sum())

    rmse_model   = math.sqrt(sq_model   / max(n_valid, 1))
    rmse_persist = math.sqrt(sq_persist / max(n_valid, 1))
    ss = 1.0 - rmse_model / (rmse_persist + 1e-12)

    # ── PSD at native 1/12° (no upsample smearing) ───────────────────────────
    # Downsample MUR target and mask to native resolution for fair comparison
    targ_native     = downsample_2d(targ_norm,            native_h, native_w)
    valid_native    = downsample_2d(valid.astype(np.float32), native_h, native_w) > 0.5

    psd_pred = band_mean_psd(pred_native_norm * valid_native,
                              native_dx_km, args.lam_lo_km, args.lam_hi_km)
    psd_targ = band_mean_psd(targ_native      * valid_native,
                              native_dx_km, args.lam_lo_km, args.lam_hi_km)
    psd_ratio    = psd_pred / (psd_targ + 1e-30)
    psd_log_bias = float(np.log10(psd_ratio + 1e-30))

    # ── Optional full PSD curves ──────────────────────────────────────────────
    psd_curves_data = None
    if args.psd_curves:
        k_c, pp = azimuthal_psd(pred_native_norm * valid_native, native_dx_km)
        _,   pt = azimuthal_psd(targ_native      * valid_native, native_dx_km)
        psd_curves_data = {
            'k_cycles_per_km': k_c.tolist(),
            'wavelength_km':   (1.0 / (k_c + 1e-30)).tolist(),
            'psd_pred':        pp.tolist(),
            'psd_targ':        pt.tolist(),
            'psd_ratio':       (pp / (pt + 1e-30)).tolist(),
        }

    # ── Assemble results ──────────────────────────────────────────────────────
    results = {
        'model':              'WenHai-ONNX',
        'ic_date':            str(ic_date.date()),
        'target_date':        str(mur_pd[mur_i].date()),
        'eval_type':          'single_step_1day_forecast',
        'n_valid_pixels':     n_valid,
        'native_slice':       f'{native_h}×{native_w} at 1/12°',
        'zarr_resolution':    f'{H}×{W}',
        'rmse_model_norm':    round(rmse_model,               6),
        'rmse_persist_norm':  round(rmse_persist,             6),
        'rmse_model_K':       round(rmse_model   * sst_std,   4),
        'rmse_persist_K':     round(rmse_persist * sst_std,   4),
        'skill_score':        round(ss,                        6),
        'psd_pred_band':      round(psd_pred,                  6),
        'psd_targ_band':      round(psd_targ,                  6),
        'psd_ratio':          round(psd_ratio,                 6),
        'psd_log10_bias':     round(psd_log_bias,              4),
        'psd_band_km':        f'{args.lam_lo_km}–{args.lam_hi_km} km',
        'dx_km_native':       native_dx_km,
    }
    if psd_curves_data:
        results['psd_curves'] = psd_curves_data

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f'\n{"═"*60}')
    print(f'  WenHai ONNX — 1-Day Forecast Results (GSL)')
    print(f'{"═"*60}')
    print(f'  IC → target        : {results["ic_date"]} → {results["target_date"]}')
    print(f'  Valid pixels       : {n_valid:,}')
    print(f'  RMSE model    (°C) : {results["rmse_model_K"]:.4f}')
    print(f'  RMSE persist  (°C) : {results["rmse_persist_K"]:.4f}')
    print(f'  RMSE model   (norm): {results["rmse_model_norm"]:.5f}')
    print(f'  RMSE persist (norm): {results["rmse_persist_norm"]:.5f}')
    ss_val = results['skill_score']
    print(f'  Skill Score SS     : {ss_val:+.5f}  '
          f'({"↑ better" if ss_val > 0 else "↓ worse"} than IC persistence)')
    print(f'  PSD band           : {results["psd_band_km"]}  |  dx={native_dx_km} km native')
    print(f'  PSD pred  (band)   : {psd_pred:.4e}')
    print(f'  PSD targ  (band)   : {psd_targ:.4e}')
    print(f'  PSD ratio          : {psd_ratio:.4f}  '
          f'(log₁₀ bias = {psd_log_bias:+.3f})')
    print(f'{"═"*60}\n')

    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, indent=2))
    print(f'Results written → {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx',          required=True,
                   help='Path to WenHai.onnx')
    p.add_argument('--sample_glorys', required=True,
                   help='WenHai sample_GLORYS_23lev.nc (IC, Jan 1 2019)')
    p.add_argument('--sample_era5',   default=None,
                   help='WenHai sample_ERA5_d0.083.nc (bulk flux; zeros if absent)')
    p.add_argument('--output',        default='results_wenhai.json')
    p.add_argument('--dx_km',         type=float, default=9.0,
                   help='Native grid spacing in km (~9 km at GSL latitudes for 1/12°)')
    p.add_argument('--lam_lo_km',     type=float, default=20.0,
                   help='PSD band lower wavelength (km)')
    p.add_argument('--lam_hi_km',     type=float, default=200.0,
                   help='PSD band upper wavelength (km)')
    p.add_argument('--psd_curves',    action='store_true',
                   help='Save full PSD curves to output JSON')
    return p.parse_args()


if __name__ == '__main__':
    fix_cfg_paths()
    args = parse_args()
    print(f'\n{"═"*60}')
    print(f'  WenHai ONNX Evaluation — Single-Step 1-Day Forecast')
    print(f'{"═"*60}')
    print(f'  ONNX          : {args.onnx}')
    print(f'  sample_glorys : {args.sample_glorys}')
    print(f'  sample_era5   : {args.sample_era5 or "(none — zero flux)"}')
    print(f'  Stats dir     : {Path(args.onnx).parent}')
    print(f'  Global grid   : {GLOBAL_H} × {GLOBAL_W}')
    print(f'  dx_km native  : {args.dx_km}')
    print(f'  PSD band      : {args.lam_lo_km}–{args.lam_hi_km} km\n')
    evaluate(args)