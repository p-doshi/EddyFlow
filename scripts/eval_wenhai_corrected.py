"""
eval_wenhai_corrected.py — WenHai evaluation with ERA5-SST persistence denominator
and multi-domain (GSL + BOF + GOM) cropping in a single inference pass.

Key differences from original wenhai_eval.py:
  1. Skill denominator = ERA5 SST persistence (same as EddyFlow), not GLORYS IC hold.
     This makes WenHai skill scores directly comparable to EddyFlow.
  2. All three domains (GSL, BOF, GOM) evaluated from one global inference.
  3. PSD computed at MUR fine-grid resolution after bilinear upsample of WenHai output,
     in the same 5-50 km band used by EddyFlow (not WenHai's original 20-200 km band).

Usage (requires onnxruntime-gpu in venv):
    cd /scratch/pdoshi/my_project/eddyflow
    python scripts/eval_wenhai_corrected.py \\
        --onnx          scripts/WenHai.onnx \\
        --sample_glorys scripts/sample_GLORYS_23lev.nc \\
        --output_prefix results_wenhai_corrected

Outputs:
    results_wenhai_corrected_gsl.json
    results_wenhai_corrected_bof.json   (if BOF MUR data available)
    results_wenhai_corrected_gom.json   (if GOM MUR data available)
"""

import argparse, json, math
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
import xarray as xr
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom as nd_zoom

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from config import cfg

THETAO_SFC_CH = 0   # WenHai channel layout: thetao surface = ch 0
ERA5_SST_CH   = 3   # ERA5 sst_era5 is channel index 3
DX_KM_FINE    = 1.11
LAM_LO_KM     = 5.0
LAM_HI_KM     = 50.0

DOMAIN_BOUNDS = {
    'gsl': dict(lat_lo=44.0, lat_hi=52.8, lon_lo=-69.5, lon_hi=-56.0,
                mur_zarr   = 'data/domains/gsl/mur.zarr',
                mur_times  = 'data/domains/gsl/mur_times.npy',
                era5_zarr  = 'data/domains/gsl/era5.zarr',
                era5_times = 'data/domains/gsl/era5_times.npy',
                fine_h=501,  fine_w=1201),
    'bof': dict(lat_lo=44.0, lat_hi=47.0, lon_lo=-67.0, lon_hi=-63.0,
                mur_zarr   = 'data/domains/bof/mur/mur_bof.zarr',
                mur_times  = 'data/domains/bof/mur/mur_bof_times.npy',
                era5_zarr  = None,   # use GSL era5 and crop
                era5_times = None,
                fine_h=301, fine_w=401),
    'gom': dict(lat_lo=18.0, lat_hi=31.0, lon_lo=-98.0, lon_hi=-80.0,
                mur_zarr   = 'data/domains/gom/mur/mur_gom.zarr',
                mur_times  = 'data/domains/gom/mur/mur_gom_times.npy',
                era5_zarr  = False,   # no GOM ERA5 zarr; skip skill vs ERA5
                era5_times = None,
                fine_h=1301, fine_w=1801),
}


def band_mean_psd(field: np.ndarray, dx_km: float,
                  lam_lo_km: float, lam_hi_km: float) -> float:
    H, W   = field.shape
    f2d    = np.fft.rfft2(field)
    p2d    = (np.abs(f2d)**2) / (H * W)
    ky     = np.fft.fftfreq(H, d=dx_km)
    kx     = np.fft.rfftfreq(W, d=dx_km)
    K2D    = np.sqrt(np.meshgrid(kx, ky)[0]**2 + np.meshgrid(kx, ky)[1]**2)
    lo, hi = 1.0 / lam_hi_km, 1.0 / lam_lo_km
    mask   = (K2D >= lo) & (K2D <= hi)
    return float(p2d[mask].mean()) if mask.sum() > 0 else float('nan')


def detect_slice(glb_lat, glb_lon, lat_lo, lat_hi, lon_lo, lon_hi):
    lat_idx = np.where((glb_lat >= lat_lo - 0.5) & (glb_lat <= lat_hi + 0.5))[0]
    lon_idx = np.where((glb_lon >= lon_lo - 0.5) & (glb_lon <= lon_hi + 0.5))[0]
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        return None, None
    return slice(lat_idx[0], lat_idx[-1]+1), slice(lon_idx[0], lon_idx[-1]+1)


def upsample_native(arr, h, w):
    zh, zw = h / arr.shape[0], w / arr.shape[1]
    return nd_zoom(arr.astype(np.float32), (zh, zw), order=1)


def eval_domain(domain_key, pred_global_norm, ic_global_norm,
                mn, mx, glb_lat, glb_lon, ic_date,
                stats_dir, root):
    dc = DOMAIN_BOUNDS[domain_key]

    # ── Crop domain from WenHai global output ────────────────────────────────
    lat_sl, lon_sl = detect_slice(glb_lat, glb_lon,
                                  dc['lat_lo'], dc['lat_hi'],
                                  dc['lon_lo'], dc['lon_hi'])
    if lat_sl is None:
        print(f'  [{domain_key.upper()}] Domain not found in global grid — skipping')
        return None

    pred_native = pred_global_norm[0, THETAO_SFC_CH, lat_sl, lon_sl]  # [nh, nw]
    ic_native   = ic_global_norm  [0, THETAO_SFC_CH, lat_sl, lon_sl]

    # Denorm to physical °C at native 1/12° resolution
    pred_phys_native = pred_native.astype(np.float32) * (mx - mn) + mn
    ic_phys_native   = ic_native.astype(np.float32)   * (mx - mn) + mn
    print(f'  [{domain_key.upper()}] native slice {pred_native.shape}  '
          f'T_pred range [{pred_phys_native.min():.1f}, {pred_phys_native.max():.1f}]°C')

    # ── Upsample to fine MUR grid ─────────────────────────────────────────────
    H_f, W_f = dc['fine_h'], dc['fine_w']
    pred_fine = upsample_native(pred_phys_native, H_f, W_f)
    ic_fine   = upsample_native(ic_phys_native,   H_f, W_f)

    # ── Load MUR target ───────────────────────────────────────────────────────
    mur_zarr_path  = root / dc['mur_zarr']
    mur_times_path = root / dc['mur_times']
    if not mur_zarr_path.exists():
        print(f'  [{domain_key.upper()}] MUR zarr not found: {mur_zarr_path} — skipping')
        return None

    mur_z     = zarr.open(str(mur_zarr_path), mode='r')
    mur_times = pd.DatetimeIndex(np.load(str(mur_times_path), allow_pickle=True))
    target_date = ic_date + pd.Timedelta(days=1)
    delta   = np.abs((mur_times - target_date).total_seconds())
    mi      = int(delta.argmin())
    if delta[mi] / 86400 > 2:
        print(f'  [{domain_key.upper()}] No MUR close to {target_date.date()} — skipping')
        return None

    # GSL zarr is a flat normalised array; BOF/GOM are zarr Groups with 'analysed_sst'
    import zarr as _zarr
    if isinstance(mur_z, _zarr.hierarchy.Group):
        mur_raw = mur_z['analysed_sst'][mi].astype(np.float32)
        # BOF/GOM store raw Kelvin
        mur_raw = np.where(np.isfinite(mur_raw), mur_raw - 273.15, np.nan)
    else:
        # GSL: flat array storing raw physical °C — use directly
        mur_raw = mur_z[mi].astype(np.float32)
    mur_phys = mur_raw
    valid    = np.isfinite(mur_phys) & (mur_phys > -3)

    # Resize pred_fine/ic_fine to match actual MUR zarr shape
    if pred_fine.shape != mur_phys.shape:
        pred_fine = upsample_native(pred_fine, mur_phys.shape[0], mur_phys.shape[1])
        ic_fine   = upsample_native(ic_fine,   mur_phys.shape[0], mur_phys.shape[1])

    # ── ERA5 SST persistence for same date ───────────────────────────────────
    # BOF falls within the GSL ERA5 domain so we can use GSL ERA5 as fallback.
    # GOM is out-of-domain (era5_zarr=False) — skip skill vs ERA5 for GOM.
    _ez = dc['era5_zarr']
    era5_persist_rmse = None
    if _ez is not False:
        era5_zarr_path  = root / (_ez if _ez else 'data/domains/gsl/era5.zarr')
        era5_times_path = root / (dc['era5_times'] if dc['era5_times']
                                  else 'data/domains/gsl/era5_times.npy')
    if _ez is not False and era5_zarr_path.exists():
        era5_z      = zarr.open(str(era5_zarr_path), mode='r')
        era5_times  = pd.DatetimeIndex(np.load(str(era5_times_path), allow_pickle=True))
        delta_e     = np.abs((era5_times - ic_date).total_seconds())
        ei          = int(delta_e.argmin())
        if delta_e[ei] / 86400 <= 2:
            # ERA5 zarr stores raw physical values; ch3 is raw Kelvin
            sst_phys = era5_z[ei, ERA5_SST_CH].astype(np.float32) - 273.15
            sst_phys = np.nan_to_num(sst_phys, nan=0.0)
            sst_t    = torch.from_numpy(sst_phys)[None, None]
            sst_fine = F.interpolate(sst_t, size=mur_phys.shape,
                                     mode='bilinear', align_corners=False
                                     ).squeeze().numpy()
            diff_e = np.where(valid, (sst_fine - mur_phys)**2, 0.0)
            era5_persist_rmse = float(np.sqrt(diff_e.sum() / max(int(valid.sum()), 1)))

    # ── RMSE vs MUR ──────────────────────────────────────────────────────────
    diff_pred = np.where(valid, (pred_fine - mur_phys)**2, 0.0)
    diff_ic   = np.where(valid, (ic_fine   - mur_phys)**2, 0.0)
    rmse_pred = float(np.sqrt(diff_pred.sum() / max(int(valid.sum()), 1)))
    rmse_ic   = float(np.sqrt(diff_ic.sum()   / max(int(valid.sum()), 1)))

    skill_vs_ic   = 1.0 - rmse_pred / max(rmse_ic, 1e-6)
    skill_vs_era5 = (1.0 - rmse_pred / era5_persist_rmse
                     if era5_persist_rmse else None)

    # ── PSD at fine grid ─── use np.where so NaN land pixels become 0 ────────
    pred_m = np.where(valid, pred_fine, 0.0).astype(np.float32)
    targ_m = np.where(valid, mur_phys,  0.0).astype(np.float32)
    psd_pred = band_mean_psd(pred_m, DX_KM_FINE, LAM_LO_KM, LAM_HI_KM)
    psd_targ = band_mean_psd(targ_m, DX_KM_FINE, LAM_LO_KM, LAM_HI_KM)
    psd_ratio = psd_pred / max(psd_targ, 1e-30)

    result = dict(
        model              = 'WenHai-ONNX',
        domain             = domain_key,
        ic_date            = str(ic_date.date()),
        target_date        = str(target_date.date()),
        native_slice       = f'{pred_native.shape[0]}×{pred_native.shape[1]} at 1/12°',
        fine_grid          = f'{mur_phys.shape[0]}×{mur_phys.shape[1]} at 0.01°',
        n_valid_pixels     = int(valid.sum()),
        rmse_model_K       = rmse_pred,
        rmse_glorys_ic_K   = rmse_ic,
        rmse_era5_persist_K= era5_persist_rmse,
        skill_vs_glorys_ic = skill_vs_ic,
        skill_vs_era5      = skill_vs_era5,
        psd_pred_band      = float(psd_pred),
        psd_targ_band      = float(psd_targ),
        psd_ratio          = float(psd_ratio),
        psd_log10_bias     = float(np.log10(max(psd_ratio, 1e-20))),
        psd_band_km        = f'{LAM_LO_KM}–{LAM_HI_KM} km',
        dx_km_fine         = DX_KM_FINE,
        note = ('skill_vs_era5 uses same ERA5-SST-persistence denominator as EddyFlow '
                '— directly comparable to EddyFlow skill scores'),
    )

    skill_str = f'{skill_vs_era5:.4f}' if skill_vs_era5 is not None else 'n/a'
    print(f'  [{domain_key.upper()}] RMSE={rmse_pred:.3f}K  '
          f'skill_vs_IC={skill_vs_ic:.4f}  '
          f'skill_vs_ERA5={skill_str}  '
          f'PSD_ratio={psd_ratio:.4f}')
    return result


def main(args):
    import onnxruntime as ort

    root      = Path(__file__).resolve().parent.parent
    stats_dir = root / 'scripts'

    min_GLORYS = np.load(stats_dir / 'min_GLORYS.npy').reshape(1, -1, 1, 1)
    max_GLORYS = np.load(stats_dir / 'max_GLORYS.npy').reshape(1, -1, 1, 1)
    min_flux   = np.load(stats_dir / 'min_flux.npy').reshape(-1, 1, 1)
    max_flux   = np.load(stats_dir / 'max_flux.npy').reshape(-1, 1, 1)

    mn = float(min_GLORYS[0, THETAO_SFC_CH, 0, 0])
    mx = float(max_GLORYS[0, THETAO_SFC_CH, 0, 0])

    # ── Load GLORYS IC ────────────────────────────────────────────────────────
    print(f'Loading GLORYS IC: {args.sample_glorys}')
    ds = xr.open_dataset(args.sample_glorys)
    ic_date = pd.Timestamp(ds.time.values[0] if ds.time.values.ndim > 0
                           else ds.time.values)
    print(f'IC date: {ic_date.date()}')

    def g(name, fb=None):
        for c in [name] + (fb or []):
            if c in ds: return ds[c].values.astype(np.float32)
        raise KeyError(name)

    def e4d(x):
        if x.ndim == 2: x = x[None, None]
        if x.ndim == 3: x = x[None]
        return x

    thetao = e4d(g('thetao'))
    so     = e4d(g('so'))
    uo     = e4d(g('uo', ['u']))
    vo     = e4d(g('vo', ['v']))
    zos    = e4d(g('zos', ['ssh', 'adt']))
    ds.close()

    init_global = np.concatenate([thetao, so, uo, vo, zos], axis=1).astype(np.float32)
    init_global = np.nan_to_num(init_global)   # replace land NaNs with 0 before norm
    init_norm   = np.clip((init_global - min_GLORYS) / (max_GLORYS - min_GLORYS + 1e-30),
                          0.0, 1.0).astype(np.float16)

    # Bulk flux (zeros if no ERA5 forcing file)
    bulk_norm = np.zeros((1, 8, init_norm.shape[2], init_norm.shape[3]),
                         dtype=np.float16)

    # ── Global grid ───────────────────────────────────────────────────────────
    glb_lat = np.linspace(-80.0, 90.0,  init_norm.shape[2])
    glb_lon = np.linspace(-180.0, 180.0, init_norm.shape[3])

    # ── WenHai inference ─────────────────────────────────────────────────────
    print('Running WenHai ONNX inference ...')
    sess = ort.InferenceSession(args.onnx,
                                providers=['CUDAExecutionProvider',
                                           'CPUExecutionProvider'])
    n1, n2 = sess.get_inputs()[0].name, sess.get_inputs()[1].name
    print(f'  Provider: {sess.get_providers()[0]}')

    out_norm = sess.run(None, {n1: init_norm, n2: bulk_norm})[0]
    out_norm = np.clip(
        out_norm.astype(np.float32) + init_norm.astype(np.float32),
        0.0, 1.0
    ).astype(np.float16)
    print(f'  out shape: {out_norm.shape}')

    # ── Evaluate each domain ──────────────────────────────────────────────────
    for domain in ['gsl', 'bof', 'gom']:
        print(f'\n--- {domain.upper()} ---')
        res = eval_domain(domain, out_norm, init_norm, mn, mx,
                          glb_lat, glb_lon, ic_date,
                          stats_dir, root)
        if res is not None:
            out_path = Path(f'{args.output_prefix}_{domain}.json')
            out_path.write_text(json.dumps(res, indent=2))
            print(f'  Saved: {out_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--onnx',          default='scripts/WenHai.onnx')
    p.add_argument('--sample_glorys', default='scripts/sample_GLORYS_23lev.nc')
    p.add_argument('--output_prefix', default='results_wenhai_corrected')
    main(p.parse_args())
