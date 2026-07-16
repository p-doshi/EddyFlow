"""
eval_era5_sst_baseline.py — Compute PSD ratio and RMSE of ERA5 SST
bilinearly interpolated to fine-grid MUR resolution over the 2022-2023 test set.

This quantifies the spectral gap between a global-model-interpolated product
and high-resolution MUR SST — the key argument for why dedicated downscaling
is needed.  The ERA5 SST at 0.25° carries essentially no information in the
5-50 km band that EddyFlow is designed to recover.

Usage:
    python scripts/eval_era5_sst_baseline.py --output results_era5_sst_baseline.json
"""

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from config import cfg

# ─────────────────────────────────────────────────────────────────────────────
ERA5_SST_CH  = 3      # sst_era5 is channel index 3 in the 21-channel ERA5 array
TEST_YEARS   = (2022, 2023)
DX_KM        = 1.11   # MUR pixel spacing at ~49°N (0.01° ≈ 1.11 km)
LAM_LO_KM    = 5.0    # lower wavelength bound for PSD band
LAM_HI_KM    = 50.0   # upper wavelength bound for PSD band

# ─────────────────────────────────────────────────────────────────────────────

def band_mean_psd(field: np.ndarray, dx_km: float,
                  lam_lo_km: float, lam_hi_km: float) -> float:
    H, W   = field.shape
    f2d    = np.fft.rfft2(field)
    p2d    = (np.abs(f2d) ** 2) / (H * W)
    ky     = np.fft.fftfreq(H, d=dx_km)
    kx     = np.fft.rfftfreq(W, d=dx_km)
    KX, KY = np.meshgrid(kx, ky)
    K2D    = np.sqrt(KX**2 + KY**2)
    lo, hi = 1.0 / lam_hi_km, 1.0 / lam_lo_km
    mask   = (K2D >= lo) & (K2D <= hi)
    return float(p2d[mask].mean()) if mask.sum() > 0 else float('nan')


def main(args):
    root = Path(__file__).resolve().parent.parent

    # ── Load stats (all paths relative to project root) ──────────────────────
    stats_dir = root / 'scripts' / 'data' / 'stats'
    era5_mean = np.load(stats_dir / 'era5_mean.npy').astype(np.float32)  # [21]
    era5_std  = np.load(stats_dir / 'era5_std.npy').astype(np.float32)   # [21]
    sst_mean  = float(np.load(stats_dir / 'sst_mean.npy'))
    sst_std   = float(np.load(stats_dir / 'sst_std.npy'))

    # ── Open zarrs ───────────────────────────────────────────────────────────
    gsl = root / 'data' / 'domains' / 'gsl'
    era5_z = zarr.open(str(gsl / 'era5.zarr'), mode='r')   # [T, 21, 20, 48]
    mur_z  = zarr.open(str(gsl / 'mur.zarr'),  mode='r')   # [T, H_f, W_f]

    era5_times = pd.DatetimeIndex(
        np.load(str(gsl / 'era5_times.npy'), allow_pickle=True))
    mur_times  = pd.DatetimeIndex(
        np.load(str(gsl / 'mur_times.npy'),  allow_pickle=True))

    H_f, W_f = mur_z.shape[1], mur_z.shape[2]
    print(f'ERA5 shape : {era5_z.shape}')
    print(f'MUR  shape : {mur_z.shape}')

    # ── Test-split dates (ERA5 time aligns to 18Z; MUR to 09Z next day) ──────
    test_mask  = np.array([t.year in range(TEST_YEARS[0], TEST_YEARS[1]+1)
                           for t in era5_times])
    test_idxs  = np.where(test_mask)[0]
    print(f'\nTest samples: {len(test_idxs)} ERA5 frames in {TEST_YEARS[0]}–{TEST_YEARS[1]}')

    # Build (era5_idx, mur_idx) pairs: ERA5 at t → MUR SST at t
    pairs = []
    for ei in test_idxs:
        et = era5_times[ei]
        # find closest MUR time within ±2 days
        delta = np.abs((mur_times - et).total_seconds())
        mi    = int(delta.argmin())
        if delta[mi] / 86400 <= 2:
            pairs.append((ei, mi))

    print(f'Matched pairs: {len(pairs)}')

    # ── Accumulate metrics ───────────────────────────────────────────────────
    sq_err_model   = 0.0   # (era5_sst_interp - mur)^2 sum
    sq_err_persist = 0.0   # same but persistence = era5 sst itself (denominator)
    psd_pred_sum   = 0.0
    psd_targ_sum   = 0.0
    n_valid_px     = 0
    n_samples      = 0

    for ei, mi in pairs:
        # ERA5 zarr stores raw physical values (not normalised).
        # ch3 (sst_era5) is in Kelvin → subtract 273.15 to get Celsius.
        era5_sst_phys = era5_z[ei, ERA5_SST_CH].astype(np.float32) - 273.15
        era5_sst_phys = np.nan_to_num(era5_sst_phys, nan=0.0)

        # MUR SST: GSL zarr stores raw physical Celsius — use directly
        mur_phys = mur_z[mi].astype(np.float32)

        # Valid ocean mask
        valid = np.isfinite(mur_phys) & (mur_phys > -3)

        # Bilinear upsample ERA5 SST to fine grid
        era5_t = torch.from_numpy(era5_sst_phys)[None, None]   # [1,1,20,48]
        era5_f = F.interpolate(era5_t, size=(H_f, W_f),
                               mode='bilinear', align_corners=False
                               ).squeeze().numpy()              # [H_f, W_f]

        diff_sq = np.where(valid, (era5_f - mur_phys)**2, 0.0)
        sq_err_model   += float(diff_sq.sum())
        sq_err_persist += float(diff_sq.sum())   # ERA5 SST IS the persistence baseline
        n_valid_px     += int(valid.sum())

        # PSD at fine resolution — use np.where so NaN land pixels become 0
        pred_m = np.where(valid, era5_f,   0.0).astype(np.float32)
        targ_m = np.where(valid, mur_phys, 0.0).astype(np.float32)
        psd_pred_sum += band_mean_psd(pred_m, DX_KM, LAM_LO_KM, LAM_HI_KM)
        psd_targ_sum += band_mean_psd(targ_m, DX_KM, LAM_LO_KM, LAM_HI_KM)

        n_samples += 1
        if n_samples % 50 == 0:
            print(f'  {n_samples}/{len(pairs)} done …', flush=True)

    rmse_model   = float(np.sqrt(sq_err_model   / max(n_valid_px, 1)))
    psd_pred_avg = psd_pred_sum / max(n_samples, 1)
    psd_targ_avg = psd_targ_sum / max(n_samples, 1)
    psd_ratio    = psd_pred_avg / max(psd_targ_avg, 1e-30)
    psd_log10    = float(np.log10(max(psd_ratio, 1e-20)))

    # ERA5 SST skill vs itself = 0 by definition (it IS the persist baseline).
    # Report skill vs oracle (MUR yesterday) if available, or just note 0.
    skill_vs_persist = 0.0  # ERA5 SST is the persistence denominator

    result = dict(
        model              = 'ERA5_SST_bilinear',
        description        = 'ERA5 sst_era5 (0.25°) bilinearly interpolated to MUR 0.01° grid',
        test_years         = list(TEST_YEARS),
        n_samples          = n_samples,
        n_valid_pixels     = int(n_valid_px),
        rmse_model_K       = rmse_model,
        skill_vs_persist   = skill_vs_persist,
        note_skill         = 'ERA5 SST IS the persistence baseline so skill=0 by definition',
        psd_pred_band      = float(psd_pred_avg),
        psd_targ_band      = float(psd_targ_avg),
        psd_ratio          = float(psd_ratio),
        psd_log10_bias     = float(psd_log10),
        psd_band_km        = f'{LAM_LO_KM}–{LAM_HI_KM} km',
        dx_km              = DX_KM,
    )

    print(f'\n=== ERA5 SST Baseline (bilinear interpolation to MUR fine grid) ===')
    print(f'  RMSE          : {rmse_model:.4f} °C')
    print(f'  Skill         : {skill_vs_persist:.4f} (0 by definition)')
    print(f'  PSD ratio     : {psd_ratio:.6f}  (log10={psd_log10:.3f})')
    print(f'  Interpretation: ERA5 SST carries {psd_ratio*100:.3f}% of target '
          f'mesoscale PSD in {LAM_LO_KM}–{LAM_HI_KM} km band')
    print(f'  → EddyFlow Stage4 PSD ratio ≈ 0.974 ({97.4:.1f}% of target band energy)')

    out = Path(args.output)
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(result, indent=2))
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='results_era5_sst_baseline.json')
    main(p.parse_args())
