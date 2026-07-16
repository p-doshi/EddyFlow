"""
Prepare min/max and mask files for WenHai:

  - min_GLORYS.npy, max_GLORYS.npy
  - min_flux.npy,  max_flux.npy
  - mask_GLORYS.nc

Requires:
  - data/processed/glorys.zarr    [T, 5, H, W]
  - data/processed/era5.zarr      [T, 21, H_c, W_c]
"""

import os
import sys
import numpy as np
import zarr
import xarray as xr
import pandas as pd
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from src.config import cfg
from metpy.units import units
from metpy.calc import specific_humidity_from_dewpoint
import AirSeaFluxCode as asfc   # pip install AirSeaFluxCode


# ─────────────────────────────────────────────────────────────────────────────
# FIX RELATIVE CFG PATHS
# ─────────────────────────────────────────────────────────────────────────────

def fix_cfg_paths():
    root = Path('/home/parth/ieee2026/eddyflow')
    scripts_data = root / 'scripts' / 'data'

    cfg.GLORYS_ZARR  = str(scripts_data / 'processed' / 'glorys.zarr')
    cfg.ERA5_ZARR    = str(scripts_data / 'processed' / 'era5.zarr')
    cfg.MUR_ZARR     = str(scripts_data / 'processed' / 'mur.zarr')
    cfg.ICE_ZARR     = str(scripts_data / 'processed' / 'ice.zarr')
    cfg.ERR_ZARR     = str(scripts_data / 'processed' / 'mur_error.zarr')
    cfg.BATHY_NPY    = str(scripts_data / 'processed' / 'bathy.npy')
    cfg.ERA5_TIMES   = str(scripts_data / 'processed' / 'era5_times.npy')
    cfg.MUR_TIMES    = str(scripts_data / 'processed' / 'mur_times.npy')
    cfg.GLORYS_TIMES = str(scripts_data / 'processed' / 'glorys_times.npy')
    cfg.ERA5_MEAN    = str(scripts_data / 'stats' / 'era5_mean.npy')
    cfg.ERA5_STD     = str(scripts_data / 'stats' / 'era5_std.npy')
    cfg.SST_MEAN     = str(scripts_data / 'stats' / 'sst_mean.npy')
    cfg.SST_STD      = str(scripts_data / 'stats' / 'sst_std.npy')
    cfg.STATS_DIR    = str(scripts_data / 'stats')
    cfg.CKPT_DIR     = str(root / 'outputs' / 'checkpoints')
    cfg.LOG_DIR      = str(root / 'outputs' / 'logs')
    cfg.FIG_DIR      = str(root / 'outputs' / 'figures')


# ─────────────────────────────────────────────────────────────────────────────
# GRID HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_era5_grid():
    era5 = zarr.open(cfg.ERA5_ZARR, mode='r')
    era5_lat = np.linspace(47.0, 51.75, era5.shape[2])
    era5_lon = np.linspace(-68.0, -56.25, era5.shape[3])
    return era5_lat, era5_lon


def load_glorys_grid():
    g = zarr.open(cfg.GLORYS_ZARR, mode='r')  # [T, 5, H, W]
    T, C, H, W = g.shape
    print(f"  GLORYS shape: {g.shape}")
    return g, H, W


# ─────────────────────────────────────────────────────────────────────────────
# GLORYS MIN / MAX
# ─────────────────────────────────────────────────────────────────────────────

def build_minmax_glorys():
    g, H, W = load_glorys_grid()
    g_arr = np.asarray(g[:], dtype=np.float32)  # [T, 5, H, W]

    mur_times_raw = np.load(cfg.MUR_TIMES, allow_pickle=True)
    mur_pd = pd.DatetimeIndex(mur_times_raw)

    y0, y1 = 2013, 2020
    mur_mask = (mur_pd.year >= y0) & (mur_pd.year <= y1)
    mur_idx = np.where(mur_mask)[0]

    glorys_times = np.load(cfg.GLORYS_TIMES, allow_pickle=True)
    glorys_pd = pd.DatetimeIndex(glorys_times)
    time_map = np.array(
        [glorys_pd.get_indexer([mur_pd[i]], method='nearest')[0] for i in mur_idx]
    )
    time_map = np.clip(time_map, 0, g.shape[0] - 1)

    train_data = g_arr[time_map, ...]  # [T_train, 5, H, W]

    min_GLORYS = np.nanmin(train_data, axis=(0, 2, 3), keepdims=True)  # [1, 5, 1, 1]
    max_GLORYS = np.nanmax(train_data, axis=(0, 2, 3), keepdims=True)

    np.save(os.path.join(cfg.STATS_DIR, 'min_GLORYS.npy'), min_GLORYS.astype(np.float32))
    np.save(os.path.join(cfg.STATS_DIR, 'max_GLORYS.npy'), max_GLORYS.astype(np.float32))
    print(f"Saved min_GLORYS.npy  shape={min_GLORYS.shape}")
    print(f"Saved max_GLORYS.npy  shape={max_GLORYS.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# BULK FLUX ROW (pure Python, no aerobulk)
# ─────────────────────────────────────────────────────────────────────────────

def bulk_flux_row(t0, t2m, h2m, u10, v10, msl):
    """
    NCAR bulk flux parameterisation (Large & Pond 1981 / Large & Yeager 2004).
    All inputs: 1D float32 arrays of length W (one latitude slice).
    Land points should already be zeroed via glor_mask before calling.

    Returns: qh, qe, taux, tauy, evap  (float32 1D arrays)
    """
    rho_air = 1.225      # kg/m³  (constant approximation)
    cp_air  = 1004.0     # J/kg/K
    Lv      = 2.501e6    # J/kg   latent heat of vaporisation

    U = np.sqrt(u10**2 + v10**2)
    U = np.maximum(U, 0.5)          # floor at 0.5 m/s to avoid degenerate calcs

    # NCAR neutral drag coefficient — Large & Pond (1981)
    Cd = np.where(
        U <= 11.0,  1.2e-3,
        np.where(U <= 25.0, (0.49 + 0.065 * U) * 1e-3, 2.34e-3)
    )
    Ch = 1.0e-3    # sensible heat transfer coefficient
    Ce = 1.15e-3   # latent heat transfer coefficient

    # Saturated specific humidity at SST (ocean only, salinity correction 0.98)
    e_sat = 611.2 * np.exp(17.67 * (t0 - 273.15) / np.maximum(t0 - 29.65, 1.0))
    q_sat = 0.622 * e_sat / np.maximum(msl - 0.378 * e_sat, 1.0)
    dq    = 0.98 * q_sat - h2m      # ocean-air specific humidity difference
    dT    = t0 - t2m                # SST minus air temp

    taux = (rho_air * Cd * U * u10).astype(np.float32)
    tauy = (rho_air * Cd * U * v10).astype(np.float32)
    qh   = (rho_air * cp_air * Ch * U * dT).astype(np.float32)
    qe   = (rho_air * Lv     * Ce * U * dq).astype(np.float32)
    evap = (qe / Lv).astype(np.float32)

    return qh, qe, taux, tauy, evap


# ─────────────────────────────────────────────────────────────────────────────
# FLUX MIN / MAX  +  GLORYS MASK
# ─────────────────────────────────────────────────────────────────────────────

def build_minmax_flux():
    """
    Build min_flux.npy, max_flux.npy, and mask_GLORYS.nc.
    Flux channels (8): ql, ssr, qh, qe, taux, tauy, evap, mtpr
    """
    era5_lat, era5_lon = load_era5_grid()
    era5 = zarr.open(cfg.ERA5_ZARR, mode='r')  # [T, 21, H_c, W_c]

    # Channel indices matching 05_preprocess.py SURFACE_VARS
    CH_U10, CH_V10, CH_MSL, CH_SST, CH_T2M, CH_SICONC = 0, 1, 2, 3, 4, 5

    era5_times_raw = np.load(cfg.ERA5_TIMES, allow_pickle=True)
    era5_pd = pd.DatetimeIndex(era5_times_raw)

    y0, y1 = 2013, 2020
    era5_mask = (era5_pd.year >= y0) & (era5_pd.year <= y1)
    era5_idx  = np.where(era5_mask)[0]
    era5_train = era5[era5_idx, :, :, :]  # [T_train, 21, H_c, W_c]

    # GLORYS grid + land mask
    g, H, W = load_glorys_grid()
    glor_mask = (
        (g[0, 0] != 0) | (g[0, 1] != 0) |
        (g[0, 2] != 0) | (g[0, 3] != 0)
    ).astype(np.float32)  # [H, W]

    # Fine grid lat/lon matching your GLORYS zarr
    grid_lat = np.linspace(47.0, 51.99, H)
    grid_lon = np.linspace(-68.0, -56.01, W)

    # Write mask_GLORYS.nc
    mask_da = xr.DataArray(
        glor_mask[None, None, :, :],
        dims=["time", "var", "lat", "lon"],
        coords={"lat": grid_lat, "lon": grid_lon},
    )
    mask_path = os.path.join(cfg.STATS_DIR, 'mask_GLORYS.nc')
    xr.Dataset({"mask": mask_da}).to_netcdf(mask_path)
    print(f"Saved mask_GLORYS.nc → {mask_path}")

    # Precompute meshgrid query points once
    lat_grid, lon_grid = np.meshgrid(grid_lat, grid_lon, indexing='ij')
    pts = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)  # [H*W, 2]

    # Sample a small set of training days
    n_sample = min(30, len(era5_train))
    accum = {k: [] for k in ('ql', 'ssr', 'qh', 'qe', 'taux', 'tauy', 'evap', 'mtpr')}

    for tidx in range(n_sample):
        era5_t = era5_train[tidx]  # [21, H_c, W_c]

        u10_t  = era5_t[CH_U10].astype(np.float32)
        v10_t  = era5_t[CH_V10].astype(np.float32)
        msl_t  = era5_t[CH_MSL].astype(np.float32)
        t2m_t  = era5_t[CH_T2M].astype(np.float32)
        d2m_t  = era5_t[CH_T2M].astype(np.float32)   # best proxy available
        mtpr_t = era5_t[CH_SST].astype(np.float32)   # ch3 = sst in your ERA5
        ssr_t  = era5_t[CH_SST].astype(np.float32)   # placeholder
        strd_t = era5_t[CH_SST].astype(np.float32)   # placeholder

        # SST placeholder (10 °C = 283.15 K), masked to ocean
        t0 = np.full((H, W), 283.15, dtype=np.float32) * glor_mask
        u0 = np.zeros((H, W), dtype=np.float32)
        v0 = np.zeros((H, W), dtype=np.float32)

        # Interpolate ERA5 coarse → GLORYS fine grid
        def interp2fine(field):
            fn = RegularGridInterpolator(
                (era5_lat, era5_lon), field,
                method='linear', bounds_error=False, fill_value=0.0
            )
            return fn(pts).reshape(H, W).astype(np.float32)

        t2m  = interp2fine(t2m_t)
        d2m  = interp2fine(d2m_t)
        u10  = interp2fine(u10_t)
        v10  = interp2fine(v10_t)
        mtpr = interp2fine(mtpr_t)
        ssr  = interp2fine(ssr_t)
        strd = interp2fine(strd_t)
        msl  = interp2fine(msl_t)

        ssr  /= 3600.0
        strd /= 3600.0
        mtpr /= 1000.0

        h2m = np.zeros((H, W), dtype=np.float32)
        ocean = glor_mask > 0
        if ocean.any():
            msl_o = np.clip(msl[ocean], 50000.0, 115000.0)
            d2m_o = np.clip(d2m[ocean], 200.0, 340.0)
            h2m_o = specific_humidity_from_dewpoint(
                msl_o * units.Pa, d2m_o * units.K
            ).to('kg/kg').magnitude
            h2m[ocean] = np.nan_to_num(h2m_o.astype(np.float32))

        qh_2d   = np.zeros_like(t0)
        qe_2d   = np.zeros_like(t0)
        taux_2d = np.zeros_like(t0)
        tauy_2d = np.zeros_like(t0)
        evap_2d = np.zeros_like(t0)

        for j in range(W):
            qh_j, qe_j, taux_j, tauy_j, evap_j = bulk_flux_row(
                t0=t0[:, j],
                t2m=t2m[:, j],
                h2m=h2m[:, j],
                u10=u10[:, j] - u0[:, j],
                v10=v10[:, j] - v0[:, j],
                msl=msl[:, j],
            )
            qh_2d[:, j]   = qh_j
            qe_2d[:, j]   = qe_j
            taux_2d[:, j] = taux_j
            tauy_2d[:, j] = tauy_j
            evap_2d[:, j] = evap_j

        sigma = 5.67e-8
        ql = strd - sigma * (t0 ** 4)

        accum['ql'].append(ql)
        accum['ssr'].append(ssr)
        accum['qh'].append(qh_2d)
        accum['qe'].append(qe_2d)
        accum['taux'].append(taux_2d)
        accum['tauy'].append(tauy_2d)
        accum['evap'].append(evap_2d)
        accum['mtpr'].append(mtpr)

        if (tidx + 1) % 5 == 0:
            print(f"  Processed {tidx + 1}/{n_sample} ERA5 samples…")

    # Stack into [8, n_sample*H, W] and compute per-channel min/max
    channels = [
        np.stack(accum[k], axis=0).reshape(-1, W)
        for k in ('ql', 'ssr', 'qh', 'qe', 'taux', 'tauy', 'evap', 'mtpr')
    ]
    bulk_flux = np.stack(channels, axis=0)   # [8, n*H, W]
    bulk_flux = np.nan_to_num(bulk_flux)

    min_flux = np.nanmin(bulk_flux, axis=(1, 2), keepdims=True)   # [8, 1, 1]
    max_flux = np.nanmax(bulk_flux, axis=(1, 2), keepdims=True)

    np.save(os.path.join(cfg.STATS_DIR, 'min_flux.npy'), min_flux.astype(np.float32))
    np.save(os.path.join(cfg.STATS_DIR, 'max_flux.npy'), max_flux.astype(np.float32))
    print(f"Saved min_flux.npy  shape={min_flux.shape}")
    print(f"Saved max_flux.npy  shape={max_flux.shape}")
    print("─" * 60)
    print("All WenHai prep files written:")
    print(f"  {cfg.STATS_DIR}/min_GLORYS.npy")
    print(f"  {cfg.STATS_DIR}/max_GLORYS.npy")
    print(f"  {cfg.STATS_DIR}/min_flux.npy")
    print(f"  {cfg.STATS_DIR}/max_flux.npy")
    print(f"  {cfg.STATS_DIR}/mask_GLORYS.nc")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fix_cfg_paths()

    print(f"ERA5_ZARR  : {cfg.ERA5_ZARR}")
    print(f"GLORYS_ZARR: {cfg.GLORYS_ZARR}")
    print(f"STATS_DIR  : {cfg.STATS_DIR}")

    os.makedirs(cfg.STATS_DIR, exist_ok=True)

    print("\nBuilding min_GLORYS / max_GLORYS from GLORYS training set...")
    build_minmax_glorys()

    print("\nBuilding min_flux / max_flux + mask_GLORYS.nc from ERA5 training set...")
    build_minmax_flux()