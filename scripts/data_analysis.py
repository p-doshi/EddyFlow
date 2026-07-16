#!/usr/bin/env python3
"""
analyze_domain.py

Complete analysis pipeline for downloaded ERA5, MUR, and GEBCO data.
Produces per-year plots for YEARLY_PLOT_YEARS instead of a single multi-year average.

Expected folder structure:
  data/domains/<domain>/
      era5/era5_<domain>_<year>.nc
      mur/mur_<domain>.zarr
      mur/mur_<domain>_times.npy
      gebco_<domain>.nc

Example:
  python analyze_domain.py --domain bof --root data/domains
  python analyze_domain.py --domain bof --root data/domains --start-year 2013 --end-year 2023
  python analyze_domain.py --domain lab --root data/domains --out analysis_custom
"""
import os
from pathlib import Path
from pyproj import datadir


def _pick_proj_dir():
    # 1. Check environment variables first
    for key in ("PROJ_DATA", "PROJ_LIB"):
        val = os.environ.get(key)
        if val and Path(val).expanduser().is_dir():
            return Path(val).expanduser()

    # 2. Try pyproj's built-in resolver
    try:
        proj_dir = Path(datadir.get_data_dir())
        if proj_dir.exists() and (proj_dir / "proj.db").exists():
            return proj_dir
    except Exception:
        pass

    # 3. Search common locations (HPC clusters, conda envs, system installs)
    import shutil, sys
    candidates = [
        Path(sys.prefix) / "share" / "proj",           # active venv/conda
        Path(sys.prefix) / "lib" / "proj",
        Path(shutil.which("proj") or "").parent.parent / "share" / "proj",
        Path("/usr/share/proj"),
        Path("/usr/local/share/proj"),
    ]
    for p in candidates:
        if p.exists() and (p / "proj.db").exists():
            return p

    return None


PROJ_DIR = _pick_proj_dir()
if PROJ_DIR is None:
    raise RuntimeError("No valid PROJ directory found.")

datadir.set_data_dir(str(PROJ_DIR))
os.environ["PROJ_DATA"] = str(PROJ_DIR)
os.environ["PROJ_LIB"] = str(PROJ_DIR)

print(f"[PROJ] Using data directory: {PROJ_DIR}")

import argparse
import math
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import cartopy.crs as ccrs
import cartopy.feature as cfeature

try:
    import dask
    dask.config.set({"array.slicing.split_large_chunks": True})
except Exception:
    pass


# ── Years to produce individual annual plots for ──────────────────────────────
YEARLY_PLOT_YEARS = [2015, 2018, 2020, 2023]
# ─────────────────────────────────────────────────────────────────────────────


DOMAINS = {
    'bof': {'name': 'Bay of Fundy',       'lat': (44.0, 47.0), 'lon': (-67.0, -63.0)},
    'lab': {'name': 'Labrador Sea',        'lat': (53.0, 65.0), 'lon': (-62.0, -42.0)},
    'gom': {'name': 'Gulf of Mexico',      'lat': (18.0, 31.0), 'lon': (-98.0, -80.0)},
    'med': {'name': 'Mediterranean Sea',   'lat': (30.0, 47.0), 'lon': ( -6.0,  37.0)},
    'red': {'name': 'Red Sea',             'lat': (12.0, 30.0), 'lon': ( 32.0,  44.0)},
    'bls': {'name': 'Black Sea',           'lat': (40.0, 47.0), 'lon': ( 27.0,  42.0)},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True, choices=list(DOMAINS.keys()))
    p.add_argument("--root", default="data/domains")
    p.add_argument("--out", default=None)
    p.add_argument("--start-year", type=int, default=None)
    p.add_argument("--end-year",   type=int, default=None)
    p.add_argument("--dpi",        type=int, default=180)
    p.add_argument("--era5-chunks", type=int, default=64)
    p.add_argument("--mur-chunks",  type=int, default=64)
    p.add_argument("--no-ice", action="store_true")
    p.add_argument(
        "--plot-years", type=int, nargs="+", default=YEARLY_PLOT_YEARS,
        help="Years to produce individual annual plots for "
             f"(default: {YEARLY_PLOT_YEARS})"
    )
    return p.parse_args()


# ── Utilities ─────────────────────────────────────────────────────────────────

def ensure_dirs(base):
    base = Path(base)
    maps_dir   = base / "maps"
    charts_dir = base / "charts"
    tables_dir = base / "tables"
    for d in [base, maps_dir, charts_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return maps_dir, charts_dir, tables_dir


def standardize_latlon(ds):
    rename = {}
    if "latitude"  in ds.coords and "lat" not in ds.coords: rename["latitude"]  = "lat"
    if "longitude" in ds.coords and "lon" not in ds.coords: rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)

    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    if "valid_time" in ds.dims   and "time" not in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    if "lon" in ds.coords:
        lon = ds["lon"]
        if float(lon.max()) > 180:
            ds = ds.assign_coords(lon=(((lon + 180) % 360) - 180)).sortby("lon")

    if "lat" in ds.coords and ds["lat"][0] > ds["lat"][-1]:
        ds = ds.sortby("lat")
    if "lon" in ds.coords and ds["lon"][0] > ds["lon"][-1]:
        ds = ds.sortby("lon")

    return ds


def subset_domain(ds, lat_bounds, lon_bounds):
    ds = standardize_latlon(ds)
    return ds.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))


def area_weights_lat(da):
    return np.cos(np.deg2rad(da["lat"]))


def weighted_mean_3d(da3d):
    return da3d.weighted(area_weights_lat(da3d)).mean(("lat", "lon"))


# ── Data loaders ──────────────────────────────────────────────────────────────

def find_era5_files(domain_dir, domain, start_year=None, end_year=None):
    era5_dir = domain_dir / "era5"
    out = []
    for fp in sorted(era5_dir.glob(f"era5_{domain}_*.nc")):
        try:
            year = int(fp.stem.split("_")[-1])
        except Exception:
            continue
        if start_year is not None and year < start_year: continue
        if end_year   is not None and year > end_year:   continue
        out.append(fp)
    return out


def open_era5(domain_dir, domain, lat_bounds, lon_bounds,
              chunks=64, start_year=None, end_year=None):
    files = find_era5_files(domain_dir, domain, start_year, end_year)
    if not files:
        raise FileNotFoundError(
            f"No ERA5 files found for domain={domain} in {domain_dir/'era5'}")
    ds = xr.open_mfdataset(
        [str(f) for f in files],
        combine="by_coords", engine="h5netcdf",
        chunks={"time": chunks}, parallel=False,
    )
    ds = standardize_latlon(ds)
    ds = subset_domain(ds, lat_bounds, lon_bounds)
    return ds.sortby("time")


def open_mur(domain_dir, domain, lat_bounds, lon_bounds, chunks=64):
    zarr_path = domain_dir / "mur" / f"mur_{domain}.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"MUR Zarr not found: {zarr_path}")
    ds = xr.open_zarr(str(zarr_path), consolidated=True)
    ds = standardize_latlon(ds)
    ds = subset_domain(ds, lat_bounds, lon_bounds)
    ds = ds.chunk({"time": chunks})
    return ds.sortby("time")


def open_gebco(domain_dir, domain, lat_bounds, lon_bounds):
    fp = domain_dir / f"gebco_{domain}.nc"
    if not fp.exists():
        raise FileNotFoundError(f"GEBCO file not found: {fp}")
    ds = xr.open_dataset(fp)
    ds = standardize_latlon(ds)
    ds = subset_domain(ds, lat_bounds, lon_bounds)
    if "elevation" not in ds.data_vars:
        candidates = [v for v in ds.data_vars if v.lower() in ("z", "band1", "topo")]
        if candidates:
            ds = ds.rename({candidates[0]: "elevation"})
        else:
            raise KeyError(
                f"Could not find bathymetry variable in {fp}. "
                f"Variables={list(ds.data_vars)}")
    return ds


# ── Variable name helpers ─────────────────────────────────────────────────────

def infer_mur_sst_name(ds):
    for v in ["analysed_sst", "sst", "sea_surface_temperature"]:
        if v in ds.data_vars: return v
    raise KeyError(f"No MUR SST variable. Variables={list(ds.data_vars)}")

def infer_mur_ice_name(ds):
    for v in ["sea_ice_fraction", "ice_fraction", "sea_ice_cover"]:
        if v in ds.data_vars: return v
    return None

def infer_era5_sst_name(ds):
    for v in ["sst_era5", "sst", "sea_surface_temperature"]:
        if v in ds.data_vars: return v
    raise KeyError(f"No ERA5 SST variable. Variables={list(ds.data_vars)}")

def infer_era5_ice_name(ds):
    for v in ["siconc", "sea_ice_cover", "ice_fraction"]:
        if v in ds.data_vars: return v
    return None

def maybe_kelvin_to_celsius(da):
    if float(da.quantile(0.95).compute()) > 200:
        return da - 273.15
    return da


# ── Harmonised dataset ────────────────────────────────────────────────────────

def prep_common_products(mur_ds, era5_ds, gebco_ds, no_ice=False):
    mur_sst_name  = infer_mur_sst_name(era5_ds if False else mur_ds)  # keep original logic
    era5_sst_name = infer_era5_sst_name(era5_ds)
    mur_ice_name  = None if no_ice else infer_mur_ice_name(mur_ds)
    era5_ice_name = None if no_ice else infer_era5_ice_name(era5_ds)

    mur_sst  = maybe_kelvin_to_celsius(mur_ds[mur_sst_name]).astype("float32")
    era5_sst = maybe_kelvin_to_celsius(era5_ds[era5_sst_name]).astype("float32")

    mur_ice  = mur_ds[mur_ice_name].astype("float32")  if mur_ice_name  else None
    era5_ice = era5_ds[era5_ice_name].astype("float32") if era5_ice_name else None

    bathy = gebco_ds["elevation"].astype("float32")

    mur_daily  = mur_sst.resample(time="1D").mean()
    era5_daily = era5_sst.resample(time="1D").mean()

    common_start = max(pd.Timestamp(mur_daily.time.min().values),
                       pd.Timestamp(era5_daily.time.min().values))
    common_end   = min(pd.Timestamp(mur_daily.time.max().values),
                       pd.Timestamp(era5_daily.time.max().values))

    mur_daily  = mur_daily.sel( time=slice(common_start, common_end))
    era5_daily = era5_daily.sel(time=slice(common_start, common_end))

    era5_on_mur = era5_daily.interp(
        lat=mur_daily["lat"], lon=mur_daily["lon"], method="nearest")

    mur_ice_daily     = None
    era5_ice_on_mur   = None

    if mur_ice is not None:
        mur_ice_daily = mur_ice.resample(time="1D").mean().sel(
            time=slice(common_start, common_end))

    if era5_ice is not None:
        era5_ice_daily  = era5_ice.resample(time="1D").mean().sel(
            time=slice(common_start, common_end))
        era5_ice_on_mur = era5_ice_daily.interp(
            lat=mur_daily["lat"], lon=mur_daily["lon"], method="linear")

    bathy_on_mur = bathy.interp(
        lat=mur_daily["lat"], lon=mur_daily["lon"], method="linear")

    ds = xr.Dataset({
        "mur_sst":  mur_daily,
        "era5_sst": era5_on_mur,
        "bathy":    bathy_on_mur,
    })
    if mur_ice_daily   is not None: ds["mur_ice"]  = mur_ice_daily
    if era5_ice_on_mur is not None: ds["era5_ice"] = era5_ice_on_mur

    ds["bias"]      = ds["era5_sst"] - ds["mur_sst"]
    ds["cold_bias"] = ds["mur_sst"]  - ds["era5_sst"]
    ds["abs_error"] = np.abs(ds["bias"])
    ds["sq_error"]  = ds["bias"] ** 2

    return ds


def add_gradient_magnitude(ds):
    mur     = ds["mur_sst"]
    lat_rad = np.deg2rad(mur["lat"])
    dy = mur["lat"].differentiate("lat") * 111.32
    dx = xr.DataArray(
        111.32 * np.cos(lat_rad), coords={"lat": mur["lat"]}, dims=("lat",)
    ) * mur["lon"].differentiate("lon")
    dTdy = mur.differentiate("lat") / dy
    dTdx = mur.differentiate("lon") / dx
    ds["mur_grad_mag_km"] = np.sqrt(dTdx**2 + dTdy**2).astype("float32")
    return ds


# ── Subset helpers ─────────────────────────────────────────────────────────────

def ds_for_year(ds, year):
    """Return a view of ds restricted to a single calendar year."""
    return ds.sel(time=ds.time.dt.year == year)


def available_plot_years(ds, requested_years):
    """Filter requested_years to only those with data in ds."""
    data_years = set(int(y) for y in ds.time.dt.year.values)
    found = [y for y in requested_years if y in data_years]
    missing = [y for y in requested_years if y not in data_years]
    if missing:
        print(f"  [warn] No data for years {missing}; skipping those panels.")
    return found


# ── Metric tables (unchanged logic, only operate on what is passed) ───────────

def compute_overall_metrics(ds):
    bias_ts  = weighted_mean_3d(ds["bias"])
    mur_ts   = weighted_mean_3d(ds["mur_sst"])
    era5_ts  = weighted_mean_3d(ds["era5_sst"])
    abs_ts   = weighted_mean_3d(ds["abs_error"])
    mse_ts   = weighted_mean_3d(ds["sq_error"])

    metrics = {
        "time_start":                str(pd.Timestamp(ds.time.min().values).date()),
        "time_end":                  str(pd.Timestamp(ds.time.max().values).date()),
        "n_days":                    int(ds.sizes["time"]),
        "mur_mean_sst_c":            float(mur_ts.mean().compute()),
        "era5_mean_sst_c":           float(era5_ts.mean().compute()),
        "mean_bias_c_era5_minus_mur":float(bias_ts.mean().compute()),
        "mean_cold_bias_c_mur_minus_era5": float((-bias_ts).mean().compute()),
        "mae_c":                     float(abs_ts.mean().compute()),
        "rmse_c":                    float(np.sqrt(mse_ts.mean().compute())),
        "bias_std_c":                float(bias_ts.std().compute()),
        "mur_spatial_std_mean":      float(ds["mur_sst"].std(("lat","lon")).mean().compute()),
        "era5_spatial_std_mean":     float(ds["era5_sst"].std(("lat","lon")).mean().compute()),
    }
    metrics["timeseries_corr"] = float(
        xr.corr(mur_ts, era5_ts, dim="time").compute())
    return pd.DataFrame([metrics])


def compute_monthly_metrics(ds):
    mur_ts  = weighted_mean_3d(ds["mur_sst"])
    era5_ts = weighted_mean_3d(ds["era5_sst"])
    bias_ts = weighted_mean_3d(ds["bias"])
    abs_ts  = weighted_mean_3d(ds["abs_error"])
    sq_ts   = weighted_mean_3d(ds["sq_error"])

    rows = []
    for month in range(1, 13):
        mask = ds["time.month"] == month
        if not mask.any(): continue
        rows.append({
            "month":              month,
            "mur_mean_sst_c":     float(mur_ts.where( mask, drop=True).mean().compute()),
            "era5_mean_sst_c":    float(era5_ts.where(mask, drop=True).mean().compute()),
            "bias_c_era5_minus_mur": float(bias_ts.where(mask, drop=True).mean().compute()),
            "mae_c":              float(abs_ts.where( mask, drop=True).mean().compute()),
            "rmse_c":             float(np.sqrt(sq_ts.where(mask, drop=True).mean().compute())),
            "n_days":             int(mask.sum().compute()),
        })
    return pd.DataFrame(rows)


def compute_seasonal_metrics(ds):
    season_map = {12:"DJF",1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",
                  6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON"}
    bias_ts = weighted_mean_3d(ds["bias"])
    abs_ts  = weighted_mean_3d(ds["abs_error"])
    sq_ts   = weighted_mean_3d(ds["sq_error"])
    mur_ts  = weighted_mean_3d(ds["mur_sst"])
    era5_ts = weighted_mean_3d(ds["era5_sst"])

    df = pd.DataFrame({
        "time":    pd.to_datetime(bias_ts["time"].values),
        "mur_sst": mur_ts.values,  "era5_sst": era5_ts.values,
        "bias":    bias_ts.values, "mae": abs_ts.values,
        "rmse":    np.sqrt(sq_ts.values),
    })
    df["season"] = df["time"].dt.month.map(season_map)
    out = df.groupby("season", sort=False).agg(
        mur_mean_sst_c=("mur_sst","mean"), era5_mean_sst_c=("era5_sst","mean"),
        bias_c_era5_minus_mur=("bias","mean"), mae_c=("mae","mean"),
        rmse_c=("rmse","mean"), n_days=("bias","size"),
    ).reset_index()
    season_order = ["DJF","MAM","JJA","SON"]
    out["season"] = pd.Categorical(out["season"], categories=season_order, ordered=True)
    return out.sort_values("season").reset_index(drop=True)


def compute_depth_binned_metrics(ds):
    mean_bias = ds["bias"].mean("time")
    bathy     = ds["bathy"]
    bins   = [-10000,-4000,-2000,-1000,-200,0,10000]
    labels = ["< -4000","-4000:-2000","-2000:-1000","-1000:-200","-200:0","land_or_above0"]
    rows = []
    for lo, hi, label in zip(bins[:-1], bins[1:], labels):
        mask  = (bathy >= lo) & (bathy < hi)
        vals  = mean_bias.where(mask)
        valid = int(vals.count().compute())
        if valid == 0: continue
        rows.append({
            "depth_bin_m": label,
            "mean_depth_m": float(bathy.where(mask).mean().compute()),
            "mean_bias_c":  float(vals.mean().compute()),
            "mae_c":        float(np.abs(mean_bias.where(mask)).mean().compute()),
            "n_cells":      valid,
        })
    return pd.DataFrame(rows)


def save_csv(df, path):
    df.to_csv(path, index=False)


# ── Map helpers ───────────────────────────────────────────────────────────────

def add_map_features(ax, lon_bounds, lat_bounds, title):
    ax.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]],
                  crs=ccrs.PlateCarree())
    ax.coastlines(resolution="10m", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.4, alpha=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--", alpha=0.5)
    gl.top_labels   = False
    gl.right_labels = False
    ax.set_title(title, fontsize=10)


def _pcolormesh_on_ax(ax, da2d, cmap, vmin, vmax):
    return ax.pcolormesh(
        da2d["lon"], da2d["lat"], da2d,
        transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
    )


# ── NEW: 4-panel yearly map ───────────────────────────────────────────────────

def plot_yearly_maps(ds_full, var, years, out_path,
                     suptitle, cmap, lon_bounds, lat_bounds,
                     dpi=180, center_zero=False,
                     vmin_override=None, vmax_override=None):
    """
    4-panel (2×2) figure — one panel per year in `years`.
    Each panel shows the annual mean of `var`.
    """
    n     = len(years)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(11 * ncols, 7 * nrows),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
    )

    # Compute per-year means first (needed for shared colour scale)
    means = {}
    for yr in years:
        sub = ds_full[var].sel(time=ds_full.time.dt.year == yr)
        if sub.sizes["time"] == 0:
            means[yr] = None
        else:
            means[yr] = sub.mean("time").compute()

    # Shared colour limits
    all_vals = np.concatenate([
        m.values.ravel() for m in means.values() if m is not None
    ])
    all_vals = all_vals[np.isfinite(all_vals)]

    if center_zero:
        abs_max = float(np.nanpercentile(np.abs(all_vals), 98))
        vmin = vmin_override if vmin_override is not None else -abs_max
        vmax = vmax_override if vmax_override is not None else  abs_max
    else:
        vmin = vmin_override if vmin_override is not None else float(np.nanpercentile(all_vals, 2))
        vmax = vmax_override if vmax_override is not None else float(np.nanpercentile(all_vals, 98))

    mesh_ref = None
    for idx, yr in enumerate(years):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        if means[yr] is None:
            ax.set_visible(False)
            continue
        mesh = _pcolormesh_on_ax(ax, means[yr], cmap, vmin, vmax)
        add_map_features(ax, lon_bounds, lat_bounds, str(yr))
        mesh_ref = mesh

    # Hide unused panels
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    if mesh_ref is not None:
        cbar = fig.colorbar(mesh_ref, ax=axes, shrink=0.55, pad=0.03)
        cbar.ax.set_ylabel(var)

    fig.suptitle(suptitle, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── Original single-panel map (kept for bathy & ice) ─────────────────────────

def plot_map(da2d, out_path, title, cmap="viridis", vmin=None, vmax=None,
             lon_bounds=None, lat_bounds=None, dpi=180, center_zero=False):
    proj = ccrs.PlateCarree()
    fig  = plt.figure(figsize=(10, 7))
    ax   = plt.axes(projection=proj)

    if center_zero:
        vmax_auto = float(np.nanmax(np.abs(da2d.values)))
        vmin = vmin or -vmax_auto
        vmax = vmax or  vmax_auto

    mesh = ax.pcolormesh(
        da2d["lon"], da2d["lat"], da2d,
        transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
    )
    add_map_features(ax, lon_bounds, lat_bounds, title)
    cbar = plt.colorbar(mesh, ax=ax, shrink=0.86, pad=0.03)
    cbar.ax.set_ylabel(str(da2d.name) if da2d.name else "")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_bathy_map(bathy, out_path, lon_bounds, lat_bounds, dpi=180):
    fig = plt.figure(figsize=(10, 7))
    ax  = plt.axes(projection=ccrs.PlateCarree())

    vmin = float(np.nanpercentile(bathy.values, 2))
    vmax = float(np.nanpercentile(bathy.values, 98))

    mesh = ax.pcolormesh(
        bathy["lon"], bathy["lat"], bathy,
        transform=ccrs.PlateCarree(),
        cmap="Blues_r", vmin=vmin, vmax=vmax, shading="auto",
    )
    try:
        ax.contour(bathy["lon"], bathy["lat"], bathy,
                   levels=[-200,-1000,-2000],
                   colors="k", linewidths=0.6, alpha=0.7,
                   transform=ccrs.PlateCarree())
    except Exception:
        pass
    add_map_features(ax, lon_bounds, lat_bounds, "GEBCO Bathymetry / Elevation (m)")
    cbar = plt.colorbar(mesh, ax=ax, shrink=0.86, pad=0.03)
    cbar.ax.set_ylabel("meters")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── NEW: per-year chart helpers ───────────────────────────────────────────────

def _monthly_df_for_year(ds, year):
    """Compute monthly metrics for a single year."""
    sub = ds_for_year(ds, year)
    if sub.sizes["time"] == 0:
        return None
    return compute_monthly_metrics(sub)


def plot_yearly_monthly_climatology(ds, years, out_path, dpi=180):
    """2×2 grid: monthly SST climatology (MUR vs ERA5) for each year."""
    ncols = 2
    nrows = math.ceil(len(years) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows), sharey=False)
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, yr in enumerate(years):
        row, col = divmod(idx, ncols)
        ax  = axes[row][col]
        mdf = _monthly_df_for_year(ds, yr)
        if mdf is None or mdf.empty:
            ax.set_visible(False)
            continue
        ax.plot(mdf["month"], mdf["mur_mean_sst_c"],  marker="o", lw=2, label="MUR")
        ax.plot(mdf["month"], mdf["era5_mean_sst_c"], marker="s", lw=2, label="ERA5")
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("Month"); ax.set_ylabel("SST (°C)")
        ax.set_title(str(yr), fontsize=11)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    for idx in range(len(years), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Monthly SST Climatology — per year", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_yearly_monthly_bars(ds, years, ycol, out_path, suptitle, ylabel,
                              color="steelblue", dpi=180):
    """2×2 grid: monthly bar chart (`ycol`) for each year."""
    ncols = 2
    nrows = math.ceil(len(years) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows), sharey=False)
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, yr in enumerate(years):
        row, col = divmod(idx, ncols)
        ax  = axes[row][col]
        mdf = _monthly_df_for_year(ds, yr)
        if mdf is None or mdf.empty:
            ax.set_visible(False)
            continue
        ax.bar(mdf["month"].astype(str), mdf[ycol],
               color=color, edgecolor="black", alpha=0.85)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("Month"); ax.set_ylabel(ylabel)
        ax.set_title(str(yr), fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)

    for idx in range(len(years), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_yearly_seasonal_bars(ds, years, out_path, dpi=180):
    """2×2 grid: seasonal mean bias bar chart for each year."""
    ncols = 2
    nrows = math.ceil(len(years) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows), sharey=False)
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, yr in enumerate(years):
        row, col = divmod(idx, ncols)
        ax  = axes[row][col]
        sub = ds_for_year(ds, yr)
        if sub.sizes["time"] == 0:
            ax.set_visible(False)
            continue
        sdf = compute_seasonal_metrics(sub)
        if sdf.empty:
            ax.set_visible(False)
            continue
        ax.bar(sdf["season"].astype(str), sdf["bias_c_era5_minus_mur"],
               color="seagreen", edgecolor="black", alpha=0.85)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("Season"); ax.set_ylabel("Bias (°C)")
        ax.set_title(str(yr), fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)

    for idx in range(len(years), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Seasonal Mean Bias (ERA5 − MUR) — per year", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_yearly_bias_histogram(ds, years, out_path, dpi=180):
    """2×2 grid: bias histogram for each year."""
    ncols = 2
    nrows = math.ceil(len(years) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows), sharey=False)
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, yr in enumerate(years):
        row, col = divmod(idx, ncols)
        ax  = axes[row][col]
        sub = ds_for_year(ds, yr)
        if sub.sizes["time"] == 0:
            ax.set_visible(False)
            continue
        vals = sub["bias"].stack(z=("time","lat","lon")).values
        vals = vals[np.isfinite(vals)]
        if vals.size > 2_000_000:
            vals = np.random.default_rng(42).choice(vals, size=2_000_000, replace=False)
        ax.hist(vals, bins=80, color="indianred", edgecolor="black", alpha=0.8)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_xlabel("Bias (°C)"); ax.set_ylabel("Count")
        ax.set_title(str(yr), fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)

    for idx in range(len(years), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Bias Distribution (ERA5 − MUR) — per year", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_depth_binned(depth_df, out_path, dpi=180):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(depth_df["depth_bin_m"], depth_df["mean_bias_c"],
           color="slateblue", edgecolor="black", alpha=0.85)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_title("Mean Bias by Bathymetry Bin")
    ax.set_xlabel("Depth Bin (m)"); ax.set_ylabel("Mean Bias (°C)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args        = parse_args()
    domain      = args.domain
    domain_meta = DOMAINS[domain]
    lat_bounds  = domain_meta["lat"]
    lon_bounds  = domain_meta["lon"]

    root       = Path(args.root)
    domain_dir = root / domain
    if not domain_dir.exists():
        raise FileNotFoundError(f"Domain folder not found: {domain_dir}")

    out_dir = Path(args.out) if args.out else domain_dir / "analysis"
    maps_dir, charts_dir, tables_dir = ensure_dirs(out_dir)

    print(f"Opening data for domain={domain} ({domain_meta['name']})")
    print(f"Input folder : {domain_dir}")
    print(f"Output folder: {out_dir}")

    era5_ds  = open_era5(domain_dir, domain, lat_bounds, lon_bounds,
                         chunks=args.era5_chunks,
                         start_year=args.start_year, end_year=args.end_year)
    mur_ds   = open_mur(domain_dir, domain, lat_bounds, lon_bounds,
                        chunks=args.mur_chunks)
    gebco_ds = open_gebco(domain_dir, domain, lat_bounds, lon_bounds)

    print("Preparing harmonised products...")
    ds = prep_common_products(mur_ds, era5_ds, gebco_ds, no_ice=args.no_ice)
    ds = add_gradient_magnitude(ds)

    # Determine which requested years actually have data
    plot_years = available_plot_years(ds, args.plot_years)
    if not plot_years:
        raise ValueError(
            f"None of the requested plot years {args.plot_years} are present in the data.")
    print(f"Producing annual plots for years: {plot_years}")

    # ── Tables (whole dataset) ────────────────────────────────────────────────
    print("Computing tables...")
    save_csv(compute_overall_metrics(ds),       tables_dir / "overall_metrics.csv")
    save_csv(compute_monthly_metrics(ds),       tables_dir / "monthly_metrics.csv")
    save_csv(compute_seasonal_metrics(ds),      tables_dir / "seasonal_metrics.csv")
    save_csv(compute_depth_binned_metrics(ds),  tables_dir / "depth_binned_metrics.csv")

    # Per-year tables
    for yr in plot_years:
        sub = ds_for_year(ds, yr)
        if sub.sizes["time"] == 0:
            continue
        save_csv(compute_overall_metrics(sub),  tables_dir / f"overall_metrics_{yr}.csv")
        save_csv(compute_monthly_metrics(sub),  tables_dir / f"monthly_metrics_{yr}.csv")
        save_csv(compute_seasonal_metrics(sub), tables_dir / f"seasonal_metrics_{yr}.csv")

    # ── Maps — 4-panel (one panel per year) ───────────────────────────────────
    print("Preparing map products (per-year panels)...")

    # Shared SST colour limits across both products and all years
    all_sst = []
    for yr in plot_years:
        sub = ds.sel(time=ds.time.dt.year == yr)
        if sub.sizes["time"] == 0: continue
        all_sst.append(sub["mur_sst"].mean("time").compute().values.ravel())
        all_sst.append(sub["era5_sst"].mean("time").compute().values.ravel())
    all_sst_flat = np.concatenate(all_sst)
    sst_vmin = float(np.nanpercentile(all_sst_flat, 2))
    sst_vmax = float(np.nanpercentile(all_sst_flat, 98))

    plot_yearly_maps(
        ds, "mur_sst", plot_years,
        maps_dir / "yearly_mean_mur_sst.png",
        f"{domain.upper()} Annual Mean MUR SST",
        cmap="turbo", lon_bounds=lon_bounds, lat_bounds=lat_bounds, dpi=args.dpi,
        vmin_override=sst_vmin, vmax_override=sst_vmax,
    )

    plot_yearly_maps(
        ds, "era5_sst", plot_years,
        maps_dir / "yearly_mean_era5_sst.png",
        f"{domain.upper()} Annual Mean ERA5 SST (on MUR grid)",
        cmap="turbo", lon_bounds=lon_bounds, lat_bounds=lat_bounds, dpi=args.dpi,
        vmin_override=sst_vmin, vmax_override=sst_vmax,
    )

    plot_yearly_maps(
        ds, "bias", plot_years,
        maps_dir / "yearly_mean_bias.png",
        f"{domain.upper()} Annual Mean Bias (ERA5 − MUR)",
        cmap="RdBu_r", lon_bounds=lon_bounds, lat_bounds=lat_bounds, dpi=args.dpi,
        center_zero=True,
    )

    plot_yearly_maps(
        ds, "cold_bias", plot_years,
        maps_dir / "yearly_mean_cold_bias.png",
        f"{domain.upper()} Annual Mean Cold Bias (MUR − ERA5)",
        cmap="RdBu_r", lon_bounds=lon_bounds, lat_bounds=lat_bounds, dpi=args.dpi,
        center_zero=True,
    )

    # RMSE — compute sq_error mean per year, then sqrt
    # Build a temporary dataset with per-year RMSE maps stored as separate DataArrays
    rmse_das = {}
    for yr in plot_years:
        sub = ds.sel(time=ds.time.dt.year == yr)
        if sub.sizes["time"] == 0:
            rmse_das[yr] = None
        else:
            rmse_das[yr] = np.sqrt(sub["sq_error"].mean("time")).compute()

    # Shared RMSE colour scale
    all_rmse = np.concatenate([v.values.ravel() for v in rmse_das.values() if v is not None])
    all_rmse = all_rmse[np.isfinite(all_rmse)]
    rmse_vmin = float(np.nanpercentile(all_rmse, 2))
    rmse_vmax = float(np.nanpercentile(all_rmse, 98))

    ncols = 2; nrows = math.ceil(len(plot_years) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11 * ncols, 7 * nrows),
                             subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
    mesh_ref = None
    for idx, yr in enumerate(plot_years):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        if rmse_das.get(yr) is None:
            ax.set_visible(False); continue
        mesh = _pcolormesh_on_ax(ax, rmse_das[yr], "magma", rmse_vmin, rmse_vmax)
        add_map_features(ax, lon_bounds, lat_bounds, str(yr))
        mesh_ref = mesh
    for idx in range(len(plot_years), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)
    if mesh_ref is not None:
        cbar = fig.colorbar(mesh_ref, ax=axes, shrink=0.55, pad=0.03)
        cbar.ax.set_ylabel("RMSE (°C)")
    fig.suptitle(f"{domain.upper()} Annual RMSE Map", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(maps_dir / "yearly_rmse_map.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    plot_yearly_maps(
        ds, "mur_grad_mag_km", plot_years,
        maps_dir / "yearly_mur_gradient_magnitude.png",
        f"{domain.upper()} Annual Mean MUR SST Gradient Magnitude",
        cmap="viridis", lon_bounds=lon_bounds, lat_bounds=lat_bounds, dpi=args.dpi,
    )

    # Bathymetry — single panel (time-invariant)
    bathy = ds["bathy"].compute()
    plot_bathy_map(bathy, maps_dir / "gebco_bathymetry.png",
                   lon_bounds=lon_bounds, lat_bounds=lat_bounds, dpi=args.dpi)

    # Ice — per-year panels if available
    if not args.no_ice and "mur_ice" in ds:
        plot_yearly_maps(
            ds, "mur_ice", plot_years,
            maps_dir / "yearly_mean_mur_ice_fraction.png",
            f"{domain.upper()} Annual Mean MUR Ice Fraction",
            cmap="Blues", lon_bounds=lon_bounds, lat_bounds=lat_bounds, dpi=args.dpi,
            vmin_override=0, vmax_override=1,
        )

    if not args.no_ice and "era5_ice" in ds:
        plot_yearly_maps(
            ds, "era5_ice", plot_years,
            maps_dir / "yearly_mean_era5_ice_fraction.png",
            f"{domain.upper()} Annual Mean ERA5 Ice Fraction",
            cmap="Blues", lon_bounds=lon_bounds, lat_bounds=lat_bounds, dpi=args.dpi,
            vmin_override=0, vmax_override=1,
        )

    # ── Charts — 2×2 grids (one subplot per year) ─────────────────────────────
    print("Preparing charts (per-year panels)...")

    plot_yearly_monthly_climatology(
        ds, plot_years,
        charts_dir / "yearly_monthly_sst_climatology.png",
        dpi=args.dpi,
    )

    plot_yearly_monthly_bars(
        ds, plot_years, "bias_c_era5_minus_mur",
        charts_dir / "yearly_monthly_bias_bars.png",
        "Monthly Mean Bias (ERA5 − MUR) — per year", "Bias (°C)",
        color="tomato", dpi=args.dpi,
    )

    plot_yearly_monthly_bars(
        ds, plot_years, "rmse_c",
        charts_dir / "yearly_monthly_rmse_bars.png",
        "Monthly RMSE — per year", "RMSE (°C)",
        color="darkorange", dpi=args.dpi,
    )

    plot_yearly_seasonal_bars(
        ds, plot_years,
        charts_dir / "yearly_seasonal_bias_bars.png",
        dpi=args.dpi,
    )

    plot_yearly_bias_histogram(
        ds, plot_years,
        charts_dir / "yearly_bias_histogram.png",
        dpi=args.dpi,
    )

    depth_df = compute_depth_binned_metrics(ds)
    if len(depth_df) > 0:
        save_csv(depth_df, tables_dir / "depth_binned_metrics.csv")
        plot_depth_binned(depth_df, charts_dir / "depth_binned_bias.png", dpi=args.dpi)

    print("\nDone.")
    print(f"Saved maps   -> {maps_dir}")
    print(f"Saved charts -> {charts_dir}")
    print(f"Saved tables -> {tables_dir}")


if __name__ == "__main__":
    main()