# scripts/02_download_mur.py
"""
Downloads NASA MUR SST granule by granule (one .nc per day),
crops each one immediately, deletes the raw file, and appends
to a per-year output. No dask, no open_mfdataset, no hangs.

Strategy:
  1. Search granules for a date range
  2. Download one granule at a time (~200 MB each, global)
  3. Crop to GSL bbox immediately in memory
  4. Append cropped slice to list
  5. Concatenate year and write to netCDF
  6. Delete raw granule to save disk

Expected per-year:  ~1.8 GB output, ~45 min download time
Total (2013-2023):  ~20 GB output
"""

import earthaccess
import xarray as xr
import numpy as np
import gc
import shutil
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
LAT      = (47.0, 52.0)
LON      = (-68.0, -56.0)
YEARS    = range(2013, 2024)
OUT_DIR  = Path('data/raw/mur_sst')
TEMP_DIR = Path('data/raw/mur_tmp')   # raw global granules land here temporarily

OUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ── Login ─────────────────────────────────────────────────────────────────────
earthaccess.login(strategy='interactive')

# ── Download ──────────────────────────────────────────────────────────────────
for year in YEARS:
    out_path = OUT_DIR / f'mur_{year}.nc'
    if out_path.exists():
        print(f'[{year}] Already exists, skipping.')
        continue

    print(f'\n[{year}] Starting...')

    # Search — returns one granule per day for the year
    results = earthaccess.search_data(
        short_name   = 'MUR-JPL-L4-GLOB-v4.1',
        temporal     = (f'{year}-01-01', f'{year}-12-31'),
        bounding_box = (LON[0], LAT[0], LON[1], LAT[1]),
    )
    print(f'[{year}] Found {len(results)} granules')

    daily_slices = []

    for i, granule in enumerate(results):

        # ── Download single granule to temp dir ───────────────────────────
        try:
            downloaded = earthaccess.download(granule, str(TEMP_DIR))
        except Exception as e:
            print(f'  [{i+1}/{len(results)}] Download failed: {e} — skipping')
            continue

        if not downloaded:
            print(f'  [{i+1}/{len(results)}] No file returned — skipping')
            continue

        raw_file = Path(downloaded[0])

        # ── Crop immediately, never load full global file into memory ─────
        try:
            with xr.open_dataset(
                raw_file,
                engine           = 'h5netcdf',
                mask_and_scale   = True,
                decode_timedelta = False,    # silences FutureWarning
            ) as ds:
                crop = ds[['analysed_sst', 'analysis_error', 'sea_ice_fraction']].sel(
                    lat = slice(LAT[0], LAT[1]),
                    lon = slice(LON[0], LON[1]),
                ).load()   # pulls only the cropped region into RAM (~3 MB)

                # Convert SST Kelvin → Celsius
                crop['analysed_sst'] = (crop['analysed_sst'] - 273.15).astype(np.float32)
                crop['analysed_sst'].attrs['units'] = 'degC'
                crop['analysis_error']   = crop['analysis_error'].astype(np.float32)
                crop['sea_ice_fraction'] = crop['sea_ice_fraction'].astype(np.float32)

                daily_slices.append(crop.copy(deep=True))

        except Exception as e:
            print(f'  [{i+1}/{len(results)}] Crop failed: {e} — skipping')

        finally:
            # Always delete the raw global granule immediately (~200 MB each)
            raw_file.unlink(missing_ok=True)

        if (i + 1) % 30 == 0:
            pct = (i + 1) / len(results) * 100
            print(f'  [{year}] {i+1}/{len(results)} done ({pct:.0f}%)')

        gc.collect()

    if not daily_slices:
        print(f'[{year}] No data collected — skipping year.')
        continue

    # ── Concatenate year and write ─────────────────────────────────────────
    print(f'[{year}] Concatenating {len(daily_slices)} days...')
    year_ds = xr.concat(daily_slices, dim='time')

    # netcdf4 engine required for zlib compression (scipy backend doesn't support it)
    encoding = {
        v: {
            'zlib':      True,
            'complevel': 4,
            'dtype':     'float32',
        }
        for v in year_ds.data_vars
    }

    print(f'[{year}] Writing {out_path}...')
    year_ds.to_netcdf(
        out_path,
        encoding = encoding,
        engine   = 'netcdf4',   # explicit — avoids scipy fallback
    )

    sz = out_path.stat().st_size / 1e9
    print(f'[{year}] Done. Size: {sz:.2f} GB  |  Grid: {dict(year_ds.dims)}')

    # Free memory before next year
    del daily_slices, year_ds
    gc.collect()

# ── Cleanup ───────────────────────────────────────────────────────────────────
shutil.rmtree(TEMP_DIR, ignore_errors=True)
print('\nAll years complete.')