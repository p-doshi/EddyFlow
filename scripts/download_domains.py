"""
download_domains.py
Downloads MUR SST, ERA5, and GEBCO bathymetry for all study domains.

MUR is processed one granule at a time:
  raw download -> preprocess -> append compressed Zarr -> delete raw file

Usage:
  python download_domains.py --domain bof --years 2013 2013
  python download_domains.py --domain bof --years 2013 2013 --restart   # wipe and re-download
"""

import gc
import argparse
import shutil
from pathlib import Path
import time
import cdsapi
import earthaccess
import numpy as np
import logging

DOMAINS = {
    'bof': {
        'name': 'Bay of Fundy',
        'lat': (44.0, 47.0),
        'lon': (-67.0, -63.0),
        'note': 'Extreme tidal forcing',
    },
    'lab': {
        'name': 'Labrador Sea',
        'lat': (53.0, 65.0),
        'lon': (-62.0, -42.0),
        'note': 'Deep convection, ice edge',
    },
    'gom': {
        'name': 'Gulf of Mexico',
        'lat': (18.0, 31.0),
        'lon': (-98.0, -80.0),
        'note': 'Loop Current eddies',
    },
    'med': {
        'name': 'Mediterranean Sea',
        'lat': (30.0, 47.0),
        'lon': (-6.0, 37.0),
        'note': 'Multi sub-basin dynamics',
    },
    'red': {
        'name': 'Red Sea',
        'lat': (12.0, 30.0),
        'lon': (32.0, 44.0),
        'note': 'Dust-driven SST forcing',
    },
    'bls': {
        'name': 'Black Sea',
        'lat': (40.0, 47.0),
        'lon': (27.0, 42.0),
        'note': 'Enclosed basin, halocline',
    },
}

ERA5_VARS = ['t2m', 'msl', 'u10', 'v10', 'sst', 'ssr', 'str', 'tp']

MUR_VARS = ['analysed_sst', 'analysis_error', 'sea_ice_fraction']

def _setup_logging(out_root: Path) -> logging.Logger:
    """
    Set up a logger that writes to both stdout and a persistent log file.
    The log file accumulates across runs so you can always trace what has been
    done without starting over.
    """
    log_dir = Path(out_root)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'download_progress.log'

    logger = logging.getLogger('download_domains')
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter('%(asctime)s  %(levelname)-8s  %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    # File handler — append mode so every resume adds to the same log
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# Module-level placeholder — replaced by _setup_logging() in main()
log = logging.getLogger('download_domains')
def _inventory_existing_files(out_root: Path, domains: dict, years: tuple) -> None:
    """
    Walk the output tree and log what already exists before any download starts.
    This gives a clear picture of where we are resuming from without modifying
    any files.

      For MUR  : reads the *_times.npy sidecar and reports how many granules
                 are stored and which date range they cover.
      For ERA5 : lists which yearly NetCDF files are present.
      For GEBCO: reports whether the domain bathymetry file exists.
    """
    log.info('=' * 70)
    log.info('INVENTORY — scanning existing files before download')
    log.info(f'  Output root : {out_root}')
    log.info(f'  Domains     : {list(domains.keys())}')
    log.info(f'  Year range  : {years[0]}–{years[1]}')
    log.info('=' * 70)

    year_range = list(range(years[0], years[1] + 1))

    for key, dom in domains.items():
        log.info(f'  ── {key.upper()} ({dom["name"]}) ──')
        domain_dir = Path(out_root) / key

        # ── MUR ──────────────────────────────────────────────────────────────
        times_path = domain_dir / 'mur' / f'mur_{key}_times.npy'
        zarr_path  = domain_dir / 'mur' / f'mur_{key}.zarr'

        if times_path.exists():
            try:
                arr = np.load(times_path, allow_pickle=False)
                n   = len(arr)
                t0  = np.datetime_as_string(arr.min(), unit='D') if n > 0 else 'n/a'
                t1  = np.datetime_as_string(arr.max(), unit='D') if n > 0 else 'n/a'
                log.info(f'    MUR  zarr   : {zarr_path}  [exists={zarr_path.exists()}]')
                log.info(f'    MUR  times  : {n} granule(s) stored  ({t0} → {t1})')
            except Exception as e:
                log.warning(f'    MUR  times  : could not read {times_path.name}: {e}')
        else:
            log.info(f'    MUR  times  : not started (no {times_path.name})')

        # ── ERA5 ─────────────────────────────────────────────────────────────
        era5_dir  = domain_dir / 'era5'
        era5_done = []
        era5_miss = []
        for yr in year_range:
            fp = era5_dir / f'era5_{key}_{yr}.nc'
            (era5_done if fp.exists() else era5_miss).append(yr)

        if era5_done:
            log.info(f'    ERA5 done   : {era5_done}')
        if era5_miss:
            log.info(f'    ERA5 missing: {era5_miss}')
        if not era5_done and not era5_miss:
            log.info(f'    ERA5        : directory not found')

        # ── GEBCO ────────────────────────────────────────────────────────────
        gebco_path = domain_dir / f'gebco_{key}.nc'
        if gebco_path.exists():
            size_mb = gebco_path.stat().st_size / 1_048_576
            log.info(f'    GEBCO       : {gebco_path.name}  ({size_mb:.1f} MB)  ✓')
        else:
            log.info(f'    GEBCO       : not downloaded yet')

    log.info('=' * 70)
    log.info('INVENTORY complete — starting downloads')
    log.info('=' * 70)

def _download_with_retry(granule, local_path, max_retries=5, base_delay=30):
    """Exponential backoff retry for transient PO.DAAC 502 errors."""
    for attempt in range(max_retries):
        try:
            return earthaccess.download([granule], local_path=str(local_path))
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)   # 30, 60, 120, 240, 480s
            print(f'      Download failed (attempt {attempt+1}/{max_retries}): {e}')
            print(f'      Retrying in {delay}s …')
            time.sleep(delay)


def _cleanup_files(paths):
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass


def _remove_tree(path: Path):
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _mur_encoding(chunk_lat, chunk_lon):
    from numcodecs import Blosc

    compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
    return {
        'analysed_sst': {
            'compressor': compressor,
            'dtype': 'float32',
            'chunks': (1, chunk_lat, chunk_lon),
            '_FillValue': np.float32(np.nan),
        },
        'analysis_error': {
            'compressor': compressor,
            'dtype': 'float32',
            'chunks': (1, chunk_lat, chunk_lon),
            '_FillValue': np.float32(np.nan),
        },
        'sea_ice_fraction': {
            'compressor': compressor,
            'dtype': 'float32',
            'chunks': (1, chunk_lat, chunk_lon),
            '_FillValue': np.float32(0.0),
        },
    }


def _load_completed_granules(times_path: Path) -> set:
    """
    Return the set of already-written time values as int64 nanoseconds since epoch,
    so we can skip granules whose timestamp is already present in the Zarr store.
    Using int64 avoids type-mismatch bugs when comparing np.datetime64 vs Python objects.
    """
    if not times_path.exists():
        return set()
    arr = np.load(times_path, allow_pickle=False)
    return set(arr.astype('datetime64[ns]').astype('int64').tolist())


def _preprocess_mur_file(fpath, domain):
    import xarray as xr

    lat_min, lat_max = domain['lat']
    lon_min, lon_max = domain['lon']

    ds = xr.open_dataset(fpath, decode_timedelta=False)

    lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
    lon_name = 'lon' if 'lon' in ds.coords else 'longitude'

    lat_vals = ds[lat_name].values
    lon_vals = ds[lon_name].values

    lat_slice = slice(lat_min, lat_max) if lat_vals[0] < lat_vals[-1] else slice(lat_max, lat_min)
    lon_slice = slice(lon_min, lon_max) if lon_vals[0] < lon_vals[-1] else slice(lon_max, lon_min)

    ds = ds[MUR_VARS].sel({lat_name: lat_slice, lon_name: lon_slice})

    if lat_name != 'lat' or lon_name != 'lon':
        ds = ds.rename({lat_name: 'lat', lon_name: 'lon'})

    ds = ds.astype(np.float32)
    ds['sea_ice_fraction'] = ds['sea_ice_fraction'].fillna(0.0).astype(np.float32)
    return ds

def _search_with_retry(short_name, temporal, bounding_box, max_retries=5, base_delay=30):
    for attempt in range(max_retries):
        try:
            return earthaccess.search_data(
                short_name=short_name,
                temporal=temporal,
                bounding_box=bounding_box,
            )
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            log.warning(f'CMR search failed (attempt {attempt+1}/{max_retries}): {e}')
            log.warning(f'Retrying in {delay}s ...')
            time.sleep(delay)

def download_mur(domain_key, domain, years, out_root, restart=False):
    earthaccess.login(strategy='netrc')

    out_dir = Path(out_root) / domain_key / 'mur'
    tmp_dir = out_dir / '_tmp_download'
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    zarr_path = out_dir / f'mur_{domain_key}.zarr'
    times_path = out_dir / f'mur_{domain_key}_times.npy'

    # --restart: wipe existing outputs and start fresh
    if restart:
        if zarr_path.exists():
            print(f'  [restart] Removing existing MUR store: {zarr_path}')
            shutil.rmtree(zarr_path, ignore_errors=True)
        if times_path.exists():
            print(f'  [restart] Removing existing times file: {times_path}')
            times_path.unlink()
    else:
        # Resume: report what we already have
        completed = _load_completed_granules(times_path)
        if completed:
            print(f'  Resuming — {len(completed)} granule(s) already written, skipping those.')

    lat_min, lat_max = domain['lat']
    lon_min, lon_max = domain['lon']

    for year in range(years[0], years[1] + 1):
        print(f"  MUR {domain['name']} {year}...")

        # Reload completed set each year so granules written in a previous
        # year loop iteration are also recognised as done.
        completed = _load_completed_granules(times_path)

        results = _search_with_retry(
            short_name='MUR-JPL-L4-GLOB-v4.1',
            temporal=(f'{year}-01-01', f'{year}-12-31'),
            bounding_box=(lon_min, lat_min, lon_max, lat_max),
        )

        if not results:
            print('    No granules found, skipping')
            continue

        print(f'    Found {len(results)} granules')

        first_write = not zarr_path.exists()

        for i, granule in enumerate(results, 1):
            native_id = granule['meta']['native-id']
            fname     = native_id.split('/')[-1]
            if not fname.endswith('.nc'):
                fname = fname + '.nc'
            expected  = tmp_dir / fname

            # ------------------------------------------------------------------
            # Resume guard: peek at the granule's time without downloading.
            # MUR native-IDs encode the date as YYYYMMDD; parse it to build a
            # datetime64 and compare against the already-written set.
            # Example native-id: …/20130103090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc
            # ------------------------------------------------------------------
            try:
                date_str = fname.split('-')[0]          # '20130103090000'
                granule_time = np.datetime64(
                    f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T{date_str[8:10]}:{date_str[10:12]}:{date_str[12:14]}'
                )
                granule_time_int = int(granule_time.astype('datetime64[ns]').astype('int64'))
                if granule_time_int in completed:
                    print(f'    [{i}/{len(results)}] Already written ({granule_time}), skipping')
                    continue
            except Exception:
                # If we can't parse the date, fall through to the normal path;
                # the xarray-level check below will catch true duplicates.
                pass

            print(f'    Granule {i}/{len(results)}: {fname}')

            # Disk-level cache: file present from a previous interrupted attempt
            if expected.exists() and expected.stat().st_size > 1_000_000:
                print('      Already on disk, skipping download')
                files = [expected]
            else:
                try:
                    raw   = _download_with_retry(granule, tmp_dir)
                    files = [Path(f) for f in raw]
                except Exception as e:
                    print(f'      Gave up after retries: {e}  — skipping granule')
                    continue

            if not files:
                print('      No files returned, skipping')
                continue

            ds = None
            try:
                ds = _preprocess_mur_file(files[0], domain)

                # Secondary guard: check actual time coordinate against written set
                # (handles edge cases where the filename date differs from the data)
                ds_times = set(ds['time'].values.tolist())
                if ds_times.issubset(completed):
                    print(f'      Time(s) already in store, skipping')
                    continue

                chunk_lat = min(256, int(ds.sizes['lat']))
                chunk_lon = min(256, int(ds.sizes['lon']))
                ds = ds.chunk({'time': 1, 'lat': chunk_lat, 'lon': chunk_lon})
                enc = _mur_encoding(chunk_lat, chunk_lon)

                if first_write:
                    ds.to_zarr(
                        str(zarr_path),
                        mode='w',
                        consolidated=True,
                        encoding=enc,
                        align_chunks=True,
                    )
                    first_write = False
                else:
                    ds.to_zarr(
                        str(zarr_path),
                        mode='a',
                        append_dim='time',
                        consolidated=True,
                        align_chunks=True,
                    )

                batch_times = ds['time'].values
                if times_path.exists():
                    old_times = np.load(times_path, allow_pickle=False)
                    np.save(times_path, np.concatenate([old_times, batch_times]))
                else:
                    np.save(times_path, batch_times)

                # Keep the in-memory completed set current within this year loop
                completed.update(batch_times.tolist())

                print(f"      Appended {ds.sizes['time']} timestep(s)")
            finally:
                if ds is not None:
                    ds.close()
                _cleanup_files(files)
                gc.collect()

        _remove_tree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

    _remove_tree(tmp_dir)


def download_era5(domain_key, domain, years, out_root, restart=False):
    import xarray as xr
    import calendar

    c = cdsapi.Client()

    out_dir = Path(out_root) / domain_key / 'era5'
    out_dir.mkdir(parents=True, exist_ok=True)

    lat_min, lat_max = domain['lat']
    lon_min, lon_max = domain['lon']

    def _days_in_month(year, month):
        nd = calendar.monthrange(year, month)[1]
        return [f'{d:02d}' for d in range(1, nd + 1)]

    def _open_fix(path):
        ds = xr.open_dataset(path, engine='h5netcdf')
        if 'valid_time' in ds and 'time' not in ds.dims:
            ds = ds.rename({'valid_time': 'time'})
        if 'expver' in ds.coords:
            ds = ds.drop_vars('expver')
        if 'number' in ds.coords:
            ds = ds.drop_vars('number')
        return ds

    def _pick(ds, v):
        if v in ds.data_vars:
            return ds[v]
        raise KeyError(f"Missing variable {v} in {list(ds.data_vars)}")

    sl_var_map = {
        '10m_u_component_of_wind': 'u10',
        '10m_v_component_of_wind': 'v10',
        'mean_sea_level_pressure': 'msl',
        'sea_surface_temperature': 'sst_era5',
        '2m_temperature': 't2m',
        'sea_ice_cover': 'siconc',
    }

    pl_var_map = {
        'u_component_of_wind': 'u',
        'v_component_of_wind': 'v',
        'temperature': 't',
        'geopotential': 'z',
        'specific_humidity': 'q',
    }

    train_order = [
        'u10', 'v10', 'msl', 'sst_era5', 't2m', 'siconc',
        'u850', 'v850', 't850', 'z850', 'q850',
        'u700', 'v700', 't700', 'z700', 'q700',
        'u500', 'v500', 't500', 'z500', 'q500',
    ]

    for year in range(years[0], years[1] + 1):
        out_file = out_dir / f'era5_{domain_key}_{year}.nc'

        if out_file.exists():
            if restart:
                print(f"  ERA5 {domain['name']} {year} — [restart] removing")
                out_file.unlink()
            else:
                print(f"  ERA5 {domain['name']} {year} — already exists, skipping")
                continue

        print(f"  ERA5 {domain['name']} {year} — downloading in smallest chunks...")

        sl_tmp_files = []
        pl_tmp_files = []

        try:
            # ── Single levels: month × single variable ───────────────────────
            for month in range(1, 13):
                month_str = f'{month:02d}'
                day_list = _days_in_month(year, month)

                for cds_var, short_name in sl_var_map.items():
                    tmp_sl = out_dir / f'_tmp_{domain_key}_{year}_sl_{month_str}_{short_name}.nc'
                    print(
                        f"  ERA5 {domain['name']} {year} month {month_str} "
                        f"{short_name} — single level..."
                    )

                    c.retrieve(
                        'reanalysis-era5-single-levels',
                        {
                            'product_type': 'reanalysis',
                            'variable': [cds_var],
                            'year': str(year),
                            'month': month_str,
                            'day': day_list,
                            'time': ['00:00', '06:00', '12:00', '18:00'],
                            'area': [lat_max, lon_min, lat_min, lon_max],
                            'data_format': 'netcdf',
                            'download_format': 'unarchived',
                        },
                        str(tmp_sl),
                    )
                    sl_tmp_files.append((tmp_sl, short_name))

            # ── Pressure levels: month × level × single variable ──────────────
            for month in range(1, 13):
                month_str = f'{month:02d}'
                day_list = _days_in_month(year, month)

                for level in ['850', '700', '500']:
                    for cds_var, short_name in pl_var_map.items():
                        tmp_pl = out_dir / f'_tmp_{domain_key}_{year}_pl_{month_str}_{level}_{short_name}.nc'
                        print(
                            f"  ERA5 {domain['name']} {year} month {month_str} "
                            f"level {level} {short_name} — pressure level..."
                        )

                        c.retrieve(
                            'reanalysis-era5-pressure-levels',
                            {
                                'product_type': 'reanalysis',
                                'variable': [cds_var],
                                'pressure_level': [level],
                                'year': str(year),
                                'month': month_str,
                                'day': day_list,
                                'time': ['00:00', '06:00', '12:00', '18:00'],
                                'area': [lat_max, lon_min, lat_min, lon_max],
                                'data_format': 'netcdf',
                                'download_format': 'unarchived',
                            },
                            str(tmp_pl),
                        )
                        pl_tmp_files.append((tmp_pl, level, short_name))

            # ── Open and assemble single-level data ───────────────────────────
            sl_month_parts = []
            for month in range(1, 13):
                month_str = f'{month:02d}'
                month_ds_vars = []

                for fp, short_name in sl_tmp_files:
                    token = f"_sl_{month_str}_{short_name}.nc"
                    if token in fp.name:
                        ds = _open_fix(fp)
                        var_name = list(ds.data_vars)[0]
                        da = ds[var_name].astype('float32').rename(short_name)
                        month_ds_vars.append(da.to_dataset())

                if not month_ds_vars:
                    raise RuntimeError(f"No single-level chunks found for {year}-{month_str}")

                ds_month = xr.merge(month_ds_vars, compat='override', combine_attrs='override')
                ds_month = ds_month.sortby('time')
                sl_month_parts.append(ds_month)

            ds_sl = xr.concat(sl_month_parts, dim='time').sortby('time')

            # ── Open and assemble pressure-level data ─────────────────────────
            pl_month_parts = []
            for month in range(1, 13):
                month_str = f'{month:02d}'
                ds_month_vars = []

                for level in ['850', '700', '500']:
                    level_vars = []
                    for fp, lev, short_name in pl_tmp_files:
                        if lev == level and f"_pl_{month_str}_{level}_{short_name}.nc" in fp.name:
                            ds = _open_fix(fp)
                            var_name = list(ds.data_vars)[0]
                            da = ds[var_name].astype('float32').rename(short_name)
                            level_vars.append(da.to_dataset())

                    if not level_vars:
                        raise RuntimeError(f"No pressure-level chunks found for {year}-{month_str} level {level}")

                    ds_level = xr.merge(level_vars, compat='override', combine_attrs='override')

                    rename_map = {
                        'u': f'u{level}',
                        'v': f'v{level}',
                        't': f't{level}',
                        'z': f'z{level}',
                        'q': f'q{level}',
                    }
                    ds_level = ds_level.rename(rename_map)
                    ds_level = ds_level.sortby('time')
                    ds_month_vars.append(ds_level)

                ds_month = xr.merge(ds_month_vars, compat='override', combine_attrs='override')
                ds_month = ds_month.sortby('time')
                pl_month_parts.append(ds_month)

            ds_pl = xr.concat(pl_month_parts, dim='time').sortby('time')

            # ── Final merged dataset in training order ────────────────────────
            out = xr.Dataset()

            out['u10']      = _pick(ds_sl, 'u10').astype('float32')
            out['v10']      = _pick(ds_sl, 'v10').astype('float32')
            out['msl']      = _pick(ds_sl, 'msl').astype('float32')
            out['sst_era5'] = _pick(ds_sl, 'sst_era5').astype('float32')
            out['t2m']      = _pick(ds_sl, 't2m').astype('float32')
            out['siconc']   = _pick(ds_sl, 'siconc').astype('float32')

            for v in [
                'u850', 'v850', 't850', 'z850', 'q850',
                'u700', 'v700', 't700', 'z700', 'q700',
                'u500', 'v500', 't500', 'z500', 'q500',
            ]:
                out[v] = _pick(ds_pl, v).astype('float32')

            if 'valid_time' in out and 'time' not in out.dims:
                out = out.rename({'valid_time': 'time'})

            out = out[train_order]
            out = out.sortby('time')

            out.to_netcdf(str(out_file), engine='h5netcdf')
            print(f"    Saved → {out_file}  vars={list(out.data_vars)}")

        finally:
            for fp, _ in sl_tmp_files:
                Path(fp).unlink(missing_ok=True)
            for fp, _, _ in pl_tmp_files:
                Path(fp).unlink(missing_ok=True)

def download_gebco(domain_key, domain, out_root, restart=False):
    import subprocess

    out_file = Path(out_root) / domain_key / f'gebco_{domain_key}.nc'
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists() and not restart:
        print(f"  GEBCO {domain['name']} — already exists, skipping")
        return

    lat_min, lat_max = domain['lat']
    lon_min, lon_max = domain['lon']

    # GEBCO subset API — returns native NetCDF with 'elevation' variable
    # No login or API key required
    url = (
        f'https://download.gebco.net/api/grd/sub/'
        f'?longitude_range={lon_min},{lon_max}'
        f'&latitude_range={lat_min},{lat_max}'
        f'&format=netcdf'
    )

    print(f"  GEBCO {domain['name']} — downloading from GEBCO API...")
    print(f"    URL: {url}")

    import urllib.request
    tmp_file = out_file.with_suffix('.tmp.nc')
    try:
        urllib.request.urlretrieve(url, str(tmp_file))
        tmp_file.rename(out_file)
        print(f"    Saved to {out_file}")
    except Exception as e:
        tmp_file.unlink(missing_ok=True)
        print(f"  ERROR: GEBCO API download failed: {e}")
        print(f"  Fallback: manually download from https://download.gebco.net")
        print(f"    Bbox — S:{lat_min} N:{lat_max} W:{lon_min} E:{lon_max}")
        print(f"    Format: NetCDF → save to {out_file}")
        return

    # Verify the file has the expected variable
    try:
        import xarray as xr
        ds = xr.open_dataset(out_file)
        if 'elevation' not in ds:
            # Some GEBCO API versions use 'z' — rename to 'elevation'
            rename_map = {v: 'elevation' for v in ds.data_vars
                          if v in ('z', 'Band1', 'topo')}
            if rename_map:
                ds = ds.rename(rename_map)
                tmp = out_file.with_suffix('.fixed.nc')
                ds.to_netcdf(str(tmp))
                ds.close()
                tmp.rename(out_file)
                print(f"    Renamed variable → 'elevation'")
            else:
                print(f"    WARNING: unexpected variables: {list(ds.data_vars)}")
                ds.close()
        else:
            ds.close()
            print(f"    Variable 'elevation' confirmed ✓")
    except Exception as e:
        print(f"    WARNING: could not verify NetCDF contents: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='all',
                        help='Domain key (bof/lab/gom/med/red/bls) or "all"')
    parser.add_argument('--years', nargs=2, type=int, default=[2013, 2023])
    parser.add_argument('--out', default='../data/domains')
    parser.add_argument('--restart', action='store_true',
                        help='Wipe existing outputs and re-download from scratch. '
                             'Without this flag the script resumes from where it left off.')
    parser.add_argument('--skip_mur',   action='store_true')
    parser.add_argument('--skip_era5',  action='store_true')
    parser.add_argument('--skip_gebco', action='store_true')
    args = parser.parse_args()

    global log
    log = _setup_logging(Path(args.out))
    log.info('▶ download_domains.py starting')
    if args.restart:
        log.warning('--restart flag set: existing outputs will be removed before downloading')

    targets = DOMAINS if args.domain == 'all' else {args.domain: DOMAINS[args.domain]}

    _inventory_existing_files(Path(args.out), targets, tuple(args.years))
    
    for key, dom in targets.items():
        print(f'\n{"─" * 60}')
        print(f'Domain: {dom["name"]}  ({dom["note"]})')
        print(f'  Lat: {dom["lat"]}  Lon: {dom["lon"]}')
        print(f'{"─" * 60}')

        if not args.skip_mur:
            download_mur(key, dom, args.years, args.out, restart=args.restart)
        if not args.skip_era5:
            download_era5(key, dom, args.years, args.out, restart=args.restart)
        if not args.skip_gebco:
            download_gebco(key, dom, args.out, restart=args.restart)

    print('\nAll downloads complete.')


if __name__ == '__main__':
    main()