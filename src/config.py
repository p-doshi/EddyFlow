# src/config.py

import numpy as np
import os

class Config:
    # ── Domain ────────────────────────────────────────────────────────────
    # These come from what preprocess.py actually produced — read from zarr
    # rather than hardcoding, so nothing drifts again
    LAT_C = 20
    LON_C = 48      # was 48 — arange gave one extra
    LAT_F = 501     # was 500
    LON_F = 1201    # was 1200
    TRAIN_YEARS = range(2010, 2018)   # 2010-2017 inclusive
    VAL_YEARS   = range(2018, 2019)   # 2018 only
    TEST_YEARS  = range(2019, 2020)
    # ── ERA5 channels ─────────────────────────────────────────────────────
    N_CH = 21
    CHANNEL_NAMES = [
        'u10', 'v10', 'msl', 'sst_era5', 't2m', 'siconc',
        'u850', 'v850', 't850', 'z850', 'q850',
        'u700', 'v700', 't700', 'z700', 'q700',
        'u500', 'v500', 't500', 'z500', 'q500',
    ]
    
    NAN_CHANNELS = []
    DATA_DIR = "../data/domains/gsl"
    MUR_LAT_NPY = f'{DATA_DIR}/mur_lat.npy'
    MUR_LON_NPY = f'{DATA_DIR}/mur_lon.npy'
    # Channels with NaN over land/ocean — need fill before normalisation
    NAN_FILL_CHANNELS = {
        CHANNEL_NAMES.index('sst_era5'): 0.0,   # fill with 0 (will be masked)
        CHANNEL_NAMES.index('siconc'):   0.0,    # no ice = 0
    }

    # ── Model input ───────────────────────────────────────────────────────
    # ERA5 channels (21) + bathy (1) + ice from MUR (1) = 23 total input channels
    N_INPUT = N_CH + 2

    # Temporal window
    T_ATM = 28   # 7 ERA5 frames at 6-hourly = 42 hours context
    T = T_ATM
    T_OCE = 60 # should b 112, check next time when running
    # ── Model architecture ────────────────────────────────────────────────
    PATCH_SIZE = 4       # 4×4 patches on coarse grid
    # Tokens per frame: ceil(20/4) × ceil(49/4) = 5 × 13 = 65
    N_TOKENS_H = LAT_C // PATCH_SIZE          # 5
    N_TOKENS_W = (LON_C + PATCH_SIZE - 1) // PATCH_SIZE   # 13 (49 not divisible by 4)
    N_TOKENS   = N_TOKENS_H * N_TOKENS_W     # 65

    D_MODEL    = 256
    N_HEADS    = 8
    N_LAYERS   = 8
    DROPOUT    = 0.1

    # ── Diffusion ─────────────────────────────────────────────────────────
    DIFF_STEPS        = 1000
    DIFF_SAMPLE_STEPS = 20
    DIFF_CHANNELS     = 64
    SIGMA_DATA    = 0.06    # tune: std of (sst - x_base) on GSL training data
    VAE_LATENT_CH = 4       # latent channels in ResidualVAE
    VAE_KL_WEIGHT = 1e-4    # KL regularisation weight

    # ── A100 training ─────────────────────────────────────────────────────
    BATCH_SIZE  = 2
    GRAD_ACCUM  = 8       # effective batch = 16
    LR          = 3e-4
    WEIGHT_DECAY= 1e-4
    EPOCHS      = 100
    GRAD_CLIP   = 1.0

    NUM_WORKERS = 8
    PREFETCH    = 4

    # ── Paths ─────────────────────────────────────────────────────────────
    ERA5_ZARR  = '../data/domains/gsl/era5.zarr'
    ERA5_MEAN  = '../scripts/data/stats/era5_mean.npy'
    ERA5_STD   = '../scripts/data/stats/era5_std.npy'
    SST_MEAN   = '../scripts/data/stats/sst_mean.npy'
    SST_STD    = '../scripts/data/stats/sst_std.npy'
    MUR_ZARR   = '../data/domains/gsl/mur.zarr'
    ERR_ZARR   = '../data/domains/gsl/mur_error.zarr'
    ICE_ZARR   = '../data/domains/gsl/ice.zarr'
    GLORYS_ZARR = '../data/domains/gsl/glorys.zarr'
    BATHY_NPY  = '../data/domains/gsl/bathy.npy'
    ERA5_TIMES = '../data/domains/gsl/era5_times.npy'
    MUR_TIMES  = '../data/domains/gsl/mur_times.npy'
    STATS_DIR  = '../scripts/data/stats'
    CKPT_DIR   = '../outputs/checkpoints'
    LOG_DIR    = '../outputs/logs'
    FIG_DIR    = '../outputs/figures'

    def __post_init__(self):
        for d in [self.CKPT_DIR, self.LOG_DIR, self.FIG_DIR]:
            os.makedirs(d, exist_ok=True)


cfg = Config()

# ── Sanity check against actual zarr shapes ───────────────────────────────────
if __name__ == '__main__':
    import zarr
    e = zarr.open(cfg.ERA5_ZARR, 'r')
    m = zarr.open(cfg.MUR_ZARR,  'r')
    print(f'ERA5: {e.shape}  expected (T, {cfg.N_CH}, {cfg.LAT_C}, {cfg.LON_C})')
    print(f'MUR:  {m.shape}  expected (T, {cfg.LAT_F}, {cfg.LON_F})')
    assert e.shape[1] == cfg.N_CH,  f"Channel mismatch: {e.shape[1]} vs {cfg.N_CH}"
    assert e.shape[2] == cfg.LAT_C, f"LAT_C mismatch:   {e.shape[2]} vs {cfg.LAT_C}"
    assert e.shape[3] == cfg.LON_C, f"LON_C mismatch:   {e.shape[3]} vs {cfg.LON_C}"
    assert m.shape[1] == cfg.LAT_F, f"LAT_F mismatch:   {m.shape[1]} vs {cfg.LAT_F}"
    assert m.shape[2] == cfg.LON_F, f"LON_F mismatch:   {m.shape[2]} vs {cfg.LON_F}"
    print('Config matches zarr shapes. Ready.')