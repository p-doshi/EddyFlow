import numpy as np
import zarr
import torch
from torch.utils.data import Dataset
import pandas as pd
from config import cfg


class GSLDataset(Dataset):
    """
    Each sample:
        era5_window  [T, N_INPUT, LAT_C, LON_C]   normalised float32
        sst_target   [1, LAT_F, LON_F]             normalised float32
        loss_weight  [1, LAT_F, LON_F]             float32  (0=mask, >0=train)
        bathy        [1, LAT_F, LON_F]             normalised float32 (static)

    Index is built on MUR daily timestamps. For each MUR day t:
        - find 7 ERA5 frames ending at t (t-36h, t-30h, ..., t-0h)
        - load MUR SST, error, and ice for day t
        - append static bathy
    """

    def __init__(self, split='train', augment=False, stage=1):
        assert split in ('train', 'val', 'test')
        self.augment = augment
        self.stage   = stage
        self.T     = cfg.T_ATM if stage == 4 else cfg.T
        self.T_oce = cfg.T_OCE if stage == 4 else None
        # ── Open zarr stores (memory-mapped, no RAM cost) ─────────────────
        self.era5 = zarr.open(cfg.ERA5_ZARR, 'r')  # [T_e, 21, 20, 49]
        self.mur  = zarr.open(cfg.MUR_ZARR,  'r')  # [T_m, 501, 1201]
        self.err  = zarr.open(cfg.ERR_ZARR,  'r')  # [T_m, 501, 1201]
        self.ice  = zarr.open(cfg.ICE_ZARR,  'r')  # [T_m, 501, 1201]

        # ── Static bathy — load once, keep in RAM (~2.4 MB) ───────────────
        bathy_raw  = np.load(cfg.BATHY_NPY).astype(np.float32)
        bathy_log  = np.log1p(bathy_raw)
        bm = np.load(f'{cfg.STATS_DIR}/bathy_mean.npy')
        bs = np.load(f'{cfg.STATS_DIR}/bathy_std.npy')
        bathy_norm = (bathy_log - bm) / (bs + 1e-6)
        self.bathy = torch.from_numpy(bathy_norm).unsqueeze(0)  # [1, 501, 1201]


        # ── Normalisation stats ───────────────────────────────────────────
        self.era5_mean = np.load(f'{cfg.STATS_DIR}/era5_mean.npy')
        self.era5_std  = np.load(f'{cfg.STATS_DIR}/era5_std.npy')
        for ch in cfg.NAN_CHANNELS:
            self.era5_mean[ch] = 0.0
            self.era5_std[ch]  = 1.0
        self.sst_mean  = float(np.load(f'{cfg.STATS_DIR}/sst_mean.npy'))
        self.sst_std   = float(np.load(f'{cfg.STATS_DIR}/sst_std.npy'))

        # ── Build time indices ────────────────────────────────────────────
        era5_times_raw  = np.load(cfg.ERA5_TIMES, allow_pickle=True)
        mur_times_raw   = np.load(cfg.MUR_TIMES,  allow_pickle=True)
        self.mur_times  = mur_times_raw   # keep raw for later dict lookups
        

        try:
            self.era5_pd = pd.DatetimeIndex(era5_times_raw)
        except Exception:
            self.era5_pd = pd.DatetimeIndex([str(t) for t in era5_times_raw])

       
        try:
            self.mur_pd = pd.DatetimeIndex(mur_times_raw)
        except Exception:
            self.mur_pd = pd.DatetimeIndex([str(t) for t in mur_times_raw])
        # ── Select split years ────────────────────────────────────────────
        year_ranges = {
            'train': (2013, 2020),
            'val':   (2021, 2021),
            'test':  (2022, 2023),
        }
       
        y0, y1 = year_ranges[split]
        print(f'  Split={split}  years={y0}–{y1}')

        mur_mask = (self.mur_pd.year >= y0) & (self.mur_pd.year <= y1)
        candidate_mur_idx = np.where(mur_mask)[0]

        # ── Filter: only keep MUR days where all 7 ERA5 frames exist ──────
        # ERA5 frame spacing: 6 hours. T=7 frames → need 36 hours lookback.
        # For MUR day t at 00:00 UTC, we want ERA5 at:
        #   t-36h, t-30h, t-24h, t-18h, t-12h, t-6h, t+0h
        self.samples = []   # list of (mur_idx, [era5_idx × 7])

        self.era5_time_to_idx = {t: i for i, t in enumerate(self.era5_pd)}
        self.mur_tod         = self.mur_pd[0] - self.mur_pd[0].normalize()
        self.mur_time_to_idx = {
            pd.Timestamp(t): i
            for i, t in enumerate(self.mur_times)
        }
        era5_step_h = 6

        for mur_i in candidate_mur_idx:
            mur_day = pd.Timestamp(self.mur_times[mur_i])   # keep 09:00:00 intact

            # ERA5 frames
            era5_frames = []
            valid = True
            for hours_back in range((self.T - 1) * era5_step_h, -1, -era5_step_h):
                target_ts = (mur_day - pd.Timedelta(hours=hours_back)).floor('6h')
                if target_ts not in self.era5_time_to_idx:
                    valid = False
                    break
                era5_frames.append(self.era5_time_to_idx[target_ts])

            if not valid or len(era5_frames) != self.T:
                continue

            # MUR history frames (Stage 4 only)
            mur_hist = None
            if self.stage == 4:
                mur_hist = []
                for days_back in range(self.T_oce, 0, -1):
                    past_day = (mur_day - pd.Timedelta(days=days_back)).normalize() + self.mur_tod
                    if past_day not in self.mur_time_to_idx:
                        valid = False
                        break
                    mur_hist.append(self.mur_time_to_idx[past_day])

                if not valid or len(mur_hist) != self.T_oce:
                    continue

            self.samples.append((mur_i, era5_frames, mur_hist))


        print(f'[GSLDataset/{split}] {len(self.samples)} samples '
              f'({y0}–{y1})')

    def __len__(self):
        return len(self.samples)

    def _load_mur_frame(self, mi):
        sst = self.mur[mi].astype(np.float32)
        sst = (sst - self.sst_mean) / (self.sst_std + 1e-6)
        sst = np.where(np.isfinite(sst), sst, 0.0)
        return sst[np.newaxis]   # [1, H_f, W_f]

    def __getitem__(self, idx):
        mur_i, era5_idxs, mur_hist_idxs = self.samples[idx]

        # ── ERA5 window: [T, 21, LAT_C, LON_C] ───────────────────────────
        # ── ERA5 window ───────────────────────────────────────────────────
        frames = []
        for ei in era5_idxs:
            frame = self.era5[ei, :, :, :cfg.LON_C].astype(np.float32)  # [21, 20, 48]

            # Normalise first, THEN fill NaN
            # This ensures land pixels = 0.0 in z-score space (channel mean)
            # rather than -44 (which happens when 0 K is z-scored)
            frame = (frame - self.era5_mean[:, None, None]) \
                  / (self.era5_std[:, None, None] + 1e-6)

            # Now fill: NaN land pixels → 0.0 (= channel mean in z-score space)
            frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
            frames.append(frame)

        era5_window = np.stack(frames, axis=0)   # [T, 21, 20, 49]

        # ── MUR SST target: [1, LAT_F, LON_F] ────────────────────────────
        sst_raw  = self.mur[mur_i].astype(np.float32)    # [501, 1201]
        sst_norm = (sst_raw - self.sst_mean) / (self.sst_std + 1e-6)
        sst_valid = np.isfinite(sst_norm)
        sst_norm  = np.where(sst_valid, sst_norm, 0.0)
        sst_target = torch.from_numpy(sst_norm).unsqueeze(0)   # [1, 501, 1201]

        # ── Loss weight: ice mask + valid SST mask ────────────────────────
        # err is currently unused; keep only if you plan confidence weighting later
        ice = self.ice[mur_i].astype(np.float32)         # [H_f, W_f]

        # Pixels with SIC > 0.15 or non-finite SST → weight 0
        ice_mask   = (ice > 0.15) | (~sst_valid)   # ice OR invalid SST
        loss_weight = torch.from_numpy(
            (~ice_mask).astype(np.float32)
        ).unsqueeze(0)   # [1, 501, 1201]  binary: 1=ocean, 0=ice/land

        # ── Ice channel for encoder input: [1, LAT_C, LON_C] ─────────────
        era5_h, era5_w = self.era5.shape[2], self.era5.shape[3]  # 20, 49

        import torch.nn.functional as F
        ice_coarse = F.adaptive_avg_pool2d(
            torch.from_numpy(ice).unsqueeze(0).unsqueeze(0),
            (cfg.LAT_C, cfg.LON_C)   # (20, 48) — cfg is single source of truth
        ).squeeze(0)

        bathy_coarse = F.adaptive_avg_pool2d(
            self.bathy.unsqueeze(0), (cfg.LAT_C, cfg.LON_C)
        ).squeeze(0)

        static     = torch.cat([bathy_coarse, ice_coarse], dim=0)             # [2, 20, 49]
        static_seq = static.unsqueeze(0).expand(self.T, -1, -1, -1).contiguous()  # [T, 2, 20, 49]

        era5_tensor = torch.from_numpy(era5_window)              # [T, 21, 20, 49]
        era5_input  = torch.cat([era5_tensor, static_seq], dim=1) # [T, 23, 20, 49]
        # Final NaN guard on the fully assembled input
        era5_input = torch.nan_to_num(era5_input, nan=0.0, posinf=0.0, neginf=0.0)
        mur_prev      = max(mur_i - 1, 0)
        sst_prev_raw  = self.mur[mur_prev].astype(np.float32)
        sst_prev_norm = (sst_prev_raw - self.sst_mean) / (self.sst_std + 1e-6)
        sst_prev_norm = np.where(np.isfinite(sst_prev_norm), sst_prev_norm, 0.0)
        sst_prev      = torch.from_numpy(sst_prev_norm).unsqueeze(0)  # [1, 501, 1201]
        delta_sst = sst_target - sst_prev
        sample_ts = pd.Timestamp(self.mur_pd[mur_i])

        out = {
            'era5':     era5_input,
            'sst':      delta_sst,
            'sst_abs':  sst_target,
            'weight':   loss_weight,
            'bathy':    self.bathy,
            'sst_prev': sst_prev,
            'sample_idx': idx,
            'mur_idx':    mur_i,
            # tensors for seasonal logic
            'year':       torch.tensor(sample_ts.year,  dtype=torch.long),
            'month':      torch.tensor(sample_ts.month, dtype=torch.long),
            'day':        torch.tensor(sample_ts.day,   dtype=torch.long),
            'hour':       torch.tensor(sample_ts.hour,  dtype=torch.long),
            'minute':     torch.tensor(sample_ts.minute, dtype=torch.long),
            # strings for naming / HTML
            'date_str':   sample_ts.strftime('%Y-%m-%d'),
            'time_str':   sample_ts.strftime('%H-%M-%S'),
        }


        if self.stage == 4:
            mur_seq = np.stack(
                [self._load_mur_frame(mi) for mi in mur_hist_idxs],
                axis=0
            )  # [T_oce, 1, H_f, W_f]
            out['mur_seq'] = torch.from_numpy(mur_seq)

        return out


def make_loaders(batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, stage=1):
    train_ds = GSLDataset(split='train', augment=True, stage=stage)
    val_ds   = GSLDataset(split='val',   augment=False, stage=stage)
    test_ds  = GSLDataset(split='test',  augment=False, stage=stage)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        drop_last=True, prefetch_factor=cfg.PREFETCH,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_ds