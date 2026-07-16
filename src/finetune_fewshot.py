"""
finetune_fewshot.py — Few-shot domain adaptation of the diffusion UNet.

Loads a merged (phase1+phase2) checkpoint, freezes encoder+baseline,
fine-tunes the diffusion UNet on n_shots randomly sampled days from the
target domain's non-test period, and saves a new checkpoint.

Domains:
  BOF -- ERA5 available 2013-2021 (non-test); ~2920 candidate days
  GOM -- ERA5 available 2021 only (only non-test year downloaded); ~305-365 days

Usage:
  python finetune_fewshot.py \\
      --stage 4 \\
      --ckpt  ../outputs/checkpoints/stage4_sweep_4_123_lr0_sw0_merged.pt \\
      --domain bof \\
      --n_shots 30 \\
      --seed 42 \\
      --epochs 20 \\
      --run_tag fs_bof_n30
"""

import argparse, json, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
import zarr

from config import cfg
from model import TemporalDownscaler, TemporalDownscalerV2, TemporalDownscalerV5


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers (same as finetune_diffusion.py / finetune_sweep_v2.py)
# ─────────────────────────────────────────────────────────────────────────────

def spectral_loss_weighted(pred, target, weight=None):
    if weight is not None:
        pred   = pred   * weight
        target = target * weight
    pf = torch.fft.rfft2(pred)
    tf = torch.fft.rfft2(target)
    ky = torch.fft.fftfreq(pred.shape[-2], device=pred.device).abs()
    kx = torch.fft.rfftfreq(pred.shape[-1], device=pred.device).abs()
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    w  = (1.0 + 20.0 * k2).sqrt()
    return (w * (pf.abs() - tf.abs()).abs()).mean()


def x0_from_noise(r_noisy, noise_pred, sqrt_ab_t, sqrt_1mab_t):
    return (r_noisy - sqrt_1mab_t * noise_pred) / sqrt_ab_t.clamp(min=1e-5)

STAGE_CHOICES = ('1', '2', '3', '4', '4v2', '5a')

ERA5_CHANNELS_FULL = [
    'u10', 'v10', 'msl', 'sst_era5', 't2m', 'siconc',
    'u850', 'v850', 't850', 'z850', 'q850',
    'u700', 'v700', 't700', 'z700', 'q700',
    'u500', 'v500', 't500', 'z500', 'q500',
]
TEST_YEARS = (2022, 2023)

DOMAIN_CFG = {
    'bof': dict(
        data_dir    = Path('../data/domains/bof'),
        era5_prefix = 'era5/era5_bof_',
        mur_zarr    = 'mur/mur_bof.zarr',
        mur_times   = 'mur/mur_bof_times.npy',
        bathy_file  = 'gebco_bof.nc',
        train_years = (2013, 2021),
    ),
    'gom': dict(
        data_dir    = Path('../data/domains/gom'),
        era5_prefix = 'era5/era5_gom_',
        mur_zarr    = 'mur/mur_gom.zarr',
        mur_times   = 'mur/mur_gom_times.npy',
        bathy_file  = 'gebco_gom.nc',
        train_years = (2021, 2021),  # only year with ERA5 not in test split
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def freeze(module):
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()

def unfreeze(module):
    for p in module.parameters():
        p.requires_grad_(True)
    module.train()

def load_model(ckpt_path, stage, device):
    stage_str = str(stage)
    if stage_str == '4v2':
        model = TemporalDownscalerV2().to(device)
    elif stage_str in ('5a', '5b'):
        model = TemporalDownscalerV5().to(device)
    else:
        model = TemporalDownscaler(stage=int(stage_str)).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get('model', ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt)))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f'  [WARN] Missing keys: {missing[:5]}')
    if unexpected:
        print(f'  [WARN] Unexpected keys: {unexpected[:5]}')
    return model


def compute_era5_stats(era5_ds):
    means, stds = [], []
    for v in ERA5_CHANNELS_FULL:
        if v in era5_ds:
            data = era5_ds[v].values.astype(np.float32)
            valid = data[np.isfinite(data)]
            m = float(valid.mean()) if valid.size else 0.0
            s = float(valid.std())  if valid.size else 1.0
            s = max(s, 1e-6)
        else:
            m, s = 0.0, 1.0
        means.append(m)
        stds.append(s)
    return np.array(means, np.float32), np.array(stds, np.float32)


def load_bathy(dcfg, stats):
    bathy_nc = dcfg['data_dir'] / dcfg['bathy_file']
    ds = None
    for eng in ('h5netcdf', 'scipy', 'netcdf4', None):
        try:
            ds = xr.open_dataset(bathy_nc, engine=eng) if eng else xr.open_dataset(bathy_nc)
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError(f'Cannot open bathy: {bathy_nc}')
    ev = 'elevation' if 'elevation' in ds.data_vars else next(iter(ds.data_vars))
    elev = ds[ev].values.astype(np.float32)
    depth = np.where(elev < 0, -elev, 0.0)
    blog = np.log1p(depth)
    bm = float(np.load(Path(cfg.STATS_DIR) / 'bathy_mean.npy'))
    bs = float(np.load(Path(cfg.STATS_DIR) / 'bathy_std.npy'))
    bnorm = (blog - bm) / (bs + 1e-6)
    ds.close()
    return torch.from_numpy(bnorm).unsqueeze(0)  # [1, H_b, W_b]


def open_era5(dcfg):
    yr0, yr1 = dcfg['train_years']
    files = []
    for yr in range(yr0, yr1 + 1):
        p = dcfg['data_dir'] / f"{dcfg['era5_prefix']}{yr}.nc"
        if p.exists():
            files.append(str(p))
    if not files:
        raise FileNotFoundError(f"No ERA5 files found for {dcfg['data_dir']}")
    ds = xr.open_mfdataset(files, combine='by_coords',
                           decode_timedelta=False, engine='h5netcdf')
    if 'valid_time' in ds and 'time' not in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    return ds


def open_mur(dcfg):
    store = zarr.open(str(dcfg['data_dir'] / dcfg['mur_zarr']))
    times = np.load(str(dcfg['data_dir'] / dcfg['mur_times']), allow_pickle=True)
    return store, times


def build_sample_index(era5_ds, mur_times_raw, T_atm, T_oce):
    """Return list of valid (mur_i, era5_frames, mur_prev_i) from non-test years only."""
    era5_pd = pd.DatetimeIndex(era5_ds['time'].values)
    try:
        mur_pd = pd.DatetimeIndex(mur_times_raw)
    except Exception:
        mur_pd = pd.DatetimeIndex([str(t) for t in mur_times_raw])

    era5_date_to_idx = {t.date(): i for i, t in enumerate(era5_pd)}
    test_years_set   = set(range(TEST_YEARS[0], TEST_YEARS[1] + 1))

    samples = []
    for mur_i, mur_ts in enumerate(mur_pd):
        mur_date = mur_ts.date()
        # Exclude test years
        if mur_date.year in test_years_set:
            continue
        # Need T_oce prior MUR frames
        if T_oce > 0 and mur_i < T_oce:
            continue

        era5_frames, valid = [], True
        for days_back in range(T_atm - 1, -1, -1):
            target = (pd.Timestamp(mur_date) - pd.Timedelta(days=days_back)).date()
            if target not in era5_date_to_idx:
                valid = False
                break
            era5_frames.append(era5_date_to_idx[target])

        if valid and len(era5_frames) == T_atm:
            samples.append((mur_i, era5_frames, max(mur_i - 1, 0)))

    return samples


def build_batch_tensor(sample, era5_ds, mur_store, era5_mean, era5_std,
                       sst_mean, sst_std, bathy, T_atm, T_oce, LAT_C, LON_C):
    """Build a single batch dict (CPU tensors) from one sample."""
    mur_i, era5_frames, mur_prev_i = sample
    n_ch = len(ERA5_CHANNELS_FULL)

    # ERA5 window [T_atm, 21, H_e, W_e] → resize to [T_atm, 21, LAT_C, LON_C]
    frames = []
    for ei in era5_frames:
        chans = []
        for ci, v in enumerate(ERA5_CHANNELS_FULL):
            if v in era5_ds:
                arr = era5_ds[v].isel(time=ei).values.astype(np.float32)
                if arr.ndim == 3:
                    arr = arr[0]
            else:
                ref = era5_ds[list(era5_ds.data_vars)[0]].isel(time=ei).values
                arr = np.zeros_like(ref if ref.ndim == 2 else ref[0], dtype=np.float32)
            arr = (arr - era5_mean[ci]) / (era5_std[ci] + 1e-6)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            chans.append(arr)
        frames.append(np.stack(chans, 0))
    era5_win = torch.from_numpy(np.stack(frames, 0))  # [T, 21, H, W]

    era5_t = F.interpolate(
        era5_win.reshape(T_atm * n_ch, 1, era5_win.shape[-2], era5_win.shape[-1]),
        size=(LAT_C, LON_C), mode='bilinear', align_corners=False,
    ).reshape(T_atm, n_ch, LAT_C, LON_C)

    bathy_c = F.adaptive_avg_pool2d(bathy.unsqueeze(0), (LAT_C, LON_C)).squeeze(0)
    ice_c   = torch.zeros(1, LAT_C, LON_C)
    static  = torch.cat([bathy_c, ice_c], 0).unsqueeze(0).expand(T_atm, -1, -1, -1)
    era5_input = torch.cat([era5_t, static], 1)  # [T, 23, LAT_C, LON_C]
    era5_input = torch.nan_to_num(era5_input, nan=0.0)

    def mur_norm(idx):
        raw  = mur_store['analysed_sst'][idx].astype(np.float32)
        if np.nanmean(raw[raw > 0]) > 200:
            raw = np.where(np.isfinite(raw), raw - 273.15, raw)
        norm = (raw - sst_mean) / (sst_std + 1e-6)
        return torch.from_numpy(np.nan_to_num(norm, nan=0.0)).unsqueeze(0)

    sst_abs  = mur_norm(mur_i)
    sst_prev = mur_norm(mur_prev_i)
    delta    = sst_abs - sst_prev

    fine_size = sst_abs.shape[-2:]
    if bathy.shape[-2:] != fine_size:
        bathy_f = F.interpolate(bathy.unsqueeze(0), size=fine_size,
                                mode='bilinear', align_corners=False).squeeze(0)
    else:
        bathy_f = bathy

    try:
        ice_fine = torch.from_numpy(mur_store['sea_ice_fraction'][mur_i].astype(np.float32)).unsqueeze(0)
        ice_mask = ice_fine > 0.15
    except Exception:
        ice_mask = torch.zeros_like(sst_abs, dtype=torch.bool)
    raw_valid = torch.isfinite(torch.from_numpy(mur_store['analysed_sst'][mur_i].astype(np.float32)).unsqueeze(0))
    weight = (~ice_mask & raw_valid).float()

    batch = dict(
        era5    = era5_input.unsqueeze(0),   # [1, T, 23, LC, WC]
        sst     = delta.unsqueeze(0),         # [1, 1, H_f, W_f]
        sst_abs = sst_abs.unsqueeze(0),
        bathy   = bathy_f.unsqueeze(0),
        weight  = weight.unsqueeze(0),
    )
    if T_oce > 0:
        oce_seq = []
        for j in range(mur_i - T_oce, mur_i):
            oce_seq.append(mur_norm(j))
        batch['mur_seq'] = torch.stack(oce_seq, 0).unsqueeze(0)  # [1, T_oce, 1, H_f, W_f]

    return batch


# ─────────────────────────────────────────────────────────────────────────────
# Main fine-tune routine
# ─────────────────────────────────────────────────────────────────────────────

def finetune_fewshot(stage, ckpt_path, domain, n_shots, seed,
                     lr, epochs, spec_weight, run_tag):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"═"*60}')
    print(f'Few-shot finetune: stage={stage}  domain={domain}')
    print(f'  n_shots={n_shots}  seed={seed}  epochs={epochs}  lr={lr}')
    print(f'  spec_weight={spec_weight}  run_tag={run_tag}')
    print(f'  checkpoint: {ckpt_path}')
    print(f'  device: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}  '
              f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

    # ── Load stats ────────────────────────────────────────────────────────────
    sst_mean = float(np.load(Path(cfg.STATS_DIR) / 'sst_mean.npy'))
    sst_std  = float(np.load(Path(cfg.STATS_DIR) / 'sst_std.npy'))

    # ── Domain data ───────────────────────────────────────────────────────────
    dcfg = DOMAIN_CFG[domain]

    print(f'\nLoading ERA5 ({domain}) …')
    era5_ds = open_era5(dcfg)
    era5_mean, era5_std = compute_era5_stats(era5_ds)

    # Derive coarse grid from ERA5 data
    for dim in ('latitude', 'lat', 'y'):
        if dim in era5_ds.dims:
            LAT_C = len(era5_ds[dim]); break
    for dim in ('longitude', 'lon', 'x'):
        if dim in era5_ds.dims:
            LON_C = len(era5_ds[dim]); break

    print(f'  ERA5 coarse grid: {LAT_C} × {LON_C}')

    print(f'\nLoading MUR ({domain}) …')
    mur_store, mur_times = open_mur(dcfg)

    print(f'\nLoading bathy ({domain}) …')
    bathy = load_bathy(dcfg, {})  # bathy_mean/std loaded inside

    # ── Sample index ──────────────────────────────────────────────────────────
    stage_str = str(stage)
    T_atm = cfg.T_ATM if stage_str in ('4', '4v2', '5a') else cfg.T
    T_oce = cfg.T_OCE if stage_str in ('4', '4v2', '5a') else 0

    # All stages use SinCos2DPE for the atmospheric stream, hardcoded to the
    # GSL patch grid (20×48 → 5×12=60 tokens). Override for all OOD domains.
    LAT_C, LON_C = cfg.LAT_C, cfg.LON_C
    print(f'  Stage {stage_str}: using GSL coarse grid {LAT_C}×{LON_C} for atm PE')

    all_samples = build_sample_index(era5_ds, mur_times, T_atm, T_oce)
    print(f'\nCandidate training samples (non-test): {len(all_samples)}')

    if len(all_samples) < n_shots:
        print(f'  [WARN] Only {len(all_samples)} samples available; using all.')
        n_shots = len(all_samples)

    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(all_samples), size=n_shots, replace=False)
    chosen_samples = [all_samples[i] for i in sorted(chosen)]
    print(f'Sampled {n_shots} training days (seed={seed})')

    # ── Preload batches to GPU ────────────────────────────────────────────────
    print('\nPreloading batches …')
    gpu_batches = []
    for s in chosen_samples:
        b = build_batch_tensor(s, era5_ds, mur_store,
                               era5_mean, era5_std, sst_mean, sst_std,
                               bathy, T_atm, T_oce, LAT_C, LON_C)
        b = {k: v.to(device) for k, v in b.items()}
        gpu_batches.append(b)
    print(f'  Loaded {len(gpu_batches)} batches on {device}')
    era5_ds.close()

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f'\nLoading model (stage={stage}) …')
    model = load_model(ckpt_path, stage, device)

    freeze(model.encoder)
    freeze(model.baseline)
    unfreeze(model.diffusion)

    trainable = sum(p.numel() for p in model.diffusion.parameters() if p.requires_grad)
    print(f'  Trainable diffusion params: {trainable/1e6:.1f}M')

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.diffusion.parameters(), lr=lr,
        weight_decay=cfg.WEIGHT_DECAY, betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01,
    )
    scaler = torch.amp.GradScaler('cuda')

    # ── Training loop ─────────────────────────────────────────────────────────
    best_loss = float('inf')
    log = []
    ckpt_dir = Path(cfg.CKPT_DIR)
    ckpt_dir.mkdir(exist_ok=True)

    out_tag = f'stage{stage}_fewshot_{domain}_{run_tag}'

    for epoch in range(epochs):
        model.encoder.eval()
        model.baseline.eval()
        model.diffusion.train()

        # Shuffle samples each epoch
        perm = torch.randperm(len(gpu_batches)).tolist()
        acc_diff, acc_spec, acc_total, n_steps = 0., 0., 0., 0

        for idx in perm:
            batch = gpu_batches[idx]
            sst    = torch.nan_to_num(batch['sst'],    nan=0.0)
            weight = torch.nan_to_num(batch['weight'], nan=0.0)
            bathy_b = batch['bathy']

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    latent = model._encode(batch)
                    x_base = model.baseline(latent, bathy_b)

                r_gt = (sst - x_base).detach()
                B    = sst.shape[0]

                if stage_str in ('4v2', '5a'):
                    # EDM path
                    dm = batch.get('domain_sst_mean', torch.zeros(B, device=device))
                    ds = batch.get('domain_sst_std',  torch.ones(B,  device=device))
                    if stage_str == '4v2':
                        ln_sigma = (torch.randn(B, device=device)
                                    * model.diffusion.P_std + model.diffusion.P_mean)
                        sigma   = ln_sigma.exp().clamp(model.diffusion.sigma_min,
                                                       model.diffusion.sigma_max)
                        s       = sigma.view(B, 1, 1, 1)
                        r_noisy = r_gt + s * torch.randn_like(r_gt)
                        D_x     = model.diffusion.D_theta(
                            r_noisy, sigma, x_base.detach(), latent.detach(),
                            bathy_b, dm, ds)
                        lam      = ((sigma**2 + model.diffusion.sigma_data**2)
                                    / (sigma * model.diffusion.sigma_data)**2)
                        loss_diff = (lam.view(B, 1, 1, 1) * (D_x - r_gt)**2).mean()
                        pred_m   = torch.nan_to_num((x_base + D_x) * weight, nan=0.0)
                    else:  # 5a VAE
                        r_rec, kl = model.diffusion.vae(r_gt, bathy_b)
                        loss_diff = F.l1_loss(r_rec, r_gt) + 1e-4 * kl
                        pred_m   = torch.nan_to_num((x_base + r_rec) * weight, nan=0.0)
                    targ_m    = torch.nan_to_num(sst * weight, nan=0.0)
                    loss_spec = spectral_loss_weighted(pred_m, targ_m)

                else:
                    # DDIM path (stages 1-4)
                    t           = torch.randint(0, cfg.DIFF_STEPS, (B,), device=device)
                    noise       = torch.randn_like(r_gt)
                    sqrt_ab_t   = model.sqrt_ab[t].view(B, 1, 1, 1)
                    sqrt_1mab_t = model.sqrt_1mab[t].view(B, 1, 1, 1)
                    r_noisy     = sqrt_ab_t * r_gt + sqrt_1mab_t * noise
                    t_norm      = t.float() / cfg.DIFF_STEPS
                    noise_pred  = model.diffusion(
                        r_noisy, x_base.detach(), latent.detach(), bathy_b, t_norm)

                    def wmean(x):
                        return (x * weight).sum() / (weight.sum() + 1e-6)
                    loss_diff = wmean(F.mse_loss(noise_pred, noise, reduction='none'))

                    r0_hat = x0_from_noise(r_noisy, noise_pred, sqrt_ab_t, sqrt_1mab_t)
                    r0_hat = torch.nan_to_num(r0_hat, nan=0.0)
                    loss_spec = spectral_loss_weighted(x_base + r0_hat, sst, weight)

                loss = loss_diff + spec_weight * loss_spec

            if not torch.isfinite(loss):
                print(f'  [SKIP] NaN/Inf loss at epoch {epoch}')
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.diffusion.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            acc_diff  += loss_diff.item()
            acc_spec  += loss_spec.item()
            acc_total += loss.item()
            n_steps   += 1

        scheduler.step()
        avg_loss = acc_total / max(n_steps, 1)
        avg_diff = acc_diff  / max(n_steps, 1)
        avg_spec = acc_spec  / max(n_steps, 1)

        log.append({'epoch': epoch, 'loss': avg_loss,
                    'loss_diff': avg_diff, 'loss_spec': avg_spec})
        print(f'  Epoch {epoch:3d}  loss={avg_loss:.4f}  '
              f'diff={avg_diff:.4f}  spec={avg_spec:.4f}')

        # Save best + latest
        save_dict = dict(
            model         = model.state_dict(),
            epoch         = epoch,
            loss          = avg_loss,
            domain        = domain,
            n_shots       = n_shots,
            seed          = seed,
            spec_weight   = spec_weight,
            run_tag       = run_tag,
        )
        torch.save(save_dict, ckpt_dir / f'{out_tag}_latest.pt')
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(save_dict, ckpt_dir / f'{out_tag}_best.pt')

    log_path = Path(cfg.LOG_DIR) / f'{out_tag}_log.json'
    log_path.parent.mkdir(exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2))
    print(f'\nSaved: {ckpt_dir}/{out_tag}_best.pt')
    print(f'Log:   {log_path}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Few-shot domain adaptation (diffusion only)')
    p.add_argument('--stage',       required=True, choices=STAGE_CHOICES)
    p.add_argument('--ckpt',        required=True, help='Merged phase1+phase2 checkpoint')
    p.add_argument('--domain',      required=True, choices=('bof', 'gom'))
    p.add_argument('--n_shots',     type=int, default=30,
                   help='Number of target-domain training days (default 30)')
    p.add_argument('--seed',        type=int, default=0)
    p.add_argument('--lr',          type=float, default=5e-6,
                   help='Learning rate (default 5e-6, lower than phase2 to avoid forgetting)')
    p.add_argument('--epochs',      type=int, default=20)
    p.add_argument('--spec_weight', type=float, default=5.0)
    p.add_argument('--run_tag',     type=str,  default='fs')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    finetune_fewshot(
        stage        = args.stage,
        ckpt_path    = args.ckpt,
        domain       = args.domain,
        n_shots      = args.n_shots,
        seed         = args.seed,
        lr           = args.lr,
        epochs       = args.epochs,
        spec_weight  = args.spec_weight,
        run_tag      = args.run_tag,
    )
