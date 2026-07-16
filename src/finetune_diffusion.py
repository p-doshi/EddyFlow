"""
finetune_diffusion.py — Phase 2: Frozen baseline, diffusion-only fine-tuning.

Strategy:
  - Loads a Phase 1 checkpoint (encoder + baseline already converged)
  - Freezes encoder + baseline decoder
  - Trains ONLY the ResidualDiffusion UNet
  - Adds spectral loss on the x0-estimate to fix PSD ratio
  - Uses a lower, stable LR (baseline target is fixed → no moving target problem)

Usage:
  python finetune_diffusion.py --stage 1 --ckpt path/to/stage1_best.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import time
import argparse

from config import cfg
from dataset import make_loaders
from model import TemporalDownscaler


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()

def move_batch_to_device(batch: dict, device, non_blocking: bool = True):
    return {
        k: v.to(device, non_blocking=non_blocking)
        for k, v in batch.items()
        if torch.is_tensor(v)
    }

def unfreeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(True)
    module.train()


def count_params(module: nn.Module, label: str):
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f'  {label:<20s}  total={total/1e6:.2f}M  trainable={trainable/1e6:.2f}M')


def spectral_loss_weighted(pred, target, weight=None):
    """
    Spectral loss identical to TemporalDownscaler._spectral_loss,
    but optionally masked by weight before FFT.
    """
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
    """
    DDPM x0 estimate from the predicted noise:
        x0_hat = (r_noisy - sqrt(1-ab_t) * noise_pred) / sqrt(ab_t)
    """
    return (r_noisy - sqrt_1mab_t * noise_pred) / sqrt_ab_t.clamp(min=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def diffusion_diagnostic(model, loader, device):
    """
    Quick sanity check before Phase 2 training.
    Verifies baseline is frozen and diffusion is unfrozen.
    Also prints PSD ratio of the current baseline output to set expectations.
    """
    print('\n' + '─' * 60)
    print('Phase 2 pre-training diagnostic')
    print('─' * 60)

    # Param status
    enc_grad  = any(p.requires_grad for p in model.encoder.parameters())
    base_grad = any(p.requires_grad for p in model.baseline.parameters())
    diff_grad = any(p.requires_grad for p in model.diffusion.parameters())
    print(f'\n  Encoder  trainable: {enc_grad}   ← should be False')
    print(f'  Baseline trainable: {base_grad}  ← should be False')
    print(f'  Diffusion trainable:{diff_grad}  ← should be True')

    # PSD ratio of current baseline
    batch = next(iter(loader))
    batch_cpu = batch
    batch = move_batch_to_device(batch, device, non_blocking=False)

    model.eval()
    latent = model._encode(batch)
    x_base = model.baseline(latent, batch['bathy'])
    sst    = batch['sst']
    weight = batch['weight']

    def band_power(x, w):
        xw = torch.nan_to_num(x * w, nan=0.0)
        f  = torch.fft.rfft2(xw)
        ky = torch.fft.fftfreq(x.shape[-2], device=x.device).abs()
        kx = torch.fft.rfftfreq(x.shape[-1], device=x.device).abs()
        k2 = (ky[:, None]**2 + kx[None, :]**2)
        band = (k2 > 0.01) & (k2 < 0.25)
        return (f.abs()**2)[:, :, band].mean().item()

    psd_pred = band_power(x_base, weight)
    psd_targ = band_power(sst,    weight)
    ratio    = psd_pred / (psd_targ + 1e-12)
    r_gt     = (sst - x_base)
    rmse_r   = r_gt[weight > 0].pow(2).mean().sqrt().item()

    print(f'\n  Baseline PSD ratio (pred/targ): {ratio:.4f}  ← ideally ~1.0')
    print(f'  Residual RMSE (normalised):     {rmse_r:.6f}')
    print(f'  This is the target distribution the diffusion UNet must learn.')
    print('\n' + '─' * 60 + '\n')
    model.train()
    model.encoder.eval()
    model.baseline.eval()


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate_phase2(model, loader, device, n_sample_steps=50):
    """
    Reports:
      - baseline RMSE  (frozen, should not change)
      - sample() RMSE  (diffusion path — what we're improving)
      - PSD ratio of sample() output
    """
    model.eval()

    sq_base, sq_samp, n_sum = 0.0, 0.0, 0

    psd_pred_acc, psd_targ_acc, n_psd = 0.0, 0.0, 0

    for batch in loader:
        batch_cpu = batch
        batch = move_batch_to_device(batch, device, non_blocking=False)
        bathy  = batch['bathy']
        sst    = batch['sst'].float()
        weight = batch['weight'].float()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            latent = model._encode(batch)
            x_base = model.baseline(latent, bathy)
            pred_s = model.sample(batch, n_steps=n_sample_steps)

        x_base = x_base.float()
        pred_s = pred_s.float()

        valid = (weight > 0) & torch.isfinite(sst) & torch.isfinite(x_base)

        sq_base += (((x_base - sst)**2) * valid.float()).sum().item()
        sq_samp += (((pred_s  - sst)**2) * valid.float()).sum().item()
        n_sum   += valid.sum().item()

        # PSD of sample output
        def band_power(x, w):
            xw = torch.nan_to_num(x * w, nan=0.0)
            f  = torch.fft.rfft2(xw)
            ky = torch.fft.fftfreq(x.shape[-2], device=x.device).abs()
            kx = torch.fft.rfftfreq(x.shape[-1], device=x.device).abs()
            k2 = ky[:, None]**2 + kx[None, :]**2
            band = (k2 > 0.01) & (k2 < 0.25)
            return (f.abs()**2)[:, :, band].mean().item()

        psd_pred_acc += band_power(pred_s, weight)
        psd_targ_acc += band_power(sst,    weight)
        n_psd        += 1

    rmse_base = (sq_base / max(n_sum, 1)) ** 0.5
    rmse_samp = (sq_samp / max(n_sum, 1)) ** 0.5
    psd_ratio = psd_pred_acc / (psd_targ_acc + 1e-12)

    model.train()
    model.encoder.eval()
    model.baseline.eval()

    return rmse_base, rmse_samp, psd_ratio


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def finetune(stage=1, ckpt_path=None, epochs=None, lr=None, spec_weight=None,
             run_tag='new2try1', seed=None):

    print(f'\n{"═"*60}')
    print(f'  Phase 2 — Diffusion fine-tuning  (Stage {stage})')
    print(f'{"═"*60}\n')

    # ── Config overrides ──────────────────────────────────────────────────
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    FT_EPOCHS      = epochs      or getattr(cfg, 'FT_EPOCHS',      30)
    FT_LR          = lr          or getattr(cfg, 'FT_LR',          5e-5)
    FT_SPEC_WEIGHT = spec_weight or getattr(cfg, 'FT_SPEC_WEIGHT', 0.5)
    FT_GRAD_CLIP   = getattr(cfg, 'FT_GRAD_CLIP',   1.0)
    FT_GRAD_ACCUM  = getattr(cfg, 'GRAD_ACCUM',      1)

    print(f'  run_tag:      {run_tag}')
    print(f'  seed:         {seed}')
    print(f'  LR:           {FT_LR}')
    print(f'  Epochs:       {FT_EPOCHS}')
    print(f'  Spec weight:  {FT_SPEC_WEIGHT}')
    print(f'  Grad clip:    {FT_GRAD_CLIP}\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        torch.cuda.init()  # lock in CUDA runtime before DataLoader pin_memory thread races
        print(f'GPU:    {torch.cuda.get_device_name(0)}')
        print(f'VRAM:   {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = make_loaders(
        batch_size  = cfg.BATCH_SIZE,
        num_workers = cfg.NUM_WORKERS,
        stage       = stage,
    )
    print(f'\nTrain batches: {len(train_loader)}  Val batches: {len(val_loader)}')

    # ── Model ─────────────────────────────────────────────────────────────
    model = TemporalDownscaler(stage=stage).to(device)

    # Load Phase 1 checkpoint
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f'\nLoaded Phase 1 checkpoint: {ckpt_path}')
        print(f'  (Phase 1 epoch: {ckpt.get("epoch", "?")}  '
              f'best_val_rmse: {ckpt.get("best_val_rmse", "?")})')
    else:
        print('\nWARNING: No checkpoint provided — starting from random weights.')
        print('  Pass --ckpt path/to/stage1_best.pt for proper Phase 2 training.\n')

    # ── Freeze encoder + baseline; train diffusion only ───────────────────
    freeze(model.encoder)
    freeze(model.baseline)
    unfreeze(model.diffusion)

    print('\nParameter status:')
    count_params(model.encoder,   'encoder')
    count_params(model.baseline,  'baseline')
    count_params(model.diffusion, 'diffusion  ← training')

    # ── Optimiser — diffusion params only ─────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.diffusion.parameters(),
        lr           = FT_LR,
        weight_decay = cfg.WEIGHT_DECAY,
        betas        = (0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FT_EPOCHS, eta_min=FT_LR * 0.01
    )
    scaler = torch.amp.GradScaler('cuda')

    # ── Resume Phase 2 checkpoint if available ────────────────────────────
    ft_ckpt_path = (f'{cfg.CKPT_DIR}/stage{stage}_phase2_{run_tag}_latest.pt')
    start_epoch  = 0
    best_rmse_s  = float('inf')

    if os.path.exists(ft_ckpt_path):
        ft_ckpt = torch.load(ft_ckpt_path, map_location=device)
        # Only load diffusion weights — encoder/baseline unchanged
        model.diffusion.load_state_dict(ft_ckpt['diffusion'])
        optimizer.load_state_dict(ft_ckpt['optimizer'])
        scheduler.load_state_dict(ft_ckpt['scheduler'])
        start_epoch = ft_ckpt['epoch'] + 1
        best_rmse_s = ft_ckpt.get('best_rmse_sample', float('inf'))
        print(f'\nResumed Phase 2 from epoch {start_epoch}')

    # ── Pre-training diagnostic ───────────────────────────────────────────
    diffusion_diagnostic(model, train_loader, device)

    # ── Training loop ─────────────────────────────────────────────────────
    log = []
    r_std_ema = None
    EMA_DECAY  = 0.99

    for epoch in range(start_epoch, FT_EPOCHS):

        # Keep frozen modules in eval mode every epoch
        model.encoder.eval()
        model.baseline.eval()
        model.diffusion.train()

        t0 = time.time()
        acc_diff, acc_spec, acc_total = 0.0, 0.0, 0.0
        nan_steps = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch_cpu = batch
            batch = move_batch_to_device(batch, device, non_blocking=False)
            sst    = torch.nan_to_num(batch['sst'],    nan=0.0)
            weight = torch.nan_to_num(batch['weight'], nan=0.0)
            bathy  = batch['bathy']
            B      = sst.shape[0]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):

                # Frozen baseline — no grad needed
                with torch.no_grad():
                    latent = model._encode(batch)
                    x_base = model.baseline(latent, bathy)

                # Ground truth residual (stable — baseline is frozen)
                r_gt = (sst - x_base).detach()
                r_std = r_gt[weight > 0].std().clamp(min=1e-4).detach()
                if r_std_ema is None:
                    r_std_ema = r_std.item()
                else:
                    r_std_ema = EMA_DECAY * r_std_ema + (1 - EMA_DECAY) * r_std.item()
                r_gt_n = r_gt / r_std  
                # Sample noise schedule
                t       = torch.randint(0, cfg.DIFF_STEPS, (B,), device=device)
                noise   = torch.randn_like(r_gt_n)
                sqrt_ab_t   = model.sqrt_ab[t].view(B, 1, 1, 1)
                sqrt_1mab_t = model.sqrt_1mab[t].view(B, 1, 1, 1)
                r_noisy = sqrt_ab_t * r_gt_n + sqrt_1mab_t * noise

                t_norm     = t.float() / cfg.DIFF_STEPS
                noise_pred = model.diffusion(
                    r_noisy, x_base, latent, bathy, t_norm
                )

                # ── Loss 1: noise prediction MSE (standard DDPM) ──────────
                def wmean(x):
                    return (x * weight).sum() / (weight.sum() + 1e-6)

                loss_diff = wmean(
                    F.mse_loss(noise_pred, noise, reduction='none')
                )

                # ── Loss 2: spectral loss on x0 estimate ──────────────────
                # x0_hat = (r_noisy - sqrt(1-ab)*eps_pred) / sqrt(ab)
                r0_hat = x0_from_noise(
                    r_noisy, noise_pred, sqrt_ab_t, sqrt_1mab_t
                )
                r0_hat = torch.nan_to_num(r0_hat, nan=0.0, posinf=0.0, neginf=0.0)
                #spec_mask = (t < 200).float().view(B, 1, 1, 1) # try 2, reducing noise stps
                # loss_spec = spectral_loss_weighted(
                #     r0_hat * spec_mask, r_gt * spec_mask, weight

                # )
                loss_spec = spectral_loss_weighted(
                    r0_hat, r_gt, weight
                )
                loss = loss_diff + FT_SPEC_WEIGHT * loss_spec

            # Guard NaN
            if not torch.isfinite(loss):
                nan_steps += 1
                optimizer.zero_grad()
                continue

            scaled = loss / FT_GRAD_ACCUM
            scaler.scale(scaled).backward()

            if (step + 1) % FT_GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.diffusion.parameters(), FT_GRAD_CLIP
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            acc_diff  += loss_diff.item()
            acc_spec  += loss_spec.item()
            acc_total += loss.item()

            if step % 100 == 0:
                lr_now = optimizer.param_groups[0]['lr']
                print(f'  E{epoch:03d} S{step:04d}  '
                      f'diff={loss_diff.item():.5f}  '
                      f'spec={loss_spec.item():.5f}  '
                      f'total={loss.item():.5f}  '
                      f'lr={lr_now:.2e}  '
                      f'nan={nan_steps}')

        # Flush any remaining gradients
        if len(train_loader) % FT_GRAD_ACCUM != 0:
            if scaler._scale is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.diffusion.parameters(), FT_GRAD_CLIP
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        scheduler.step()

        n = max(len(train_loader) - nan_steps, 1)
        avg_diff  = acc_diff  / n
        avg_spec  = acc_spec  / n
        avg_total = acc_total / n

        # ── Validation ────────────────────────────────────────────────────
        rmse_base, rmse_samp, psd_ratio = validate_phase2(
            model, val_loader, device
        )

        elapsed = time.time() - t0
        print(f'\nEpoch {epoch:03d} | '
              f'diff={avg_diff:.5f}  spec={avg_spec:.5f}  total={avg_total:.5f} | '
              f'rmse_base={rmse_base:.5f}  rmse_sample={rmse_samp:.5f}  '
              f'psd_ratio={psd_ratio:.4f} | '
              f'{elapsed:.0f}s\n')

        # ── Log ───────────────────────────────────────────────────────────
        row = {
            'epoch':        epoch,
            'loss_diff':    avg_diff,
            'loss_spec':    avg_spec,
            'loss_total':   avg_total,
            'rmse_base':    rmse_base,
            'rmse_sample':  rmse_samp,
            'psd_ratio':    psd_ratio,
            'nan_skipped':  nan_steps,
        }
        log.append(row)
        with open(f'{cfg.LOG_DIR}/stage{stage}_phase2_{run_tag}_log.json', 'w') as f:
            json.dump(log, f, indent=2)

        # ── Checkpoint — save diffusion weights only ───────────────────────
        ft_ckpt_data = {
            'epoch':             epoch,
            'diffusion':         model.diffusion.state_dict(),
            'optimizer':         optimizer.state_dict(),
            'r_std_estimate':    r_std_ema,
            'scheduler':         scheduler.state_dict(),
            'best_rmse_sample':  best_rmse_s,
            'stage':             stage,
            'phase1_ckpt':       ckpt_path,
        }
        torch.save(ft_ckpt_data, ft_ckpt_path)

        if rmse_samp < best_rmse_s:
            best_rmse_s = rmse_samp
            torch.save(
                ft_ckpt_data,
                f'{cfg.CKPT_DIR}/stage{stage}_phase2_{run_tag}_best.pt'
            )
            print(f'  ★ New best sample RMSE: {rmse_samp:.5f}  '
                  f'PSD ratio: {psd_ratio:.4f}')

        # ── Early stop hint ───────────────────────────────────────────────
        if psd_ratio > 0.5 and psd_ratio < 2.0 and rmse_samp < rmse_base * 1.1:
            print('  ✓ PSD ratio normalised and sample RMSE close to baseline.')
            print('    Consider evaluating with eval.py and moving to Stage 2.')

    print(f'\nPhase 2 complete. Best sample RMSE: {best_rmse_s:.5f}')
    return best_rmse_s


# ─────────────────────────────────────────────────────────────────────────────
# MERGE UTILITY — combine Phase 1 + Phase 2 into a single deployable checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def merge_checkpoints(stage, phase1_ckpt, phase2_ckpt, out_path=None):
    """
    Loads encoder+baseline from Phase 1, diffusion from Phase 2,
    and saves a single merged checkpoint compatible with TemporalDownscaler.

    Usage:
        python finetune_diffusion.py --merge \
            --stage 1 \
            --ckpt checkpoints/stage1_best.pt \
            --phase2_ckpt checkpoints/stage1_phase2_best.pt
    """
    device = torch.device('cpu')
    model  = TemporalDownscaler(stage=stage)

    p1 = torch.load(phase1_ckpt, map_location=device)
    model.load_state_dict(p1['model'])
    print(f'Loaded Phase 1: {phase1_ckpt}')

    p2 = torch.load(phase2_ckpt, map_location=device)
    model.diffusion.load_state_dict(p2['diffusion'])
    print(f'Loaded Phase 2 diffusion: {phase2_ckpt}')

    out_path = out_path or phase1_ckpt.replace('.pt', '_merged.pt')
    merged = {
        'epoch':         p2['epoch'],
        'model':         model.state_dict(),
        'stage':         stage,
        'phase1_source': phase1_ckpt,
        'phase2_source': phase2_ckpt,
    }
    torch.save(merged, out_path)
    print(f'Saved merged checkpoint: {out_path}')
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Phase 2: Frozen-baseline diffusion fine-tuning'
    )
    parser.add_argument('--stage',       type=int,   default=1,
                        help='Encoder stage 1–4')
    parser.add_argument('--ckpt',        type=str,   default=None,
                        help='Path to Phase 1 checkpoint (required)')
    parser.add_argument('--epochs',      type=int,   default=None,
                        help='Override number of fine-tuning epochs')
    parser.add_argument('--lr',          type=float, default=None,
                        help='Override learning rate (default 5e-5)')
    parser.add_argument('--spec_weight', type=float, default=None,
                        help='Weight for spectral loss (default 0.5)')
    parser.add_argument('--run_tag',     type=str,   default='new2try1',
                        help='Tag used in checkpoint/log filenames (default: new2try1)')
    parser.add_argument('--seed',        type=int,   default=None,
                        help='Global random seed for reproducibility')
    parser.add_argument('--merge',       action='store_true',
                        help='Merge Phase 1 + Phase 2 into one checkpoint')
    parser.add_argument('--phase2_ckpt', type=str,   default=None,
                        help='Phase 2 checkpoint path (for --merge)')
    parser.add_argument('--out',         type=str,   default=None,
                        help='Output path for merged checkpoint')

    args = parser.parse_args()

    if args.merge:
        assert args.ckpt and args.phase2_ckpt, \
            '--merge requires both --ckpt (phase1) and --phase2_ckpt'
        merge_checkpoints(
            args.stage, args.ckpt, args.phase2_ckpt, args.out
        )
    else:
        finetune(
            stage       = args.stage,
            ckpt_path   = args.ckpt,
            epochs      = args.epochs,
            lr          = args.lr,
            spec_weight = args.spec_weight,
            run_tag     = args.run_tag,
            seed        = args.seed,
        )

