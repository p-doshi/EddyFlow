"""
finetune_sweep_v2.py — Phase 2 diffusion finetuning for stages 4v2 and 5a.

Sweep-aware replacement for finetune_diffusion_v2.py: adds --run_tag and
--seed args so each sweep run writes its own checkpoint and log files.

Usage (finetune):
    python finetune_sweep_v2.py --stage 4v2 \
        --ckpt ../outputs/checkpoints/stage4v2_T_28_sweep_4v2_42_lr0_best.pt \
        --run_tag sweep_4v2_42_lr0_sw1 \
        --spec_weight 5.0 --lr 1e-5 --seed 42 --epochs 40

Usage (merge phase1 + phase2):
    python finetune_sweep_v2.py --merge --stage 4v2 \
        --ckpt ../outputs/checkpoints/stage4v2_T_28_sweep_4v2_42_lr0_best.pt \
        --phase2_ckpt ../outputs/checkpoints/stage4v2_phase2_sweep_4v2_42_lr0_sw1_best.pt \
        --out ../outputs/checkpoints/stage4v2_sweep_4v2_42_lr0_sw1_merged.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import time
import argparse
import random

from config import cfg
from dataset import make_loaders
from model import TemporalDownscalerV2, TemporalDownscalerV5, TemporalDownscaler

STAGE_CHOICES = ('4v2', '5a')


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch, device, non_blocking=True):
    return {k: v.to(device, non_blocking=non_blocking)
            for k, v in batch.items() if torch.is_tensor(v)}


def freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


def spectral_loss_weighted(pred, target):
    """Spectral loss weighted by sqrt(1 + 20*k^2) — emphasises mesoscale."""
    pf = torch.fft.rfft2(pred)
    tf = torch.fft.rfft2(target)
    ky = torch.fft.fftfreq(pred.shape[-2], device=pred.device).abs()
    kx = torch.fft.rfftfreq(pred.shape[-1], device=pred.device).abs()
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    w  = (1.0 + 20.0 * k2).sqrt()
    return (w * (pf.abs() - tf.abs()).abs()).mean()


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate_sample(model, loader, device, stage, n_steps=10):
    """RMSE using diffusion samples from the validation set."""
    model.eval()
    sq_sum, n_sum = 0.0, 0

    for batch in loader:
        batch = move_batch(batch, device, non_blocking=False)

        if stage == '4v2':
            pred = model.sample(batch, n_steps=n_steps)
        else:
            # 5a: encoder+baseline frozen; use VAE reconstruction
            with torch.no_grad():
                latent = model._encode(batch)
                x_base = model.baseline(latent, batch['bathy'])
            r_gt  = torch.nan_to_num(batch['sst'] - x_base, nan=0.0)
            r_rec, _ = model.diffusion.vae(r_gt, batch['bathy'])
            pred  = x_base + r_rec

        sst    = batch['sst'].float()
        weight = batch['weight'].float()
        valid  = (weight > 0) & torch.isfinite(sst) & torch.isfinite(pred)
        sq_sum += ((pred - sst) ** 2 * valid.float()).sum().item()
        n_sum  += valid.sum().item()

    model.train()
    # Re-freeze after .train() resets eval mode
    freeze(model.encoder)
    freeze(model.baseline)
    return (sq_sum / max(n_sum, 1)) ** 0.5


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FINETUNE
# ─────────────────────────────────────────────────────────────────────────────

def finetune(stage: str, ckpt_path: str, run_tag: str,
             spec_weight: float = 5.0, lr: float = 1e-5,
             seed: int = None, epochs: int = 40):

    if seed is not None:
        set_seed(seed)

    print(f'\n{"═"*60}')
    print(f'  Phase 2 — {stage} diffusion finetune')
    print(f'  run_tag={run_tag}  spec_weight={spec_weight}  lr={lr}  seed={seed}  epochs={epochs}')
    print(f'{"═"*60}\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.init()  # lock in CUDA runtime before DataLoader pin_memory thread races
        print(f'GPU:  {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

    train_loader, val_loader, _ = make_loaders(
        batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, stage=4
    )
    print(f'Train batches: {len(train_loader)}  Val batches: {len(val_loader)}\n')

    # ── Model ─────────────────────────────────────────────────────────────
    if stage == '4v2':
        model = TemporalDownscalerV2().to(device)
    else:
        model = TemporalDownscalerV5().to(device)

    print(f'Loading phase-1 checkpoint: {ckpt_path}')
    p1 = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(p1['model'])

    # Freeze encoder + baseline; only diffusion trains
    freeze(model.encoder)
    freeze(model.baseline)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params (diffusion only): {n_trainable/1e6:.2f}M\n')

    # ── Optimiser + scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    scaler = torch.amp.GradScaler('cuda')

    # ── Resume from phase-2 checkpoint if available ───────────────────────
    phase2_latest = f'{cfg.CKPT_DIR}/stage{stage}_phase2_{run_tag}_latest.pt'
    start_epoch   = 0
    best_rmse     = float('inf')

    if os.path.exists(phase2_latest):
        p2 = torch.load(phase2_latest, map_location=device)
        sd = model.state_dict()
        for k, v in p2['model'].items():
            if k.startswith('diffusion.') and k in sd and sd[k].shape == v.shape:
                sd[k] = v
        model.load_state_dict(sd)
        optimizer.load_state_dict(p2['optimizer'])
        scheduler.load_state_dict(p2['scheduler'])
        start_epoch = p2['epoch'] + 1
        best_rmse   = p2.get('best_sample_rmse', float('inf'))
        print(f'Resumed from phase-2 epoch {start_epoch}')

    # ── Training loop ─────────────────────────────────────────────────────
    log = []

    for epoch in range(start_epoch, epochs):
        model.diffusion.train()
        t0 = time.time()
        tot_diff, tot_spec, tot_loss = 0.0, 0.0, 0.0
        nan_steps = 0

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = move_batch(batch, device, non_blocking=False)

            sst    = torch.nan_to_num(batch['sst'],    nan=0.0, posinf=0.0, neginf=0.0)
            weight = torch.nan_to_num(batch['weight'], nan=0.0, posinf=0.0, neginf=0.0)
            bathy  = batch['bathy']
            B      = sst.shape[0]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Frozen encoder + baseline
                with torch.no_grad():
                    latent = model._encode(batch)
                    x_base = model.baseline(latent, bathy)

                r_gt = (sst - x_base).detach()
                dm   = batch.get('domain_sst_mean', torch.zeros(B, device=device))
                ds   = batch.get('domain_sst_std',  torch.ones(B,  device=device))

                if stage == '4v2':
                    # EDM loss + spectral loss on D_theta x0 estimate
                    ln_sigma = (torch.randn(B, device=device) * model.diffusion.P_std
                                + model.diffusion.P_mean)
                    sigma = ln_sigma.exp().clamp(model.diffusion.sigma_min,
                                                 model.diffusion.sigma_max)
                    s     = sigma.view(B, 1, 1, 1)
                    noise    = torch.randn_like(r_gt)
                    r_noisy  = r_gt + s * noise
                    D_x      = model.diffusion.D_theta(
                        r_noisy, sigma, x_base.detach(), latent.detach(),
                        bathy, dm, ds
                    )
                    lam      = ((sigma**2 + model.diffusion.sigma_data**2)
                                / (sigma * model.diffusion.sigma_data)**2)
                    lam      = lam.view(B, 1, 1, 1)
                    loss_diff = (lam * (D_x - r_gt)**2).mean()

                    pred_m   = torch.nan_to_num((x_base + D_x) * weight, nan=0.0)
                    targ_m   = torch.nan_to_num(sst * weight,             nan=0.0)
                    loss_spec = spectral_loss_weighted(pred_m, targ_m)

                else:
                    # 5a: VAE loss + spectral on reconstruction
                    r_rec, kl = model.diffusion.vae(r_gt, bathy)
                    loss_diff = (F.l1_loss(r_rec, r_gt) + 1e-4 * kl)

                    pred_m   = torch.nan_to_num((x_base + r_rec) * weight, nan=0.0)
                    targ_m   = torch.nan_to_num(sst * weight,               nan=0.0)
                    loss_spec = spectral_loss_weighted(pred_m, targ_m)

                loss = loss_diff + spec_weight * loss_spec

            if not torch.isfinite(loss):
                nan_steps += 1
                if nan_steps <= 3:
                    print(f'  WARNING: NaN loss at E{epoch} S{step}')
                optimizer.zero_grad()
                continue

            scaler.scale(loss / cfg.GRAD_ACCUM).backward()

            if (step + 1) % cfg.GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    cfg.GRAD_CLIP
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            tot_diff += loss_diff.item()
            tot_spec += loss_spec.item()
            tot_loss += loss.item()

            if step % 100 == 0:
                print(f'  E{epoch:03d} S{step:04d}  '
                      f'loss={loss.item():.4f}  '
                      f'diff={loss_diff.item():.4f}  '
                      f'spec={loss_spec.item():.4f}  '
                      f'nan={nan_steps}')

        # Flush remaining grad accum buffer
        if len(train_loader) % cfg.GRAD_ACCUM != 0:
            if scaler._scale is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    cfg.GRAD_CLIP
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        scheduler.step()

        n = max(len(train_loader) - nan_steps, 1)
        sample_rmse = validate_sample(model, val_loader, device, stage, n_steps=10)
        elapsed     = time.time() - t0

        print(f'\nEpoch {epoch:03d} | '
              f'loss={tot_loss/n:.4f}  diff={tot_diff/n:.4f}  spec={tot_spec/n:.4f} | '
              f'sample_rmse={sample_rmse:.4f} | {elapsed:.0f}s\n')

        row = {
            'epoch':       epoch,
            'loss':        tot_loss / n,
            'loss_diff':   tot_diff / n,
            'loss_spec':   tot_spec / n,
            'sample_rmse': sample_rmse,
            'nan_steps':   nan_steps,
        }
        log.append(row)
        log_path = f'{cfg.LOG_DIR}/stage{stage}_phase2_{run_tag}_log.json'
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2)

        save_ckpt = {
            'epoch':            epoch,
            'model':            model.state_dict(),
            'optimizer':        optimizer.state_dict(),
            'scheduler':        scheduler.state_dict(),
            'best_sample_rmse': best_rmse,
            'run_tag':          run_tag,
            'spec_weight':      spec_weight,
        }
        torch.save(save_ckpt, phase2_latest)

        if sample_rmse < best_rmse:
            best_rmse = sample_rmse
            torch.save(save_ckpt,
                       f'{cfg.CKPT_DIR}/stage{stage}_phase2_{run_tag}_best.pt')
            print(f'  ★ New best sample RMSE: {sample_rmse:.4f}')

    print(f'\nFinetuning complete. Best sample RMSE: {best_rmse:.4f}')


# ─────────────────────────────────────────────────────────────────────────────
# MERGE
# ─────────────────────────────────────────────────────────────────────────────

def merge_checkpoints(stage: str, phase1_ckpt: str, phase2_ckpt: str,
                      out_path: str = None) -> str:
    """Merge phase-1 encoder+baseline with phase-2 diffusion weights."""
    print(f'Merging phase-1 + phase-2 for stage {stage}')
    print(f'  Phase 1: {phase1_ckpt}')
    print(f'  Phase 2: {phase2_ckpt}')

    p1 = torch.load(phase1_ckpt, map_location='cpu')
    p2 = torch.load(phase2_ckpt, map_location='cpu')

    merged_sd = {k: v.clone() for k, v in p1['model'].items()}
    overwritten = 0
    for k, v in p2['model'].items():
        if k.startswith('diffusion.'):
            merged_sd[k] = v
            overwritten += 1

    print(f'  Overwrote {overwritten} diffusion keys from phase-2')

    if out_path is None:
        out_path = (phase1_ckpt
                    .replace('_best.pt',   '_merged.pt')
                    .replace('_latest.pt', '_merged.pt'))

    torch.save({'model': merged_sd, 'stage': stage}, out_path)
    print(f'  Saved merged checkpoint: {out_path}')
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Phase 2 EDM diffusion finetuning for stages 4v2 and 5a'
    )
    parser.add_argument('--stage',       required=True, choices=STAGE_CHOICES)
    parser.add_argument('--ckpt',        required=True,
                        help='Phase-1 checkpoint path')
    parser.add_argument('--run_tag',     required=True,
                        help='Tag for checkpoint/log filenames')
    parser.add_argument('--spec_weight', type=float, default=5.0,
                        help='Spectral loss weight (default: 5.0)')
    parser.add_argument('--lr',          type=float, default=1e-5,
                        help='Finetune learning rate (default: 1e-5)')
    parser.add_argument('--seed',        type=int,   default=None,
                        help='Global random seed')
    parser.add_argument('--epochs',      type=int,   default=40,
                        help='Number of finetune epochs (default: 40)')
    parser.add_argument('--merge',       action='store_true',
                        help='Merge phase-1 + phase-2 into one checkpoint')
    parser.add_argument('--phase2_ckpt', type=str,   default=None,
                        help='Phase-2 checkpoint for --merge')
    parser.add_argument('--out',         type=str,   default=None,
                        help='Output path for merged checkpoint')
    args = parser.parse_args()

    if args.merge:
        assert args.phase2_ckpt, '--merge requires --phase2_ckpt'
        merge_checkpoints(args.stage, args.ckpt, args.phase2_ckpt, args.out)
    else:
        finetune(
            stage       = args.stage,
            ckpt_path   = args.ckpt,
            run_tag     = args.run_tag,
            spec_weight = args.spec_weight,
            lr          = args.lr,
            seed        = args.seed,
            epochs      = args.epochs,
        )
