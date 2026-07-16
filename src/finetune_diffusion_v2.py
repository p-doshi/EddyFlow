"""
finetune_diffusion_v2.py — Phase 2 fine-tuning for Stage 4v2, 5a, 5b.

Strategy mirrors finetune_diffusion.py but adapts for:
  - Stage 4v2: freeze encoder + baseline, fine-tune EDMDiffusion only
                (EDM loss + spectral loss on D_theta x0 estimate)
  - Stage 5a:  freeze encoder + baseline, fine-tune ResidualVAE only
                (L1 reconstruction + KL — same as train_v2.py phase a,
                 but with lower LR for stable convergence from a warm VAE)
  - Stage 5b:  freeze encoder + vae.encoder, fine-tune latent diffusion UNet
                (EDM latent loss + spectral loss decoded to pixel space)

Usage:
  python finetune_diffusion_v2.py --stage 4v2 --ckpt checkpoints/stage4v2_T_28_best.pt
  python finetune_diffusion_v2.py --stage 5a  --ckpt checkpoints/stage5a_T_28_best.pt
  python finetune_diffusion_v2.py --stage 5b  --ckpt checkpoints/stage5b_T_28_best.pt

  # Merge Phase 1 + Phase 2 weights into a single deployable checkpoint:
  python finetune_diffusion_v2.py --merge --stage 4v2 \
      --ckpt checkpoints/stage4v2_T_28_best.pt \
      --phase2_ckpt checkpoints/stage4v2_phase2_best.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import time
import argparse

from config import cfg
from dataset import make_loaders
from model import (
    TemporalDownscalerV2,
    TemporalDownscalerV5,
    TemporalDownscaler,   # for _spectral_loss static method only
)

STAGE_CHOICES = ('4v2', '5a', '5b')


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()

def unfreeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(True)
    module.train()

def move_batch_to_device(batch: dict, device, non_blocking: bool = True):
    return {
        k: v.to(device, non_blocking=non_blocking)
        for k, v in batch.items()
        if torch.is_tensor(v)
    }

def count_params(module: nn.Module, label: str):
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f'  {label:<30s}  total={total/1e6:.2f}M  trainable={trainable/1e6:.2f}M')

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

def band_power(x, w):
    xw = torch.nan_to_num(x * w, nan=0.0)
    f  = torch.fft.rfft2(xw)
    ky = torch.fft.fftfreq(x.shape[-2], device=x.device).abs()
    kx = torch.fft.rfftfreq(x.shape[-1], device=x.device).abs()
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    band = (k2 > 0.01) & (k2 < 0.25)
    return (f.abs()**2)[:, :, band].mean().item()

def apply_freeze(model, stage):
    """
    Apply correct freeze pattern per stage — called at start of every epoch
    so .train() calls don't accidentally unfreeze frozen modules.
    """
    if stage == '4v2':
        freeze(model.encoder)
        freeze(model.baseline)
        unfreeze(model.diffusion)

    elif stage == '5a':
        freeze(model.encoder)
        freeze(model.baseline)
        unfreeze(model.diffusion.vae)
        freeze(model.diffusion.time_mlp)
        freeze(model.diffusion.domain_norm)
        freeze(model.diffusion.cond_merge)
        # latent UNet blocks frozen too — only VAE trains in 5a
        freeze(model.diffusion.enc1)
        freeze(model.diffusion.enc2)
        freeze(model.diffusion.mid)
        freeze(model.diffusion.dec2)
        freeze(model.diffusion.dec1)
        freeze(model.diffusion.out)
        freeze(model.diffusion.cond_proj)

    elif stage == '5b':
        freeze(model.diffusion.vae.encoder)
        unfreeze(model.encoder)
        unfreeze(model.baseline)
        unfreeze(model.diffusion)
        # VAE encoder stays frozen but decoder can fine-tune
        freeze(model.diffusion.vae.encoder)


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def diffusion_diagnostic(model, loader, device, stage):
    print('\n' + '─' * 60)
    print(f'Phase 2 pre-training diagnostic — stage {stage}')
    print('─' * 60)

    # Param status
    enc_grad  = any(p.requires_grad for p in model.encoder.parameters())
    base_grad = any(p.requires_grad for p in model.baseline.parameters())
    diff_grad = any(p.requires_grad for p in model.diffusion.parameters())
    print(f'\n  encoder   trainable: {str(enc_grad):<5}  ← 4v2/5a: False | 5b: True')
    print(f'  baseline  trainable: {str(base_grad):<5}  ← 4v2/5a: False | 5b: True')
    print(f'  diffusion trainable: {str(diff_grad):<5}  ← should be True')

    if stage == '5a':
        vae_enc_grad = any(p.requires_grad for p in model.diffusion.vae.encoder.parameters())
        vae_dec_grad = any(p.requires_grad for p in model.diffusion.vae.decoder.parameters())
        print(f'  vae.encoder trainable: {str(vae_enc_grad):<5}  ← should be True (5a)')
        print(f'  vae.decoder trainable: {str(vae_dec_grad):<5}  ← should be True (5a)')

    elif stage == '5b':
        vae_enc_grad = any(p.requires_grad for p in model.diffusion.vae.encoder.parameters())
        print(f'  vae.encoder trainable: {str(vae_enc_grad):<5}  ← should be False (5b)')

    batch = move_batch_to_device(next(iter(loader)), device, non_blocking=False)
    model.eval()

    latent = model._encode(batch)
    x_base = model.baseline(latent, batch['bathy'])
    sst    = batch['sst']
    weight = batch['weight']
    r_gt   = (sst - x_base)

    psd_pred = band_power(x_base, weight)
    psd_targ = band_power(sst,    weight)
    ratio    = psd_pred / (psd_targ + 1e-12)
    rmse_r   = r_gt[weight > 0].pow(2).mean().sqrt().item()

    print(f'\n  Baseline PSD ratio (pred/targ): {ratio:.4f}  ← ideally ~1.0')
    print(f'  Residual RMSE (normalised):     {rmse_r:.6f}')

    if stage in ('5a', '5b'):
        z_mean, z_logvar = model.diffusion.vae.encode(
            torch.nan_to_num(r_gt, nan=0.0), batch['bathy']
        )
        print(f'  VAE latent mean:  min={z_mean.min():.4f}  max={z_mean.max():.4f}')
        print(f'  VAE latent logvar: min={z_logvar.min():.4f}  max={z_logvar.max():.4f}')

    print('\n' + '─' * 60 + '\n')
    apply_freeze(model, stage)
    model.diffusion.train()


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate_phase2(model, loader, device, stage, n_sample_steps=20):
    """
    Reports baseline RMSE, sample RMSE, and PSD ratio.
    For 5a also reports VAE reconstruction RMSE.
    """
    model.eval()
    sq_base, sq_samp, n_sum            = 0.0, 0.0, 0
    psd_pred_acc, psd_targ_acc, n_psd  = 0.0, 0.0, 0
    sq_vae_rec                         = 0.0   # 5a only

    for batch in loader:
        batch  = move_batch_to_device(batch, device, non_blocking=False)
        bathy  = batch['bathy']
        sst    = batch['sst'].float()
        weight = batch['weight'].float()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            latent = model._encode(batch)
            x_base = model.baseline(latent, bathy).float()

            if stage == '5a':
                r_gt  = torch.nan_to_num(sst - x_base, nan=0.0)
                r_rec, _ = model.diffusion.vae(r_gt, bathy)
                r_rec = r_rec.float()
                sq_vae_rec += (((r_rec - r_gt)**2) *
                               (weight > 0).float()).sum().item()
                pred_s = x_base  # no diffusion sample in 5a val
            else:
                dm     = batch.get('domain_sst_mean',
                         torch.zeros(sst.shape[0], device=device))
                ds     = batch.get('domain_sst_std',
                         torch.ones(sst.shape[0],  device=device))
                pred_s = model.sample(batch, n_steps=n_sample_steps).float()

        valid   = (weight > 0) & torch.isfinite(sst) & torch.isfinite(x_base)
        sq_base += (((x_base - sst)**2) * valid.float()).sum().item()
        sq_samp += (((pred_s  - sst)**2) * valid.float()).sum().item()
        n_sum   += valid.sum().item()

        psd_pred_acc += band_power(pred_s, weight)
        psd_targ_acc += band_power(sst,    weight)
        n_psd        += 1

    rmse_base = (sq_base   / max(n_sum, 1)) ** 0.5
    rmse_samp = (sq_samp   / max(n_sum, 1)) ** 0.5
    rmse_vae  = (sq_vae_rec / max(n_sum, 1)) ** 0.5 if stage == '5a' else None
    psd_ratio = psd_pred_acc / (psd_targ_acc + 1e-12)

    apply_freeze(model, stage)
    model.diffusion.train()
    return rmse_base, rmse_samp, psd_ratio, rmse_vae


# ═══════════════════════════════════════════════════════════════════════════════
# PER-STAGE LOSS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_loss(model, batch, stage, FT_SPEC_WEIGHT, device):
    """
    Returns loss dict with keys: loss, loss_diff, loss_spec, loss_vae.
    Keeps all stage-specific logic in one place.
    """
    sst    = torch.nan_to_num(batch['sst'],    nan=0.0)
    weight = torch.nan_to_num(batch['weight'], nan=0.0)
    bathy  = batch['bathy']
    B      = sst.shape[0]

    def wmean(x):
        return (x * weight).sum() / (weight.sum() + 1e-6)

    # Frozen baseline — always no_grad
    with torch.no_grad():
        latent = model._encode(batch)
        x_base = model.baseline(latent, bathy)

    r_gt = (sst - x_base).detach()

    dm = batch.get('domain_sst_mean', torch.zeros(B, device=device))
    ds = batch.get('domain_sst_std',  torch.ones(B,  device=device))

    # ── Stage 4v2 ─────────────────────────────────────────────────────────────
    if stage == '4v2':
        # EDM training loss
        loss_diff = model.diffusion.forward_train(
            r_gt, x_base, latent.detach(), bathy, dm, ds
        )

        # Spectral loss on D_theta x0 estimate at a mid-noise sigma
        sigma_mid = torch.full((B,), 0.3, device=device)
        noise_mid = torch.randn_like(r_gt)
        r_noisy   = r_gt + 0.3 * noise_mid
        with torch.no_grad():
            pass  # x0 estimate computed below with grad
        D_x = model.diffusion.D_theta(
            r_noisy, sigma_mid, x_base, latent.detach(), bathy, dm, ds
        )
        D_x = torch.nan_to_num(D_x, nan=0.0, posinf=0.0, neginf=0.0)
        loss_spec = spectral_loss_weighted(D_x, r_gt, weight)
        loss_vae  = torch.tensor(0.0, device=device)

    # ── Stage 5a ──────────────────────────────────────────────────────────────
    elif stage == '5a':
        r_rec, kl = model.diffusion.vae(r_gt, bathy)
        loss_vae  = wmean(F.l1_loss(r_rec, r_gt, reduction='none')) + kl
        loss_diff = loss_vae                     # only VAE trains in 5a
        loss_spec = spectral_loss_weighted(r_rec, r_gt, weight)
        loss_vae  = loss_vae.detach()            # log separately

    # ── Stage 5b ──────────────────────────────────────────────────────────────
    elif stage == '5b':
        loss_diff = model.diffusion.forward_train(
            r_gt, x_base, latent.detach(), bathy, dm, ds
        )

        # Spectral loss: decode a latent D_theta estimate back to pixel space
        with torch.no_grad():
            z_mean, _ = model.diffusion.vae.encode(r_gt, bathy)
            lat_hw    = z_mean.shape[-2:]
            lat_cond  = model.diffusion._spatial_cond(
                latent.detach(), x_base, lat_hw
            )
        sigma_mid = torch.full((B,), 0.3, device=device)
        z_noisy   = z_mean + 0.3 * torch.randn_like(z_mean)
        D_z       = model.diffusion.D_theta_latent(
            z_noisy, sigma_mid, lat_cond, dm, ds
        )
        r_decoded = model.diffusion.vae.decode(D_z, bathy)
        r_decoded = torch.nan_to_num(r_decoded, nan=0.0, posinf=0.0, neginf=0.0)
        loss_spec = spectral_loss_weighted(r_decoded, r_gt, weight)
        loss_vae  = torch.tensor(0.0, device=device)

    loss = loss_diff + FT_SPEC_WEIGHT * loss_spec

    return {
        'loss':      loss,
        'loss_diff': loss_diff.item() if torch.is_tensor(loss_diff) else loss_diff,
        'loss_spec': loss_spec.item(),
        'loss_vae':  loss_vae.item()  if torch.is_tensor(loss_vae)  else loss_vae,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def finetune(stage, ckpt_path=None, epochs=None, lr=None, spec_weight=None):
    assert stage in STAGE_CHOICES, \
        f'stage must be one of {STAGE_CHOICES}, got {stage}'

    print(f'\n{"═"*60}')
    print(f'  Phase 2 diffusion fine-tuning — stage {stage}')
    print(f'{"═"*60}\n')

    FT_EPOCHS      = epochs      or getattr(cfg, 'FT_EPOCHS',      30)
    FT_LR          = lr          or getattr(cfg, 'FT_LR',          5e-5)
    FT_SPEC_WEIGHT = spec_weight or getattr(cfg, 'FT_SPEC_WEIGHT',  0.5)
    FT_GRAD_CLIP   = getattr(cfg, 'FT_GRAD_CLIP',  1.0)
    FT_GRAD_ACCUM  = getattr(cfg, 'GRAD_ACCUM',    1)

    print(f'  LR:           {FT_LR}')
    print(f'  Epochs:       {FT_EPOCHS}')
    print(f'  Spec weight:  {FT_SPEC_WEIGHT}')
    print(f'  Grad clip:    {FT_GRAD_CLIP}\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU:    {torch.cuda.get_device_name(0)}')
        print(f'VRAM:   {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = make_loaders(
        batch_size  = cfg.BATCH_SIZE,
        num_workers = cfg.NUM_WORKERS,
        stage       = 4,   # all v2/v5 stages use stage 4 pipeline (has mur_seq)
    )
    print(f'Train batches: {len(train_loader)}  Val batches: {len(val_loader)}')

    # ── Model ──────────────────────────────────────────────────────────────────
    if stage == '4v2':
        model = TemporalDownscalerV2().to(device)
    else:
        model = TemporalDownscalerV5().to(device)

    # Load source checkpoint
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        print(f'\nLoaded checkpoint: {ckpt_path}')
        print(f'  epoch={ckpt.get("epoch","?")}  '
              f'best_val_rmse={ckpt.get("best_val_rmse","?")}')
    else:
        print('\nWARNING: No checkpoint provided — training from random weights.')

    # ── Freeze pattern ────────────────────────────────────────────────────────
    apply_freeze(model, stage)

    print('\nParameter status:')
    count_params(model.encoder,   'encoder')
    count_params(model.baseline,  'baseline')
    count_params(model.diffusion, 'diffusion')
    if stage in ('5a', '5b'):
        count_params(model.diffusion.vae,        'diffusion.vae')
        count_params(model.diffusion.vae.encoder,'diffusion.vae.encoder')
        count_params(model.diffusion.vae.decoder,'diffusion.vae.decoder')

    # ── Optimiser — trainable params only ─────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = FT_LR,
        weight_decay = cfg.WEIGHT_DECAY,
        betas        = (0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FT_EPOCHS, eta_min=FT_LR * 0.01
    )
    scaler = torch.amp.GradScaler('cuda')

    # ── Resume Phase 2 checkpoint ─────────────────────────────────────────────
    ft_ckpt_path = f'{cfg.CKPT_DIR}/stage{stage}_phase2_try2_latest.pt'
    start_epoch  = 0
    best_rmse_s  = float('inf')

    if os.path.exists(ft_ckpt_path):
        ft = torch.load(ft_ckpt_path, map_location=device)
        # Load only the trainable component weights
        if stage == '4v2':
            model.diffusion.load_state_dict(ft['diffusion'])
        elif stage == '5a':
            model.diffusion.vae.load_state_dict(ft['vae'])
        elif stage == '5b':
            model.diffusion.load_state_dict(ft['diffusion'])
            model.encoder.load_state_dict(ft.get('encoder', model.encoder.state_dict()))
            model.baseline.load_state_dict(ft.get('baseline', model.baseline.state_dict()))
        optimizer.load_state_dict(ft['optimizer'])
        scheduler.load_state_dict(ft['scheduler'])
        start_epoch = ft['epoch'] + 1
        best_rmse_s = ft.get('best_rmse_sample', float('inf'))
        print(f'\nResumed Phase 2 from epoch {start_epoch}  '
              f'(best sample RMSE: {best_rmse_s:.5f})')

    # ── Pre-training diagnostic ───────────────────────────────────────────────
    diffusion_diagnostic(model, train_loader, device, stage)

    # ── Training loop ─────────────────────────────────────────────────────────
    log = []

    for epoch in range(start_epoch, FT_EPOCHS):
        apply_freeze(model, stage)
        model.diffusion.train()
        t0 = time.time()

        acc = {'loss': 0.0, 'loss_diff': 0.0, 'loss_spec': 0.0, 'loss_vae': 0.0}
        nan_steps = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device, non_blocking=False)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                losses = compute_loss(
                    model, batch, stage, FT_SPEC_WEIGHT, device
                )

            if not torch.isfinite(losses['loss']):
                nan_steps += 1
                if nan_steps <= 3:
                    print(f'  WARNING: NaN loss at E{epoch} S{step}')
                optimizer.zero_grad()
                continue

            scaled = losses['loss'] / FT_GRAD_ACCUM
            scaler.scale(scaled).backward()

            if (step + 1) % FT_GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    FT_GRAD_CLIP
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            for k in acc:
                v = losses.get(k, 0.0)
                acc[k] += v if isinstance(v, float) else v.item()

            if step % 100 == 0:
                lr_now  = optimizer.param_groups[0]['lr']
                vae_str = (f'  vae={losses["loss_vae"]:.5f}'
                           if stage in ('5a', '5b') else '')
                print(f'  E{epoch:03d} S{step:04d}  '
                      f'diff={losses["loss_diff"]:.5f}  '
                      f'spec={losses["loss_spec"]:.5f}  '
                      f'total={losses["loss"].item():.5f}'
                      f'{vae_str}  '
                      f'lr={lr_now:.2e}  nan={nan_steps}')

        # Flush remaining
        if len(train_loader) % FT_GRAD_ACCUM != 0:
            if scaler._scale is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    FT_GRAD_CLIP
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        scheduler.step()

        n = max(len(train_loader) - nan_steps, 1)
        for k in acc:
            acc[k] /= n

        if nan_steps > 0:
            print(f'  WARNING: {nan_steps}/{len(train_loader)} steps '
                  f'skipped due to NaN this epoch')

        # ── Validation ────────────────────────────────────────────────────────
        rmse_base, rmse_samp, psd_ratio, rmse_vae = validate_phase2(
            model, val_loader, device, stage
        )

        elapsed = time.time() - t0
        vae_str = (f'  rmse_vae={rmse_vae:.5f}' if rmse_vae is not None else '')
        print(f'\nEpoch {epoch:03d} | '
              f'diff={acc["loss_diff"]:.5f}  spec={acc["loss_spec"]:.5f}  '
              f'total={acc["loss"]:.5f} | '
              f'rmse_base={rmse_base:.5f}  rmse_sample={rmse_samp:.5f}  '
              f'psd_ratio={psd_ratio:.4f}{vae_str} | {elapsed:.0f}s\n')

        # ── Log ───────────────────────────────────────────────────────────────
        row = {
            'epoch':        epoch,
            'loss_diff':    acc['loss_diff'],
            'loss_spec':    acc['loss_spec'],
            'loss_vae':     acc['loss_vae'],
            'loss_total':   acc['loss'],
            'rmse_base':    rmse_base,
            'rmse_sample':  rmse_samp,
            'psd_ratio':    psd_ratio,
            'rmse_vae':     rmse_vae,
            'nan_skipped':  nan_steps,
        }
        log.append(row)
        with open(f'{cfg.LOG_DIR}/stage{stage}_phase2_try2_log.json', 'w') as f:
            json.dump(log, f, indent=2)

        # ── Checkpoint — save only trainable component weights ────────────────
        ft_ckpt_data = {
            'epoch':            epoch,
            'optimizer':        optimizer.state_dict(),
            'scheduler':        scheduler.state_dict(),
            'best_rmse_sample': best_rmse_s,
            'stage':            stage,
            'source_ckpt':      ckpt_path,
        }
        if stage == '4v2':
            ft_ckpt_data['diffusion'] = model.diffusion.state_dict()
        elif stage == '5a':
            ft_ckpt_data['vae']       = model.diffusion.vae.state_dict()
        elif stage == '5b':
            ft_ckpt_data['diffusion'] = model.diffusion.state_dict()
            ft_ckpt_data['encoder']   = model.encoder.state_dict()
            ft_ckpt_data['baseline']  = model.baseline.state_dict()

        torch.save(ft_ckpt_data, ft_ckpt_path)

        if rmse_samp < best_rmse_s:
            best_rmse_s = rmse_samp
            torch.save(ft_ckpt_data,
                f'{cfg.CKPT_DIR}/stage{stage}_phase2_try2_best.pt')
            print(f'  ★ New best sample RMSE: {rmse_samp:.5f}  '
                  f'PSD ratio: {psd_ratio:.4f}')

        if psd_ratio > 0.5 and psd_ratio < 2.0 and rmse_samp < rmse_base * 1.1:
            print('  ✓ PSD ratio normalised and sample RMSE near baseline.')
            print('    Consider moving to evaluation.')

    print(f'\nPhase 2 complete. Best sample RMSE: {best_rmse_s:.5f}')
    return best_rmse_s


# ═══════════════════════════════════════════════════════════════════════════════
# MERGE UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

def merge_checkpoints(stage, base_ckpt, phase2_ckpt, out_path=None):
    """
    Merges base checkpoint weights with Phase 2 fine-tuned weights
    into a single deployable checkpoint.

    4v2: base encoder+baseline + phase2 diffusion
    5a:  base encoder+baseline+diffusion + phase2 vae
    5b:  phase2 encoder+baseline+diffusion (all updated in 5b)
    """
    device = torch.device('cpu')

    if stage == '4v2':
        model = TemporalDownscalerV2()
    else:
        model = TemporalDownscalerV5()

    p1 = torch.load(base_ckpt,   map_location=device)
    p2 = torch.load(phase2_ckpt, map_location=device)

    model.load_state_dict(p1['model'])
    print(f'Loaded base checkpoint: {base_ckpt}')

    if stage == '4v2':
        model.diffusion.load_state_dict(p2['diffusion'])
        print(f'Merged phase2 diffusion from: {phase2_ckpt}')
    elif stage == '5a':
        model.diffusion.vae.load_state_dict(p2['vae'])
        print(f'Merged phase2 VAE from: {phase2_ckpt}')
    elif stage == '5b':
        model.diffusion.load_state_dict(p2['diffusion'])
        model.encoder.load_state_dict(p2['encoder'])
        model.baseline.load_state_dict(p2['baseline'])
        print(f'Merged phase2 encoder+baseline+diffusion from: {phase2_ckpt}')

    out_path = out_path or base_ckpt.replace('.pt', '_merged.pt')
    merged = {
        'epoch':          p2['epoch'],
        'model':          model.state_dict(),
        'stage':          stage,
        'base_source':    base_ckpt,
        'phase2_source':  phase2_ckpt,
    }
    torch.save(merged, out_path)
    print(f'Saved merged checkpoint: {out_path}')
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Phase 2 diffusion fine-tuning for Stage 4v2, 5a, 5b'
    )
    parser.add_argument('--stage',       type=str,   required=True,
                        choices=STAGE_CHOICES,
                        help='4v2 | 5a | 5b')
    parser.add_argument('--ckpt',        type=str,   default=None,
                        help='Path to source checkpoint')
    parser.add_argument('--epochs',      type=int,   default=None)
    parser.add_argument('--lr',          type=float, default=None)
    parser.add_argument('--spec_weight', type=float, default=None)
    parser.add_argument('--merge',       action='store_true',
                        help='Merge base + phase2 into one checkpoint')
    parser.add_argument('--phase2_ckpt', type=str,   default=None,
                        help='Phase 2 checkpoint (for --merge)')
    parser.add_argument('--out',         type=str,   default=None,
                        help='Output path for merged checkpoint')
    args = parser.parse_args()

    if args.merge:
        assert args.ckpt and args.phase2_ckpt, \
            '--merge requires both --ckpt (base) and --phase2_ckpt'
        merge_checkpoints(args.stage, args.ckpt, args.phase2_ckpt, args.out)
    else:
        finetune(
            stage       = args.stage,
            ckpt_path   = args.ckpt,
            epochs      = args.epochs,
            lr          = args.lr,
            spec_weight = args.spec_weight,
        )