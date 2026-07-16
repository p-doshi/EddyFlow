import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import time
from config import cfg
from dataset import make_loaders
from model import TemporalDownscalerV2, TemporalDownscalerV5, TemporalDownscaler


STAGE_CHOICES = ('4v2', '5a', '5b')


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def move_batch_to_device(batch: dict, device, non_blocking: bool = True):
    return {
        k: v.to(device, non_blocking=non_blocking)
        for k, v in batch.items()
        if torch.is_tensor(v)
    }


def tensor_info(name, t):
    finite = torch.isfinite(t) if torch.is_floating_point(t) else torch.ones_like(t, dtype=torch.bool)
    n_nan  = (~finite).sum().item()
    vals   = t[finite] if finite.numel() > 0 else t
    if vals.numel() > 0 and torch.is_floating_point(vals):
        print(f'  {name:<22s} shape={tuple(t.shape)}  '
              f'NaN={n_nan}  '
              f'min={vals.min().item():10.4f}  '
              f'max={vals.max().item():10.4f}  '
              f'mean={vals.mean().item():10.4f}')
    else:
        print(f'  {name:<22s} shape={tuple(t.shape)}  '
              f'dtype={t.dtype}  NaN={n_nan}')


# ═══════════════════════════════════════════════════════════════════════════════
# NaN DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════════

def nan_diagnostic(model, train_loader, device, stage):
    print('\n' + '─' * 60)
    print(f'NaN diagnostic — stage {stage}')
    print('─' * 60)
    
    batch = move_batch_to_device(next(iter(train_loader)), device, non_blocking=False)
    required = ['era5', 'sst', 'weight', 'bathy', 'mur_seq']
    for key in required:
        if key not in batch:
            raise KeyError(f'missing required batch field: {key}')
    model.eval()
    print('\n[1] Batch tensors:')
    for k, v in batch.items():
        tensor_info(k, v)

    with torch.no_grad():

        # Encoder
        print('\n[2] Encoder:')
        try:
            latent = model._encode(batch)
            tensor_info('latent', latent)
        except Exception as e:
            print(f'  ENCODER FAILED: {e}')
            model.train()
            return

        # Baseline
        print('\n[3] Baseline decoder:')
        try:
            x_base = model.baseline(latent, batch['bathy'])
            tensor_info('x_base', x_base)
        except Exception as e:
            print(f'  BASELINE FAILED: {e}')
            model.train()
            return

        # Residual
        print('\n[4] Residual (sst - x_base):')
        sst  = torch.nan_to_num(batch['sst'], nan=0.0)
        r_gt = sst - x_base
        tensor_info('r_gt', r_gt)

        # Diffusion component
        print('\n[5] Diffusion forward:')
        try:
            B = sst.shape[0]

            if stage == '4v2':
                # EDM: sample a low-noise sigma for diagnostic
                sigma = torch.full((B,), 0.1, device=device)
                dm    = batch.get('domain_sst_mean', torch.zeros(B, device=device))
                ds    = batch.get('domain_sst_std',  torch.ones(B,  device=device))
                D_x   = model.diffusion.D_theta(
                    r_gt + 0.1 * torch.randn_like(r_gt),
                    sigma, x_base, latent, batch['bathy'], dm, ds
                )
                tensor_info('D_x (denoised)', D_x)

            elif stage in ('5a',):
                # VAE encode + decode
                r_rec, kl = model.diffusion.vae(r_gt, batch['bathy'])
                tensor_info('r_rec (VAE)', r_rec)
                print(f'  kl_loss              = {kl.item():.6f}')

            elif stage == '5b':
                # Latent diffusion D_theta in VAE latent space
                z_mean, z_logvar = model.diffusion.vae.encode(r_gt, batch['bathy'])
                tensor_info('z_mean', z_mean)
                tensor_info('z_logvar', z_logvar)

        except Exception as e:
            print(f'  DIFFUSION FAILED: {e}')
            model.train()
            return

        # Losses
        print('\n[6] Losses:')
        weight = batch['weight']
        tensor_info('weight', weight)
        def wmean(x):
            return (x * weight).sum() / (weight.sum() + 1e-6)

        loss_base = wmean(F.l1_loss(x_base, sst, reduction='none'))
        print(f'  loss_base            = {loss_base.item():.6f}')

        pred_m    = torch.nan_to_num(x_base * weight, nan=0.0)
        targ_m    = torch.nan_to_num(sst    * weight, nan=0.0)
        loss_spec = TemporalDownscaler._spectral_loss(pred_m, targ_m)
        print(f'  loss_spec            = {loss_spec.item():.6f}')

    print('\n' + '─' * 60)
    print('Diagnostic complete.')
    print('─' * 60 + '\n')
    model.train()


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, loader, device, stage):
    """
    RMSE in normalised SST units over weighted (ice-free, high-confidence) pixels.
    For 5a: measures baseline + VAE reconstruction so the metric actually reflects
    what the VAE is learning (the frozen baseline alone never changes).
    """
    model.eval()
    sq_sum, n_sum = 0.0, 0

    for batch in loader:
        batch = move_batch_to_device(batch, device, non_blocking=False)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            latent = model._encode(batch)
            x_base = model.baseline(latent, batch['bathy'])

            if stage == '5a':
                sst_nan = torch.nan_to_num(batch['sst'], nan=0.0)
                r_gt    = sst_nan - x_base
                r_rec, _ = model.diffusion.vae(r_gt, batch['bathy'])
                pred    = x_base + r_rec
            else:
                pred = x_base

        pred   = pred.float()
        sst    = batch['sst'].float()
        weight = batch['weight'].float()

        valid   = (weight > 0) & torch.isfinite(sst) & torch.isfinite(pred)
        sq      = ((pred - sst) ** 2) * valid.float()
        sq_sum += sq.sum().item()
        n_sum  += valid.sum().item()

    model.train()
    return (sq_sum / max(n_sum, 1)) ** 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# FREEZE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)

def unfreeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(True)


def configure_frozen_params(model, stage):
    """
    5a: train VAE only — freeze encoder + baseline.
    5b: train latent diffusion — freeze vae.encoder only.
    4v2: all params trainable.
    """
    if stage == '5a':
        freeze(model.encoder)
        freeze(model.baseline)
        unfreeze(model.diffusion.vae)

    elif stage == '5b':
        # Encoder + baseline train alongside latent diffusion
        # VAE encoder frozen — only decoder can fine-tune if desired
        freeze(model.diffusion.vae.encoder)
        unfreeze(model.encoder)
        unfreeze(model.baseline)
        unfreeze(model.diffusion)   # UNet + domain norm + time mlp

    # 4v2: nothing frozen


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def train(stage: str, run_tag: str = 'try2', lr: float = None,
          seed: int = None, epochs: int = None):
    assert stage in STAGE_CHOICES, \
        f'stage must be one of {STAGE_CHOICES}, got {stage}'

    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    _lr     = lr     if lr     is not None else cfg.LR
    _epochs = epochs if epochs is not None else cfg.EPOCHS

    print(f'\n{"═"*60}')
    print(f'  Stage {stage} training — GSL temporal downscaler')
    print(f'  run_tag={run_tag}  lr={_lr}  seed={seed}  epochs={_epochs}')
    print(f'{"═"*60}\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU:    {torch.cuda.get_device_name(0)}')
        print(f'VRAM:   {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

    # ── Data ──────────────────────────────────────────────────────────────────
    # All v2/v5 stages use the Stage 4 data pipeline (has mur_seq)
    train_loader, val_loader, _ = make_loaders(
        batch_size  = cfg.BATCH_SIZE,
        num_workers = cfg.NUM_WORKERS,
        stage       = 4,
    )
    print(f'Train batches: {len(train_loader)}  Val batches: {len(val_loader)}')

    # ── Model ──────────────────────────────────────────────────────────────────
    if stage == '4v2':
        model = TemporalDownscalerV2().to(device)
    else:
        model = TemporalDownscalerV5().to(device)

    configure_frozen_params(model, stage)

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = count_trainable(model)
    print(f'Parameters: {n_total/1e6:.2f}M total | {n_trainable/1e6:.2f}M trainable')

    # ── Load Stage 4 encoder + baseline weights if starting v2 or 5x ─────────
    # Warm-start from best Stage 4 checkpoint so we don't train encoder from scratch
        # ── Warm-start shared Stage 4 weights only ─────────────────────────
    # Reuse encoder + baseline weights where names/shapes still match.
    # New ocean-coordinate / RoPE modules remain randomly initialized.
    stage4_ckpt = f'{cfg.CKPT_DIR}/stage4_new2_T_{cfg.T}_best.pt'
    if os.path.exists(stage4_ckpt):
        s4 = torch.load(stage4_ckpt, map_location=device)
        s4_sd = s4['model']

        model_sd = model.state_dict()
        loaded, skipped, new_keys = [], [], []

        for k, v in model_sd.items():
            if k in s4_sd and s4_sd[k].shape == v.shape:
                model_sd[k] = s4_sd[k]
                loaded.append(k)
            else:
                new_keys.append(k)

        for k in s4_sd.keys():
            if k not in model_sd or s4_sd[k].shape != model_sd.get(k, torch.empty(0)).shape:
                skipped.append(k)

        model.load_state_dict(model_sd, strict=False)
        print(f'Warm-started from Stage 4 ckpt: '
              f'{len(loaded)} keys loaded, {len(new_keys)} new keys, {len(skipped)} skipped')
    else:
        print('No Stage 4 checkpoint found — training from scratch')

    # ── Optimiser + scheduler ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = _lr,
        weight_decay = cfg.WEIGHT_DECAY,
        betas        = (0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=_epochs, eta_min=_lr * 0.01
    )
    scaler = torch.amp.GradScaler('cuda')

    # ── Resume ─────────────────────────────────────────────────────────────────
    ckpt_path     = f'{cfg.CKPT_DIR}/stage{stage}_T_{cfg.T}_{run_tag}_latest.pt'
    start_epoch   = 0
    best_val_rmse = float('inf')

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_rmse = ckpt.get('best_val_rmse', float('inf'))
        print(f'Resumed from epoch {start_epoch}  '
              f'(best val RMSE so far: {best_val_rmse:.4f})')

    # ── NaN diagnostic ─────────────────────────────────────────────────────────
    nan_diagnostic(model, train_loader, device, stage)

    # ── Training loop ──────────────────────────────────────────────────────────
    log_path = f'{cfg.LOG_DIR}/stage{stage}_T_{cfg.T}_{run_tag}_log.json'
    log = []
    if start_epoch > 0 and os.path.exists(log_path):
        try:
            with open(log_path) as f:
                log = json.load(f)
        except Exception:
            log = []
    phase = 'a' if stage == '5a' else 'b'   # used only by V5

    for epoch in range(start_epoch, _epochs):
        model.train()
        configure_frozen_params(model, stage)   # re-apply freezes after .train()
        t0 = time.time()

        epoch_losses = {
            'loss': 0.0, 'loss_base': 0.0,
            'loss_diff': 0.0, 'loss_spec': 0.0,
        }
        if stage in ('5a', '5b'):
            epoch_losses['loss_vae'] = 0.0

        optimizer.zero_grad()
        nan_steps = 0

        for step, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device, non_blocking=False)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                if stage == '4v2':
                    losses = model(batch)
                else:
                    losses = model(batch, phase=phase)

            if not torch.isfinite(losses['loss']):
                nan_steps += 1
                if nan_steps <= 3:
                    print(f'  WARNING: NaN loss at E{epoch} S{step}')
                    for k, v in batch.items():
                        n = (~torch.isfinite(v)).sum().item()
                        if n > 0:
                            print(f'    batch[{k}] has {n} non-finite values')
                optimizer.zero_grad()
                continue

            loss = losses['loss'] / cfg.GRAD_ACCUM
            scaler.scale(loss).backward()

            if (step + 1) % cfg.GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    cfg.GRAD_CLIP
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            for k in epoch_losses:
                v = losses.get(k, 0.0)
                epoch_losses[k] += v if isinstance(v, float) else v.item()

            if step % 100 == 0:
                lr = optimizer.param_groups[0]['lr']
                vae_str = (f'  vae={losses.get("loss_vae", 0.0):.4f}'
                           if stage in ('5a', '5b') else '')
                print(f'  E{epoch:03d} S{step:04d}  '
                      f'loss={losses["loss"].item():.4f}  '
                      f'base={losses.get("loss_base", 0.0):.4f}  '
                      f'diff={losses.get("loss_diff", 0.0):.4f}  '
                      f'spec={losses.get("loss_spec", 0.0):.4f}'
                      f'{vae_str}  '
                      f'lr={lr:.2e}  '
                      f'nan_skipped={nan_steps}')

        # Flush remaining gradient accumulation buffer
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
        for k in epoch_losses:
            epoch_losses[k] /= n

        if nan_steps > 0:
            print(f'  WARNING: {nan_steps}/{len(train_loader)} steps '
                  f'skipped due to NaN this epoch')

        # ── Validation ────────────────────────────────────────────────────
        val_rmse = validate(model, val_loader, device, stage)

        elapsed = time.time() - t0
        print(f'\nEpoch {epoch:03d} | '
              f'train_loss={epoch_losses["loss"]:.4f} | '
              f'val_rmse={val_rmse:.4f} | '
              f'nan_skipped={nan_steps} | '
              f'{elapsed:.0f}s\n')

        # ── Logging ───────────────────────────────────────────────────────
        row = {
            'epoch':       epoch,
            'val_rmse':    val_rmse,
            'nan_skipped': nan_steps,
            **epoch_losses,
        }
        log.append(row)
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2)

        # ── Checkpointing ─────────────────────────────────────────────────
        ckpt = {
            'epoch':         epoch,
            'model':         model.state_dict(),
            'optimizer':     optimizer.state_dict(),
            'scheduler':     scheduler.state_dict(),
            'best_val_rmse': best_val_rmse,
            'stage':         stage,
        }
        torch.save(ckpt, ckpt_path)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(ckpt,
                f'{cfg.CKPT_DIR}/stage{stage}_T_{cfg.T}_{run_tag}_best.pt')
            print(f'  ★ New best val RMSE: {val_rmse:.4f}')

    print(f'\nTraining complete. Best val RMSE: {best_val_rmse:.4f}')
    return best_val_rmse


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',   type=str,   required=True,
                        choices=STAGE_CHOICES,
                        help='4v2 = EDM diffusion | 5a = VAE pretraining | 5b = latent diffusion')
    parser.add_argument('--run_tag', type=str,   default='try2',
                        help='Tag used in checkpoint and log filenames')
    parser.add_argument('--lr',      type=float, default=None,
                        help='Override learning rate (default: cfg.LR=3e-4)')
    parser.add_argument('--seed',    type=int,   default=None,
                        help='Global random seed for reproducibility')
    parser.add_argument('--epochs',  type=int,   default=None,
                        help='Override number of training epochs (default: cfg.EPOCHS=100)')
    args = parser.parse_args()
    train(stage=args.stage, run_tag=args.run_tag, lr=args.lr,
          seed=args.seed, epochs=args.epochs)