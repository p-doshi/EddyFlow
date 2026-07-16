import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import time
from config import cfg
from dataset import make_loaders
from model import TemporalDownscaler


def nan_diagnostic(model, train_loader, device):
    """
    Runs before training to pinpoint exactly where NaN first appears.
    Checks: batch data → encoder → baseline → residual → diffusion → losses.
    """
    print('\n' + '─' * 60)
    print('NaN diagnostic')
    print('─' * 60)

    model.eval()
    batch = next(iter(train_loader))
    batch_cpu = batch
    batch = move_batch_to_device(batch, device, non_blocking=False)

    def tensor_info(name, t):
        finite = torch.isfinite(t) if torch.is_floating_point(t) or torch.is_complex(t) else torch.ones_like(t, dtype=torch.bool)
        n_nan  = (~finite).sum().item() if finite.numel() > 0 else 0
        vals   = t[finite] if finite.numel() > 0 else t

        if vals.numel() > 0:
            if torch.is_floating_point(vals) or torch.is_complex(vals):
                print(f'  {name:<20s} shape={tuple(t.shape)}  '
                    f'NaN={n_nan}  '
                    f'min={vals.min().item():10.4f}  '
                    f'max={vals.max().item():10.4f}  '
                    f'mean={vals.mean().item():10.4f}')
            else:
                print(f'  {name:<20s} shape={tuple(t.shape)}  '
                    f'dtype={t.dtype}  min={vals.min().item()}  '
                    f'max={vals.max().item()}  mean=N/A')
        else:
            print(f'  {name:<20s} shape={tuple(t.shape)}  ALL NaN — completely invalid')

    # 1. Raw batch
    print('\n[1] Batch tensors:')
    for k, v in batch.items():
        tensor_info(k, v)

    with torch.no_grad():

        # 2. Encoder
        print('\n[2] Encoder:')
        try:
            latent = model._encode(batch)
            tensor_info('latent', latent)
        except Exception as e:
            print(f'  ENCODER FAILED: {e}')
            return

        # 3. Baseline decoder
        print('\n[3] Baseline decoder:')
        try:
            x_base = model.baseline(latent, batch['bathy'])
            tensor_info('x_base', x_base)
        except Exception as e:
            print(f'  BASELINE FAILED: {e}')
            return

        # 4. Residual
        print('\n[4] Residual (sst - x_base):')
        sst   = batch['sst']
        r_gt  = sst - x_base
        tensor_info('r_gt', r_gt)

        # 5. Diffusion
        print('\n[5] Diffusion (t=0, no noise):')
        try:
            B      = batch['sst'].shape[0]
            t_zero = torch.zeros(B, dtype=torch.long, device=device)
            t_norm = t_zero.float() / cfg.DIFF_STEPS
            noise_pred = model.diffusion(
                r_gt, x_base, latent, batch['bathy'], t_norm
            )
            tensor_info('noise_pred', noise_pred)
        except Exception as e:
            print(f'  DIFFUSION FAILED: {e}')
            return

        # 6. Individual losses
        print('\n[6] Losses:')
        weight = batch['weight']
        tensor_info('weight', weight)

        def wmean(x):
            return (x * weight).sum() / (weight.sum() + 1e-6)

        loss_base = wmean(F.l1_loss(x_base, sst, reduction='none'))
        print(f'  loss_base            = {loss_base.item():.6f}')

        noise     = torch.randn_like(r_gt)
        loss_diff = wmean(F.mse_loss(noise_pred, noise, reduction='none'))
        print(f'  loss_diff            = {loss_diff.item():.6f}')

        pred_m    = torch.nan_to_num(x_base * weight, nan=0.0)
        targ_m    = torch.nan_to_num(sst    * weight, nan=0.0)
        loss_spec = model._spectral_loss(pred_m, targ_m)
        print(f'  loss_spec            = {loss_spec.item():.6f}')

        total = loss_base + loss_diff + 0.1 * loss_spec
        print(f'  total                = {total.item():.6f}')

    print('\n' + '─' * 60)
    print('Diagnostic complete.')
    print('─' * 60 + '\n')
    model.train()

def move_batch_to_device(batch: dict, device, non_blocking: bool = True):
    return {
        k: v.to(device, non_blocking=non_blocking)
        for k, v in batch.items()
        if torch.is_tensor(v)
    }

def train(stage=1, run_tag='new2', lr=None, seed=None, epochs=None):
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

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = make_loaders(
        batch_size  = cfg.BATCH_SIZE,
        num_workers = cfg.NUM_WORKERS,
        stage       = stage,
    )
    print(f'Train batches: {len(train_loader)}  '
          f'Val batches: {len(val_loader)}')

    # ── Model ─────────────────────────────────────────────────────────────
    model = TemporalDownscaler(stage=stage).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {n_params/1e6:.2f}M')

    # ── Optimiser + scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = _lr,
        weight_decay = cfg.WEIGHT_DECAY,
        betas        = (0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=_epochs, eta_min=_lr * 0.01
    )
    scaler = torch.amp.GradScaler('cuda')

    # ── Resume from checkpoint if available ───────────────────────────────
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
        print(f'Resumed from epoch {start_epoch}')

    # ── NaN diagnostic (always run before first epoch) ────────────────────
    nan_diagnostic(model, train_loader, device)

    # ── Training loop ─────────────────────────────────────────────────────
    log = []

    for epoch in range(start_epoch, _epochs):
        model.train()
        t0 = time.time()

        epoch_losses = {
            'loss': 0.0, 'loss_base': 0.0,
            'loss_diff': 0.0, 'loss_spec': 0.0,
        }
        optimizer.zero_grad()
        nan_steps = 0

        for step, batch in enumerate(train_loader):
            batch_cpu = batch
            batch = move_batch_to_device(batch, device, non_blocking=False)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                losses = model(batch)

            # Guard against NaN loss — skip update but keep training
            if not torch.isfinite(losses['loss']):
                nan_steps += 1
                if nan_steps <= 3:
                    # Print batch stats to help diagnose
                    print(f'  WARNING: NaN loss at E{epoch} S{step}')
                    for k, v in batch.items():
                        n = (~torch.isfinite(v)).sum().item()
                        if n > 0:
                            print(f'    batch[{k}] has {n} non-finite values')
                optimizer.zero_grad()
                continue

            # Gradient accumulation
            loss = losses['loss'] / cfg.GRAD_ACCUM
            scaler.scale(loss).backward()

            if (step + 1) % cfg.GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            for k in epoch_losses:
                v = losses[k]
                epoch_losses[k] += v if isinstance(v, float) else v.item()

            if step % 100 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f'  E{epoch:03d} S{step:04d}  '
                      f'loss={losses["loss"].item():.4f}  '
                      f'base={losses["loss_base"]:.4f}  '
                      f'diff={losses["loss_diff"]:.4f}  '
                      f'spec={losses["loss_spec"]:.4f}  '
                      f'lr={lr:.2e}  '
                      f'nan_skipped={nan_steps}')

        if (len(train_loader)) % cfg.GRAD_ACCUM != 0:
            if scaler._scale is not None:  # only flush if scaler was actually used
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        scheduler.step()

        n = max(len(train_loader) - nan_steps, 1)
        for k in epoch_losses:
            epoch_losses[k] /= n

        if nan_steps > 0:
            print(f'  WARNING: {nan_steps}/{len(train_loader)} steps '
                  f'skipped due to NaN loss this epoch')

        # ── Validation ────────────────────────────────────────────────────
        val_rmse = validate(model, val_loader, device)

        elapsed = time.time() - t0
        print(f'\nEpoch {epoch:03d} | '
              f'train_loss={epoch_losses["loss"]:.4f} | '
              f'val_rmse={val_rmse:.4f} | '
              f'nan_skipped={nan_steps} | '
              f'{elapsed:.0f}s\n')

        # ── Logging ───────────────────────────────────────────────────────
        row = {
            'epoch':      epoch,
            'val_rmse':   val_rmse,
            'nan_skipped': nan_steps,
            **epoch_losses,
        }
        log.append(row)
        with open(f'{cfg.LOG_DIR}/stage{stage}_T_{cfg.T}_{run_tag}_log.json', 'w') as f:
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
            torch.save(ckpt, f'{cfg.CKPT_DIR}/stage{stage}_T_{cfg.T}_{run_tag}_best.pt')
            print(f'  ★ New best val RMSE: {val_rmse:.4f}')

    print(f'\nTraining complete. Best val RMSE: {best_val_rmse:.4f}')
    return best_val_rmse


@torch.no_grad()
def validate(model, loader, device):
    """
    RMSE in normalised SST units over weighted (ice-free, high-confidence) pixels.
    Multiply by sst_std (~6.3 °C) to get physical units.
    """
    model.eval()
    sq_sum, n_sum = 0.0, 0

    for batch in loader:
        batch_cpu = batch
        batch = move_batch_to_device(batch, device, non_blocking=False)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            latent = model._encode(batch)
            pred = model.baseline(latent, batch['bathy'])

        # Cast back to float32 for metrics
        pred   = pred.float()
        sst    = batch['sst'].float()
        weight = batch['weight'].float()

        valid   = (weight > 0) & torch.isfinite(sst) & torch.isfinite(pred)
        sq      = ((pred - sst) ** 2) * valid.float()
        sq_sum += sq.sum().item()
        n_sum  += valid.sum().item()

    rmse = (sq_sum / max(n_sum, 1)) ** 0.5
    model.train()
    return rmse


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',   type=int,   default=1,
                        help='Encoder stage: 1 | 2 | 3 | 4')
    parser.add_argument('--run_tag', type=str,   default='new2',
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