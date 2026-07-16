"""
evaluate_v2.py — Evaluation for Stage 4v2, 5a, 5b.
Wraps evaluate.py's Evaluator/PSD/viz machinery unchanged.
Only overrides: model loading, stage arg parsing, data pipeline stage.

Usage:
  python evaluate_v2.py --stage 4v2 --ckpt checkpoints/stage4v2_phase2_best.pt
  python evaluate_v2.py --stage 5b  --ckpt checkpoints/stage5b_phase2_best.pt \
      --psd_curves --save_examples
"""
import os
import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

# ── PROJ patch — must run before evaluate.py (and therefore cartopy) loads ───
# evaluate.py imports cartopy at module level; if PROJ_DATA is unset that
# import crashes immediately, before any user code runs.
_proj_candidates = [
    os.environ.get('PROJ_DATA', ''),
    os.environ.get('PROJ_LIB',  ''),
    '/home/pdoshi/eddyflow-venv/share/proj',
    '/usr/share/proj',
]
for _p in _proj_candidates:
    if _p and os.path.isfile(os.path.join(_p, 'proj.db')):
        os.environ['PROJ_DATA'] = _p
        os.environ['PROJ_LIB']  = _p
        print(f'[proj] using PROJ data dir: {_p}')
        break
else:
    # No valid dir found — patch evaluate.py to make cartopy import optional
    # so metrics still run even without map output.
    import sys, types, unittest.mock as _mock
    for _mod in ('cartopy', 'cartopy.crs', 'cartopy.feature'):
        sys.modules.setdefault(_mod, _mock.MagicMock())
    print('[proj] WARNING: PROJ data dir not found — cartopy mocked, '
          'map plots disabled. Metrics and JSON output unaffected.')
# ─────────────────────────────────────────────────────────────────────────────

from config  import cfg
from dataset import make_loaders
from model   import TemporalDownscalerV2, TemporalDownscalerV5

# Now safe to import from evaluate.py — cartopy either has PROJ data or is mocked
from evaluate import (
    Evaluator,
    compute_psd_curves,
    load_normalisation_stats,
)

# ... rest of the file unchanged ...

STAGE_CHOICES = ('4v2', '5a', '5b')


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_model_v2(checkpoint: str, stage: str, device: torch.device,
                  n_diff_steps: int | None = None):
    """
    Loads the correct model class for each new stage.
    Handles both plain Phase 1 checkpoints and merged Phase 2 checkpoints.
    """
    if stage == '4v2':
        model = TemporalDownscalerV2()
    elif stage in ('5a', '5b'):
        model = TemporalDownscalerV5()
    else:
        raise ValueError(f'Unknown stage: {stage}')

    ckpt = torch.load(checkpoint, map_location=device)

    # Support plain {model: ...} and merged Phase 2 checkpoints
    sd = ckpt.get('model',
         ckpt.get('model_state_dict',
         ckpt.get('state_dict', ckpt)))

    # If it's a Phase 2 checkpoint (has 'diffusion' key but no 'model' key),
    # load the diffusion weights only — encoder/baseline come from Phase 1
    if 'diffusion' in ckpt and 'model' not in ckpt:
        print(f'  Detected Phase 2 checkpoint — loading diffusion weights only.')
        print(f'  WARNING: encoder+baseline weights will be random unless you')
        print(f'  pass a merged checkpoint. Use merge_checkpoints() first.')
        model.diffusion.load_state_dict(ckpt['diffusion'])
    else:
        model.load_state_dict(sd, strict=True)

    if n_diff_steps is not None:
        cfg.DIFF_SAMPLE_STEPS = n_diff_steps

    return model.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# All new stages use the stage-4 data pipeline (has mur_seq, sst_abs, etc.)
# ─────────────────────────────────────────────────────────────────────────────

def make_test_loader_v2(batch_size: int) -> torch.utils.data.DataLoader:
    _, _, test_ds = make_loaders(
        batch_size  = cfg.BATCH_SIZE,
        num_workers = cfg.NUM_WORKERS,
        stage       = 4,   # always stage-4 pipeline for 4v2/5a/5b
    )
    return torch.utils.data.DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = min(cfg.NUM_WORKERS, 4),
        pin_memory  = True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluation for Stage 4v2, 5a, 5b'
    )
    p.add_argument('--stage',        required=True, choices=STAGE_CHOICES)
    p.add_argument('--ckpt',         required=True,
                   help='Path to merged checkpoint (Phase1+2) or Phase 1 only')
    p.add_argument('--batch_size',   type=int,   default=4)
    p.add_argument('--n_diff_steps', type=int,   default=None)
    p.add_argument('--dx_km',        type=float, default=6.0)
    p.add_argument('--lam_lo_km',    type=float, default=5.0)
    p.add_argument('--lam_hi_km',    type=float, default=50.0)
    p.add_argument('--sst_std',      type=float, default=None)
    p.add_argument('--psd_curves',   action='store_true')
    p.add_argument('--save_examples',action='store_true')
    p.add_argument('--example_dir',  default='../outputs/figures/runs')
    p.add_argument('--max_examples', type=int,   default=8)
    p.add_argument('--output',       default=None,
                   help='Output JSON path. Default: eval_results_<stage>.json')
    p.add_argument('--device',
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device(args.device)
    stats  = load_normalisation_stats()
    sst_std = args.sst_std if args.sst_std is not None else stats['sst_std']

    out_path = args.output or f'eval_results_{args.stage}.json'

    print(f'\n{"═"*60}')
    print(f'  Evaluation — Stage {args.stage}')
    print(f'{"═"*60}')
    print(f'  Checkpoint : {args.ckpt}')
    print(f'  Device     : {device}')
    if device.type == 'cuda':
        print(f'  GPU        : {torch.cuda.get_device_name(0)}')
    print(f'  sst_std    : {sst_std:.4f} °C  (display only)')

    model = load_model_v2(args.ckpt, args.stage, device, args.n_diff_steps)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters : {n_params / 1e6:.2f} M\n')

    test_loader = make_test_loader_v2(args.batch_size)
    print(f'  Test batches : {len(test_loader)}\n')

    # Evaluator is unchanged from evaluate.py — pass model directly
    evaluator = Evaluator(
        model, device,
        stats        = stats,
        dx_km        = args.dx_km,
        lam_lo_km    = args.lam_lo_km,
        lam_hi_km    = args.lam_hi_km,
        sst_std      = sst_std,
        save_examples= args.save_examples,
        example_dir  = args.example_dir,
        max_examples = args.max_examples,
    )

    for batch in tqdm(test_loader, desc='Evaluating'):
        evaluator.update(batch)

    results = evaluator.compute()
    results['stage'] = args.stage
    results['checkpoint'] = args.ckpt

    print(f'\n{"═"*60}')
    print('  RESULTS')
    print(f'{"═"*60}')
    print(f'  Samples              : {results["n_samples"]}')
    print(f'  Valid pixels         : {results["n_valid_pixels"]:,}')
    print(f'  RMSE  model   (norm) : {results["rmse_model_norm"]:.5f}')
    print(f'  RMSE  persist (norm) : {results["rmse_persist_norm"]:.5f}')
    print(f'  RMSE  model   (°C)   : {results["rmse_model_K"]:.4f}')
    print(f'  RMSE  persist (°C)   : {results["rmse_persist_K"]:.4f}')
    if 'rmse_murprev_K' in results:
        print(f'  RMSE  MUR-prev (°C)  : {results["rmse_murprev_K"]:.4f}')
    ss = results["skill_score"]
    print(f'  Skill Score SS       : {ss:+.5f}  '
          f'({"↑ better" if ss > 0 else "↓ worse"} than persistence)')
    print(f'  PSD ratio   p/t      : {results["psd_ratio"]:.4f}  '
          f'(log₁₀ bias = {results["psd_log10_bias"]:+.3f})')
    print(f'{"═"*60}\n')

    if args.psd_curves:
        print('Computing full PSD curves (second pass) …')
        results['psd_curves'] = compute_psd_curves(
            model, make_test_loader_v2(args.batch_size), device, args.dx_km
        )

    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f'Results written → {out_path}')


if __name__ == '__main__':
    main()