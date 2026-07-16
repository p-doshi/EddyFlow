#!/usr/bin/env python3
"""
generate_experiments.py -- Build experiments.json for the hyperparameter sweep.

Edit the CONFIG block below, then run:
    python scripts/sweep/generate_experiments.py

This creates scripts/sweep/experiments.json which the SLURM array scripts read
at runtime via SLURM_ARRAY_TASK_ID.

Naming convention:
  pretrain run_tag : sweep_{stage}_{seed}_lr{lr_idx}
  finetune run_tag : sweep_{stage}_{seed}_lr{lr_idx}_sw{sw_idx}

Checkpoint names produced:
  Phase 1  : stage{s}_T_28_{run_tag}_best.pt
  Phase 2  : stage{s}_phase2_{ft_run_tag}_best.pt
  Merged   : stage{s}_{ft_run_tag}_merged.pt
  GSL eval : eval_gsl_{ft_run_tag}.json
  BOF eval : eval_bof_{ft_run_tag}.json
"""
import json, os

# ── EDIT THIS CONFIG ──────────────────────────────────────────────────────────
STAGES          = ['4', '4v2', '5a']
LRS             = [1e-4, 3e-4]            # pretrain learning rates  (index 0, 1, ...)
SEEDS           = [0, 42, 123]            # global random seeds
SPEC_WEIGHTS    = [2.5, 5.0, 7.5]        # finetune spectral-loss weights (index 0, 1, 2)
PRETRAIN_EPOCHS = 100
FINETUNE_EPOCHS = 40
FINETUNE_LR     = 1e-5                   # phase-2 learning rate (fixed)
# ─────────────────────────────────────────────────────────────────────────────

pretrain = []
finetune = []

for stage in STAGES:
    for lr_idx, lr in enumerate(LRS):
        for seed in SEEDS:
            pt_tag = f'sweep_{stage}_{seed}_lr{lr_idx}'
            pt_idx = len(pretrain)
            pretrain.append({
                'idx':     pt_idx,
                'stage':   stage,
                'lr':      lr,
                'seed':    seed,
                'run_tag': pt_tag,
                'epochs':  PRETRAIN_EPOCHS,
            })
            for sw_idx, sw in enumerate(SPEC_WEIGHTS):
                ft_tag = f'{pt_tag}_sw{sw_idx}'
                finetune.append({
                    'idx':             len(finetune),
                    'pretrain_idx':    pt_idx,
                    'stage':           stage,
                    'spec_weight':     sw,
                    'lr':              FINETUNE_LR,
                    'seed':            seed,
                    'run_tag':         ft_tag,
                    'pretrain_run_tag':pt_tag,
                    'epochs':          FINETUNE_EPOCHS,
                })

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments.json')
with open(out_path, 'w') as f:
    json.dump({'pretrain': pretrain, 'finetune': finetune}, f, indent=2)

print(f'Generated {len(pretrain)} pretrain + {len(finetune)} finetune experiments')
print(f'  stages       : {STAGES}')
print(f'  LRs          : {LRS}')
print(f'  seeds        : {SEEDS}')
print(f'  spec_weights : {SPEC_WEIGHTS}')
print(f'Written to: {out_path}')
