#!/usr/bin/env python3
"""
plot_sweep.py -- Visualise all sweep training logs and eval results.

Produces (in outputs/figures/sweep/):
  curves_stage{N}_{metric}.png   training/sample-RMSE curves per stage
  final_metrics_{gsl,bof}.png   skill score + RMSE bars
  seed_variance.png              box plot across seeds
  psd_vs_skill.png               scatter: PSD ratio vs skill score
  heatmap_stage{N}_{domain}.png  lr x spec_weight grid of mean skill

Usage:
    cd /scratch/pdoshi/my_project/eddyflow/src
    python plot_sweep.py
    python plot_sweep.py --log_dir ../outputs/logs --eval_dir . --out_dir ../outputs/figures/sweep
"""

import json, os, re, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches

STAGE_COLORS = {
    '4':  '#1f77b4',
    '4v2':'#ff7f0e',
    '5a': '#2ca02c',
    '5b': '#d62728',
    'unknown': '#7f7f7f',
}

# ── loaders ──────────────────────────────────────────────────────────────────

def load_training_logs(log_dir):
    logs = {}
    if not os.path.isdir(log_dir):
        return logs
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith('_log.json'):
            continue
        try:
            with open(os.path.join(log_dir, fname)) as f:
                data = json.load(f)
            if data:
                logs[fname[:-len('_log.json')]] = data
        except Exception:
            pass
    return logs

def load_eval_jsons(eval_dir):
    evals = {}
    if not os.path.isdir(eval_dir):
        return evals
    for fname in sorted(os.listdir(eval_dir)):
        if not (fname.endswith('.json') and
                (fname.startswith('eval_gsl_') or fname.startswith('eval_bof_')
                 or fname.startswith('eval_gom_'))):
            continue
        try:
            with open(os.path.join(eval_dir, fname)) as f:
                evals[fname[:-5]] = json.load(f)
        except Exception:
            pass
    return evals

# ── helpers ───────────────────────────────────────────────────────────────────

def extract_stage(key):
    m = re.match(r'stage([^_]+)(?:_|$)', key)
    return m.group(1) if m else 'unknown'

def short_label(key, n=35):
    lbl = re.sub(r'^stage[^_]+_T_\d+_', '', key)
    lbl = re.sub(r'^stage[^_]+_phase2_', 'p2_', lbl)
    return lbl[:n]

# ── training curves ───────────────────────────────────────────────────────────

def plot_training_curves(logs, out_dir):
    by_stage = {}
    for key, data in logs.items():
        by_stage.setdefault(extract_stage(key), []).append((key, data))

    for stage, runs in sorted(by_stage.items()):
        for metric in ('val_rmse', 'sample_rmse', 'loss'):
            if not any(any(metric in row for row in d) for _, d in runs):
                continue
            fig, ax = plt.subplots(figsize=(12, 6))
            palette = cm.tab20(np.linspace(0, 1, max(len(runs), 1)))
            for (key, data), col in zip(sorted(runs), palette):
                vals = [row[metric] for row in data if metric in row]
                if vals:
                    ax.plot(range(len(vals)), vals, color=col,
                            label=short_label(key), linewidth=1.5, alpha=0.9)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'Stage {stage} — {metric}')
            ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.01, 1), loc='upper left')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            out = os.path.join(out_dir, f'curves_stage{stage}_{metric}.png')
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  saved {out}')

# ── final metrics bar charts ───────────────────────────────────────────────────

def plot_final_metrics(evals, out_dir):
    for prefix, domain in [('eval_gsl_', 'GSL'), ('eval_bof_', 'BOF')]:
        sub = {k: v for k, v in evals.items() if k.startswith(prefix)}
        if not sub:
            continue
        keys    = sorted(sub.keys())
        labels  = [k[len(prefix):][:30] for k in keys]
        skills  = [sub[k].get('skill_score',    float('nan')) for k in keys]
        rmses   = [sub[k].get('rmse_model_K',   float('nan')) for k in keys]
        persist = [sub[k].get('rmse_persist_K', float('nan')) for k in keys]
        x = np.arange(len(keys))

        fig, axes = plt.subplots(2, 1, figsize=(max(8, len(keys)*0.9+2), 11))

        colors = ['#2ca02c' if s > 0 else '#d62728' for s in skills]
        axes[0].bar(x, skills, color=colors, alpha=0.85, edgecolor='k', linewidth=0.5)
        axes[0].axhline(0, color='k', linewidth=1.2, linestyle='--')
        axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        axes[0].set_ylabel('Skill Score')
        axes[0].set_title(f'{domain} Skill Score  (green=better than persistence)')
        axes[0].grid(axis='y', alpha=0.3)

        axes[1].bar(x, rmses,   color='steelblue', alpha=0.85, label='Model',    edgecolor='k', linewidth=0.5)
        if any(np.isfinite(p) for p in persist):
            axes[1].bar(x, persist, color='orange',    alpha=0.5,  label='Persist', edgecolor='k', linewidth=0.5)
        axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        axes[1].set_ylabel('RMSE (°C)')
        axes[1].set_title(f'{domain} RMSE (°C)')
        axes[1].legend(fontsize=9)
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        out = os.path.join(out_dir, f'final_metrics_{domain.lower()}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  saved {out}')

# ── seed variance box plot ────────────────────────────────────────────────────

def plot_seed_variance(evals, out_dir):
    """Box plot of skill score variance across seeds, separate panels for GSL and BOF."""
    pat = re.compile(r'sweep_([^_]+)_(\d+)_lr(\d+)_sw(\d+)')
    groups_gsl = {}; groups_bof = {}
    for key, data in evals.items():
        m = pat.search(key)
        if not m:
            continue
        stage, _, lr_idx, sw_idx = m.groups()
        gk = f'{stage}_lr{lr_idx}_sw{sw_idx}'
        skill = data.get('skill_score', float('nan'))
        if not np.isfinite(skill):
            continue
        if key.startswith('eval_gsl_'):
            groups_gsl.setdefault(gk, []).append(skill)
        elif key.startswith('eval_bof_'):
            groups_bof.setdefault(gk, []).append(skill)

    all_keys = sorted(set(list(groups_gsl.keys()) + list(groups_bof.keys())))
    if not all_keys:
        print('  [info] no sweep patterns found for seed_variance plot')
        return

    colors = [STAGE_COLORS.get(l.split('_')[0], 'grey') for l in all_keys]
    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(all_keys)*0.9+2), 11))

    for ax, (groups, domain) in zip(axes, [(groups_gsl, 'GSL'), (groups_bof, 'BOF')]):
        data_by_key = [groups.get(k, [float('nan')]) for k in all_keys]
        bp = ax.boxplot(data_by_key, tick_labels=all_keys, patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 2})
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        # Overlay individual seed points
        for i, vals in enumerate(data_by_key, start=1):
            valid = [v for v in vals if np.isfinite(v)]
            ax.scatter([i]*len(valid), valid, color='k', s=20, zorder=5, alpha=0.6)
        ax.axhline(0, color='red', linestyle='--', linewidth=1.2)
        ax.set_xticklabels(all_keys, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Skill Score')
        ax.set_title(f'{domain} Skill Variance Across Seeds\n(box = 3 seeds per lr×sw combo; dots = individual runs)')
        stage_handles = [mpatches.Patch(color=c, label=f'Stage {s}')
                         for s, c in STAGE_COLORS.items() if s != 'unknown']
        ax.legend(handles=stage_handles, fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, 'seed_variance.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out}')

# ── PSD vs skill scatter ──────────────────────────────────────────────────────

def plot_psd_vs_skill(evals, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sweep_pat = re.compile(r'sweep_([^_]+)_')
    single_runs = [  # (label, stage, gsl_key, bof_key)
        ('Stage 1', '1', 'eval_gsl_stage1_try2', 'eval_bof_stage1_try2'),
        ('Stage 2', '2', 'eval_gsl_stage2_try3', 'eval_bof_stage2_try3'),
        ('Stage 3', '3', 'eval_gsl_stage3_try5', 'eval_bof_stage3_try5'),
    ]
    for ax, (prefix, domain) in zip(axes, [('eval_gsl_', 'GSL'), ('eval_bof_', 'BOF')]):
        sub = {k: v for k, v in evals.items() if k.startswith(prefix)}
        by_stage = {}
        # Sweep runs
        for key, data in sub.items():
            m = sweep_pat.search(key)
            if not m:
                continue
            stage = m.group(1)
            psd = data.get('psd_ratio', float('nan'))
            sk  = data.get('skill_score', float('nan'))
            if np.isfinite(psd) and np.isfinite(sk):
                xs, ys = by_stage.setdefault(stage, ([], []))
                xs.append(psd); ys.append(sk)
        for stage, (xs, ys) in sorted(by_stage.items()):
            ax.scatter(xs, ys, label=f'Stage {stage} (sweep)',
                       color=STAGE_COLORS.get(stage, 'grey'),
                       alpha=0.8, s=60, edgecolors='k', linewidths=0.4)
        # Single-run stages (1/2/3) as stars
        for lbl, stage, gk, bk in single_runs:
            k = gk if domain == 'GSL' else bk
            if k in sub:
                psd = sub[k].get('psd_ratio', float('nan'))
                sk  = sub[k].get('skill_score', float('nan'))
                if np.isfinite(psd) and np.isfinite(sk):
                    ax.scatter([psd], [sk], marker='*', s=200,
                               color=STAGE_COLORS.get(stage, 'grey'),
                               edgecolors='k', linewidths=0.8,
                               label=f'{lbl} (single run)', zorder=5)
        ax.axhline(0, color='red',  linestyle='--', linewidth=1)
        ax.axvline(1, color='blue', linestyle='--', linewidth=1)
        ax.set_xlabel('PSD Ratio (pred/target)')
        ax.set_ylabel('Skill Score')
        ax.set_title(f'{domain} — PSD Ratio vs Skill Score\n(stars=Stages 1-3; dots=sweep Stages 4/4v2/5a; ideal: psd≈1)')
        ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(out_dir, 'psd_vs_skill.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out}')

# ── lr × spec_weight heatmap ──────────────────────────────────────────────────

def plot_heatmap(evals, out_dir):
    pat = re.compile(r'eval_(gsl|bof)_sweep_([^_]+)_\d+_lr(\d+)_sw(\d+)')
    data = {}
    for key, val in evals.items():
        m = pat.search(key)
        if not m:
            continue
        dom, stage, lr_i, sw_i = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        skill = val.get('skill_score', float('nan'))
        if np.isfinite(skill):
            data.setdefault(stage, {}).setdefault((dom, lr_i, sw_i), []).append(skill)

    for stage, combos in sorted(data.items()):
        for domain in ('gsl', 'bof'):
            sub = {(lr, sw): np.mean(v)
                   for (d, lr, sw), v in combos.items() if d == domain}
            if not sub:
                continue
            lrs = sorted(set(k[0] for k in sub))
            sws = sorted(set(k[1] for k in sub))
            mat = np.full((len(lrs), len(sws)), float('nan'))
            for (lr, sw), val in sub.items():
                mat[lrs.index(lr), sws.index(sw)] = val
            fig, ax = plt.subplots(figsize=(6, 4))
            valid_vals = mat[np.isfinite(mat)]
            if len(valid_vals):
                vmin_d = max(valid_vals.min() - 0.02, 0.0)
                vmax_d = min(valid_vals.max() + 0.02, 1.0)
            else:
                vmin_d, vmax_d = 0.0, 1.0
            im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=vmin_d, vmax=vmax_d)
            ax.set_xticks(range(len(sws))); ax.set_xticklabels([f'sw{i}' for i in sws])
            ax.set_yticks(range(len(lrs))); ax.set_yticklabels([f'lr{i}' for i in lrs])
            ax.set_xlabel('spec_weight index'); ax.set_ylabel('lr index')
            ax.set_title(f'Stage {stage} {domain.upper()} — mean skill score\n'
                         '(averaged over seeds; darker green = higher skill)')
            plt.colorbar(im, ax=ax, label='skill score')
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if np.isfinite(mat[i, j]):
                        ax.text(j, i, f'{mat[i,j]:.3f}', ha='center', va='center', fontsize=9)
            plt.tight_layout()
            out = os.path.join(out_dir, f'heatmap_stage{stage}_{domain}.png')
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'  saved {out}')

# ── GSL vs BOF skill trade-off scatter ───────────────────────────────────────

def plot_gsl_vs_bof(evals, out_dir):
    """Scatter: GSL skill vs BOF skill — reveals the generalisation trade-off."""
    pat = re.compile(r'sweep_([^_]+)_')
    gsl_by_run = {k.replace('eval_gsl_', ''): v for k, v in evals.items()
                  if k.startswith('eval_gsl_')}
    bof_by_run = {k.replace('eval_bof_', ''): v for k, v in evals.items()
                  if k.startswith('eval_bof_')}

    by_stage = {}
    for tag in sorted(set(gsl_by_run) & set(bof_by_run)):
        m = pat.search(tag)
        stage = m.group(1) if m else 'unknown'
        if 'sweep_' not in tag:
            continue
        gs = gsl_by_run[tag].get('skill_score', float('nan'))
        bs = bof_by_run[tag].get('skill_score', float('nan'))
        if np.isfinite(gs) and np.isfinite(bs):
            xs, ys, tags = by_stage.setdefault(stage, ([], [], []))
            xs.append(gs); ys.append(bs); tags.append(tag)

    if not by_stage:
        print('  [info] no sweep data for gsl_vs_bof scatter')
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    for stage, (xs, ys, tags) in sorted(by_stage.items()):
        col = STAGE_COLORS.get(stage, 'grey')
        sc = ax.scatter(xs, ys, color=col, alpha=0.8, s=60,
                        edgecolors='k', linewidths=0.4, label=f'Stage {stage}')

    # Oracle / persist reference lines
    ax.axhline(0, color='red',   linestyle='--', linewidth=1, alpha=0.6, label='Persist baseline')
    ax.axvline(0, color='red',   linestyle='--', linewidth=1, alpha=0.6)
    ax.set_xlabel('GSL Skill Score (training domain)', fontsize=12)
    ax.set_ylabel('BOF Skill Score (zero-shot transfer)', fontsize=12)
    ax.set_title('GSL vs BOF Skill Score Trade-off\n'
                 '(each dot = one sweep run; ideal = upper-right)', fontsize=12)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(out_dir, 'gsl_vs_bof_skill.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out}')

# ── stage comparison summary (paper figure) ───────────────────────────────────

def plot_stage_comparison(evals, out_dir):
    """Grouped bar: mean skill (±std) for each stage on GSL and BOF side-by-side.
    Stages 1-3 use single-run (no std bar). Stage 4/4v2/5a use sweep mean ± std.
    Also shows MUR-yesterday oracle as a horizontal reference line.
    """
    import re
    sweep_pat = re.compile(r'sweep_([^_]+)_')

    # Collect single-run stages: prefer best_bias, fall back to regular
    single_keys = [
        ('Stage 1', 'eval_gsl_stage1_try2',      'eval_bof_stage1_try2'),
        ('Stage 2', 'eval_gsl_stage2_try3',       'eval_bof_stage2_try3'),
        ('Stage 3', 'eval_gsl_stage3_try5',       'eval_bof_stage3_try5'),
        ('Stage 4\nbest', 'eval_gsl_stage4_best_bias',   'eval_bof_stage4_best_bias'),
        ('Stage 4v2\nbest','eval_gsl_stage4v2_best_bias','eval_bof_stage4v2_best_bias'),
        ('Stage 5a\nbest', 'eval_gsl_stage5a_best_bias', 'eval_bof_stage5a_best_bias'),
    ]
    # Sweep mean ± std for Stage 4/4v2/5a
    sweep_stages = ['4', '4v2', '5a']
    sweep_gsl = {s: [] for s in sweep_stages}
    sweep_bof = {s: [] for s in sweep_stages}
    for key, val in evals.items():
        m = sweep_pat.search(key)
        if not m:
            continue
        stage = m.group(1)
        if stage not in sweep_stages:
            continue
        skill = val.get('skill_score', float('nan'))
        if not np.isfinite(skill):
            continue
        if key.startswith('eval_gsl_'):
            sweep_gsl[stage].append(skill)
        elif key.startswith('eval_bof_'):
            sweep_bof[stage].append(skill)

    stage_names, gsl_mean, gsl_err, bof_mean, bof_err, colors = [], [], [], [], [], []
    stage_color_map = {'Stage 1': '#1f77b4', 'Stage 2': '#ff7f0e', 'Stage 3': '#2ca02c',
                       'Stage 4\nbest': '#d62728', 'Stage 4v2\nbest': '#9467bd',
                       'Stage 5a\nbest': '#8c564b',
                       'Stage 4 sweep': '#e377c2', 'Stage 4v2 sweep': '#bcbd22',
                       'Stage 5a sweep': '#17becf'}

    for lbl, gk, bk in single_keys:
        if gk not in evals:
            continue
        gm = evals[gk].get('skill_score', float('nan'))
        bm = evals.get(bk, {}).get('skill_score', float('nan'))
        if not np.isfinite(gm):
            continue
        stage_names.append(lbl)
        gsl_mean.append(gm); gsl_err.append(0)
        bof_mean.append(bm if np.isfinite(bm) else 0); bof_err.append(0)
        colors.append(stage_color_map.get(lbl, 'grey'))

    for s in sweep_stages:
        if not sweep_gsl[s]:
            continue
        gm_arr = np.array(sweep_gsl[s]); bm_arr = np.array(sweep_bof[s])
        stage_names.append(f'Stage {s}\nsweep')
        gsl_mean.append(np.mean(gm_arr)); gsl_err.append(np.std(gm_arr))
        bof_mean.append(np.mean(bm_arr) if len(bm_arr) else 0)
        bof_err.append(np.std(bm_arr) if len(bm_arr) else 0)
        colors.append(stage_color_map.get(f'Stage {s} sweep', 'grey'))

    if not stage_names:
        return

    n = len(stage_names)
    x = np.arange(n)
    w = 0.35

    # Oracles from domain-tagged evals
    oracle_gsl = float('nan'); oracle_bof = float('nan')
    for k, v in evals.items():
        dom = v.get('domain', 'gsl' if k.startswith('eval_gsl_') else 'bof')
        og = v.get('rmse_murprev_K', float('nan'))
        rp = v.get('rmse_persist_K', float('nan'))
        if not (np.isfinite(og) and np.isfinite(rp)):
            continue
        if dom == 'gsl' and not np.isfinite(oracle_gsl):
            oracle_gsl = 1 - og / rp
        elif dom == 'bof' and not np.isfinite(oracle_bof):
            oracle_bof = 1 - og / rp
        if np.isfinite(oracle_gsl) and np.isfinite(oracle_bof):
            break

    fig, ax = plt.subplots(figsize=(max(10, n*1.2+2), 5))
    bars_gsl = ax.bar(x - w/2, gsl_mean, w, label='GSL (training)', color='steelblue',
                      alpha=0.85, edgecolor='k', linewidth=0.5,
                      yerr=gsl_err, capsize=4, error_kw={'linewidth': 1.5})
    bars_bof = ax.bar(x + w/2, bof_mean, w, label='BOF (zero-shot)', color='coral',
                      alpha=0.85, edgecolor='k', linewidth=0.5,
                      yerr=bof_err, capsize=4, error_kw={'linewidth': 1.5})
    if np.isfinite(oracle_gsl):
        ax.axhline(oracle_gsl, color='steelblue', linestyle=':', linewidth=1.5,
                   label=f'GSL oracle (MUR-prev, skill={oracle_gsl:.3f})')
    if np.isfinite(oracle_bof):
        ax.axhline(oracle_bof, color='coral', linestyle=':', linewidth=1.5,
                   label=f'BOF oracle (MUR-prev, skill={oracle_bof:.3f})')
    ax.set_xticks(x); ax.set_xticklabels(stage_names, fontsize=9)
    ax.set_ylabel('Skill Score  (1 − RMSE/persist)', fontsize=11)
    ax.set_title('EddyFlow Ablation: Skill Score by Stage\n'
                 '(error bars = std over 18 sweep runs; best = highest combined GSL+BOF)')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=max(0, min(bof_mean + gsl_mean) - 0.05))
    plt.tight_layout()
    out = os.path.join(out_dir, 'stage_comparison.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out}')

# ── seasonal bias comparison ──────────────────────────────────────────────────

def plot_seasonal_bias(evals, out_dir):
    """Bar chart of seasonal bias (°C) for each key architecture on GSL and BOF."""
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    # Prefer best-bias runs; fall back to regular evals for stages 1-3
    stage_keys = [
        ('Stage 1',   'eval_gsl_stage1_try2',       'eval_bof_stage1_try2'),
        ('Stage 2',   'eval_gsl_stage2_try3',        'eval_bof_stage2_try3'),
        ('Stage 3',   'eval_gsl_stage3_try5',        'eval_bof_stage3_try5'),
        ('Stage 4',   'eval_gsl_stage4_best_bias',   'eval_bof_stage4_best_bias'),
        ('Stage 4v2', 'eval_gsl_stage4v2_best_bias', 'eval_bof_stage4v2_best_bias'),
        ('Stage 5a',  'eval_gsl_stage5a_best_bias',  'eval_bof_stage5a_best_bias'),
    ]
    present = [(lbl, gk, bk) for lbl, gk, bk in stage_keys
               if gk in evals and 'seasonal_bias_K' in evals[gk]]
    if not present:
        return  # nothing to plot yet

    for domain_label, key_idx in [('GSL', 0), ('BOF', 1)]:
        rows = [(lbl, evals[keys[key_idx]]['seasonal_bias_K'])
                for lbl, *keys in present
                if keys[key_idx] in evals and 'seasonal_bias_K' in evals[keys[key_idx]]]
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(9, 4))
        n = len(rows)
        x = np.arange(len(seasons))
        width = 0.8 / n
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, (lbl, bias) in enumerate(rows):
            vals = [bias.get(s, float('nan')) for s in seasons]
            offset = (i - n / 2 + 0.5) * width
            ax.bar(x + offset, vals, width=width * 0.9, label=lbl, color=colors[i % len(colors)])
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_xticks(x); ax.set_xticklabels(seasons)
        ax.set_ylabel('Mean bias (°C)')
        ax.set_title(f'{domain_label} — seasonal mean bias by stage\n(positive = warm bias, negative = cold bias)')
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        out = os.path.join(out_dir, f'seasonal_bias_{domain_label.lower()}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  saved {out}')

# ── GOM zero-shot transfer plot ───────────────────────────────────────────────

def plot_gom_comparison(evals, out_dir):
    """Bar chart comparing GSL (training), BOF (zero-shot), GOM (zero-shot) skill scores
    for the best-run of each Stage 4/4v2/5a. Only shown if GOM evals are available."""
    gom_keys = [k for k in evals if k.startswith('eval_gom_')]
    if not gom_keys:
        return  # GOM evals not ready yet

    # Stage mapping: GOM key → (GSL key, BOF key, label)
    stage_map = [
        ('eval_gom_stage4_sw0lr0',   'eval_gsl_stage4_best_bias',
         'eval_bof_stage4_best_bias',   'Stage 4\nbest'),
        ('eval_gom_stage4v2_sw0lr0', 'eval_gsl_stage4v2_best_bias',
         'eval_bof_stage4v2_best_bias', 'Stage 4v2\nbest'),
        ('eval_gom_stage5a_sw0lr0',  'eval_gsl_stage5a_best_bias',
         'eval_bof_stage5a_best_bias',  'Stage 5a\nbest'),
    ]
    rows = []
    for gk_gom, gk_gsl, gk_bof, lbl in stage_map:
        if gk_gom not in evals:
            continue
        gsl_skill = evals.get(gk_gsl, {}).get('skill_score', float('nan'))
        bof_skill = evals.get(gk_bof, {}).get('skill_score', float('nan'))
        gom_skill = evals[gk_gom].get('skill_score', float('nan'))
        rows.append((lbl, gsl_skill, bof_skill, gom_skill))

    if not rows:
        return

    labels = [r[0] for r in rows]
    gsl_skills = [r[1] for r in rows]
    bof_skills = [r[2] for r in rows]
    gom_skills = [r[3] for r in rows]

    n = len(rows)
    x = np.arange(n)
    w = 0.26

    # GOM oracle
    gom_oracle = float('nan')
    for k in gom_keys:
        og = evals[k].get('rmse_murprev_K', float('nan'))
        rp = evals[k].get('rmse_persist_K', float('nan'))
        if np.isfinite(og) and np.isfinite(rp):
            gom_oracle = 1 - og / rp
            break

    fig, ax = plt.subplots(figsize=(max(8, n*2.5), 5))
    ax.bar(x - w, gsl_skills, w, label='GSL (training)', color='steelblue',  alpha=0.85, edgecolor='k', lw=0.5)
    ax.bar(x,     bof_skills, w, label='BOF (zero-shot)', color='coral',     alpha=0.85, edgecolor='k', lw=0.5)
    ax.bar(x + w, gom_skills, w, label='GOM (zero-shot)', color='mediumseagreen', alpha=0.85, edgecolor='k', lw=0.5)

    # Oracle lines (GSL from stage4_best_bias, BOF idem)
    gsl_oracle = evals.get('eval_gsl_stage4_best_bias', {}).get('rmse_murprev_K')
    gsl_persist = evals.get('eval_gsl_stage4_best_bias', {}).get('rmse_persist_K')
    if gsl_oracle and gsl_persist:
        ax.axhline(1 - gsl_oracle / gsl_persist, color='steelblue', ls=':', lw=1.4,
                   label=f'GSL oracle ({1-gsl_oracle/gsl_persist:.3f})')
    bof_oracle = evals.get('eval_bof_stage4_best_bias', {}).get('rmse_murprev_K')
    bof_persist = evals.get('eval_bof_stage4_best_bias', {}).get('rmse_persist_K')
    if bof_oracle and bof_persist:
        ax.axhline(1 - bof_oracle / bof_persist, color='coral', ls=':', lw=1.4,
                   label=f'BOF oracle ({1-bof_oracle/bof_persist:.3f})')
    if np.isfinite(gom_oracle):
        ax.axhline(gom_oracle, color='mediumseagreen', ls=':', lw=1.4,
                   label=f'GOM oracle ({gom_oracle:.3f})')

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Skill Score  (1 − RMSE/persist)', fontsize=11)
    ax.set_title('Zero-Shot Transfer: GSL (training) → BOF → GOM\n'
                 '(best-run for each stage; dotted = MUR-yesterday oracle)')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=max(0, min(gom_skills + bof_skills + gsl_skills) - 0.05))
    plt.tight_layout()
    out = os.path.join(out_dir, 'gom_transfer.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out}')

    # Also save GOM seasonal bias if available
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    gom_rows = [(k, evals[k].get('seasonal_bias_K', {})) for k in gom_keys
                if 'seasonal_bias_K' in evals[k]]
    if gom_rows:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        n2 = len(gom_rows)
        x2 = np.arange(len(seasons))
        w2 = 0.8 / n2
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, (tag, bias) in enumerate(gom_rows):
            lbl = tag.replace('eval_gom_', '')
            vals = [bias.get(s, float('nan')) for s in seasons]
            offset = (i - n2/2 + 0.5) * w2
            ax2.bar(x2 + offset, vals, w2 * 0.9, label=lbl, color=colors[i % len(colors)])
        ax2.axhline(0, color='k', lw=0.8, ls='--')
        ax2.set_xticks(x2); ax2.set_xticklabels(seasons)
        ax2.set_ylabel('Mean bias (°C)')
        ax2.set_title('GOM — seasonal mean bias by stage')
        ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        out2 = os.path.join(out_dir, 'seasonal_bias_gom.png')
        fig2.savefig(out2, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f'  saved {out2}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log_dir',  default='../outputs/logs')
    p.add_argument('--eval_dir', default='.')
    p.add_argument('--out_dir',  default='../outputs/figures/sweep')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Loading logs from  {args.log_dir}')
    logs  = load_training_logs(args.log_dir)
    print(f'  {len(logs)} files')

    print(f'Loading evals from {args.eval_dir}')
    evals = load_eval_jsons(args.eval_dir)
    print(f'  {len(evals)} files\n')

    plot_training_curves(logs,  args.out_dir)
    plot_final_metrics(evals,   args.out_dir)
    plot_seed_variance(evals,   args.out_dir)
    plot_psd_vs_skill(evals,    args.out_dir)
    plot_heatmap(evals,         args.out_dir)
    plot_gsl_vs_bof(evals,      args.out_dir)
    plot_stage_comparison(evals, args.out_dir)
    plot_seasonal_bias(evals,   args.out_dir)
    plot_gom_comparison(evals,  args.out_dir)   # no-op if GOM not ready

    print(f'\nDone. All plots in {args.out_dir}')

if __name__ == '__main__':
    main()
