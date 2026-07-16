"""
make_paper_figures.py — Publication-quality figures for EddyFlow paper.

Run from project root:
  python scripts/make_paper_figures.py

Outputs: outputs/figures/paper/fig*.png
"""

import json, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import Normalize
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
EVAL = ROOT / 'src' / 'eval_fixed'
OUT  = ROOT / 'outputs' / 'figures' / 'paper'
OUT.mkdir(parents=True, exist_ok=True)

FIGS_BOF = ROOT / 'outputs' / 'figures' / 'bof' / 'try3'
FIGS_GSL = ROOT / 'outputs' / 'figures' / 'runs'
FIGS_GOM = ROOT / 'outputs' / 'figures' / 'gom'

# ── Validated palette ──────────────────────────────────────────────────────────
C_GSL    = '#2a78d6'
C_BOF    = '#1baf7a'
C_GOM    = '#eda100'
C_ORACLE = '#898781'
C_WENHAI = '#e34948'
C_DARK   = '#0b0b0b'
C_MUTED  = '#52514e'
C_GRID   = '#e1e0d9'

STAGE_COLORS = {
    'Stage 1':   '#6da7ec',
    'Stage 2':   '#1baf7a',
    'Stage 3':   '#eda100',
    'Stage 4':   '#4a3aa7',
    'Stage 4v2': '#2a78d6',
    'Stage 5a':  '#e34948',
}

GOM_STD = 6.3153   # GOM sst_std for physical conversion

# ── Global style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          10,
    'axes.labelsize':     11,
    'axes.titlesize':     12,
    'axes.titleweight':   'bold',
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.color':         C_GRID,
    'grid.linewidth':     0.5,
    'grid.alpha':         0.9,
    'legend.fontsize':    9,
    'legend.frameon':     False,
    'figure.facecolor':   '#fcfcfb',
    'axes.facecolor':     '#fcfcfb',
    'savefig.facecolor':  '#fcfcfb',
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.12,
})

# ── Data loading ───────────────────────────────────────────────────────────────

def load(fname):
    p = EVAL / fname
    return json.load(open(p)) if p.exists() else None

wenhai = json.load(open('/home/pdoshi/scratch/my_project/eddyflow/results_wenhai_try1.json'))

BEST = {
    'gsl': {
        'Stage 1':   load('eval_gsl_stage1_try2.json'),
        'Stage 2':   load('eval_gsl_stage2_try3.json'),
        'Stage 3':   load('eval_gsl_stage3_try5.json'),
        'Stage 4':   load('eval_gsl_stage4_best_bias.json'),
        'Stage 4v2': load('eval_gsl_stage4v2_best_bias.json'),
        'Stage 5a':  load('eval_gsl_stage5a_best_bias.json'),
    },
    'bof': {
        'Stage 1':   load('eval_bof_stage1_try2.json'),
        'Stage 2':   load('eval_bof_stage2_try3.json'),
        'Stage 3':   load('eval_bof_stage3_try5.json'),
        'Stage 4':   load('eval_bof_stage4_best_bias.json'),
        'Stage 4v2': load('eval_bof_stage4v2_best_bias.json'),
        'Stage 5a':  load('eval_bof_stage5a_best_bias.json'),
    },
    'gom': {
        'Stage 1':   load('eval_gom_stage1_try2.json'),
        'Stage 2':   load('eval_gom_stage2_try3.json'),
        'Stage 3':   load('eval_gom_stage3_try5.json'),
        'Stage 4':   load('eval_gom_stage4_sw0lr0.json'),
        'Stage 4v2': load('eval_gom_stage4v2_sw0lr0.json'),
        'Stage 5a':  load('eval_gom_stage5a_sw0lr0.json'),
    }
}

SWEEP = {}
for sk in ['4', '4v2', '5a']:
    for domain in ['gsl', 'bof']:
        files = sorted(glob.glob(str(EVAL / f'eval_{domain}_sweep_{sk}_*.json')))
        if files:
            data = [json.load(open(f)) for f in files]
            SWEEP[(sk, domain)] = {
                'skill': [d['skill_score']  for d in data],
                'rmse':  [d['rmse_model_K'] for d in data],
                'psd':   [d.get('psd_ratio', float('nan')) for d in data],
            }

STAGES       = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 4v2', 'Stage 5a']
STAGES_SHORT = ['S1',      'S2',      'S3',      'S4',      'S4v2',      'S5a']

def sk(domain, stage):
    d = BEST[domain].get(stage)
    return d['skill_score'] if d else float('nan')

def rmse_phys(domain, stage):
    d = BEST[domain].get(stage)
    if d is None: return float('nan')
    v = d['rmse_model_K']
    return v * GOM_STD if domain == 'gom' else v

def persist_phys(domain, stage):
    d = BEST[domain].get(stage)
    if d is None: return float('nan')
    v = d['rmse_persist_K']
    return v * GOM_STD if domain == 'gom' else v

def psd(domain, stage):
    d = BEST[domain].get(stage)
    return d.get('psd_ratio', float('nan')) if d else float('nan')

def sweep_ci(stage_key, domain, metric='skill'):
    key = (stage_key, domain)
    if key not in SWEEP: return None, None
    arr = np.array(SWEEP[key][metric])
    if metric == 'rmse' and domain == 'gom': arr *= GOM_STD
    return np.percentile(arr, 10), np.percentile(arr, 90)

# ── Helper: annotate bars with values ─────────────────────────────────────────

def label_bar(ax, bar, val, fmt='.3f', dy=0.004, fontsize=7.5):
    if np.isfinite(val):
        ax.text(bar.get_x() + bar.get_width()/2, val + dy,
                f'{val:{fmt}}', ha='center', va='bottom',
                fontsize=fontsize, color=C_DARK, fontweight='bold')

def add_sweep_errbar(ax, x_pos, stage_key, domain, metric='skill'):
    lo, hi = sweep_ci(stage_key, domain, metric)
    if lo is None: return
    key = (stage_key, domain)
    arr = np.array(SWEEP[key][metric])
    if metric == 'rmse' and domain == 'gom': arr *= GOM_STD
    ax.plot([x_pos]*2, [lo, hi], color=C_DARK, lw=1.5,
            solid_capstyle='round', zorder=5)
    ax.plot(x_pos, np.mean(arr), 'o', color=C_DARK, ms=3.5, zorder=6)

KEY_MAP = {'Stage 4': '4', 'Stage 4v2': '4v2', 'Stage 5a': '5a'}


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Skill score ablation (3 panels, one per domain)
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_ablation_skill():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    domains = ['gsl', 'bof', 'gom']
    titles  = ['Gulf of St. Lawrence\n(Training Domain)', 'Bay of Fundy\n(Zero-Shot)', 'Gulf of Mexico\n(Zero-Shot)']
    ylims   = [(0.56, 0.80), (0.70, 0.91), (0.76, 0.90)]
    dcols   = [C_GSL, C_BOF, C_GOM]

    for ax, domain, title, ylim, dcol in zip(axes, domains, titles, ylims, dcols):
        x      = np.arange(len(STAGES))
        vals   = [sk(domain, s) for s in STAGES]
        colors = [STAGE_COLORS[s] for s in STAGES]
        bars   = ax.bar(x, vals, color=colors, width=0.62, linewidth=0, zorder=3, alpha=0.88)

        for i, (stage, bar, val) in enumerate(zip(STAGES, bars, vals)):
            sk_key = KEY_MAP.get(stage)
            if sk_key:
                add_sweep_errbar(ax, x[i], sk_key, domain, 'skill')
            label_bar(ax, bar, val, dy=0.004, fontsize=7.2)

        # WenHai (only GSL)
        if domain == 'gsl':
            ax.axhline(wenhai['skill_score'], color=C_WENHAI, lw=1.6, ls='--', zorder=2)
            ax.text(5.42, wenhai['skill_score'] + 0.005, 'WenHai',
                    color=C_WENHAI, fontsize=8, ha='right', va='bottom')

        # Oracle
        ref = BEST[domain].get('Stage 5a') or BEST[domain].get('Stage 4v2')
        if ref:
            o_rmse = ref.get('rmse_murprev_K', float('nan'))
            p_rmse = ref.get('rmse_persist_K', float('nan'))
            if domain == 'gom': o_rmse *= GOM_STD; p_rmse *= GOM_STD
            oracle_skill = 1.0 - o_rmse / p_rmse
            ax.axhline(oracle_skill, color=C_ORACLE, lw=1.2, ls=':', zorder=2)
            ax.text(5.42, oracle_skill - 0.007, 'Oracle',
                    color=C_ORACLE, fontsize=8, ha='right', va='top')

        # Domain label strip
        ax.fill_between([-0.5, len(STAGES)-0.5], ylim[1]-0.002, ylim[1]+0.002,
                        color=dcol, alpha=0.35, linewidth=0, zorder=0)

        ax.set_xticks(x); ax.set_xticklabels(STAGES_SHORT, fontsize=9)
        ax.set_ylim(ylim); ax.set_title(title, fontsize=11, pad=9)
        ax.set_ylabel('Skill Score', fontsize=10)
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))
        ax.set_xlim(-0.55, len(STAGES)-0.45)
        ax.set_axisbelow(True)

    patches = [mpatches.Patch(color=STAGE_COLORS[s], label=s) for s in STAGES]
    fig.legend(handles=patches, ncol=6, loc='lower center',
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=9)
    fig.suptitle('Architecture Ablation — Skill Score  (1 − RMSE / ERA5 coarse persist)',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.text(0.5, 0.97, 'Error bars: 10th–90th percentile across 18 sweep runs  ·  Circle: sweep mean',
             ha='center', fontsize=8, color=C_MUTED)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(OUT / 'fig1_ablation_skill.png')
    plt.close()
    print('✓  fig1_ablation_skill.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Physical RMSE
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_ablation_rmse():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    domains = ['gsl', 'bof', 'gom']
    titles  = ['Gulf of St. Lawrence\n(Training)', 'Bay of Fundy\n(Zero-Shot)', 'Gulf of Mexico\n(Zero-Shot)']

    for ax, domain, title in zip(axes, domains, titles):
        x      = np.arange(len(STAGES))
        vals   = [rmse_phys(domain, s) for s in STAGES]
        colors = [STAGE_COLORS[s] for s in STAGES]
        bars   = ax.bar(x, vals, color=colors, width=0.62, linewidth=0, zorder=3, alpha=0.88)

        for i, (stage, bar, val) in enumerate(zip(STAGES, bars, vals)):
            sk_key = KEY_MAP.get(stage)
            if sk_key:
                add_sweep_errbar(ax, x[i], sk_key, domain, 'rmse')
            label_bar(ax, bar, val, fmt='.2f', dy=0.015)

        # Reference lines
        p_val = persist_phys(domain, BEST[domain] and next((s for s in ['Stage 4v2','Stage 5a','Stage 4'] if BEST[domain].get(s)), 'Stage 1'))
        if np.isfinite(p_val):
            ax.axhline(p_val, color=C_MUTED, lw=1.6, ls='-', zorder=2)
            ax.text(5.42, p_val + 0.02, f'ERA5 persist\n({p_val:.2f}°C)',
                    color=C_MUTED, fontsize=7.5, ha='right', va='bottom')

        # Oracle line
        ref = BEST[domain].get('Stage 5a') or BEST[domain].get('Stage 4v2')
        if ref:
            o = ref.get('rmse_murprev_K', float('nan'))
            if domain == 'gom': o *= GOM_STD
            ax.axhline(o, color=C_ORACLE, lw=1.2, ls=':', zorder=2)
            ax.text(5.42, o - 0.02, f'Oracle ({o:.2f}°C)',
                    color=C_ORACLE, fontsize=7.5, ha='right', va='top')

        if domain == 'gsl':
            ax.axhline(wenhai['rmse_model_K'], color=C_WENHAI, lw=1.4, ls='--', zorder=2)
            ax.text(5.42, wenhai['rmse_model_K'] + 0.02,
                    f'WenHai ({wenhai["rmse_model_K"]:.2f}°C)',
                    color=C_WENHAI, fontsize=7.5, ha='right', va='bottom')

        ax.set_xticks(x); ax.set_xticklabels(STAGES_SHORT, fontsize=9)
        ax.set_title(title, fontsize=11, pad=9)
        ax.set_ylabel('Physical RMSE  (°C)', fontsize=10)
        ax.set_xlim(-0.55, len(STAGES)-0.45)
        ax.set_axisbelow(True)

    patches = [mpatches.Patch(color=STAGE_COLORS[s], label=s) for s in STAGES]
    fig.legend(handles=patches, ncol=6, loc='lower center',
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=9)
    fig.suptitle('Architecture Ablation — Physical RMSE (°C)',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.text(0.5, 0.97, 'GOM values converted from normalised space (× 6.315°C)',
             ha='center', fontsize=8, color=C_MUTED)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(OUT / 'fig2_ablation_rmse.png')
    plt.close()
    print('✓  fig2_ablation_rmse.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — PSD ratio (spectral energy)
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_psd():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0), sharey=True)
    domains = ['gsl', 'bof', 'gom']
    titles  = ['Gulf of St. Lawrence', 'Bay of Fundy', 'Gulf of Mexico']

    for ax, domain, title in zip(axes, domains, titles):
        x    = np.arange(len(STAGES))
        vals = [psd(domain, s) for s in STAGES]
        colors = [STAGE_COLORS[s] for s in STAGES]

        for xi, (val, col, stage) in enumerate(zip(vals, colors, STAGES)):
            if np.isfinite(val):
                lo = min(val, 1.0); hi = max(val, 1.0)
                bar = ax.bar(xi, hi - lo, bottom=lo, color=col, width=0.62,
                             linewidth=0, zorder=3, alpha=0.88)
                ax.text(xi, val + (0.006 if val >= 1 else -0.014), f'{val:.3f}',
                        ha='center', va='bottom' if val >= 1 else 'top',
                        fontsize=7.2, color=C_DARK, fontweight='bold')
            else:
                ax.text(xi, 0.94, 'N/A', ha='center', va='bottom',
                        fontsize=7, color=C_ORACLE)

            sk_key = KEY_MAP.get(stage)
            if sk_key and (sk_key, domain) in SWEEP:
                arr = [v for v in SWEEP[(sk_key, domain)]['psd'] if np.isfinite(v)]
                if arr:
                    lo_e, hi_e = np.percentile(arr, 10), np.percentile(arr, 90)
                    ax.plot([xi]*2, [lo_e, hi_e], color=C_DARK, lw=1.5,
                            solid_capstyle='round', zorder=5)
                    ax.plot(xi, np.mean(arr), 'o', color=C_DARK, ms=3.5, zorder=6)

        ax.axhline(1.0, color=C_DARK, lw=1.8, zorder=4)
        ax.fill_between([-0.55, len(STAGES)-0.45], 0.95, 1.05,
                        color='#0b0b0b', alpha=0.05, zorder=1)

        if domain == 'gsl':
            ax.axhline(wenhai['psd_ratio'], color=C_WENHAI, lw=1.4, ls='--', zorder=2)
            ax.text(5.42, wenhai['psd_ratio'] + 0.008, f'WenHai\n({wenhai["psd_ratio"]:.4f})',
                    color=C_WENHAI, fontsize=7.5, ha='right', va='bottom')
            ax.set_ylabel('PSD Ratio  (pred / MUR,  5–50 km band)', fontsize=10)

        ax.set_xticks(x); ax.set_xticklabels(STAGES_SHORT, fontsize=9)
        ax.set_title(title, fontsize=11, pad=9)
        ax.set_ylim(0.0, 1.15)
        ax.set_xlim(-0.55, len(STAGES)-0.45)
        ax.set_axisbelow(True)

    patches = [mpatches.Patch(color=STAGE_COLORS[s], label=s) for s in STAGES]
    fig.legend(handles=patches, ncol=6, loc='lower center',
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=9)
    fig.suptitle('Spectral Energy Recovery — PSD Ratio (5–50 km band)\n'
                 'Shaded band: ±5 % of target  ·  Below 1.0 = under-energised (too smooth)',
                 fontsize=11, fontweight='bold', y=1.04)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(OUT / 'fig3_psd.png')
    plt.close()
    print('✓  fig3_psd.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Seed variance box plots
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_seed_variance():
    stage_keys   = ['4',        '4v2',      '5a']
    stage_labels = ['Stage 4\n(DDIM)', 'Stage 4v2\n(EDM)', 'Stage 5a\n(VAE+EDM)']
    domain_pairs = [('gsl', 'GSL', C_GSL), ('bof', 'BOF', C_BOF)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))

    for ax, metric, ylabel in zip(axes, ['skill', 'rmse'], ['Skill Score', 'Physical RMSE (°C)']):
        pos = np.arange(len(stage_keys))
        w   = 0.32
        for di, (domain, dlabel, dcol) in enumerate(domain_pairs):
            offset = (di - 0.5) * w
            for pi, (sk_key, slabel) in enumerate(zip(stage_keys, stage_labels)):
                key = (sk_key, domain)
                if key not in SWEEP: continue
                arr = np.array(SWEEP[key][metric])
                if metric == 'rmse' and domain == 'gom': arr *= GOM_STD
                bp = ax.boxplot([arr], positions=[pos[pi]+offset], widths=w*0.82,
                                patch_artist=True, showfliers=True,
                                medianprops=dict(color=C_DARK, lw=2.0),
                                boxprops=dict(facecolor=dcol+'BB', linewidth=0.6),
                                whiskerprops=dict(lw=0.8, color=C_MUTED, linestyle='--'),
                                capprops=dict(lw=1.0, color=C_MUTED),
                                flierprops=dict(marker='o', ms=2.5, color=dcol,
                                                alpha=0.5, linewidth=0))
                # CV annotation on GSL plot
                if di == 0 and metric == 'skill':
                    cv = np.std(arr) / np.mean(arr) * 100
                    ax.text(pos[pi]+offset, np.min(arr) - 0.012,
                            f'CV {cv:.1f}%', ha='center', fontsize=6.5, color=C_MUTED)

        ax.set_xticks(pos); ax.set_xticklabels(stage_labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f'Distribution across 18 sweep runs\n(2 LR × 3 seeds × 3 spectral weights)', fontsize=9.5)
        ax.set_axisbelow(True)

    patch_gsl = mpatches.Patch(color=C_GSL+'BB', label='GSL (training domain)')
    patch_bof = mpatches.Patch(color=C_BOF+'BB', label='BOF (zero-shot)')
    fig.legend(handles=[patch_gsl, patch_bof], ncol=2, loc='lower center',
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=9)
    fig.suptitle('Reproducibility — Skill & RMSE Distribution Across Seeds, LRs, and Spectral Weights',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(OUT / 'fig4_seed_variance.png')
    plt.close()
    print('✓  fig4_seed_variance.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Seasonal bias heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_seasonal_bias():
    domains = ['gsl', 'bof', 'gom']
    dtitles = ['Gulf of St. Lawrence\n(Training)', 'Bay of Fundy\n(Zero-Shot)', 'Gulf of Mexico\n(Zero-Shot)']
    seasons = ['DJF', 'MAM', 'JJA', 'SON']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, domain, dtitle in zip(axes, domains, dtitles):
        mat = np.full((len(STAGES), len(seasons)), float('nan'))
        for si, stage in enumerate(STAGES):
            d = BEST[domain].get(stage)
            if d is None: continue
            sb = d.get('seasonal_bias_K', {})
            for sj, season in enumerate(seasons):
                val = sb.get(season, float('nan'))
                # GOM seasonal_bias is already physical °C (converted in eval loop)
                mat[si, sj] = val

        finite_vals = mat[np.isfinite(mat)]
        vlim = np.percentile(np.abs(finite_vals), 97) if len(finite_vals) else 0.3
        vlim = max(vlim, 0.05)

        im = ax.imshow(mat, cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                       aspect='auto', interpolation='nearest')

        for si in range(len(STAGES)):
            for sj in range(len(seasons)):
                val = mat[si, sj]
                if np.isfinite(val):
                    txt = f'{val:+.2f}'
                    brightness = abs(val) / vlim
                    col = 'white' if brightness > 0.55 else C_DARK
                else:
                    txt = '–'
                    col = C_ORACLE
                ax.text(sj, si, txt, ha='center', va='center',
                        fontsize=8.5, color=col, fontweight='normal')

        ax.set_xticks(range(len(seasons))); ax.set_xticklabels(seasons, fontsize=9.5)
        ax.set_yticks(range(len(STAGES))); ax.set_yticklabels(STAGES_SHORT, fontsize=9)
        ax.set_title(dtitle, fontsize=10.5, pad=8)
        ax.grid(False)
        ax.tick_params(length=0)

        cb = plt.colorbar(im, ax=ax, shrink=0.80, pad=0.03)
        cb.set_label('Bias (°C)', fontsize=8.5)
        cb.ax.tick_params(labelsize=7.5)

    fig.suptitle('Seasonal Mean Bias  (model − MUR target, °C)\n'
                 'Positive = model too warm  ·  Negative = model too cold',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT / 'fig5_seasonal_bias.png')
    plt.close()
    print('✓  fig5_seasonal_bias.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Cross-domain summary (grouped bars, main paper figure)
# ═══════════════════════════════════════════════════════════════════════════════

def fig6_cross_domain():
    available = [s for s in STAGES
                 if any(np.isfinite(sk(d, s)) for d in ['gsl','bof','gom'])]
    n  = len(available)
    x  = np.arange(n)
    w  = 0.24

    fig, ax = plt.subplots(figsize=(12, 5.8))

    gsl_vals = [sk('gsl', s) for s in available]
    bof_vals = [sk('bof', s) for s in available]
    gom_vals = [sk('gom', s) for s in available]

    bars_gsl = ax.bar(x - w, gsl_vals, color=C_GSL,    width=w, linewidth=0, zorder=3, alpha=0.88, label='GSL (training)')
    bars_bof = ax.bar(x,     bof_vals, color=C_BOF,    width=w, linewidth=0, zorder=3, alpha=0.88, label='BOF (zero-shot)')
    bars_gom = ax.bar(x + w, gom_vals, color=C_GOM,    width=w, linewidth=0, zorder=3, alpha=0.88, label='GOM (zero-shot)')

    for i, stage in enumerate(available):
        sk_key = KEY_MAP.get(stage)
        if sk_key:
            for offset, domain in [(-w, 'gsl'), (0, 'bof')]:
                add_sweep_errbar(ax, x[i]+offset, sk_key, domain, 'skill')

    for bars, vals in [(bars_gsl, gsl_vals), (bars_bof, bof_vals), (bars_gom, gom_vals)]:
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                label_bar(ax, bar, val, dy=0.003, fontsize=6.8)
            else:
                ax.text(bar.get_x()+bar.get_width()/2, 0.62, 'pending',
                        ha='center', va='bottom', fontsize=6, color=C_ORACLE, style='italic')

    # WenHai dashed line
    ax.axhline(wenhai['skill_score'], color=C_WENHAI, lw=1.6, ls='--', zorder=2, alpha=0.85)
    ax.text(n - 0.25, wenhai['skill_score'] + 0.005,
            f'WenHai (GSL baseline, {wenhai["skill_score"]:.3f})',
            color=C_WENHAI, fontsize=8, va='bottom', ha='right')

    ax.set_xticks(x); ax.set_xticklabels(available, fontsize=10.5)
    ax.set_ylabel('Skill Score  (1 − RMSE / ERA5 coarse persist)', fontsize=10.5)
    ax.set_ylim(0.58, 0.95)
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', fontsize=9.5, frameon=True,
              fancybox=False, edgecolor=C_GRID)

    ax.annotate('Error bars: 10th–90th pct across 18 sweep runs\nGOM Stage 4 / 5a: evaluation still running',
                xy=(0.99, 0.02), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=7.5, color=C_MUTED, style='italic')

    fig.suptitle('EddyFlow Architecture Comparison — Skill Score Across Three Ocean Domains',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT / 'fig6_cross_domain.png')
    plt.close()
    print('✓  fig6_cross_domain.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — GSL vs BOF skill tradeoff scatter
# ═══════════════════════════════════════════════════════════════════════════════

def fig7_tradeoff():
    fig, ax = plt.subplots(figsize=(8, 6.5))

    for sk_key, marker, col, slabel in [('4','o',STAGE_COLORS['Stage 4'],'Stage 4 (DDIM)'),
                                          ('4v2','s',STAGE_COLORS['Stage 4v2'],'Stage 4v2 (EDM)'),
                                          ('5a','^',STAGE_COLORS['Stage 5a'],'Stage 5a (VAE)')]:
        gsl_arr = SWEEP.get(('sk_key','gsl'),{}).get('skill', None)
        bof_arr = SWEEP.get(('sk_key','bof'),{}).get('skill', None)
        gsl_arr = SWEEP.get((sk_key,'gsl'),{}).get('skill')
        bof_arr = SWEEP.get((sk_key,'bof'),{}).get('skill')
        if gsl_arr is None: continue
        ax.scatter(gsl_arr, bof_arr, c=col, marker=marker, s=42, zorder=3,
                   alpha=0.78, linewidths=0.5, edgecolors=C_DARK, label=f'{slabel}  (n=18)')
        ax.scatter(np.mean(gsl_arr), np.mean(bof_arr), c=col, marker=marker,
                   s=180, zorder=5, linewidths=1.8, edgecolors=C_DARK)

    for stage, marker, col in [('Stage 1','D',STAGE_COLORS['Stage 1']),
                                 ('Stage 2','P',STAGE_COLORS['Stage 2']),
                                 ('Stage 3','*',STAGE_COLORS['Stage 3'])]:
        gv = sk('gsl', stage); bv = sk('bof', stage)
        if np.isfinite(gv) and np.isfinite(bv):
            ax.scatter(gv, bv, c=col, marker=marker, s=160, zorder=4,
                       linewidths=1.2, edgecolors=C_DARK, label=f'{stage} (best single run)')

    # WenHai vertical
    ax.axvline(wenhai['skill_score'], color=C_WENHAI, lw=1.3, ls='--', alpha=0.7)
    ax.text(wenhai['skill_score'] + 0.002, 0.725,
            f'WenHai\n(GSL={wenhai["skill_score"]:.3f})',
            color=C_WENHAI, fontsize=8, va='bottom')

    # Equal transfer diagonal
    d = np.linspace(0.58, 0.92, 100)
    ax.plot(d, d, color=C_ORACLE, lw=1.1, ls=':', alpha=0.7, label='Equal GSL/BOF transfer')

    # Annotations for outlier
    ax.annotate('Seed catastrophe\n(Stage 4, seed=0, lr0)',
                xy=(0.673, 0.573), xytext=(0.645, 0.60),
                fontsize=7.5, color=C_WENHAI,
                arrowprops=dict(arrowstyle='->', color=C_WENHAI, lw=0.8))

    ax.set_xlabel('GSL Skill Score  (Training Domain)', fontsize=11)
    ax.set_ylabel('BOF Skill Score  (Zero-Shot Transfer)', fontsize=11)
    ax.set_xlim(0.60, 0.77); ax.set_ylim(0.53, 0.91)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', fontsize=8.5, frameon=True,
              fancybox=False, edgecolor=C_GRID)
    ax.annotate('Large markers = sweep mean  ·  Small = individual runs',
                xy=(0.99, 0.01), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=7.5, color=C_MUTED, style='italic')

    fig.suptitle('GSL vs BOF Skill Score — Architecture Generalisation Tradeoff\n'
                 'Upper-right = both domains good  ·  Above diagonal = better transfer than training skill suggests',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT / 'fig7_tradeoff.png')
    plt.close()
    print('✓  fig7_tradeoff.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Domain map panels (existing PNGs stitched into one figure)
# ═══════════════════════════════════════════════════════════════════════════════

def fig8_domain_maps():
    """3-column (GSL/BOF/GOM) × 3-row (ERA5/Model/MUR) multi-panel map figure."""
    from PIL import Image

    # Pick June (summer) as the representative season
    SEASON = 'jun'

    def find_dir(base, season):
        """Find the seasonal subdir in base directory."""
        for d in sorted(base.iterdir()) if base.exists() else []:
            if d.is_dir() and season in d.name.lower():
                return d
        return None

    gsl_dir = find_dir(FIGS_GSL, 'june') or find_dir(FIGS_GSL, 'jun')
    bof_dir = find_dir(FIGS_BOF, 'jun')
    gom_dir = find_dir(FIGS_GOM, 'jun')

    dirs = [gsl_dir, bof_dir, gom_dir]
    domain_names = ['Gulf of St. Lawrence\n(Training)', 'Bay of Fundy\n(Zero-Shot)', 'Gulf of Mexico\n(Zero-Shot)']
    row_labels   = ['ERA5 Coarse\n(Persist Baseline)', 'EddyFlow\n(Best Model)', 'MUR Satellite\n(Target)']
    row_files    = ['era5.png', 'me.png', 'mur.png']

    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor('#fcfcfb')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.05, wspace=0.04,
                             left=0.08, right=0.97, top=0.90, bottom=0.02)

    for col_i, (d, dname) in enumerate(zip(dirs, domain_names)):
        for row_i, (rfile, rlabel) in enumerate(zip(row_files, row_labels)):
            ax = fig.add_subplot(gs[row_i, col_i])
            ax.axis('off')

            if d is not None and (d / rfile).exists():
                try:
                    img = Image.open(d / rfile)
                    ax.imshow(np.array(img), aspect='equal', interpolation='bicubic')
                except Exception as e:
                    ax.text(0.5, 0.5, f'[Image error:\n{e}]', ha='center', va='center',
                            fontsize=7, color='red', transform=ax.transAxes)
            else:
                ax.set_facecolor('#f0efec')
                ax.text(0.5, 0.5, 'Not yet available', ha='center', va='center',
                        fontsize=9, color=C_ORACLE, transform=ax.transAxes)

            # Row label on left column
            if col_i == 0:
                ax.text(-0.06, 0.5, rlabel, ha='right', va='center',
                        fontsize=9, color=C_DARK, fontweight='bold',
                        transform=ax.transAxes, rotation=0)

            # Column title on top row
            if row_i == 0:
                ax.set_title(dname, fontsize=10.5, fontweight='bold', pad=6)

            # Coloured border per column domain
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
            border_col = [C_GSL, C_BOF, C_GOM][col_i]
            for spine in ax.spines.values():
                spine.set_edgecolor(border_col)

    fig.suptitle('EddyFlow SST Downscaling — June 2022  ·  ERA5 Baseline vs Model vs MUR Target',
                 fontsize=13, fontweight='bold', y=0.96)
    fig.text(0.5, 0.93,
             'All panels show normalised SST in the same colour space.  '
             'Model output is the best-BOF sweep run for each stage.',
             ha='center', fontsize=8.5, color=C_MUTED)
    fig.savefig(OUT / 'fig8_domain_maps.png')
    plt.close()
    print('✓  fig8_domain_maps.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9 — Seasonal map grid: GSL × 4 seasons
# ═══════════════════════════════════════════════════════════════════════════════

def fig9_gsl_seasonal():
    from PIL import Image

    seasons      = ['march', 'june', 'september', 'december']
    season_names = ['March (MAM)', 'June (JJA)', 'September (SON)', 'December (DJF)']
    row_files    = ['era5.png', 'me.png', 'mur.png']
    row_labels   = ['ERA5 Coarse', 'EddyFlow', 'MUR Target']

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#fcfcfb')
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.04, wspace=0.03,
                             left=0.07, right=0.98, top=0.90, bottom=0.03)

    for col_i, (season, sname) in enumerate(zip(seasons, season_names)):
        sdir = None
        for d in sorted(FIGS_GSL.iterdir()) if FIGS_GSL.exists() else []:
            if d.is_dir() and season in d.name.lower():
                sdir = d; break

        for row_i, (rfile, rlabel) in enumerate(zip(row_files, row_labels)):
            ax = fig.add_subplot(gs[row_i, col_i])
            ax.axis('off')

            if sdir and (sdir / rfile).exists():
                try:
                    img = Image.open(sdir / rfile)
                    ax.imshow(np.array(img), aspect='equal', interpolation='bicubic')
                except Exception:
                    ax.set_facecolor('#f0efec')
            else:
                ax.set_facecolor('#f0efec')
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        fontsize=9, color=C_ORACLE, transform=ax.transAxes)

            if col_i == 0:
                ax.text(-0.06, 0.5, rlabel, ha='right', va='center',
                        fontsize=9.5, color=C_DARK, fontweight='bold',
                        transform=ax.transAxes)
            if row_i == 0:
                ax.set_title(sname, fontsize=10.5, fontweight='bold', pad=5)

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.2)
                spine.set_edgecolor(C_GSL)

    fig.suptitle('GSL Seasonal Comparison — ERA5 Baseline vs EddyFlow vs MUR Target  (2022)',
                 fontsize=13, fontweight='bold', y=0.97)
    fig.savefig(OUT / 'fig9_gsl_seasonal.png')
    plt.close()
    print('✓  fig9_gsl_seasonal.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 10 — GOM domain maps (June & September)
# ═══════════════════════════════════════════════════════════════════════════════

def fig10_gom_maps():
    from PIL import Image

    seasons      = ['mar', 'jun', 'sep', 'dec']
    season_names = ['March', 'June', 'September', 'December']
    row_files    = ['era5.png', 'me.png', 'mur.png']
    row_labels   = ['ERA5 Coarse\n(Persist Baseline)', 'EddyFlow Stage 4v2\n(Zero-Shot)', 'MUR Satellite\n(Target)']

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#fcfcfb')
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.04, wspace=0.03,
                             left=0.09, right=0.98, top=0.90, bottom=0.03)

    for col_i, (season, sname) in enumerate(zip(seasons, season_names)):
        sdir = None
        for d in sorted(FIGS_GOM.iterdir()) if FIGS_GOM.exists() else []:
            if d.is_dir() and season in d.name.lower():
                sdir = d; break

        for row_i, (rfile, rlabel) in enumerate(zip(row_files, row_labels)):
            ax = fig.add_subplot(gs[row_i, col_i])
            ax.axis('off')

            if sdir and (sdir / rfile).exists():
                try:
                    img = Image.open(sdir / rfile)
                    ax.imshow(np.array(img), aspect='equal', interpolation='bicubic')
                except Exception:
                    ax.set_facecolor('#f0efec')
            else:
                ax.set_facecolor('#f0efec')
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        fontsize=9, color=C_ORACLE, transform=ax.transAxes)

            if col_i == 0:
                ax.text(-0.08, 0.5, rlabel, ha='right', va='center',
                        fontsize=8.5, color=C_DARK, fontweight='bold',
                        transform=ax.transAxes, linespacing=1.4)
            if row_i == 0:
                ax.set_title(sname, fontsize=11, fontweight='bold', pad=5)

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.2)
                spine.set_edgecolor(C_GOM)

    fig.suptitle('Gulf of Mexico — Zero-Shot Transfer  ·  ERA5 vs EddyFlow Stage 4v2 vs MUR  (2022)',
                 fontsize=13, fontweight='bold', y=0.97)
    fig.text(0.5, 0.93, 'Model was trained exclusively on Gulf of St. Lawrence — no GOM data seen during training.',
             ha='center', fontsize=8.5, color=C_MUTED, style='italic')
    fig.savefig(OUT / 'fig10_gom_maps.png')
    plt.close()
    print('✓  fig10_gom_maps.png')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f'Output directory: {OUT}')
    fig1_ablation_skill()
    fig2_ablation_rmse()
    fig3_psd()
    fig4_seed_variance()
    fig5_seasonal_bias()
    fig6_cross_domain()
    fig7_tradeoff()
    fig8_domain_maps()
    fig9_gsl_seasonal()
    fig10_gom_maps()
    print(f'\nAll 10 figures saved to {OUT}/')
