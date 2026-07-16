"""
explore_bar_charts.py — Multiple layout variants of RMSE + Skill bar charts.
Loads from fewshot_data_cache.json (no eval JSON scanning needed).

Run:
    cd /scratch/pdoshi/my_project/eddyflow
    python scripts/explore_bar_charts.py

Output: outputs/figures/explore/bars/
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).parent.parent
CACHE = ROOT / 'outputs/figures/paper/fewshot_data_cache.json'
OUT   = ROOT / 'outputs/figures/explore/bars'
OUT.mkdir(parents=True, exist_ok=True)

# ── Load cached data ───────────────────────────────────────────────────────────
if not CACHE.exists():
    print(f'ERROR: cache not found at {CACHE}')
    print('Run scripts/make_fewshot_figures.py first to build the cache.')
    sys.exit(1)

with open(CACHE) as f:
    raw = json.load(f)
D = {dom: {sk: {int(n): nv for n, nv in ns.items()} for sk, ns in sks.items()}
     for dom, sks in raw.items()}

# ── Palette ────────────────────────────────────────────────────────────────────
STAGE_COLORS = {
    'Stage 1': '#2a78d6', 'Stage 2': '#1baf7a', 'Stage 3': '#eda100',
    'Stage 4': '#008300', 'Stage 5': '#4a3aa7',  'Stage 6': '#e34948',
}
STAGE_KEYS  = ['1', '2', '3', '4', '4v2', '5a']
STAGE_NAMES = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5', 'Stage 6']
N_SHOTS     = [0, 1, 7, 30]
HIGHLIGHT   = '4'
N_COL       = {0: '#cde2fb', 1: '#5598e7', 7: '#2a78d6', 30: '#0d366b'}
C_MUTED     = '#898781'
FG          = '#fcfcfb'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#e1e0d9', 'grid.linewidth': 0.5,
    'figure.facecolor': FG, 'axes.facecolor': FG,
    'savefig.facecolor': FG, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.12,
})

def _safe(domain, sk, n, key):
    try:
        return D[domain][sk][n][key]
    except KeyError:
        return 0.0

def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / f'{name}.{ext}')
    plt.close(fig)
    print(f'  {name}')

def _stage_legend(fig):
    handles = [mpatches.Patch(color=STAGE_COLORS[sn], label=sn) for sn in STAGE_NAMES]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.01),
               ncol=6, fontsize=7.5, handlelength=1.0, handletextpad=0.4,
               columnspacing=0.8, frameon=False)


# ── Core grouped-bar draw (x=n_shots, grouped by stage) ───────────────────────
def draw_nshots(ax, domain, metric, ylabel, title,
                ylo=None, yhi=None,
                fs_hl=5.5, fs_other=3.8, val_rot=90,
                show_vals=True, group_w=0.80):
    nS, nN = len(STAGE_KEYS), len(N_SHOTS)
    bw = group_w / nS
    xs = np.arange(nN)

    for si, (sk, sn) in enumerate(zip(STAGE_KEYS, STAGE_NAMES)):
        col   = STAGE_COLORS[sn]
        is_hl = (sk == HIGHLIGHT)
        off   = xs - group_w / 2 + (si + 0.5) * bw
        vals  = [_safe(domain, sk, n, metric) for n in N_SHOTS]

        ax.bar(off, vals, bw - 0.012,
               color=col,
               alpha=1.0 if is_hl else 0.38,
               edgecolor='#111' if is_hl else 'white',
               linewidth=0.9 if is_hl else 0.2,
               label=sn, zorder=4 if is_hl else 3)

        if show_vals:
            for xo, v in zip(off, vals):
                if v == 0.0:
                    continue
                ax.text(xo, v, f'{v:.2f}',
                        ha='center', va='bottom',
                        fontsize=fs_hl if is_hl else fs_other,
                        fontweight='bold' if is_hl else 'normal',
                        color=col if is_hl else '#555',
                        alpha=1.0 if is_hl else 0.75,
                        rotation=val_rot, rotation_mode='anchor', zorder=6)

        if metric == 'rm':
            for xo, n in zip(off, N_SHOTS):
                if n > 0:
                    best = _safe(domain, sk, n, 'rbest') or _safe(domain, sk, n, 'rm')
                    ax.plot(xo, best, 'w_', markersize=3, markeredgewidth=0.9, zorder=7)

    ax.set_xticks(xs)
    ax.set_xticklabels(['0\n(zero-shot)', '1', '7', '30'], fontsize=8)
    ax.set_xlabel('n_shots', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, pad=5, fontweight='bold')
    if ylo is not None:
        ax.set_ylim(ylo, yhi)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)


# ── Alt grouping: x=stage, bars=n_shots ───────────────────────────────────────
def draw_stage_grouped(ax, domain, metric, ylabel, title, ylo=None):
    nS, nN = len(STAGE_KEYS), len(N_SHOTS)
    bw = 0.78 / nN
    xs = np.arange(nS)

    for ni, n in enumerate(N_SHOTS):
        off  = xs - 0.39 + (ni + 0.5) * bw
        vals = [_safe(domain, sk, n, metric) for sk in STAGE_KEYS]
        ax.bar(off, vals, bw - 0.010,
               color=N_COL[n], edgecolor='white', linewidth=0.3,
               label=f'n={n}', zorder=3)
        for xi, (xo, v) in enumerate(zip(off, vals)):
            if v == 0.0: continue
            is_hl = (STAGE_KEYS[xi] == HIGHLIGHT)
            ax.text(xo, v, f'{v:.2f}',
                    ha='center', va='bottom',
                    fontsize=5.5 if is_hl else 4.0,
                    fontweight='bold' if is_hl else 'normal',
                    color='#111', rotation=90, rotation_mode='anchor', zorder=5)

    hl_i = STAGE_KEYS.index(HIGHLIGHT)
    ax.axvspan(hl_i - 0.45, hl_i + 0.45, alpha=0.09,
               color=STAGE_COLORS[STAGE_NAMES[hl_i]], zorder=0)

    ax.set_xticks(xs)
    ax.set_xticklabels([sn.replace('Stage ', 'S') for sn in STAGE_NAMES], fontsize=8)
    ax.set_xlabel('Stage', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, pad=5, fontweight='bold')
    if ylo is not None: ax.set_ylim(ylo)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    handles = [mpatches.Patch(color=N_COL[n], label=f'n={n}') for n in N_SHOTS]
    ax.legend(handles=handles, fontsize=7, loc='upper right',
              handlelength=0.8, handletextpad=0.3, frameon=False)


DOM_NAMES = {'bof': 'Bay of Fundy', 'gom': 'Gulf of Mexico'}

# ═══ VARIANT A — Standard 1×2, taller, values rotated 90° (small font) ═══════
def variant_a(metric, ylabel, stem):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    for ax, dom, ltr in zip(axes, ['bof', 'gom'], 'ab'):
        draw_nshots(ax, dom, metric, ylabel,
                    f'({ltr}) {DOM_NAMES[dom]}',
                    ylo=0, fs_hl=5.5, fs_other=3.8, val_rot=90)
    _stage_legend(fig)
    fig.tight_layout(w_pad=2.5, rect=[0, 0, 1, 0.93])
    save(fig, f'{stem}_vA_1x2_tall')

# ═══ VARIANT B — Wide figure (10in), same grouping, more breathing room ════════
def variant_b(metric, ylabel, stem):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, dom, ltr in zip(axes, ['bof', 'gom'], 'ab'):
        draw_nshots(ax, dom, metric, ylabel,
                    f'({ltr}) {DOM_NAMES[dom]}',
                    ylo=0, fs_hl=6.5, fs_other=4.5, val_rot=90)
    _stage_legend(fig)
    fig.tight_layout(w_pad=3.5, rect=[0, 0, 1, 0.93])
    save(fig, f'{stem}_vB_wide')

# ═══ VARIANT C — Grouped by STAGE (x=stage, bars=n_shots) ══════════════════════
def variant_c(metric, ylabel, stem):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, dom, ltr in zip(axes, ['bof', 'gom'], 'ab'):
        draw_stage_grouped(ax, dom, metric, ylabel,
                           f'({ltr}) {DOM_NAMES[dom]}', ylo=0)
    fig.tight_layout(w_pad=2.5)
    save(fig, f'{stem}_vC_stage_grouped')

# ═══ VARIANT D — 2×1 stacked (BOF top, GOM bottom) — tall panels ═══════════════
def variant_d(metric, ylabel, stem, ylo=0):
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.8))
    for ax, dom, ltr in zip(axes, ['bof', 'gom'], 'ab'):
        draw_nshots(ax, dom, metric, ylabel,
                    f'({ltr}) {DOM_NAMES[dom]}',
                    ylo=ylo, fs_hl=7.0, fs_other=5.0, val_rot=0)
    _stage_legend(fig)
    fig.tight_layout(h_pad=3.5, rect=[0, 0, 1, 0.95])
    save(fig, f'{stem}_vD_stacked')

# ═══ VARIANT E — Stage 4 values only (clean, uncluttered) ══════════════════════
def variant_e(metric, ylabel, stem):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    for ax, dom, ltr in zip(axes, ['bof', 'gom'], 'ab'):
        draw_nshots(ax, dom, metric, ylabel,
                    f'({ltr}) {DOM_NAMES[dom]}',
                    ylo=0, fs_hl=6.5, fs_other=0,   # 0 = skip non-hl values
                    val_rot=90, show_vals=True)
    _stage_legend(fig)
    fig.tight_layout(w_pad=2.5, rect=[0, 0, 1, 0.93])
    save(fig, f'{stem}_vE_hl_vals_only')


_YLO = {'rm': 0.05, 'sk': 0.70}

if __name__ == '__main__':
    print(f'Output → {OUT}')
    for metric, ylabel, stem in [
        ('rm', 'RMSE (°C)',   'rmse'),
        ('sk', 'Skill score', 'skill'),
    ]:
        print(f'\n── {ylabel} ──')
        variant_a(metric, ylabel, stem)
        variant_b(metric, ylabel, stem)
        variant_c(metric, ylabel, stem)
        variant_d(metric, ylabel, stem, ylo=_YLO[metric])
        variant_e(metric, ylabel, stem)
    print('\nDone. 10 figures saved.')
