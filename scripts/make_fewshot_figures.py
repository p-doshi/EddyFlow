"""
make_fewshot_figures.py — Publication-quality few-shot adaptation figures.

Run from project root:
  python scripts/make_fewshot_figures.py

Outputs: outputs/figures/paper/fig_fs*.pdf + fig_fs*.png
"""

import json, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
EVAL = ROOT / 'src' / 'eval_fixed'
OUT  = ROOT / 'outputs' / 'figures' / 'paper'
OUT.mkdir(parents=True, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
STAGE_COLORS = {
    'Stage 1': '#2a78d6',
    'Stage 2': '#1baf7a',
    'Stage 3': '#eda100',
    'Stage 4': '#008300',
    'Stage 5': '#4a3aa7',
    'Stage 6': '#e34948',
}
STAGE_KEYS = ['1', '2', '3', '4', '4v2', '5a']
STAGE_NAMES = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5', 'Stage 6']

N_COLORS  = {0: '#cde2fb', 1: '#5598e7', 7: '#2a78d6', 30: '#0d366b'}
N_SHOTS   = [0, 1, 7, 30]
HIGHLIGHT = '4'   # stage key emphasised in all combined figures

CACHE_PATH = OUT / 'fewshot_data_cache.json'

C_GRID  = '#e1e0d9'
C_MUTED = '#898781'
C_INK   = '#0b0b0b'
C_INK2  = '#52514e'
FG      = '#fcfcfb'

ZERO_SHOT_FILES = {
    'bof': {'1':'eval_bof_stage1_sw_s1_sw1.json','2':'eval_bof_stage2_sw_s2_sw0.json',
            '3':'eval_bof_stage3_sw_s3_sw0.json','4':'eval_bof_sweep_4_123_lr0_sw2.json',
            '4v2':'eval_bof_sweep_4v2_0_lr1_sw1.json','5a':'eval_bof_sweep_5a_0_lr0_sw0.json'},
    'gom': {'1':'eval_gom_stage1_sw_s1_sw1.json','2':'eval_gom_stage2_sw_s2_sw0.json',
            '3':'eval_gom_stage3_sw_s3_sw0.json','4':'eval_gom_stage4_sw0lr0.json',
            '4v2':'eval_gom_stage4v2_sw0lr0.json','5a':'eval_gom_stage5a_sw0lr0.json'},
}


# ── Cache helpers — save/load D so figures can be regenerated without re-scanning JSONs ─
def _to_py(obj):
    """Convert numpy scalars to native Python for JSON serialisation."""
    return obj.item() if hasattr(obj, 'item') else obj

def save_data_cache(d, path=None):
    path = path or CACHE_PATH
    serial = {
        dom: {sk: {str(n): nv for n, nv in ns.items()} for sk, ns in sks.items()}
        for dom, sks in d.items()
    }
    with open(path, 'w') as f:
        json.dump(serial, f, default=_to_py, indent=2)
    print(f'Data cache saved → {path}')

def load_data_cache(path=None):
    path = path or CACHE_PATH
    with open(path) as f:
        raw = json.load(f)
    return {
        dom: {sk: {int(n): nv for n, nv in ns.items()} for sk, ns in sks.items()}
        for dom, sks in raw.items()
    }


# ── Global rcParams ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          8,
    'axes.labelsize':     8,
    'axes.titlesize':     9,
    'axes.titleweight':   'bold',
    'xtick.labelsize':    7.5,
    'ytick.labelsize':    7.5,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.color':         C_GRID,
    'grid.linewidth':     0.5,
    'grid.alpha':         1.0,
    'legend.fontsize':    7.5,
    'legend.frameon':     False,
    'figure.facecolor':   FG,
    'axes.facecolor':     FG,
    'savefig.facecolor':  FG,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.10,
})

def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / f'{name}.{ext}')
    plt.close(fig)
    print(f'  saved {name}')

# ── Data loading ────────────────────────────────────────────────────────────────
def load_fewshot(domain, stage_key, n):
    vals = []
    for f in sorted((EVAL).glob(f'eval_{domain}_fewshot_stage{stage_key}_n{n}_s*.json')):
        vals.append(json.load(open(f)))
    return vals

def load_zero(domain, stage_key):
    p = EVAL / ZERO_SHOT_FILES[domain][stage_key]
    return json.load(open(p)) if p.exists() else None

def _build_D():
    """Scan eval JSON files and aggregate into D[domain][stage_key][n_shots]."""
    d = {}
    for dom in ['bof', 'gom']:
        d[dom] = {}
        for sk, sn in zip(STAGE_KEYS, STAGE_NAMES):
            d[dom][sk] = {}
            z = load_zero(dom, sk)
            if z:
                d[dom][sk][0] = dict(
                    rm=float(z['rmse_model_K']), sk=float(z['skill_score']),
                    ps=float(z['psd_ratio']),    rsd=0.0,
                    rbest=float(z['rmse_model_K']))
            for n in [1, 7, 30]:
                rows = load_fewshot(dom, sk, n)
                if rows:
                    rms_arr  = np.array([r['rmse_model_K'] for r in rows])
                    sks_arr  = np.array([r['skill_score']   for r in rows])
                    psds_arr = np.array([r['psd_ratio']      for r in rows])
                    d[dom][sk][n] = dict(
                        rm=float(rms_arr.mean()),  sk=float(sks_arr.mean()),
                        ps=float(psds_arr.mean()), rsd=float(rms_arr.std()),
                        sbest=float(sks_arr.max()), rbest=float(rms_arr.min()),
                        psbest=float(psds_arr[np.argmin(np.abs(psds_arr - 1.0))]),
                    )
    return d

# Load from cache if it exists; rebuild + save otherwise.
# To force a rebuild: delete fewshot_data_cache.json  OR  set REBUILD_CACHE=1
if os.environ.get('REBUILD_CACHE', '0') == '1' or not CACHE_PATH.exists():
    print('Building data from eval JSON files...')
    D = _build_D()
    try:
        save_data_cache(D)
    except Exception as _e:
        print(f'[warn] cache save failed: {_e}')
else:
    try:
        D = load_data_cache()
        print(f'Using cached data ({CACHE_PATH.name})')
    except Exception as _e:
        print(f'Cache load failed ({_e}), rebuilding...')
        D = _build_D()
        try:
            save_data_cache(D)
        except Exception:
            pass


# ── Shared helper: bars grouped by n_shots, stages as bars within each group ──
def _draw_nshots_grouped(ax, domain, metric, ylabel, title,
                         ylo=None, yhi=None, ref_lines=None):
    """
    x-axis  = n_shots (0, 1, 7, 30) — 4 groups
    bars     = one per stage, coloured by STAGE_COLORS
    overlay  = small white tick at best-seed value (n > 0 only, RMSE metric)
    ref_lines = list of (value, label, linestyle) tuples
    """
    nS, nN   = len(STAGE_KEYS), len(N_SHOTS)
    group_w  = 0.78
    bar_w    = group_w / nS
    xs       = np.arange(nN)          # one position per n_shots level

    for si, (sk, sn) in enumerate(zip(STAGE_KEYS, STAGE_NAMES)):
        col    = STAGE_COLORS[sn]
        is_hl  = (sk == HIGHLIGHT)
        offsets = xs - group_w/2 + (si + 0.5) * bar_w
        vals    = [D[domain][sk][n][metric] for n in N_SHOTS]
        ax.bar(offsets, vals, bar_w - 0.015,
               color=col,
               alpha=1.0 if is_hl else 0.40,
               edgecolor='#1a1a1a' if is_hl else 'white',
               linewidth=0.9 if is_hl else 0.3,
               label=sn, zorder=4 if is_hl else 3)
        # best-seed tick for RMSE (lower = better)
        if metric == 'rm':
            for ni, (xo, n) in enumerate(zip(offsets, N_SHOTS)):
                if n > 0:
                    best = D[domain][sk][n].get('rbest', vals[ni])
                    ax.plot(xo, best, 'w_', markersize=3.5, markeredgewidth=1.0, zorder=5)

    # x-axis labels
    xlabels = ['0\n(zero-shot)', '1', '7', '30']
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_xlabel('n_shots (target-domain examples)', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, pad=5)
    if ylo is not None: ax.set_ylim(ylo, yhi)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)

    # optional reference lines
    for (rv, rl, rls) in (ref_lines or []):
        ax.axhline(rv, color=C_MUTED, linewidth=0.85, linestyle=rls, zorder=2)
        ax.text(nN - 0.55, rv, rl, ha='right', va='bottom',
                fontsize=6.0, color=C_MUTED)


def _stage_legend(fig, ax, ncol=6, loc='upper right'):
    """Place a horizontal legend above the figure, outside all axes."""
    handles = [mpatches.Patch(color=STAGE_COLORS[sn], label=sn) for sn in STAGE_NAMES]
    fig.legend(handles=handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.01),
               ncol=ncol, fontsize=7.5,
               handlelength=1.0, handletextpad=0.4, columnspacing=0.8,
               frameon=False)


# ═══════════════════════════════════════════════════════════════════════════════
# fig_fs01 — RMSE grouped by n_shots: BOF (left) + GOM (right)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_fs01():
    fig, (ax_b, ax_g) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    _draw_nshots_grouped(ax_b, 'bof', 'rm', 'RMSE (°C)',
                         '(a) Bay of Fundy — RMSE',
                         ylo=0)

    _draw_nshots_grouped(ax_g, 'gom', 'rm', 'RMSE (°C)',
                         '(b) Gulf of Mexico — RMSE')

    _stage_legend(fig, ax_b)
    ax_b.text(0.01, 0.03, '— tick = best seed', transform=ax_b.transAxes,
              ha='left', va='bottom', fontsize=6.0, color=C_MUTED, style='italic')

    fig.tight_layout(w_pad=2.5, rect=[0, 0, 1, 0.94])
    save(fig, 'fig_fs01_rmse_by_nshots')


# ═══════════════════════════════════════════════════════════════════════════════
# fig_fs02 — Skill score grouped by n_shots: BOF (left) + GOM (right)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_fs02():
    fig, (ax_b, ax_g) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    _draw_nshots_grouped(ax_b, 'bof', 'sk', 'Skill score',
                         '(a) Bay of Fundy — Skill Score',
                         ylo=0.74, yhi=0.875)

    _draw_nshots_grouped(ax_g, 'gom', 'sk', 'Skill score',
                         '(b) Gulf of Mexico — Skill Score',
                         ylo=0.73, yhi=0.88)

    _stage_legend(fig, ax_b)

    fig.tight_layout(w_pad=2.5, rect=[0, 0, 1, 0.94])
    save(fig, 'fig_fs02_skill_by_nshots')


# ═══════════════════════════════════════════════════════════════════════════════
# fig_fs03 — PSD ratio grouped by n_shots: BOF (left) + GOM (right)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_fs03():
    fig, (ax_b, ax_g) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    for ax, dom, letter in zip((ax_b, ax_g), ['bof', 'gom'], ['a', 'b']):
        dom_name = 'Bay of Fundy' if dom == 'bof' else 'Gulf of Mexico'
        _draw_nshots_grouped(ax, dom, 'ps', 'PSD ratio (pred / target)',
                             f'({letter}) {dom_name} — PSD Ratio  [5–50 km]')
        ax.axhline(1.0, color=C_INK2, linewidth=0.9, linestyle='--', zorder=4)
        ax.fill_between([-0.5, len(N_SHOTS) - 0.5], 0.95, 1.05,
                        color='green', alpha=0.04, zorder=0)
        ax.set_ylim(0.92, 1.10)
        ax.text(0.5, 0.97, '1.0 = ideal', transform=ax.transAxes,
                ha='center', va='top', fontsize=6.5, color=C_MUTED, style='italic')

    _stage_legend(fig, ax_b)

    fig.tight_layout(w_pad=2.5, rect=[0, 0, 1, 0.94])
    save(fig, 'fig_fs03_psd_by_nshots')


# ═══════════════════════════════════════════════════════════════════════════════
# fig_fs04 — Dumbbell: n=0 vs n=30 RMSE, BOF + GOM
# ═══════════════════════════════════════════════════════════════════════════════
def fig_fs04():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    for ax, dom, title in zip(axes, ['bof', 'gom'],
                               ['(a) BOF — Zero-shot vs 30-shot RMSE',
                                '(b) GOM — Zero-shot vs 30-shot RMSE']):
        ys   = np.arange(len(STAGE_KEYS))
        v0s  = np.array([D[dom][sk][0]['rm'] for sk in STAGE_KEYS])
        v30s = np.array([D[dom][sk][30]['rm'] for sk in STAGE_KEYS])

        ax.axvline(0, color=C_GRID, linewidth=0.8, zorder=1)

        for yi, (sk, sn) in enumerate(zip(STAGE_KEYS, STAGE_NAMES)):
            col   = STAGE_COLORS[sn]
            is_hl = (sk == HIGHLIGHT)
            v0    = v0s[yi]
            v30   = v30s[yi]
            improved = v30 < v0
            ls    = '-' if improved else '--'
            lw    = 2.2 if is_hl else 1.4
            ms    = 9   if is_hl else 7
            alpha = 1.0 if is_hl else 0.65
            zbase = 4   if is_hl else 2
            ax.plot([v0, v30], [yi, yi], color=col, linewidth=lw,
                    linestyle=ls, zorder=zbase, alpha=alpha)
            # n=0 hollow circle
            ax.plot(v0,  yi, 'o', color=col, markersize=ms,
                    markerfacecolor='white', markeredgewidth=1.5 if is_hl else 1.2,
                    zorder=zbase + 1, alpha=alpha)
            # n=30 filled circle
            ax.plot(v30, yi, 'o', color=col, markersize=ms,
                    zorder=zbase + 1, alpha=alpha)
            # delta annotation
            delta = v30 - v0
            sign  = '+' if delta > 0 else ''
            xtext = max(v0, v30) + (v0s.max() - v0s.min()) * 0.01
            tc    = '#c03030' if delta > 0 else '#006300'
            ax.text(xtext, yi, f'{sign}{delta:.3f}',
                    va='center', ha='left', fontsize=6.5,
                    color=tc, fontfamily='monospace')

        ax.set_yticks(ys)
        ax.set_yticklabels(STAGE_NAMES, fontsize=8)
        ax.set_xlabel('RMSE (°C)', fontsize=8)
        ax.set_title(title, fontsize=9, pad=5)
        ax.invert_yaxis()
        ax.grid(axis='x'); ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)

    # Shared legend
    h1 = plt.Line2D([0],[0], marker='o', color='grey', markersize=6,
                    markerfacecolor='white', markeredgewidth=1.5, linestyle='none')
    h2 = plt.Line2D([0],[0], marker='o', color='grey', markersize=6, linestyle='none')
    fig.legend([h1, h2], ['n = 0 (zero-shot)', 'n = 30'], loc='lower center',
               ncol=2, bbox_to_anchor=(0.5, -0.04), fontsize=7.5,
               handlelength=1, handletextpad=0.4)

    fig.tight_layout(rect=[0, 0.06, 1, 1], w_pad=2.5)
    save(fig, 'fig_fs04_dumbbell')


# ═══════════════════════════════════════════════════════════════════════════════
# fig_fs05 — Delta bars: RMSE reduction n=0→30
# ═══════════════════════════════════════════════════════════════════════════════
def fig_fs05():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))

    for ax, dom, title in zip(axes, ['bof', 'gom'],
                               ['(a) BOF — RMSE Reduction (n=0 → n=30)',
                                '(b) GOM — RMSE Reduction (n=0 → n=30)']):
        deltas = [D[dom][sk][0]['rm'] - D[dom][sk][30]['rm'] for sk in STAGE_KEYS]
        ys     = np.arange(len(STAGE_KEYS))
        colors = [STAGE_COLORS[sn] for sn in STAGE_NAMES]
        alphas = [1.0 if d >= 0 else 0.45 for d in deltas]

        bars = ax.barh(ys, deltas, color=colors, alpha=1.0,
                       edgecolor='white', linewidth=0.4, height=0.55, zorder=3)
        for bar, a in zip(bars, alphas):
            bar.set_alpha(a)

        ax.axvline(0, color=C_INK2, linewidth=0.9, zorder=4)
        ax.set_yticks(ys)
        ax.set_yticklabels(STAGE_NAMES, fontsize=8)
        ax.set_xlabel('ΔRMSE = RMSE(n=0) − RMSE(n=30)  [°C]', fontsize=7.5)
        ax.set_title(title, fontsize=9, pad=5)
        ax.invert_yaxis()
        ax.grid(axis='x', zorder=0); ax.set_axisbelow(True)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)

        # value labels
        for yi, d in enumerate(deltas):
            sign = '+' if d > 0 else ''
            offset = max(abs(d) * 0.05, abs(max(deltas, key=abs)) * 0.01)
            ha = 'left' if d >= 0 else 'right'
            xpos = d + (offset if d >= 0 else -offset)
            tc = '#006300' if d > 0 else '#c03030'
            ax.text(xpos, yi, f'{sign}{d:.3f}', va='center', ha=ha,
                    fontsize=6.5, color=tc, fontweight='bold')

        ax.text(0.5, -0.18, '← degraded   |   improved →',
                transform=ax.transAxes, ha='center', fontsize=7,
                color=C_MUTED, style='italic')

    fig.tight_layout(w_pad=2.5)
    save(fig, 'fig_fs05_delta_bars')


# ═══════════════════════════════════════════════════════════════════════════════
# fig_fs06 — Seed variance at n=7
# ═══════════════════════════════════════════════════════════════════════════════
def fig_fs06():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))

    for ax, dom, title in zip(axes, ['bof', 'gom'],
                               ['(a) BOF — RMSE Seed Variance at n=7',
                                '(b) GOM — RMSE Seed Variance at n=7']):
        xs     = np.arange(len(STAGE_KEYS))
        means  = np.array([D[dom][sk][7]['rm']  for sk in STAGE_KEYS])
        stds   = np.array([D[dom][sk][7]['rsd'] for sk in STAGE_KEYS])
        colors = [STAGE_COLORS[sn] for sn in STAGE_NAMES]

        # Error bars — draw per-stage to allow individual colours
        ax.bar(xs, means, color=colors, alpha=0.25, width=0.55, zorder=2)
        for xi, (m, sd, col) in enumerate(zip(means, stds, colors)):
            ax.errorbar(xi, m, yerr=sd, fmt='none',
                        ecolor=col, elinewidth=1.8, capsize=4, capthick=1.8, zorder=4)
            ax.plot(xi, m, 'o', color=col, markersize=6, zorder=5,
                    markeredgecolor='white', markeredgewidth=1.0)

        ax.set_xticks(xs)
        ax.set_xticklabels(STAGE_NAMES, fontsize=8, rotation=20, ha='right')
        ax.set_ylabel('RMSE (°C)', fontsize=8)
        ax.set_title(title, fontsize=9, pad=5)
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)

        # Annotate Stage 5 (4v2) variance specially on BOF
        if dom == 'bof':
            s5i = STAGE_KEYS.index('4v2')
            ax.annotate(f'σ={stds[s5i]:.3f}',
                        xy=(s5i, means[s5i] + stds[s5i]),
                        xytext=(s5i + 0.5, means[s5i] + stds[s5i] + 0.03),
                        fontsize=6.5, color=STAGE_COLORS['Stage 5'],
                        arrowprops=dict(arrowstyle='->', color=STAGE_COLORS['Stage 5'],
                                        lw=0.9),
                        va='bottom')

    fig.tight_layout(w_pad=2.5)
    save(fig, 'fig_fs06_seed_variance')


# ═══════════════════════════════════════════════════════════════════════════════
# fig_fs07 — Skill score heatmap (now the annotated reference view)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_fs07():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))

    for ax, dom, title in zip(axes, ['bof', 'gom'],
                               ['(a) BOF — Skill Score', '(b) GOM — Skill Score']):
        grid = np.array([[D[dom][sk][n]['sk'] for n in N_SHOTS] for sk in STAGE_KEYS])
        im = ax.imshow(grid, aspect='auto', cmap='Blues',
                       vmin=grid.min() - 0.005, vmax=grid.max() + 0.005)
        ax.set_xticks(range(len(N_SHOTS)))
        ax.set_xticklabels([f'n={n}' for n in N_SHOTS], fontsize=7.5)
        ax.set_yticks(range(len(STAGE_NAMES)))
        ax.set_yticklabels(STAGE_NAMES, fontsize=8)
        # Bold + colour the highlighted stage row label
        hl_idx = STAGE_KEYS.index(HIGHLIGHT)
        for ti, tl in enumerate(ax.get_yticklabels()):
            if ti == hl_idx:
                tl.set_fontweight('bold')
                tl.set_color(STAGE_COLORS[STAGE_NAMES[hl_idx]])
        ax.set_title(title, fontsize=9, pad=5)
        ax.tick_params(length=0)
        ax.set_xlabel('Shot count', fontsize=8)
        for ri, sk in enumerate(STAGE_KEYS):
            for ci, n in enumerate(N_SHOTS):
                v = D[dom][sk][n]['sk']
                brightness = (v - grid.min()) / max(grid.max() - grid.min(), 1e-6)
                tc = 'white' if brightness > 0.55 else C_INK
                is_hl = (sk == HIGHLIGHT)
                ax.text(ci, ri, f'{v:.3f}', ha='center', va='center',
                        fontsize=7.0 if is_hl else 6.5, color=tc,
                        fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.03,
                     label='Skill score', format='%.3f')

    fig.tight_layout(w_pad=2.0)
    save(fig, 'fig_fs07_skill_heatmap')


# ═══════════════════════════════════════════════════════════════════════════════
# fig_fs08 — Convergence small multiples (skill vs n_shots, BOF + GOM)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_fs08():
    fig, axes = plt.subplots(1, 6, figsize=(7.0, 1.9), sharey=False)
    fig.subplots_adjust(wspace=0.3)

    # Pseudo-log x positions
    def nx(n):
        return np.log(n + 1) / np.log(31)

    xpos    = [nx(n) for n in N_SHOTS]
    xticks  = [nx(n) for n in N_SHOTS]
    xlabels = ['0', '1', '7', '30']

    all_sk_bof = [D['bof'][sk][n]['sk'] for sk in STAGE_KEYS for n in N_SHOTS]
    all_sk_gom = [D['gom'][sk][n]['sk'] for sk in STAGE_KEYS for n in N_SHOTS]
    ylo = min(all_sk_bof + all_sk_gom) * 0.997
    yhi = max(all_sk_bof + all_sk_gom) * 1.001

    for ax, sk, sn in zip(axes, STAGE_KEYS, STAGE_NAMES):
        col   = STAGE_COLORS[sn]
        is_hl = (sk == HIGHLIGHT)
        bof   = [D['bof'][sk][n]['sk'] for n in N_SHOTS]
        gom   = [D['gom'][sk][n]['sk'] for n in N_SHOTS]

        # BOF — solid filled
        ax.plot(xpos, bof, '-o', color=col,
                markersize=4.5 if is_hl else 3.5,
                linewidth=2.0 if is_hl else 1.4,
                markeredgecolor='white', markeredgewidth=0.7, zorder=3)
        # GOM — dashed hollow
        ax.plot(xpos, gom, '--o', color=col,
                markersize=4.5 if is_hl else 3.5,
                linewidth=1.6 if is_hl else 1.0,
                markerfacecolor='white', markeredgewidth=0.9, zorder=3, alpha=0.85)

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=6.5)
        ax.set_ylim(ylo, yhi)
        ax.set_title(sn.replace('Stage ', 'S'), fontsize=8, color=col, fontweight='bold', pad=3)
        ax.yaxis.grid(True, linewidth=0.4)
        ax.tick_params(labelsize=6.5)
        if ax is not axes[0]:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('Skill score', fontsize=7.5)
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

        # Highlighted panel gets a coloured border
        if is_hl:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(col)
                spine.set_linewidth(1.8)

    # Shared x-label
    fig.text(0.5, -0.04, 'n_shots (log+1 scale)', ha='center', fontsize=7.5, color=C_MUTED)

    # Shared legend
    h_bof = plt.Line2D([0],[0], color='grey', marker='o', markersize=4, linewidth=1.2,
                        markeredgecolor='white', label='BOF')
    h_gom = plt.Line2D([0],[0], color='grey', marker='o', markersize=4, linewidth=1.0,
                        linestyle='--', markerfacecolor='white', label='GOM')
    fig.legend(handles=[h_bof, h_gom], loc='lower right', bbox_to_anchor=(0.99, -0.04),
               ncol=2, fontsize=7.5, handlelength=1.4)

    fig.suptitle('Skill Score Trajectory per Stage (solid = BOF, dashed = GOM)',
                 fontsize=8.5, y=1.02)
    save(fig, 'fig_fs08_convergence_small_multiples')


# ═══════════════════════════════════════════════════════════════════════════════
# fig_fs09 — Transfer efficiency scatter: zero-shot skill vs Δskill
# ═══════════════════════════════════════════════════════════════════════════════
def fig_fs09():
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    for dom, marker, label in [('bof', 'o', 'BOF'), ('gom', 's', 'GOM')]:
        for sk, sn in zip(STAGE_KEYS, STAGE_NAMES):
            sk0  = D[dom][sk][0]['sk']
            sk30 = D[dom][sk][30]['sk']
            dsk  = sk30 - sk0
            col  = STAGE_COLORS[sn]
            fc   = col if dom == 'bof' else 'white'
            ax.plot(sk0, dsk, marker, color=col, markersize=7,
                    markerfacecolor=fc, markeredgewidth=1.5, zorder=3)
            # label
            ha = 'left' if dom == 'bof' else 'right'
            off = 0.0008 if dom == 'bof' else -0.0008
            ax.text(sk0 + off, dsk, sn.replace('Stage ', 'S'), fontsize=6.5,
                    ha=ha, va='center', color=col)

    ax.axhline(0, color=C_MUTED, linewidth=0.8, linestyle='--', zorder=1)
    ax.set_xlabel('Zero-shot skill score (n = 0)', fontsize=8)
    ax.set_ylabel('Δ skill (n=0 → n=30)', fontsize=8)
    ax.set_title('Transfer Efficiency:\nZero-shot quality vs Adaptation gain', fontsize=8.5, pad=5)
    ax.grid(True, linewidth=0.4); ax.set_axisbelow(True)

    # Quadrant labels
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.text(xlim[0] + 0.002, ylim[1] - 0.001,
            'Low zero-shot\nhigh gain', fontsize=5.5, color=C_MUTED, va='top')
    ax.text(xlim[1] - 0.002, ylim[1] - 0.001,
            'High zero-shot\nhigh gain (ideal)', fontsize=5.5, color=C_MUTED, va='top', ha='right')

    h_bof = plt.Line2D([0],[0], marker='o', color='grey', markersize=5, linestyle='none', label='BOF')
    h_gom = plt.Line2D([0],[0], marker='s', color='grey', markersize=5, linestyle='none',
                        markerfacecolor='white', markeredgewidth=1.2, label='GOM')
    ax.legend(handles=[h_bof, h_gom], fontsize=7.5, loc='lower right',
              handlelength=1, handletextpad=0.4)

    fig.tight_layout()
    save(fig, 'fig_fs09_transfer_efficiency')


# ═══════════════════════════════════════════════════════════════════════════════
# fig_fs10 — Best-achieved metric summary (normalised bars)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_fs10():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
    fig.subplots_adjust(wspace=0.4)

    METRIC_DEFS = [
        ('Best Skill\n(all n, both domains)', 'sk',  'higher',
         lambda sk: max(D['bof'][sk][n]['sk'] for n in N_SHOTS)),
        ('Best RMSE BOF (°C)\n(lowest across n, seeds)', 'rm_bof', 'lower',
         lambda sk: min(D['bof'][sk][n].get('rbest', D['bof'][sk][n]['rm']) for n in N_SHOTS if n > 0)),
        ('PSD error at n=30\n|ratio − 1| (BOF)', 'psd', 'lower',
         lambda sk: min(abs(D['bof'][sk][30]['ps'] - 1),
                        abs(D['gom'][sk][30]['ps'] - 1))),
    ]

    xs     = np.arange(len(STAGE_KEYS))
    colors = [STAGE_COLORS[sn] for sn in STAGE_NAMES]

    for ax, (title, _, better, getter) in zip(axes, METRIC_DEFS):
        vals = np.array([getter(sk) for sk in STAGE_KEYS])

        ax.bar(xs, vals, color=colors, alpha=0.85, width=0.6,
               edgecolor='white', linewidth=0.4, zorder=3)

        # Highlight best bar with full opacity
        best_i = np.argmin(vals) if better == 'lower' else np.argmax(vals)
        ax.bar(best_i, vals[best_i], color=colors[best_i], alpha=1.0, width=0.6,
               edgecolor=colors[best_i], linewidth=1.2, zorder=4)

        for xi, v in enumerate(vals):
            ax.text(xi, v + vals.max() * 0.02, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=6, rotation=90,
                    color=colors[xi], fontweight='bold' if xi == best_i else 'normal')

        ax.set_xticks(xs)
        ax.set_xticklabels([sn.replace('Stage ', 'S') for sn in STAGE_NAMES],
                           fontsize=7.5)
        ax.set_title(title, fontsize=8, pad=4)
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        ax.text(0.5, -0.22,
                '↓ lower is better' if better == 'lower' else '↑ higher is better',
                transform=ax.transAxes, ha='center', fontsize=6.5, color=C_MUTED,
                style='italic')

    fig.suptitle('Best Achieved Performance per Stage (across all shot counts)',
                 fontsize=9, y=1.02)
    fig.tight_layout()
    save(fig, 'fig_fs10_best_summary')


# ═══════════════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating few-shot paper figures ...')
    print(f'Output dir: {OUT}')
    fig_fs01()
    fig_fs02()
    fig_fs03()
    fig_fs04()
    fig_fs05()
    fig_fs06()
    fig_fs07()
    fig_fs08()
    fig_fs09()
    fig_fs10()
    print('Done.')
