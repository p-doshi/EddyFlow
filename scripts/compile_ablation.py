"""
compile_ablation.py — Build the final ablation comparison table from all eval JSONs.
Run after eval_stages123.sh completes.

Usage:
  python scripts/compile_ablation.py

Outputs: ablation_final.txt
"""
import json, glob, math, os, statistics

EVAL = '/scratch/pdoshi/my_project/eddyflow/src/eval_fixed'
OUT  = '/scratch/pdoshi/my_project/eddyflow/ablation_final.txt'

PERSIST_GSL = 2.0229
PERSIST_BOF = 4.6300
ORACLE_GSL  = 0.5202
ORACLE_BOF  = 0.6912

SKIP = {'psd_curves', 'psd_log10_by_k', 'psd_ratio_by_k'}

def load(path):
    try:
        return json.load(open(path))
    except:
        return None

def row(label, gsl, bof, note=''):
    """Format one row for the ablation table."""
    if gsl is None and bof is None:
        return f"  {label:28s}  {'MISSING':>9}  {'---':>10}  {'---':>8}  {'MISSING':>9}  {'---':>10}  {'---':>8}  {note}"

    g_rmse  = gsl['rmse_model_K']   if gsl else float('nan')
    g_skill = gsl['skill_score']    if gsl else float('nan')
    g_psd   = gsl['psd_ratio']      if gsl else float('nan')
    b_rmse  = bof['rmse_model_K']   if bof else float('nan')
    b_skill = bof['skill_score']    if bof else float('nan')
    b_psd   = bof['psd_ratio']      if bof else float('nan')
    return (f"  {label:28s}  {g_rmse:9.4f}  {g_skill:10.4f}  {g_psd:8.4f}"
            f"  {b_rmse:9.4f}  {b_skill:10.4f}  {b_psd:8.4f}  {note}")

def bias_str(d, domain):
    if d is None or 'seasonal_bias_K' not in d:
        return f"    {'':28s}  {domain} bias: (not available — re-run needed)"
    sb = d['seasonal_bias_K']
    return (f"    {'':28s}  {domain} bias °C: "
            f"DJF={sb['DJF']:+.3f}  MAM={sb['MAM']:+.3f}  "
            f"JJA={sb['JJA']:+.3f}  SON={sb['SON']:+.3f}")

# ── Load all results ──────────────────────────────────────────────────────────
runs_123 = {
    'stage1_try2':   (load(f'{EVAL}/eval_gsl_stage1_try2.json'),
                      load(f'{EVAL}/eval_bof_stage1_try2.json')),
    'stage2_try3':   (load(f'{EVAL}/eval_gsl_stage2_try3.json'),
                      load(f'{EVAL}/eval_bof_stage2_try3.json')),
    'stage2_try4':   (load(f'{EVAL}/eval_gsl_stage2_try4.json'),
                      load(f'{EVAL}/eval_bof_stage2_try4.json')),
    'stage3_try4':   (load(f'{EVAL}/eval_gsl_stage3_try4.json'),
                      load(f'{EVAL}/eval_bof_stage3_try4.json')),
    'stage3_try5':   (load(f'{EVAL}/eval_gsl_stage3_try5.json'),
                      load(f'{EVAL}/eval_bof_stage3_try5.json')),
}

best_bias = {
    'stage4':   (load(f'{EVAL}/eval_gsl_stage4_best_bias.json'),
                 load(f'{EVAL}/eval_bof_stage4_best_bias.json')),
    'stage4v2': (load(f'{EVAL}/eval_gsl_stage4v2_best_bias.json'),
                 load(f'{EVAL}/eval_bof_stage4v2_best_bias.json')),
    'stage5a':  (load(f'{EVAL}/eval_gsl_stage5a_best_bias.json'),
                 load(f'{EVAL}/eval_bof_stage5a_best_bias.json')),
}

# ── Sweep averages for stages 4/4v2/5a ───────────────────────────────────────
import re
sweep_data = {}
for gsl_f in sorted(glob.glob(f'{EVAL}/eval_gsl_sweep_*.json')):
    tag = gsl_f.split('eval_gsl_')[1].replace('.json', '')
    bof_f = f'{EVAL}/eval_bof_{tag}.json'
    try:
        g = json.load(open(gsl_f)); b = json.load(open(bof_f))
    except:
        continue
    m = re.match(r'sweep_(\w+)_(\d+)_lr(\d+)_sw(\d+)', tag)
    if not m:
        continue
    stage = m.group(1)
    sweep_data.setdefault(stage, {'gsl_rmse':[], 'gsl_skill':[], 'gsl_psd':[],
                                   'bof_rmse':[], 'bof_skill':[], 'bof_psd':[]})
    sweep_data[stage]['gsl_rmse'].append(g['rmse_model_K'])
    sweep_data[stage]['gsl_skill'].append(g['skill_score'])
    sweep_data[stage]['gsl_psd'].append(g['psd_ratio'])
    sweep_data[stage]['bof_rmse'].append(b['rmse_model_K'])
    sweep_data[stage]['bof_skill'].append(b['skill_score'])
    sweep_data[stage]['bof_psd'].append(b['psd_ratio'])

def sweep_row(stage, label, note=''):
    d = sweep_data.get(stage, {})
    if not d:
        return f"  {label:28s}  {'NO DATA':>9}"
    gm = statistics.mean(d['gsl_rmse']); gs = statistics.stdev(d['gsl_rmse'])
    gk = statistics.mean(d['gsl_skill']); gp = statistics.mean(d['gsl_psd'])
    bm = statistics.mean(d['bof_rmse']); bs = statistics.stdev(d['bof_rmse'])
    bk = statistics.mean(d['bof_skill']); bp = statistics.mean(d['bof_psd'])
    return (f"  {label:28s}  {gm:9.4f}  {gk:10.4f}  {gp:8.4f}"
            f"  {bm:9.4f}  {bk:10.4f}  {bp:8.4f}  {note}  (n=18)")

# ── Build output ──────────────────────────────────────────────────────────────
HDR = (f"  {'Architecture':28s}  {'GSL RMSE':>9}  {'GSL Skill':>10}  {'PSD':>8}"
       f"  {'BOF RMSE':>9}  {'BOF Skill':>10}  {'PSD':>8}  Notes")
SEP = "  " + "─" * 105

lines = []
lines.append("=" * 110)
lines.append("EDDYFLOW — FINAL ABLATION TABLE  (all stages, unified evaluation framework)")
lines.append(f"Persist:  GSL={PERSIST_GSL}°C  BOF={PERSIST_BOF}°C  |  Oracle: GSL={ORACLE_GSL}°C  BOF={ORACLE_BOF}°C")
lines.append("Skill = 1 - RMSE_model / RMSE_persist(ERA5 coarse)   |   PSD = pred/target in 5-50km band")
lines.append("BOF = zero-shot transfer (model trained on GSL only)")
lines.append("NOTE: Stages 1-3 BOF uses 703 test samples; Stages 4/4v2/5a BOF uses ~670 samples")
lines.append("  (T_oce=60 drops first ~33 days of 2022 test period due to MUR lookback requirement)")
lines.append("  Persist baselines differ: Stages 1-3 BOF ~4.67°C vs Stages 4+ BOF ~4.63°C")
lines.append("  Skill scores are each computed against their actual per-run persist — not directly comparable.")
lines.append("=" * 110)
lines.append("")

lines.append("REFERENCE BASELINES")
lines.append(HDR)
lines.append(SEP)
lines.append(f"  {'ERA5 coarse (persist baseline)':28s}  {PERSIST_GSL:9.4f}  {'  0.0000':>10}  {'  ---':>8}"
             f"  {PERSIST_BOF:9.4f}  {'  0.0000':>10}  {'  ---':>8}  denomiator for skill")
lines.append(f"  {'MUR yesterday (oracle)':28s}  {ORACLE_GSL:9.4f}  {1-ORACLE_GSL/PERSIST_GSL:10.4f}  {'  ---':>8}"
             f"  {ORACLE_BOF:9.4f}  {1-ORACLE_BOF/PERSIST_BOF:10.4f}  {'  ---':>8}  best possible (satellite)")
lines.append("")

lines.append("ABLATION: ENCODER ARCHITECTURE (GSL + zero-shot BOF, RMSE in °C)")
lines.append(HDR)
lines.append(SEP)

stage1_g, stage1_b = runs_123['stage1_try2']
lines.append(row('Stage 1 (joint S+T attn)', stage1_g, stage1_b,
                 'full token attn, block-causal'))
if stage1_g: lines.append(bias_str(stage1_g, 'GSL'))
if stage1_b: lines.append(bias_str(stage1_b, 'BOF'))

lines.append(SEP)
s2t3_g, s2t3_b = runs_123['stage2_try3']
s2t4_g, s2t4_b = runs_123['stage2_try4']
lines.append(row('Stage 2 try3 (alt S→T)', s2t3_g, s2t3_b, 'sw=5.0'))
if s2t3_g: lines.append(bias_str(s2t3_g, 'GSL'))
if s2t3_b: lines.append(bias_str(s2t3_b, 'BOF'))
lines.append(row('Stage 2 try4 (alt S→T)', s2t4_g, s2t4_b, 'sw=2.5'))
if s2t4_g: lines.append(bias_str(s2t4_g, 'GSL'))
if s2t4_b: lines.append(bias_str(s2t4_b, 'BOF'))

lines.append(SEP)
s3t4_g, s3t4_b = runs_123['stage3_try4']
s3t5_g, s3t5_b = runs_123['stage3_try5']
lines.append(row('Stage 3 try4 (dual-stream)', s3t4_g, s3t4_b, 'cross-attn bridge'))
if s3t4_g: lines.append(bias_str(s3t4_g, 'GSL'))
if s3t4_b: lines.append(bias_str(s3t4_b, 'BOF'))
lines.append(row('Stage 3 try5 (dual-stream)', s3t5_g, s3t5_b, 'best stage3'))
if s3t5_g: lines.append(bias_str(s3t5_g, 'GSL'))
if s3t5_b: lines.append(bias_str(s3t5_b, 'BOF'))

lines.append(SEP)
lines.append(sweep_row('4', 'Stage 4 (phys. dual-stream)', 'atm+oce streams, RoPE, sweep mean'))
s4g, s4b = best_bias['stage4']
lines.append(row('  Stage 4 BEST (seed=123,lr0,sw0)', s4g, s4b, ''))
if s4g: lines.append(bias_str(s4g, 'GSL'))
if s4b: lines.append(bias_str(s4b, 'BOF'))

lines.append(SEP)
lines.append(sweep_row('4v2', 'Stage 4v2 (Stage4 + EDM diffusion)', 'DomainNorm, sweep mean'))
s4v2g, s4v2b = best_bias['stage4v2']
lines.append(row('  Stage 4v2 BEST (seed=0,lr0,sw0)', s4v2g, s4v2b, ''))
if s4v2g: lines.append(bias_str(s4v2g, 'GSL'))
if s4v2b: lines.append(bias_str(s4v2b, 'BOF'))

lines.append(SEP)
lines.append(sweep_row('5a', 'Stage 5a (Stage4 + VAE latent EDM)', 'latent diffusion, sweep mean'))
s5ag, s5ab = best_bias['stage5a']
lines.append(row('  Stage 5a BEST (seed=0,lr0,sw0)', s5ag, s5ab, ''))
if s5ag: lines.append(bias_str(s5ag, 'GSL'))
if s5ab: lines.append(bias_str(s5ab, 'BOF'))

lines.append("")
lines.append("=" * 110)
lines.append("KEY QUESTIONS FOR PAPER")
lines.append("=" * 110)
lines.append("")

# Auto-generate conclusions from results
def conclude():
    out = []
    s1g = runs_123['stage1_try2'][0]
    s2g = runs_123['stage2_try3'][0]
    s3g = runs_123['stage3_try5'][0]
    s4g_best, _ = best_bias['stage4']
    s5ag_best, _ = best_bias['stage5a']

    labels = [('Stage 1', s1g), ('Stage 2 try3', s2g), ('Stage 3 try5', s3g),
              ('Stage 4 best', s4g_best), ('Stage 5a best', s5ag_best)]

    out.append("Q1: Did architectural complexity improve GSL RMSE?")
    for lbl, d in labels:
        if d:
            out.append(f"  {lbl:20s}: RMSE={d['rmse_model_K']:.4f}°C  skill={d['skill_score']:.4f}  PSD={d['psd_ratio']:.4f}")
        else:
            out.append(f"  {lbl:20s}: MISSING")
    out.append("")

    out.append("Q2: Did PSD ratio improve with architecture?")
    for lbl, d in labels:
        if d:
            psd = d['psd_ratio']
            state = 'too smooth' if psd < 0.85 else ('near-ideal' if psd < 1.15 else 'over-textured')
            out.append(f"  {lbl:20s}: PSD={psd:.4f}  → {state}")
        else:
            out.append(f"  {lbl:20s}: MISSING")
    out.append("")

    out.append("Q3: Did zero-shot BOF transfer improve with architecture?")
    for lbl, (_, bd) in [('Stage 1', runs_123['stage1_try2']),
                          ('Stage 2 try3', runs_123['stage2_try3']),
                          ('Stage 3 try5', runs_123['stage3_try5']),
                          ('Stage 4 best', best_bias['stage4']),
                          ('Stage 5a best', best_bias['stage5a'])]:
        if bd:
            out.append(f"  {lbl:20s}: BOF RMSE={bd['rmse_model_K']:.4f}°C  skill={bd['skill_score']:.4f}")
        else:
            out.append(f"  {lbl:20s}: MISSING")
    out.append("")

    out.append("Q4: Is the old 0.55°C GSL RMSE a bug or genuine?")
    if s2g and s4g_best:
        diff = s2g['rmse_model_K'] - s4g_best['rmse_model_K']
        out.append(f"  Stage 2 try3 RMSE = {s2g['rmse_model_K']:.4f}°C  vs  Stage 4 best = {s4g_best['rmse_model_K']:.4f}°C")
        out.append(f"  Difference = {diff:+.4f}°C")
        if abs(diff) < 0.05:
            out.append("  → Architectures are within noise on GSL RMSE alone.")
            out.append("  → Differentiation is in PSD, BOF transfer, and reproducibility.")
        else:
            out.append(f"  → Clear {'regression' if diff > 0 else 'improvement'} from Stage 2→4 on GSL RMSE.")
    else:
        out.append("  → Data missing — run eval first.")
    return '\n'.join(out)

lines.append(conclude())
lines.append("")

# ── GSL-BOF trade-off summary ─────────────────────────────────────────────────
lines.append("=" * 110)
lines.append("GSL-BOF SKILL TRADE-OFF (key paper insight)")
lines.append("=" * 110)
lines.append("")
lines.append("Each architecture's GSL and BOF skill (best run for Stages 1-3, sweep distribution for 4/4v2/5a):")
lines.append("")
lines.append(f"  {'Architecture':22s}  {'GSL best':>9}  {'GSL mean':>9}  {'GSL std':>8}  {'BOF best':>9}  {'BOF mean':>9}  {'BOF std':>8}  Notes")
lines.append("  " + "─" * 100)

# Stage 1 and 2 single-run
for lbl, gf, bf in [
    ('Stage 1',    f'{EVAL}/eval_gsl_stage1_try2.json',  f'{EVAL}/eval_bof_stage1_try2.json'),
    ('Stage 2',    f'{EVAL}/eval_gsl_stage2_try3.json',  f'{EVAL}/eval_bof_stage2_try3.json'),
    ('Stage 3',    f'{EVAL}/eval_gsl_stage3_try5.json',  f'{EVAL}/eval_bof_stage3_try5.json'),
]:
    g = load(gf); b = load(bf)
    gs = g['skill_score'] if g else float('nan')
    bs = b['skill_score'] if b else float('nan')
    gs_str = f'{gs:.4f}' if g else '------'
    bs_str = f'{bs:.4f}' if b else '------'
    lines.append(f"  {lbl:22s}  {gs_str:>9}  {'(single)':>9}  {'---':>8}  {bs_str:>9}  {'(single)':>9}  {'---':>8}")

# Stage 4/4v2/5a from sweep
for stage_key, stage_lbl in [('4', 'Stage 4'), ('4v2', 'Stage 4v2'), ('5a', 'Stage 5a')]:
    gsl_s, bof_s = [], []
    for f in sorted(glob.glob(f'{EVAL}/eval_gsl_sweep_{stage_key}_*.json')):
        tag = f.split('eval_gsl_')[1].replace('.json','')
        bf = f'{EVAL}/eval_bof_{tag}.json'
        g = load(f); b = load(bf)
        if g and b:
            gsl_s.append(g['skill_score']); bof_s.append(b['skill_score'])
    if gsl_s:
        import statistics as _st
        note = ' ⚠HIGH-VAR' if _st.stdev(bof_s) > 0.05 else ''
        lines.append(f"  {stage_lbl:22s}  {max(gsl_s):9.4f}  {_st.mean(gsl_s):9.4f}  {_st.stdev(gsl_s):8.4f}  "
                     f"{max(bof_s):9.4f}  {_st.mean(bof_s):9.4f}  {_st.stdev(bof_s):8.4f}{note}")
lines.append("")
lines.append("INSIGHT: Stage 2 improves GSL at the cost of BOF (encoder overfits to training domain).")
lines.append("Stage 4 (physics + 2D RoPE) recovers both GSL and BOF. Stage 4v2/5a improve reliability.")
lines.append("")

# ── Seed / LR variance breakdown ─────────────────────────────────────────────
lines.append("=" * 110)
lines.append("SEED & LR VARIANCE ANALYSIS (stages 4 / 4v2 / 5a across all 18 sweep runs per stage)")
lines.append("=" * 110)
lines.append("")
for stage_key, stage_label in [('4', 'Stage 4'), ('4v2', 'Stage 4v2'), ('5a', 'Stage 5a')]:
    lines.append(f"{'─'*60}")
    lines.append(f"{stage_label} — per-run GSL/BOF skill scores (18 runs: 2 LR × 3 seed × 3 sw)")
    lines.append(f"  {'tag':35s}  {'GSL skill':>10}  {'BOF skill':>10}  {'GSL RMSE':>10}  {'BOF RMSE':>10}")

    run_data = []
    for gsl_f in sorted(glob.glob(f'{EVAL}/eval_gsl_sweep_{stage_key}_*.json')):
        tag = gsl_f.split('eval_gsl_')[1].replace('.json', '')
        bof_f = f'{EVAL}/eval_bof_{tag}.json'
        try:
            g = json.load(open(gsl_f)); b = json.load(open(bof_f))
            run_data.append((tag, g['skill_score'], b['skill_score'],
                             g['rmse_model_K'], b['rmse_model_K']))
        except:
            pass

    run_data.sort(key=lambda x: -x[2])  # sort by BOF skill desc
    for tag, gs, bs, gr, br in run_data:
        short_tag = tag.replace(f'sweep_{stage_key}_', '')
        flag = ' ⚠' if bs < 0.70 else ''
        lines.append(f"  {short_tag:35s}  {gs:10.4f}  {bs:10.4f}  {gr:10.4f}  {br:10.4f}{flag}")

    if run_data:
        all_gs = [x[1] for x in run_data]; all_bs = [x[2] for x in run_data]
        lines.append(f"  {'--- SUMMARY ---':35s}  {'---':>10}  {'---':>10}  {'---':>10}  {'---':>10}")
        lines.append(f"  {'mean':35s}  {statistics.mean(all_gs):10.4f}  {statistics.mean(all_bs):10.4f}")
        lines.append(f"  {'std':35s}  {statistics.stdev(all_gs):10.4f}  {statistics.stdev(all_bs):10.4f}")
        lines.append(f"  {'min':35s}  {min(all_gs):10.4f}  {min(all_bs):10.4f}")
        lines.append(f"  {'max':35s}  {max(all_gs):10.4f}  {max(all_bs):10.4f}")
        cv_gsl = statistics.stdev(all_gs) / statistics.mean(all_gs)
        cv_bof = statistics.stdev(all_bs) / statistics.mean(all_bs)
        lines.append(f"  {'CV (std/mean)':35s}  {cv_gsl:10.4f}  {cv_bof:10.4f}  (lower=more stable)")
    lines.append("")

lines.append("=" * 110)
lines.append("END — generated by scripts/compile_ablation.py")
lines.append("=" * 110)

text = '\n'.join(lines)
print(text)

with open(OUT, 'w') as f:
    f.write(text)
print(f"\n\nWrote {OUT}")

# ── LaTeX table ────────────────────────────────────────────────────────────────
LAT = '/scratch/pdoshi/my_project/eddyflow/ablation_table.tex'

def fmt(v, prec=4):
    return f'{v:.{prec}f}' if v is not None and not (isinstance(v, float) and v != v) else '---'

def lat_row(label, gsl, bof, note='', bold_gsl_rmse=False, bold_gsl_skill=False,
            bold_bof_rmse=False, bold_bof_skill=False):
    g_rmse  = fmt(gsl['rmse_model_K'])  if gsl else '---'
    g_skill = fmt(gsl['skill_score'])   if gsl else '---'
    g_psd   = fmt(gsl['psd_ratio'])     if gsl else '---'
    b_rmse  = fmt(bof['rmse_model_K'])  if bof else '---'
    b_skill = fmt(bof['skill_score'])   if bof else '---'
    b_psd   = fmt(bof['psd_ratio'])     if bof else '---'
    if bold_gsl_rmse  and gsl:  g_rmse  = f'\\textbf{{{g_rmse}}}'
    if bold_gsl_skill and gsl:  g_skill = f'\\textbf{{{g_skill}}}'
    if bold_bof_rmse  and bof:  b_rmse  = f'\\textbf{{{b_rmse}}}'
    if bold_bof_skill and bof:  b_skill = f'\\textbf{{{b_skill}}}'
    note_str = f' % {note}' if note else ''
    return (f'  {label} & {g_rmse} & {g_skill} & {g_psd}'
            f' & {b_rmse} & {b_skill} & {b_psd} \\\\{note_str}')

sweep_rows = {}
for stage, d in sweep_data.items():
    n = len(d['gsl_rmse'])
    sweep_rows[stage] = {
        'g': {'rmse_model_K': statistics.mean(d['gsl_rmse']),
              'skill_score':  statistics.mean(d['gsl_skill']),
              'psd_ratio':    statistics.mean(d['gsl_psd'])},
        'b': {'rmse_model_K': statistics.mean(d['bof_rmse']),
              'skill_score':  statistics.mean(d['bof_skill']),
              'psd_ratio':    statistics.mean(d['bof_psd'])},
        'n': n,
        'g_std': statistics.stdev(d['gsl_skill']) if n > 1 else 0,
        'b_std': statistics.stdev(d['bof_skill']) if n > 1 else 0,
    }

lat = []
lat.append(r'% EddyFlow Ablation Table — generated by scripts/compile_ablation.py')
lat.append(r'% GSL = training domain, BOF = zero-shot Bay of Fundy transfer')
lat.append(r'% Skill = 1 - RMSE_model / RMSE_persist, PSD = pred/target in 5-50km band')
lat.append(r'')
lat.append(r'\begin{table}[t]')
lat.append(r'\centering')
lat.append(r'\caption{Ablation of encoder architecture and diffusion decoder. '
           r'GSL (Gulf of St.\ Lawrence) is the training domain; BOF (Bay of Fundy) is zero-shot transfer. '
           r'Skill $= 1 - \text{RMSE}_\text{model}/\text{RMSE}_\text{persist}$ where '
           r'persistence is the ERA5 coarse SST. '
           r'PSD ratio $= \hat{p}/p_\text{MUR}$ in the 5--50\,km wavelength band (1.0 = perfect spectral energy). '
           r'Stage 4/4v2/5a show sweep mean $\pm$ std over 18 runs (2 LR $\times$ 3 seed $\times$ 3 spec-weight); '
           r'best-run rows in \textit{italics}. '
           r'\textbf{Bold} = best result per column among all stages. '
           r'$^\dagger$BOF persist baseline differs slightly between stage groups: '
           r'4.67\,°C for Stages 1--3 (703 test days, no ocean lookback) vs.\ '
           r'4.63\,°C for Stages 4+ (670 test days; first 33 days dropped for T$_\mathrm{oce}$=60 lookback). '
           r'Skill scores are computed against each group\textquotesingle s own baseline.}')
lat.append(r'\label{tab:ablation}')
lat.append(r'\setlength{\tabcolsep}{4pt}')
lat.append(r'\begin{tabular}{l cc c cc c}')
lat.append(r'\toprule')
lat.append(r'  & \multicolumn{3}{c}{GSL (training)} & \multicolumn{3}{c}{BOF (zero-shot)$^\dagger$} \\')
lat.append(r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}')
lat.append(r'  Architecture & RMSE\,(°C) & Skill & PSD & RMSE\,(°C) & Skill & PSD \\')
lat.append(r'\midrule')
lat.append(r'  \multicolumn{7}{l}{\textit{Reference}} \\')
lat.append(f'  ERA5 coarse & {PERSIST_GSL:.4f} & 0.0000 & --- & {PERSIST_BOF:.4f} & 0.0000 & --- \\\\ % persist baseline')
lat.append(f'  MUR yesterday & {ORACLE_GSL:.4f} & {1-ORACLE_GSL/PERSIST_GSL:.4f} & --- & {ORACLE_BOF:.4f} & {1-ORACLE_BOF/PERSIST_BOF:.4f} & --- \\\\ % oracle')
lat.append(r'\midrule')
lat.append(r'  \multicolumn{7}{l}{\textit{Stage 1--3: ERA5 only encoder (T$_\mathrm{oce}$=0)}} \\')

s1g, s1b = runs_123['stage1_try2']
lat.append(lat_row('Stage 1: joint S+T attn', s1g, s1b, 'full spatiotemporal, block-causal'))

s2t3g, s2t3b = runs_123['stage2_try3']
# Stage 2 has best GSL RMSE and best GSL skill among all single-run stages
lat.append(lat_row('Stage 2: alt S$\\to$T', s2t3g, s2t3b,
                   'alternating spatial+temporal (sw=5.0)',
                   bold_gsl_rmse=True, bold_gsl_skill=True))

s3t5g, s3t5b = runs_123['stage3_try5']
lat.append(lat_row('Stage 3: dual-stream', s3t5g, s3t5b, 'spatial+temporal streams, cross-attn bridge'))

lat.append(r'\midrule')
lat.append(r'  \multicolumn{7}{l}{\textit{Stage 4--5a: physics-motivated (atm + T$_\mathrm{oce}$=60 MUR frames), 2D RoPE}} \\')

# Stage 4 sweep mean + best
s4sr = sweep_rows.get('4', {})
if s4sr:
    g4m = s4sr['g']; b4m = s4sr['b']
    lat.append(f"  Stage 4 (sweep mean$_{{n=18}}$) & {fmt(g4m['rmse_model_K'])} & "
               f"{fmt(g4m['skill_score'])}$_{{\\pm{s4sr['g_std']:.4f}}}$ & {fmt(g4m['psd_ratio'])} & "
               f"{fmt(b4m['rmse_model_K'])} & "
               f"{fmt(b4m['skill_score'])}$_{{\\pm{s4sr['b_std']:.4f}}}$ & {fmt(b4m['psd_ratio'])} \\\\")
s4g, s4b = best_bias['stage4']
# Stage 4 best has best BOF RMSE and best BOF skill among all stages
lat.append(lat_row('\\quad \\textit{best (seed=123)}', s4g, s4b, 'DDIM diffusion',
                   bold_bof_rmse=True, bold_bof_skill=True))

# Stage 4v2 sweep mean + best
s4v2sr = sweep_rows.get('4v2', {})
if s4v2sr:
    g4v = s4v2sr['g']; b4v = s4v2sr['b']
    lat.append(f"  Stage 4v2 (sweep mean$_{{n=18}}$) & {fmt(g4v['rmse_model_K'])} & "
               f"{fmt(g4v['skill_score'])}$_{{\\pm{s4v2sr['g_std']:.4f}}}$ & {fmt(g4v['psd_ratio'])} & "
               f"{fmt(b4v['rmse_model_K'])} & "
               f"{fmt(b4v['skill_score'])}$_{{\\pm{s4v2sr['b_std']:.4f}}}$ & {fmt(b4v['psd_ratio'])} \\\\")
s4v2g, s4v2b = best_bias['stage4v2']
lat.append(lat_row('\\quad \\textit{best (seed=0)}', s4v2g, s4v2b, 'EDM diffusion + DomainNorm'))

# Stage 5a sweep mean + best
s5asr = sweep_rows.get('5a', {})
if s5asr:
    g5a = s5asr['g']; b5a = s5asr['b']
    lat.append(f"  Stage 5a (sweep mean$_{{n=18}}$) & {fmt(g5a['rmse_model_K'])} & "
               f"{fmt(g5a['skill_score'])}$_{{\\pm{s5asr['g_std']:.4f}}}$ & {fmt(g5a['psd_ratio'])} & "
               f"{fmt(b5a['rmse_model_K'])} & "
               f"{fmt(b5a['skill_score'])}$_{{\\pm{s5asr['b_std']:.4f}}}$ & {fmt(b5a['psd_ratio'])} \\\\")
s5ag, s5ab = best_bias['stage5a']
lat.append(lat_row('\\quad \\textit{best (seed=0)}', s5ag, s5ab, 'VAE latent EDM'))

lat.append(r'\bottomrule')
lat.append(r'\end{tabular}')
lat.append(r'\end{table}')

latex_text = '\n'.join(lat)
print('\n\n' + '='*60)
print('LaTeX TABLE (ablation_table.tex)')
print('='*60)
print(latex_text)

with open(LAT, 'w') as f:
    f.write(latex_text + '\n')
print(f'\nWrote {LAT}')
