"""
compile_gom_results.py — Summarise GOM zero-shot evaluation results.

Run after eval_gom.sh completes (job 47611236).

Usage:
  python scripts/compile_gom_results.py

Reads: src/eval_fixed/eval_gom_*.json
Prints: formatted table with GOM RMSE, Skill, PSD, seasonal bias
        and comparison to GSL training domain.
"""
import json, glob, os

EVAL = '/scratch/pdoshi/my_project/eddyflow/src/eval_fixed'

# GOM reference baselines — computed dynamically from eval JSONs
# (ERA5 GOM persist RMSE varies by season/domain — take from eval JSON)

GOM_FILES = sorted(glob.glob(f'{EVAL}/eval_gom_*.json'))
if not GOM_FILES:
    print(f'No GOM eval files found in {EVAL}/')
    print('Run eval_gom.sh first (SLURM job 47611236).')
    raise SystemExit(1)

print('=' * 100)
print('EddyFlow — GOM Zero-Shot Evaluation Results')
print('=' * 100)
print()

rows = []
for f in GOM_FILES:
    d = json.load(open(f))
    tag = os.path.basename(f).replace('eval_gom_', '').replace('.json', '')
    rows.append((tag, d))

# Sort by BOF skill descending
rows.sort(key=lambda x: -x[1].get('skill_score', 0))

print(f"{'Stage':25s}  {'RMSE':>8}  {'Persist':>8}  {'Skill':>8}  {'PSD':>8}  {'Oracle':>8}  DJF   MAM   JJA   SON")
print('-' * 95)

for tag, d in rows:
    rmse   = d.get('rmse_model_K', float('nan'))
    persist = d.get('rmse_persist_K', float('nan'))
    skill  = d.get('skill_score', float('nan'))
    psd    = d.get('psd_ratio', float('nan'))
    oracle = d.get('rmse_murprev_K', float('nan'))
    sb     = d.get('seasonal_bias_K', {})
    djf = sb.get('DJF', float('nan'))
    mam = sb.get('MAM', float('nan'))
    jja = sb.get('JJA', float('nan'))
    son = sb.get('SON', float('nan'))
    print(f'  {tag:23s}  {rmse:8.4f}  {persist:8.4f}  {skill:8.4f}  {psd:8.4f}  {oracle:8.4f}  {djf:+.2f}  {mam:+.2f}  {jja:+.2f}  {son:+.2f}')

print()
print('Oracle (MUR yesterday) RMSE shown for reference. Skill = 1 - RMSE_model / RMSE_persist(ERA5 GOM).')

# Compare to GSL and BOF best-run results
print()
print('=' * 100)
print('CROSS-DOMAIN COMPARISON (best-run for each stage)')
print('=' * 100)
print()

gsl_bof_map = {
    'stage4v2_sw0lr0': ('eval_gsl_stage4v2_best_bias.json', 'eval_bof_stage4v2_best_bias.json'),
    'stage5a_sw0lr0':  ('eval_gsl_stage5a_best_bias.json',  'eval_bof_stage5a_best_bias.json'),
    'stage4_sw0lr0':   ('eval_gsl_stage4_best_bias.json',   'eval_bof_stage4_best_bias.json'),
}

print(f"{'Stage':25s}  {'GSL Skill':>10}  {'BOF Skill':>10}  {'GOM Skill':>10}  {'GOM/GSL':>9}  {'GOM/BOF':>9}")
print('-' * 80)

for gom_tag, d_gom in rows:
    # Try to find matching GSL/BOF result
    match_key = None
    for k in gsl_bof_map:
        if k.replace('_', '') in gom_tag.replace('_', '') or gom_tag in k:
            match_key = k
            break
    # More flexible matching
    if not match_key:
        for k in gsl_bof_map:
            stage_part = k.split('_sw')[0]  # e.g. 'stage4v2'
            if stage_part in gom_tag:
                match_key = k
                break

    gsl_skill = bof_skill = float('nan')
    if match_key:
        gf, bf = gsl_bof_map[match_key]
        try:
            gsl_d = json.load(open(f'{EVAL}/{gf}'))
            bof_d = json.load(open(f'{EVAL}/{bf}'))
            gsl_skill = gsl_d['skill_score']
            bof_skill = bof_d['skill_score']
        except Exception:
            pass

    gom_skill = d_gom.get('skill_score', float('nan'))
    gom_gsl = gom_skill - gsl_skill if gsl_skill == gsl_skill else float('nan')
    gom_bof = gom_skill - bof_skill if bof_skill == bof_skill else float('nan')

    gsl_str = f'{gsl_skill:.4f}' if gsl_skill == gsl_skill else '------'
    bof_str = f'{bof_skill:.4f}' if bof_skill == bof_skill else '------'
    gom_str = f'{gom_skill:.4f}' if gom_skill == gom_skill else '------'
    gg_str  = f'{gom_gsl:+.4f}' if gom_gsl == gom_gsl else '------'
    gb_str  = f'{gom_bof:+.4f}' if gom_bof == gom_bof else '------'

    print(f'  {gom_tag:23s}  {gsl_str:>10}  {bof_str:>10}  {gom_str:>10}  {gg_str:>9}  {gb_str:>9}')

print()
print('GOM/GSL, GOM/BOF = skill difference (GOM − domain). Negative = drop vs training domain.')
print('DomainNorm stages (4v2, 5a) expected to show smaller drop due to explicit domain conditioning.')
