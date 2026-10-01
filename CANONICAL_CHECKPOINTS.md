# Canonical checkpoints for reported numbers

## Downloading the checkpoints

The six merged checkpoints in the "Corrected via provenance trace" table below
(one per ladder stage: 1, 2, 3, 4, 4v2, 5a) are published as assets on the
[`checkpoints-v1`](../../releases/tag/checkpoints-v1) GitHub Release, since
`outputs/checkpoints/` is gitignored and not committed to this repository.
Download them with:

```bash
gh release download checkpoints-v1 -D outputs/checkpoints/
```

or directly from the release page if you don't have the `gh` CLI installed.
These are the only checkpoints needed to reproduce Table 5/7/8 and their
few-shot counterparts; the full sweep/few-shot/ablation checkpoint set
(529 files, ~44GB) is not published and can be regenerated from `scripts/sweep/`.

Written 2026-09-27, during the KDD-review correctness audit, because multiple
candidate checkpoints exist per stage on disk (e.g. Stage 1 alone has
`stage1_phase2_try2_merged.pt` and `stage1_phase2_sw_s1_sw{0,1,2}_merged.pt`)
with no prior single record of which one is "the" reported one. This file is
that record. Any script that generates a paper table (Table 5/7/8 and their
few-shot/ablation counterparts) must load checkpoints from this list, not
from whatever path happens to be hardcoded in an individual `eval_*.sh`
script — those scripts should themselves be updated to source this file.

These are the exact checkpoints already used consistently across today's
`_erafix` rerun (job 61842032), `save_bof_fields.sh` (job 61842095), and
match what the pre-existing per-stage `eval_gom_stage*.sh`/`eval_bof_stage*.sh`
scripts in the repo root already pointed at before today's fixes — chosen
for continuity/before-after comparability with prior (pre-audit) numbers,
not re-selected from scratch.

## SUPERSEDED 2026-09-29 — see "Corrected via provenance trace" below.
The table originally here (`try2`/`try3`/`try5`/`try2` for stages 1/2/3/4v2)
was never actually verified against what produced the published GSL Table 5
numbers — it was carried over from pre-existing `eval_*.sh` scripts on the
assumption they were right. They weren't, for 4 of 6 stages. Do not use the
old table below this notice for anything; kept struck-through for the
history, not as a live reference.

~~| Ladder stage | Paper label | Checkpoint path (relative to `outputs/checkpoints/`) |~~
~~|---|---|---|~~
~~| 1 | Stage 1 | `stage1_phase2_try2_merged.pt` |~~
~~| 2 | Stage 2 | `stage2_phase2_try3_merged.pt` |~~
~~| 3 | Stage 3 | `stage3_phase2_try5_merged.pt` |~~
~~| 4 | Stage 4 (EddyFlow) | `stage4_sweep_4_123_lr0_sw0_merged.pt` |~~
~~| 4v2 | Stage 5 | `stage4v2_try2_merged.pt` |~~
~~| 5a | Stage 6 | `stage5a_sweep_5a_0_lr0_sw0_merged.pt` |~~

## Corrected via provenance trace (2026-09-29) — verified, not assumed

Traced by matching each stage's published GSL Table 5 `rmse_model_K` against
every `eval_fixed/eval_gsl_*.json` on disk, then confirming empirically for
Stage 1 (fresh eval of the candidate checkpoint reproduced 0.5381 to within
0.09%, job 62077175) and directly for Stages 4/4v2/5a via the `checkpoint`
field recorded in their JSONs or `eval_best_bias.sh`'s explicit checkpoint
arguments. Stages 2/3 confirmed by filename-pattern + file-existence only
(not independently re-run) — same standard of evidence as the original
Stage 1 finding before its empirical confirmation, so treat as strong but
not yet at the same certainty tier.

| Ladder stage | Paper label | Correct checkpoint | Matched `_erafix`'s pick? |
|---|---|---|---|
| 1 | Stage 1 | `stage1_phase2_sw_s1_sw1_merged.pt` | **No** — `_erafix` used `try2` |
| 2 | Stage 2 | `stage2_phase2_sw_s2_sw0_merged.pt` | **No** — `_erafix` used `try3` |
| 3 | Stage 3 | `stage3_phase2_sw_s3_sw0_merged.pt` | **No** — `_erafix` used `try5` |
| 4 | Stage 4 (EddyFlow) | `stage4_sweep_4_123_lr0_sw0_merged.pt` | Yes |
| 4v2 | Stage 5 | `stage4v2_sweep_4v2_0_lr0_sw0_merged.pt` | **No** — `_erafix` used `try2` |
| 5a | Stage 6 | `stage5a_sweep_5a_0_lr0_sw0_merged.pt` | Yes |

**Consequence**: tonight's `_erafix` BOF/GOM reruns (job 61842032) for
Stages 1, 2, 3, and 4v2 do not correspond to the same checkpoint as their
own GSL column entry. Those four stages' BOF/GOM numbers need rerunning
against the checkpoints in this table before the corrected Table 5 can be
called internally consistent row-by-row. Stage 4 and Stage 5a's `_erafix`
numbers are already correct and don't need rerunning.

No `5b` checkpoint exists on disk and none is canonical — see prior
conversation notes on Stage 6 = code `5a` (not `5b`; `5b` is dead code).

## Open item

The stats-sensitivity check (job 61842183) used `stage1_phase2_try2_merged.pt`
per this table and got `rmse_model_K=0.6420`, which does not match the
paper's currently-printed Table 5 Stage 1 GSL RMSE of 0.5381. That number is
being superseded by `_erafix` regardless of which checkpoint produced it, so
tracing the old discrepancy is moot — but once `_erafix`'s new numbers land,
verify they were generated against *this* table's paths before trusting them
as the new canonical Table 5/7/8, so this exact ambiguity doesn't recur.

## Open item 2 (2026-09-28): single-checkpoint results, and one provenance mismatch

`_erafix`'s GOM numbers (job 61842032) are each from **one specific checkpoint
per stage**, not seed-averaged — confirmed directly from each JSON's
`checkpoint` field. Stage 5a used `stage5a_sweep_5a_0_lr0_sw0_merged.pt`
(seed=0 only; seeds 42/123 at the same lr/sw exist and haven't been checked
yet — job 61905082 tests them). Stage 4v2 used `stage4v2_try2_merged.pt`,
which is **not part of the proper seed-sweep family at all**
(`stage4v2_sweep_4v2_{0,42,123}_lr0_sw0_merged.pt` exists separately, 18
variants total, unrelated provenance to `try2`/`try1`). Any claim about
"the architecture" on GOM built from these two specific checkpoints should
be treated as provisional until the seed-sweep family is checked — a
consistent cold bias across seeds argues for a real architectural/domain-
shift property; a bias specific to one checkpoint argues for training-
instability or checkpoint-selection instead. This table's Stage 4v2 pick
in particular (`try2`) should probably be replaced with a proper sweep-
family member (e.g. `stage4v2_sweep_4v2_0_lr0_sw0_merged.pt`, matching
Stage 5a's lr/sw choice) once job 61905082's comparison point lands, for
consistency with how every other stage in this table was selected.
