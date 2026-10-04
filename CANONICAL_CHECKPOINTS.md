# Canonical checkpoints for reported numbers

## Downloading the checkpoints

The six merged checkpoints in the table below (one per ladder stage: 1, 2, 3,
4, 4v2, 5a) are published as assets on the
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

## Checkpoint registry

| Ladder stage | Paper label | Checkpoint path (relative to `outputs/checkpoints/`) |
|---|---|---|
| 1 | Stage 1 | `stage1_phase2_sw_s1_sw1_merged.pt` |
| 2 | Stage 2 | `stage2_phase2_sw_s2_sw0_merged.pt` |
| 3 | Stage 3 | `stage3_phase2_sw_s3_sw0_merged.pt` |
| 4 | Stage 4 (EddyFlow) | `stage4_sweep_4_123_lr0_sw0_merged.pt` |
| 4v2 | Stage 5 | `stage4v2_sweep_4v2_0_lr0_sw0_merged.pt` |
| 5a | Stage 6 | `stage5a_sweep_5a_0_lr0_sw0_merged.pt` |

Any script that generates a paper table (Table 5/7/8 and their few-shot/
ablation counterparts) loads checkpoints from this list.

Note: no `5b` checkpoint exists. Stage 6 in the paper corresponds to code
stage `5a`.
