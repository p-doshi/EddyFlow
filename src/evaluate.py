"""
evaluate.py — Evaluation script for TemporalDownscaler (all stages)

Aligned with train.py conventions:
  - Uses make_loaders() from dataset.py (same split logic as training)
  - All internal computation in normalised SST units
  - Physical °C conversion at print time only via --sst_std (display only)
  - Persistence baseline: ERA5 ch3 (SST in K) de-normalised → °C → re-normalised
    to MUR SST space, then bilinearly upsampled to fine grid
  - weight > 0 mask (ice-free, high-confidence pixels) — matches validate()
  - model.sample() for inference, same as validate()

Metrics reported:
  1. Skill Score  SS = 1 − RMSE(model) / RMSE(ΔSST persistence)
     Denominator expected ~3.28 °C (all days) / ~3.57 °C (obs-rich) on GSL test set
  2. Azimuthally-averaged PSD ratio in the 5–50 km wavelength band
  3. rmse_murprev_K — reference only (yesterday MUR SST, oracle upper bound ~0.43–0.50 °C)

dataset.py prerequisite — add to __getitem__ return dict:
    mur_prev      = max(mur_i - 1, 0)
    sst_prev_raw  = self.mur[mur_prev].astype(np.float32)
    sst_prev_norm = (sst_prev_raw - self.sst_mean) / (self.sst_std + 1e-6)
    sst_prev_norm = np.where(np.isfinite(sst_prev_norm), sst_prev_norm, 0.0)
    sst_prev      = torch.from_numpy(sst_prev_norm).unsqueeze(0)  # [1,501,1201]

Usage:
  python evaluate.py \
      --checkpoint checkpoints/stage1_T_28_best.pt \
      --stage      1 \
      --dx_km      6.0 \
      --batch_size 4 \
      --output     results_stage1.json \
      [--psd_curves]        # also save full k vs PSD arrays
      [--sst_std 6.33]      # physical display only — does not affect any metric
      [--n_diff_steps 50]   # override DDIM steps
"""

import argparse
import json
import math
import os
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config  import cfg
from dataset import make_loaders
from model   import TemporalDownscaler
SEASON_MONTHS = [3, 6, 9, 12]

# ═══════════════════════════════════════════════════════════════════════════════
# PSD UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def azimuthal_psd(field: np.ndarray, dx_km: float):
    """
    Azimuthally-averaged 2-D PSD for a single [H, W] field.

    Returns
    -------
    k_cents : wavenumber bin centres in cycles/km  [K]
    psd     : mean power per bin                   [K]
    """
    H, W  = field.shape
    f2d   = np.fft.rfft2(field)
    p2d   = (np.abs(f2d) ** 2) / (H * W)

    ky = np.fft.fftfreq(H,  d=dx_km)
    kx = np.fft.rfftfreq(W, d=dx_km)
    KX, KY = np.meshgrid(kx, ky)
    K2D    = np.sqrt(KX ** 2 + KY ** 2)

    k_max   = min(kx.max(), np.abs(ky).max())
    n_bins  = max(H, W) // 2
    k_edges = np.linspace(0.0, k_max, n_bins + 1)
    k_cents = 0.5 * (k_edges[:-1] + k_edges[1:])

    psd = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (K2D >= k_edges[i]) & (K2D < k_edges[i + 1])
        if mask.sum() > 0:
            psd[i] = p2d[mask].mean()

    return k_cents, psd


def band_mean_psd(field: np.ndarray, dx_km: float,
                  lam_lo_km: float = 5.0,
                  lam_hi_km: float = 50.0) -> float:
    """
    Mean PSD power in wavelength band [lam_lo_km, lam_hi_km].
    λ = 1/k  →  k ∈ [1/lam_hi, 1/lam_lo] cycles/km.
    """
    k_cents, psd = azimuthal_psd(field, dx_km)
    lo  = 1.0 / lam_hi_km
    hi  = 1.0 / lam_lo_km
    sel = (k_cents >= lo) & (k_cents <= hi)
    return float(psd[sel].mean()) if sel.sum() > 0 else float('nan')

GSL_EXTENT = [-69.5, -56.0, 44.0, 52.8]  # lon_min, lon_max, lat_min, lat_max

def make_run_stamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def infer_lonlat_from_field(field_2d: np.ndarray, extent=GSL_EXTENT):
    h, w = field_2d.shape
    lon = np.linspace(extent[0], extent[1], w, dtype=np.float32)
    lat = np.linspace(extent[2], extent[3], h, dtype=np.float32)
    return lon, lat

def safe_slug(s: str) -> str:
    return str(s).replace(":", "-").replace(" ", "_").replace("/", "-")

def compute_field_metrics(pred: np.ndarray, target: np.ndarray):
    mask = np.isfinite(pred) & np.isfinite(target)
    if mask.sum() == 0:
        return {
            "n_valid": 0,
            "rmse": None,
            "mae": None,
            "bias": None,
            "min_err": None,
            "max_err": None,
        }

    err = pred[mask] - target[mask]
    return {
        "n_valid": int(mask.sum()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "min_err": float(np.min(err)),
        "max_err": float(np.max(err)),
    }

def month_name(m: int) -> str:
    names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December",
    }
    return names.get(int(m), f"Month-{m}")
# ═══════════════════════════════════════════════════════════════════════════════
# STATS LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_normalisation_stats() -> dict:
    """
    Load ERA5 and MUR normalisation stats from cfg paths.
    Returns a dict with float/tensor values ready for use in persistence_pred.
    """
    era5_mean = np.load(cfg.ERA5_MEAN).astype(np.float32)  # [C]
    era5_std  = np.load(cfg.ERA5_STD ).astype(np.float32)  # [C]
    sst_mean  = float(np.load(cfg.SST_MEAN))
    sst_std   = float(np.load(cfg.SST_STD ))

    print(f'  ERA5 ch3 mean (SST) : {era5_mean[3]:.4f} K')
    print(f'  ERA5 ch3 std  (SST) : {era5_std[3]:.4f} K')
    print(f'  MUR  sst_mean       : {sst_mean:.4f} °C')
    print(f'  MUR  sst_std        : {sst_std:.4f} °C')

    return {
        'era5_mean': era5_mean,
        'era5_std':  era5_std,
        'sst_mean':  sst_mean,
        'sst_std':   sst_std,
    }

from scipy.ndimage import gaussian_filter

def save_cartopy_map_png(
    out_path: Path,
    field: np.ndarray,
    title: str,
    extent=GSL_EXTENT,
    cmap="viridis",
    vmin=None,
    vmax=None,
    lon=None,
    lat=None,
):
    # Optional: smooth only for visualization (do not use for metrics!)
    field_vis = gaussian_filter(field, sigma=0.4)

    fig = plt.figure(figsize=(12, 8), dpi=240)  # higher DPI
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(resolution="10m", linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor="#d9d3c3", zorder=10)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=11)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    # Use imshow with explicit interpolation instead of pcolormesh
    im = ax.imshow(
        field_vis,
        origin='lower',                 # match your grid layout
        extent=extent,                  # [lon_min, lon_max, lat_min, lat_max]
        transform=ccrs.PlateCarree(),   # proper projection
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation='bicubic',        # smoother look
        zorder=1,
    )

    ax.set_title(title, fontsize=12)
    cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cb.set_label("Normalised SST")

    # Increase output DPI to avoid pixelation when zooming
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    
def write_split_slider_html(
    out_path: Path,
    title: str,
    left_label: str,
    right_label: str,
    stats_html: str,
):
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    --bg:#0f1115; --panel:#171a21; --panel2:#1e2430;
    --text:#e8edf2; --muted:#9aa6b2; --border:#2a3443;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:Inter,Arial,sans-serif; background:var(--bg); color:var(--text); }}
  .wrap {{ max-width:1320px; margin:0 auto; padding:20px; }}
  h1 {{ margin:0 0 8px; font-size:1.25rem; }}
  p  {{ margin:0 0 14px; color:var(--muted); }}
  .grid {{ display:grid; grid-template-columns:1.4fr 0.9fr; gap:20px; }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:16px; }}

  /* ── Compare ── */
  .compare {{
    position:relative; width:100%; overflow:hidden;
    border-radius:12px; cursor:ew-resize;
    user-select:none; -webkit-user-select:none;
    background:#0b0d11; touch-action:none;
  }}
  .compare > img {{
    display:block; width:100%; height:auto;
    pointer-events:none; -webkit-user-drag:none;
  }}
  .img-top {{
    position:absolute; inset:0;
    will-change:clip-path;   /* GPU-composited — no repaint on drag */
  }}
  .img-top img {{
    width:100%; height:100%; object-fit:cover;
    pointer-events:none; -webkit-user-drag:none;
  }}
  .divider {{
    position:absolute; top:0; bottom:0;
    width:2px; background:#fff;
    transform:translateX(-50%);
    pointer-events:none; z-index:10;
    will-change:left;
  }}
  .knob {{
    position:absolute; top:50%; left:50%;
    transform:translate(-50%,-50%);
    width:40px; height:40px; border-radius:50%;
    background:#fff; display:flex;
    align-items:center; justify-content:center;
    box-shadow:0 2px 10px rgba(0,0,0,0.5);
  }}
  .lbl {{
    position:absolute; top:10px;
    padding:3px 9px; border-radius:5px;
    font-size:0.75rem; font-weight:700;
    background:rgba(0,0,0,0.6); color:#fff;
    pointer-events:none; letter-spacing:.04em; z-index:11;
  }}
  .lbl-l {{ left:10px; }}
  .lbl-r {{ right:10px; }}
  .loading {{
    position:absolute; inset:0;
    display:flex; align-items:center; justify-content:center;
    background:#0b0d11; color:var(--muted);
    font-size:0.85rem; z-index:20; border-radius:12px;
  }}

  /* ── Meta ── */
  .meta {{ display:grid; gap:12px; }}
  .meta-card {{ background:var(--panel2); border-radius:12px; padding:12px 14px; border:1px solid var(--border); }}
  .meta-card h3 {{ margin:0 0 8px; font-size:0.95rem; }}
  .meta-card table {{ width:100%; border-collapse:collapse; font-size:0.92rem; }}
  .meta-card td {{ padding:4px 0; vertical-align:top; }}
  .meta-card td:first-child {{ color:var(--muted); width:44%; }}
  @media (max-width:980px) {{ .grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="wrap">
  <h1>{title}</h1>
  <p>Drag the divider to compare maps over the Gulf of St. Lawrence.</p>

  <div class="grid">
    <div class="panel">
      <div class="compare" id="compare">
        <div class="loading" id="loading">Loading…</div>

        <!-- Bottom layer (right / after) — sets container height -->
        <img id="img-b" src="me.png" alt="{right_label}" style="visibility:hidden">

        <!-- Top layer (left / before) — GPU-clipped, never resized -->
        <div class="img-top" id="img-top" style="clip-path:inset(0 50% 0 0);visibility:hidden">
          <img id="img-t" src="mur.png" alt="{left_label}">
        </div>

        <div class="divider" id="divider" style="left:50%">
          <div class="knob">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
                 stroke="#333" stroke-width="2.5"
                 stroke-linecap="round" stroke-linejoin="round">
              <polyline points="15 18 9 12 15 6"/>
              <polyline points="9 18 3 12 9 6" transform="translate(12,0)"/>
            </svg>
          </div>
        </div>

        <span class="lbl lbl-l">{left_label}</span>
        <span class="lbl lbl-r">{right_label}</span>
      </div>
    </div>

    <div class="meta">{stats_html}</div>
  </div>
</div>

<script>
  const wrap    = document.getElementById('compare');
  const top     = document.getElementById('img-top');
  const divEl   = document.getElementById('divider');
  const loading = document.getElementById('loading');
  const imgB    = document.getElementById('img-b');
  const imgT    = document.getElementById('img-t');

  // Wait for both images before showing anything
  let loaded = 0;
  function onLoad() {{
    if (++loaded < 2) return;
    loading.style.display = 'none';
    imgB.style.visibility = '';
    top.style.visibility  = '';
  }}
  imgB.addEventListener('load', onLoad);
  imgT.addEventListener('load', onLoad);
  if (imgB.complete) onLoad();
  if (imgT.complete) onLoad();

  // rAF-throttled drag — one DOM write per frame, GPU does the rest
  let drag = false, raf = null, pendingX = 0;

  function apply() {{
    const r = wrap.getBoundingClientRect();
    const p = Math.max(0, Math.min(100, (pendingX - r.left) / r.width * 100));
    top.style.clipPath = `inset(0 ${{100 - p}}% 0 0)`;
    divEl.style.left   = p + '%';
    raf = null;
  }}
  function schedule(cx) {{
    pendingX = cx;
    if (!raf) raf = requestAnimationFrame(apply);
  }}

  wrap.addEventListener('mousedown',   e => {{ drag = true; schedule(e.clientX); e.preventDefault(); }});
  window.addEventListener('mousemove', e => {{ if (drag) schedule(e.clientX); }});
  window.addEventListener('mouseup',   () => drag = false);

  wrap.addEventListener('touchstart',  e => {{ drag = true;  schedule(e.touches[0].clientX); }}, {{passive:true}});
  window.addEventListener('touchmove', e => {{ if (drag) schedule(e.touches[0].clientX); }},     {{passive:true}});
  window.addEventListener('touchend',  () => drag = false);
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")

def save_interactive_comparison_set(
    out_root: Path,
    sample_dt_str: str,
    sample_date: str,
    sample_time: str,
    sample_tag: str,
    pred: np.ndarray,
    era5: np.ndarray,
    mur: np.ndarray,
    mur_prev: np.ndarray | None = None,
    extent=GSL_EXTENT,
    lon=None,
    lat=None,
):
    sample_folder = out_root / f"{safe_slug(sample_date)}_{safe_slug(sample_time)}_{sample_tag}"
    sample_folder.mkdir(parents=True, exist_ok=True)

    finite_vals = np.concatenate([
        pred[np.isfinite(pred)],
        era5[np.isfinite(era5)],
        mur[np.isfinite(mur)],
    ])
    if finite_vals.size == 0:
        raise ValueError("all fields are NaN after masking")

    vmin = float(np.nanpercentile(finite_vals, 1))
    vmax = float(np.nanpercentile(finite_vals, 99))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin == vmax):
        vmin = float(np.nanmin(finite_vals))
        vmax = float(np.nanmax(finite_vals))
        if vmin == vmax:
            vmax = vmin + 1e-6

    pred_png = sample_folder / "me.png"
    era5_png = sample_folder / "era5.png"
    mur_png  = sample_folder / "mur.png"

    save_cartopy_map_png(pred_png, pred, f"My prediction — {sample_dt_str}",
                         extent=extent, cmap="viridis", vmin=vmin, vmax=vmax, lon=lon, lat=lat)
    save_cartopy_map_png(era5_png, era5, f"ERA5 baseline — {sample_dt_str}",
                         extent=extent, cmap="viridis", vmin=vmin, vmax=vmax, lon=lon, lat=lat)
    save_cartopy_map_png(mur_png, mur, f"MUR target — {sample_dt_str}",
                         extent=extent, cmap="viridis", vmin=vmin, vmax=vmax, lon=lon, lat=lat)

    me_vs_mur = compute_field_metrics(pred, mur)
    era5_vs_mur = compute_field_metrics(era5, mur)
    me_vs_era5 = compute_field_metrics(pred, era5)

    stats_html = f"""
    <div class="meta-card">
      <h3>Sample details</h3>
      <table>
        <tr><td>Date</td><td>{sample_date}</td></tr>
        <tr><td>Time</td><td>{sample_time}</td></tr>
        <tr><td>Timestamp</td><td>{sample_dt_str}</td></tr>
      </table>
    </div>

    <div class="meta-card">
      <h3>Me vs MUR</h3>
      <table>
        <tr><td>Valid pixels</td><td>{me_vs_mur["n_valid"]}</td></tr>
        <tr><td>RMSE</td><td>{me_vs_mur["rmse"]}</td></tr>
        <tr><td>MAE</td><td>{me_vs_mur["mae"]}</td></tr>
        <tr><td>Bias</td><td>{me_vs_mur["bias"]}</td></tr>
        <tr><td>Min error</td><td>{me_vs_mur["min_err"]}</td></tr>
        <tr><td>Max error</td><td>{me_vs_mur["max_err"]}</td></tr>
      </table>
    </div>

    <div class="meta-card">
      <h3>ERA5 vs MUR</h3>
      <table>
        <tr><td>Valid pixels</td><td>{era5_vs_mur["n_valid"]}</td></tr>
        <tr><td>RMSE</td><td>{era5_vs_mur["rmse"]}</td></tr>
        <tr><td>MAE</td><td>{era5_vs_mur["mae"]}</td></tr>
        <tr><td>Bias</td><td>{era5_vs_mur["bias"]}</td></tr>
        <tr><td>Min error</td><td>{era5_vs_mur["min_err"]}</td></tr>
        <tr><td>Max error</td><td>{era5_vs_mur["max_err"]}</td></tr>
      </table>
    </div>

    <div class="meta-card">
      <h3>Me vs ERA5</h3>
      <table>
        <tr><td>Valid pixels</td><td>{me_vs_era5["n_valid"]}</td></tr>
        <tr><td>RMSE</td><td>{me_vs_era5["rmse"]}</td></tr>
        <tr><td>MAE</td><td>{me_vs_era5["mae"]}</td></tr>
        <tr><td>Bias</td><td>{me_vs_era5["bias"]}</td></tr>
      </table>
    </div>
    """

    write_split_slider_html(
        sample_folder / "me_vs_era5.html",
        title=f"{sample_dt_str} — Me vs ERA5",
        left_label="Me",
        right_label="ERA5",
        stats_html=stats_html,
    )
    write_split_slider_html(
        sample_folder / "me_vs_mur.html",
        title=f"{sample_dt_str} — Me vs MUR",
        left_label="Me",
        right_label="MUR",
        stats_html=stats_html,
    )

    return sample_folder
# ═══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE BASELINE  —  ERA5 SST channel de-normalised → MUR SST space
# ═══════════════════════════════════════════════════════════════════════════════

def persistence_pred(batch: dict, stats: dict) -> torch.Tensor:
    """ERA5 SST ch3 upsampled to fine grid — coarse ΔSST=0 baseline (~3.28 °C RMSE).
    Chain: ERA5-normalised → Kelvin → Celsius → MUR-normalised → bilinear upsample."""
    # Most recent ERA5 timestep, ch3 = sst_era5: [B, H_c, W_c]
    era5_sst_norm = batch['era5'][:, -1, 3, :, :].float()

    era5_sst_C = (era5_sst_norm * float(stats['era5_std'][3])
                  + float(stats['era5_mean'][3])
                  - 273.15)                                    # K → °C

    era5_mur = (era5_sst_C - float(stats['sst_mean'])) / (float(stats['sst_std']) + 1e-6)

    fine_h, fine_w = batch['sst_abs'].shape[-2:]
    return F.interpolate(
        era5_mur.unsqueeze(1), size=(fine_h, fine_w),
        mode='bilinear', align_corners=False,
    )  # [B, 1, H_f, W_f]

# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

class Evaluator:
    """
    Accumulates metrics across batches then computes aggregates.

    All arithmetic is in normalised SST units.
    Physical °C conversion happens only in compute() for display.

    Masking rule (identical to validate() in train.py):
        valid = (weight > 0) & isfinite(sst) & isfinite(pred)

    Skill Score denominator: RMSE of ERA5-SST ΔSST=0 persistence.
    Expected ~3.28 °C (all days) / ~3.57 °C (obs-rich) on GSL 2022–2023.

    rmse_murprev_K: oracle reference only — not used for SS.
    Expected ~0.43 °C (all days) / ~0.50 °C (obs-rich).
    """

    def __init__(self, model: TemporalDownscaler, device: torch.device,
                 stats: dict,
                 dx_km: float = 6.0,
                 lam_lo_km: float = 5.0,
                 lam_hi_km: float = 50.0,
                 sst_std: float = 1.0, save_examples: bool = False, example_dir = "../outputs/figures/runs", max_examples: int = 8):
        self.model   = model.to(device).eval()
        self.device  = device
        self.stats   = stats
        self.dx_km   = dx_km
        self.lam_lo  = lam_lo_km
        self.lam_hi  = lam_hi_km
        self.sst_std = sst_std        # display-only unit conversion
        self.save_examples = save_examples
        self.example_dir = Path(example_dir)
        self.max_examples = max_examples
        self.saved_examples = 0
        self.run_stamp = make_run_stamp()
        self.day_str, self.time_str = self.run_stamp.split("_", 1)
        self.selected_months = set(SEASON_MONTHS)
        self.month_to_saved = {m: 0 for m in self.selected_months}
        self.max_per_month = 1

        if self.save_examples:
            self.example_dir.mkdir(parents=True, exist_ok=True)
            print(f"[viz] saving examples to: {self.example_dir.resolve()}")

        self.sq_model   = 0.0
        self.sq_persist = 0.0
        self.sq_murprev = 0.0         # oracle reference — not used for SS
        self.n_valid    = 0

        self.psd_pred_band = []
        self.psd_targ_band = []
        self.n_samples     = 0

        # Seasonal mean bias: bias_sum[m] = sum(pred - sst) over valid pixels for month m
        self.bias_sum   = {m: 0.0 for m in range(1, 13)}
        self.bias_count = {m: 0   for m in range(1, 13)}
       

    @torch.no_grad()
    def update(self, batch: dict):
        batch_cpu = batch.copy()
        # move only tensors to GPU
        batch_tensors = {k: v for k, v in batch.items() if torch.is_tensor(v)}
        batch = {k: v.to(self.device) for k, v in batch_tensors.items()}
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # latent = self.model._encode(batch)
            # pred = self.model.baseline(latent, batch['bathy'])
            pred = self.model.sample(batch)

        pred    = (pred + batch['sst_prev']).float()   # delta → absolute SST
        sst     = batch['sst_abs'].float()             # compare against absolute target                                 \
        w       = batch['weight'].float()                               # [B, 1, H, W]
        persist = persistence_pred(batch, self.stats).float()          # [B, 1, H, W]

        # Valid mask — identical to validate() in train.py
        valid = (w > 0) & torch.isfinite(sst) & torch.isfinite(pred)
        vf    = valid.float()

        self.sq_model   += ((pred    - sst) ** 2 * vf).sum().item()
        self.sq_persist += ((persist - sst) ** 2 * vf).sum().item()
        self.n_valid    += valid.sum().item()

        # Oracle reference: yesterday's MUR SST (if available)
        if 'sst_prev' in batch:
            mur_prev = batch['sst_prev'].float()
            self.sq_murprev += ((mur_prev - sst) ** 2 * vf).sum().item()

        # Seasonal mean bias — accumulate per month (in physical °C)
        err_phys = ((pred - sst) * vf * self.sst_std)   # [B, 1, H, W] in °C
        months = batch['month'].cpu().tolist() if torch.is_tensor(batch.get('month')) else \
                 [batch_cpu['month'][b].item() for b in range(sst.shape[0])]
        for b_idx, m in enumerate(months):
            self.bias_sum[m]   += err_phys[b_idx].sum().item()
            self.bias_count[m] += valid[b_idx].sum().item()

        # PSD — zero-fill masked pixels to suppress land artefacts
        pred_np   = (pred * vf).squeeze(1).cpu().numpy()
        target_np = (sst  * vf).squeeze(1).cpu().numpy()

        for b in range(pred_np.shape[0]):
            self.psd_pred_band.append(
                band_mean_psd(pred_np[b],   self.dx_km, self.lam_lo, self.lam_hi)
            )
            self.psd_targ_band.append(
                band_mean_psd(target_np[b], self.dx_km, self.lam_lo, self.lam_hi)
            )

        self.n_samples += sst.shape[0]
        if self.save_examples:
            pred_np_full  = pred.squeeze(1).detach().cpu().numpy()
            sst_np_full   = sst.squeeze(1).detach().cpu().numpy()
            era5_np_full  = persist.squeeze(1).detach().cpu().numpy()
            valid_np_full = valid.squeeze(1).detach().cpu().numpy().astype(bool)

            batch_dates   = [batch_cpu['date_str'][b] for b in range(pred_np_full.shape[0])]
            batch_times   = [batch_cpu['time_str'][b] for b in range(pred_np_full.shape[0])]
            batch_months  = [batch_cpu['month'][b].item() for b in range(pred_np_full.shape[0])]

            if batch_dates is None or batch_times is None or batch_months is None:
                raise KeyError("Batch must include date_str, time_str, and month metadata from dataset.py")

            if torch.is_tensor(batch_months):
                batch_months = batch_months.detach().cpu().tolist()

            for b in range(pred_np_full.shape[0]):
                month = int(batch_months[b])
                if month not in self.selected_months:
                    continue
                if self.month_to_saved[month] >= self.max_per_month:
                    continue

                pred_b = np.where(valid_np_full[b], pred_np_full[b], np.nan)
                mur_b  = np.where(valid_np_full[b], sst_np_full[b], np.nan)
                era5_b = np.where(valid_np_full[b], era5_np_full[b], np.nan)

                all_vals = np.concatenate([
                    pred_b[np.isfinite(pred_b)],
                    mur_b[np.isfinite(mur_b)],
                    era5_b[np.isfinite(era5_b)],
                ])
                if all_vals.size == 0:
                    continue

                sample_date = str(batch_dates[b])
                sample_time = str(batch_times[b]).replace(":", "-")
                sample_dt_str = f"{sample_date} {sample_time.replace('-', ':')}"
                sample_tag = f"{month_name(month).lower()}_{self.month_to_saved[month]:02d}"

                try:
                    if not HAS_CARTOPY:
                        raise RuntimeError('cartopy unavailable — map plots skipped')
                    sample_folder = save_interactive_comparison_set(
                        out_root=self.example_dir,
                        sample_dt_str=sample_dt_str,
                        sample_date=sample_date,
                        sample_time=sample_time,
                        sample_tag=sample_tag,
                        pred=pred_b,
                        era5=era5_b,
                        mur=mur_b,
                        extent=GSL_EXTENT,
                    )
                    if sample_folder is not None:
                        self.month_to_saved[month] += 1
                        self.saved_examples += 1
                        print(f"[viz] wrote seasonal sample for {month_name(month)} -> {sample_folder}")
                except Exception as e:
                    print(f"[viz] failed seasonal sample for {sample_date} {sample_time}: {e}")
                    # Still count it so we don't retry every sample in the epoch
                    self.month_to_saved[month] += 1

                if all(v >= self.max_per_month for v in self.month_to_saved.values()):
                    break

    def compute(self) -> dict:
        n = max(self.n_valid, 1)

        rmse_model   = math.sqrt(self.sq_model   / n)
        rmse_persist = math.sqrt(self.sq_persist / n)
        rmse_murprev = math.sqrt(self.sq_murprev / n) if self.sq_murprev > 0 else None
        ss           = 1.0 - rmse_model / (rmse_persist + 1e-12)

        # Physical conversion — display only
        rmse_model_phys   = rmse_model   * self.sst_std
        rmse_persist_phys = rmse_persist * self.sst_std
        rmse_murprev_phys = (rmse_murprev * self.sst_std) if rmse_murprev else None

        psd_pred     = float(np.nanmean(self.psd_pred_band))
        psd_targ     = float(np.nanmean(self.psd_targ_band))
        psd_ratio    = psd_pred / (psd_targ + 1e-30)
        psd_log_bias = float(np.log10(psd_ratio + 1e-30))

        out = {
            'n_samples':         self.n_samples,
            'n_valid_pixels':    self.n_valid,
            # normalised units
            'rmse_model_norm':   round(rmse_model,         6),
            'rmse_persist_norm': round(rmse_persist,       6),
            # physical units — display only
            'rmse_model_K':      round(rmse_model_phys,    4),
            'rmse_persist_K':    round(rmse_persist_phys,  4),
            # skill score vs ERA5-SST ΔSST=0 persistence
            'skill_score':       round(ss,                  6),
            # spectral
            'psd_pred_band':     round(psd_pred,            6),
            'psd_targ_band':     round(psd_targ,            6),
            'psd_ratio':         round(psd_ratio,           6),
            'psd_log10_bias':    round(psd_log_bias,        4),
        }
        if rmse_murprev_phys is not None:
            out['rmse_murprev_K'] = round(rmse_murprev_phys, 4)

        # Seasonal mean bias (°C) — positive = model too warm, negative = too cold
        season_map = {
            'DJF': [12, 1, 2], 'MAM': [3, 4, 5],
            'JJA': [6, 7, 8],  'SON': [9, 10, 11],
        }
        seasonal_bias = {}
        for season, months in season_map.items():
            total_sum   = sum(self.bias_sum[m]   for m in months)
            total_count = sum(self.bias_count[m] for m in months)
            seasonal_bias[season] = round(total_sum / max(total_count, 1), 6)
        out['seasonal_bias_K'] = seasonal_bias

        # Per-month bias for full detail
        monthly_bias = {}
        month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                       'Jul','Aug','Sep','Oct','Nov','Dec']
        for m in range(1, 13):
            monthly_bias[month_names[m-1]] = round(
                self.bias_sum[m] / max(self.bias_count[m], 1), 6)
        out['monthly_bias_K'] = monthly_bias

        return out


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PSD CURVES  (optional)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_psd_curves(model, loader, device, dx_km) -> dict:
    """
    Returns mean + p10/p90 PSD curves for pred and target.
    Drop into matplotlib: plt.loglog(k, psd_pred_mean).
    """
    model.eval()
    pred_psds, targ_psds = [], []

    for batch in tqdm(loader, desc='PSD curves'):
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred = (model.sample(batch) + batch['sst_prev']).float()

        sst = batch['sst_abs'].float()
        w   = batch['weight'].float()
        vf  = (w > 0).float()

        pred_np = (pred * vf).squeeze(1).cpu().numpy()
        targ_np = (sst  * vf).squeeze(1).cpu().numpy()

        for b in range(pred_np.shape[0]):
            k_c, pp = azimuthal_psd(pred_np[b], dx_km)
            _,   pt = azimuthal_psd(targ_np[b], dx_km)
            pred_psds.append(pp)
            targ_psds.append(pt)

    pred_arr = np.array(pred_psds)
    targ_arr = np.array(targ_psds)

    return {
        'k_cycles_per_km': k_c.tolist(),
        'wavelength_km':   (1.0 / (k_c + 1e-30)).tolist(),
        'psd_pred_mean':   np.mean(pred_arr,           axis=0).tolist(),
        'psd_targ_mean':   np.mean(targ_arr,           axis=0).tolist(),
        'psd_pred_p10':    np.percentile(pred_arr, 10, axis=0).tolist(),
        'psd_pred_p90':    np.percentile(pred_arr, 90, axis=0).tolist(),
        'psd_ratio_mean':  (np.mean(pred_arr, axis=0) /
                            (np.mean(targ_arr, axis=0) + 1e-30)).tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint: str, stage: int, device: torch.device,
               n_diff_steps: int | None = None) -> TemporalDownscaler:
    model = TemporalDownscaler(stage=stage)

    ckpt = torch.load(checkpoint, map_location=device)
    sd   = ckpt.get('model', ckpt.get('model_state_dict',
                    ckpt.get('state_dict', ckpt)))
    model.load_state_dict(sd, strict=True)

    if n_diff_steps is not None:
        cfg.DIFF_SAMPLE_STEPS = n_diff_steps

    return model.to(device).eval()


def _make_test_loader(batch_size: int, stage: int) -> torch.utils.data.DataLoader:
    """Wraps test_ds (returned by make_loaders) in a DataLoader."""
    _, _, test_ds = make_loaders(
        batch_size  = cfg.BATCH_SIZE,
        num_workers = cfg.NUM_WORKERS,
        stage         = stage,
    )
    return torch.utils.data.DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = cfg.NUM_WORKERS,
        pin_memory  = True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',   required=True)
    p.add_argument('--stage',        type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument('--batch_size',   type=int, default=4)
    p.add_argument('--n_diff_steps', type=int, default=None)
    p.add_argument('--dx_km',        type=float, default=6.0)
    p.add_argument('--lam_lo_km',    type=float, default=5.0)
    p.add_argument('--lam_hi_km',    type=float, default=50.0)
    p.add_argument('--sst_std',      type=float, default=None,
                   help='Display-only: multiplies printed RMSE into °C. '
                        'Auto-loaded from cfg.SST_STD if omitted.')
    p.add_argument('--psd_curves',   action='store_true')
    p.add_argument('--output',       default='eval_results_abs_sst.json')
    p.add_argument('--device',
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--save_examples', action='store_true')
    p.add_argument('--example_dir', default='../outputs/figures/runs')
    p.add_argument('--max_examples', type=int, default=8)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    # Load normalisation stats from cfg paths
    stats = load_normalisation_stats()

    # sst_std for display — use arg if passed, else use stats value
    sst_std = args.sst_std if args.sst_std is not None else stats['sst_std']

    print(f'\n{"═"*60}')
    print(f'  Evaluation — Stage {args.stage}')
    print(f'{"═"*60}')
    print(f'  Checkpoint : {args.checkpoint}')
    print(f'  Device     : {device}')
    if device.type == 'cuda':
        print(f'  GPU        : {torch.cuda.get_device_name(0)}')
    print(f'  sst_std    : {sst_std:.4f} °C  (display only)')
    print(f'  Persistence: ERA5 ch3 (SST/K) → °C → MUR-normalised → bilinear upsample')

    model    = load_model(args.checkpoint, args.stage, device, args.n_diff_steps)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters : {n_params / 1e6:.2f} M\n')

    test_loader = _make_test_loader(args.batch_size, args.stage)
    print(f'  Test batches : {len(test_loader)}\n')

    evaluator = Evaluator(
        model, device,
        stats     = stats,
        dx_km     = args.dx_km,
        lam_lo_km = args.lam_lo_km,
        lam_hi_km = args.lam_hi_km,
        sst_std   = sst_std,
        save_examples=args.save_examples,
        example_dir=args.example_dir,
        max_examples=args.max_examples,
    )

    for batch in tqdm(test_loader, desc='Evaluating'):
        evaluator.update(batch)

    results = evaluator.compute()

    print(f'\n{"═"*60}')
    print('  RESULTS')
    print(f'{"═"*60}')
    print(f'  Samples              : {results["n_samples"]}')
    print(f'  Valid pixels         : {results["n_valid_pixels"]:,}')
    print(f'  RMSE  model   (norm) : {results["rmse_model_norm"]:.5f}')
    print(f'  RMSE  persist (norm) : {results["rmse_persist_norm"]:.5f}  '
          f'[ERA5 SST ch3, expect ~3.28–3.57 °C]')
    print(f'  RMSE  model   (°C)   : {results["rmse_model_K"]:.4f}')
    print(f'  RMSE  persist (°C)   : {results["rmse_persist_K"]:.4f}')
    if 'rmse_murprev_K' in results:
        print(f'  RMSE  MUR-prev (°C)  : {results["rmse_murprev_K"]:.4f}  '
              f'[oracle ref, expect ~0.43–0.50 °C]')
    ss = results["skill_score"]
    print(f'  Skill Score SS       : {ss:+.5f}  '
          f'({"↑ better" if ss > 0 else "↓ worse"} than ΔSST=0 persistence)')
    print(f'  PSD pred   (band)    : {results["psd_pred_band"]:.4e}')
    print(f'  PSD target (band)    : {results["psd_targ_band"]:.4e}')
    print(f'  PSD ratio   p/t      : {results["psd_ratio"]:.4f}  '
          f'(log₁₀ bias = {results["psd_log10_bias"]:+.3f})')
    print(f'{"═"*60}\n')

    if args.psd_curves:
        print('Computing full PSD curves (second pass) …')
        test_loader2 = _make_test_loader(args.batch_size, args.stage)
        results['psd_curves'] = compute_psd_curves(
            model, test_loader2, device, args.dx_km
        )
        print('  Done.\n')

    out = Path(args.output)
    out.write_text(json.dumps(results, indent=2))
    print(f'Results written → {out}')


if __name__ == '__main__':
    main()