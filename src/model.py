import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from config import cfg


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════════════════════

class SinCos2DPE(nn.Module):
    """2D sinusoidal positional encoding for spatial patch tokens."""
    def __init__(self, d_model, nH, nW):
        super().__init__()
        pe = self._build(d_model, nH, nW)
        self.register_buffer('pe', pe)   # [1, nH*nW, D]

    @staticmethod
    def _build(D, nH, nW):
        assert D % 4 == 0
        d   = D // 4
        div = torch.exp(torch.arange(0, d).float() * (-np.log(10000.0) / d))
        yp  = torch.arange(nH).unsqueeze(1).float()
        xp  = torch.arange(nW).unsqueeze(1).float()
        pey = torch.cat([torch.sin(yp * div), torch.cos(yp * div)], dim=1)
        pex = torch.cat([torch.sin(xp * div), torch.cos(xp * div)], dim=1)
        pe  = torch.cat([
            pey.unsqueeze(1).expand(nH, nW, -1),
            pex.unsqueeze(0).expand(nH, nW, -1),
        ], dim=-1)
        return pe.view(1, nH * nW, D)

    def forward(self, x):
        return x + self.pe


class TemporalPE(nn.Module):
    """1D sinusoidal PE for temporal position."""
    def __init__(self, d_model, T):
        super().__init__()
        pe  = torch.zeros(T, d_model)
        pos = torch.arange(T).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # [1, T, D]

    def forward(self, x, T, N):
        """
        x:   [B, T*N, D]
        Broadcasts temporal PE across all N spatial tokens per frame.
        """
        pe = self.pe[:, :T, :]                         # [1, T, D]
        pe = pe.unsqueeze(2).expand(1, T, N, -1)       # [1, T, N, D]
        pe = pe.reshape(1, T * N, -1)                  # [1, T*N, D]
        return x + pe


class PatchEmbed(nn.Module):
    """
    Embed a spatial ERA5 frame into patch tokens.
    Input:  [B, C, H, W]
    Output: [B, nH*nW, D]
    Handles non-divisible spatial dims with zero-padding.
    """
    def __init__(self, in_channels, patch_size, d_model, H, W):
        super().__init__()
        self.P     = patch_size
        self.nH    = (H + patch_size - 1) // patch_size
        self.nW    = (W + patch_size - 1) // patch_size
        self.pad_H = self.nH * patch_size - H
        self.pad_W = self.nW * patch_size - W
        self.proj  = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        if self.pad_H > 0 or self.pad_W > 0:
            # F.pad order: (left, right, top, bottom)
            x = F.pad(x, (0, self.pad_W, 0, self.pad_H))
        x = self.proj(x)                              # [B, D, nH, nW]
        return rearrange(x, 'b d h w -> b (h w) d')  # [B, N, D]


class FFN(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)


class ConvBlock(nn.Module):
    """Conv2d → GroupNorm → GELU × 2"""
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        g = min(groups, out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
        )
    def forward(self, x): return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — JOINT SPATIOTEMPORAL ENCODER (BASELINE)
# ═══════════════════════════════════════════════════════════════════════════════

class JointBlock(nn.Module):
    """
    Full self-attention over all T*N tokens — no spatial/temporal separation.
    Accepts optional causal attention mask.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = FFN(d_model, dropout)

    def forward(self, x, attn_mask=None):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class Stage1Encoder(nn.Module):
    """
    Single-stream joint spatiotemporal encoder.
    All T*N tokens attend to each other — no separation of concerns.
    Block-causal mask prevents future frame leakage.
    Baseline against which Stages 2–4 are compared.
    """
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(
            cfg.N_INPUT, cfg.PATCH_SIZE, cfg.D_MODEL,
            cfg.LAT_C, cfg.LON_C
        )
        nH, nW       = self.patch_embed.nH, self.patch_embed.nW
        self.nH      = nH
        self.nW      = nW
        self.N       = nH * nW

        self.spatial_pe  = SinCos2DPE(cfg.D_MODEL, nH, nW)
        self.temporal_pe = TemporalPE(cfg.D_MODEL, cfg.T)

        self.blocks = nn.ModuleList([
            JointBlock(cfg.D_MODEL, cfg.N_HEADS, cfg.DROPOUT)
            for _ in range(cfg.N_LAYERS)
        ])
        self.norm      = nn.LayerNorm(cfg.D_MODEL)
        self.unproject = nn.Linear(cfg.D_MODEL, cfg.D_MODEL)

    def _causal_mask(self, T, N, device):
        """
        Block-causal mask [T*N, T*N].
        Token i cannot attend to token j if frame(j) > frame(i).
        True = blocked (→ -inf in softmax).
        """
        frame_idx = torch.arange(T, device=device).repeat_interleave(N)
        return frame_idx.unsqueeze(0) > frame_idx.unsqueeze(1)   # [T*N, T*N]

    def forward(self, era5_window):
        """
        era5_window: [B, T, C, H, W]
        returns:     [B, D, nH, nW]
        """
        B, T, C, H, W = era5_window.shape

        # Patchify each frame independently
        x      = rearrange(era5_window, 'b t c h w -> (b t) c h w')
        tokens = self.patch_embed(x)                        # [B*T, N, D]

        # Spatial PE per frame
        tokens = self.spatial_pe(tokens)                    # [B*T, N, D]
        tokens = rearrange(tokens, '(b t) n d -> b (t n) d', b=B, t=T)

        # Temporal PE broadcast over spatial tokens
        tokens = self.temporal_pe(tokens, T, self.N)        # [B, T*N, D]

        # Causal mask
        mask = self._causal_mask(T, self.N, era5_window.device)

        for block in self.blocks:
            tokens = block(tokens, attn_mask=mask)
        tokens = self.norm(tokens)

        # Extract last frame's tokens
        tokens = rearrange(tokens, 'b (t n) d -> b t n d', t=T, n=self.N)
        last   = self.unproject(tokens[:, -1])              # [B, N, D]
        return rearrange(last, 'b (h w) d -> b d h w',
                         h=self.nH, w=self.nW)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — ALTERNATING SPATIAL / TEMPORAL BLOCKS (SINGLE STREAM)
# ═══════════════════════════════════════════════════════════════════════════════

class SpatialBlock(nn.Module):
    """Attention over spatial tokens within each frame independently."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = FFN(d_model, dropout)

    def forward(self, x, T, N):
        """x: [B, T*N, D]"""
        B   = x.shape[0]
        x_s = rearrange(x, 'b (t n) d -> (b t) n d', t=T, n=N)
        h   = self.norm1(x_s)
        h, _ = self.attn(h, h, h, need_weights=False)
        x_s = x_s + h
        x_s = x_s + self.ffn(self.norm2(x_s))
        return rearrange(x_s, '(b t) n d -> b (t n) d', b=B, t=T)


class TemporalBlock(nn.Module):
    """Attention over time axis per spatial location independently."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = FFN(d_model, dropout)

    def forward(self, x, T, N):
        """x: [B, T*N, D]"""
        B   = x.shape[0]
        x_t = rearrange(x, 'b (t n) d -> (b n) t d', t=T, n=N)
        h   = self.norm1(x_t)
        causal = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        h, _ = self.attn(h, h, h, attn_mask=causal, need_weights=False)
        x_t = x_t + h
        x_t = x_t + self.ffn(self.norm2(x_t))
        return rearrange(x_t, '(b n) t d -> b (t n) d', b=B, n=N)


class Stage2Encoder(nn.Module):
    """
    Single stream, alternating S→T→S→T blocks.
    Separation of concerns without parallel streams.
    """
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(
            cfg.N_INPUT, cfg.PATCH_SIZE, cfg.D_MODEL,
            cfg.LAT_C, cfg.LON_C
        )
        nH, nW           = self.patch_embed.nH, self.patch_embed.nW
        self.nH, self.nW = nH, nW
        self.N            = nH * nW

        self.spatial_pe  = SinCos2DPE(cfg.D_MODEL, nH, nW)
        self.temporal_pe = TemporalPE(cfg.D_MODEL, cfg.T)

        self.spatial_blocks = nn.ModuleList([
            SpatialBlock(cfg.D_MODEL, cfg.N_HEADS, cfg.DROPOUT)
            for _ in range(cfg.N_LAYERS)
        ])
        self.temporal_blocks = nn.ModuleList([
            TemporalBlock(cfg.D_MODEL, cfg.N_HEADS, cfg.DROPOUT)
            for _ in range(cfg.N_LAYERS)
        ])
        self.norm      = nn.LayerNorm(cfg.D_MODEL)
        self.unproject = nn.Linear(cfg.D_MODEL, cfg.D_MODEL)

    def forward(self, era5_window):
        B, T, C, H, W = era5_window.shape

        x      = rearrange(era5_window, 'b t c h w -> (b t) c h w')
        tokens = self.spatial_pe(self.patch_embed(x))      # [B*T, N, D]
        tokens = rearrange(tokens, '(b t) n d -> b (t n) d', b=B, t=T)
        tokens = self.temporal_pe(tokens, T, self.N)

        for s_blk, t_blk in zip(self.spatial_blocks, self.temporal_blocks):
            tokens = s_blk(tokens, T, self.N)
            tokens = t_blk(tokens, T, self.N)

        tokens = self.norm(tokens)
        tokens = rearrange(tokens, 'b (t n) d -> b t n d', t=T, n=self.N)
        last   = self.unproject(tokens[:, -1])
        return rearrange(last, 'b (h w) d -> b d h w', h=self.nH, w=self.nW)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — TRUE DUAL-STREAM ENCODER (PARALLEL SPATIAL + TEMPORAL)
# ═══════════════════════════════════════════════════════════════════════════════

class CrossAttentionBridge(nn.Module):
    """
    Lets one stream query the other's key/value space.
    query_stream and kv_stream may have different sequence lengths.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn    = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = FFN(d_model, dropout)

    def forward(self, query_stream, kv_stream):
        """Both [B, L, D]. Returns updated query_stream."""
        q  = self.norm_q(query_stream)
        kv = self.norm_kv(kv_stream)
        h, _ = self.attn(q, kv, kv, need_weights=False)
        x = query_stream + h
        x = x + self.ffn(self.norm2(x))
        return x


class Stage3Encoder(nn.Module):
    """
    True dual-stream encoder running in parallel:
      - Spatial stream:  per-frame attention only
      - Temporal stream: per-location attention only
      - Cross-attention bridges at layers 4 and 8
      - Learned fusion gate at output
    """
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(
            cfg.N_INPUT, cfg.PATCH_SIZE, cfg.D_MODEL,
            cfg.LAT_C, cfg.LON_C
        )
        nH, nW           = self.patch_embed.nH, self.patch_embed.nW
        self.nH, self.nW = nH, nW
        self.N            = nH * nW

        self.spatial_pe  = SinCos2DPE(cfg.D_MODEL, nH, nW)
        self.temporal_pe = TemporalPE(cfg.D_MODEL, cfg.T)

        self.s_blocks = nn.ModuleList([
            SpatialBlock(cfg.D_MODEL, cfg.N_HEADS, cfg.DROPOUT)
            for _ in range(cfg.N_LAYERS)
        ])
        self.t_blocks = nn.ModuleList([
            TemporalBlock(cfg.D_MODEL, cfg.N_HEADS, cfg.DROPOUT)
            for _ in range(cfg.N_LAYERS)
        ])

        # Cross-attention at layers 3 and 7 (0-indexed)
        self.bridge_layers = {3, 7}
        self.s2t_bridges   = nn.ModuleList([
            CrossAttentionBridge(cfg.D_MODEL, cfg.N_HEADS, cfg.DROPOUT)
            for _ in range(2)
        ])
        self.t2s_bridges   = nn.ModuleList([
            CrossAttentionBridge(cfg.D_MODEL, cfg.N_HEADS, cfg.DROPOUT)
            for _ in range(2)
        ])

        self.fusion_gate = nn.Sequential(
            nn.Linear(cfg.D_MODEL * 2, cfg.D_MODEL),
            nn.Sigmoid()
        )
        self.norm_s = nn.LayerNorm(cfg.D_MODEL)
        self.norm_t = nn.LayerNorm(cfg.D_MODEL)
        self.frame_embed = nn.Embedding(cfg.T, cfg.D_MODEL)


    def forward(self, era5_window):
        B, T, C, H, W = era5_window.shape

        x      = rearrange(era5_window, 'b t c h w -> (b t) c h w')
        tokens = self.patch_embed(x)                        # [B*T, N, D]

        # Spatial stream — spatial PE only
        s = self.spatial_pe(tokens)
        # Add frame embedding to each frame's tokens
        frame_idx = torch.arange(T, device=era5_window.device)
        #frame_idx = frame_idx.repeat_interleave(1)          # [T]
        f_emb = self.frame_embed(frame_idx)                 # [T, D]
        f_emb = f_emb.unsqueeze(1).expand(T, self.N, -1)   # [T, N, D]
        f_emb = rearrange(f_emb, 't n d -> (t n) d')       # [T*N, D]
        s = rearrange(s, '(b t) n d -> b (t n) d', b=B, t=T)
        s = s + f_emb.unsqueeze(0)                         # [B, T*N, D]

        # Temporal stream — temporal PE only
        t_tok = rearrange(tokens, '(b t) n d -> b (t n) d', b=B, t=T)
        t_tok = self.temporal_pe(t_tok, T, self.N)

        bridge_idx = 0
        for i, (s_blk, t_blk) in enumerate(zip(self.s_blocks, self.t_blocks)):
            s     = s_blk(s,     T, self.N)
            t_tok = t_blk(t_tok, T, self.N)

            if i in self.bridge_layers:
                s_new = self.s2t_bridges[bridge_idx](s,     t_tok)
                t_new = self.t2s_bridges[bridge_idx](t_tok, s)
                s, t_tok = s_new, t_new
                bridge_idx += 1

        s     = self.norm_s(s)
        t_tok = self.norm_t(t_tok)

        s     = rearrange(s,     'b (t n) d -> b t n d', t=T, n=self.N)[:, -1]
        t_tok = rearrange(t_tok, 'b (t n) d -> b t n d', t=T, n=self.N)[:, -1]

        gate  = self.fusion_gate(torch.cat([s, t_tok], dim=-1))
        fused = gate * s + (1 - gate) * t_tok

        return rearrange(fused, 'b (h w) d -> b d h w', h=self.nH, w=self.nW)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — DUAL-STREAM WITH PHYSICALLY MOTIVATED TEMPORAL WINDOWS
# # ═══════════════════════════════════════════════════════════════════════════════

# class Stage4Encoder(nn.Module):
#     """
#     Atmospheric stream: T_atm=7 frames (42 hr synoptic memory)
#     Ocean stream:       T_oce=28 frames (7-day eddy/SST memory)
#     Shared spatial encoder feeds both temporal streams.
#     """
#     def __init__(self, T_atm=cfg.T_ATM, T_oce=cfg.T_OCE):
#         super().__init__()
#         self.T_atm = T_atm
#         self.T_oce = T_oce

#         self.patch_embed = PatchEmbed(
#             cfg.N_INPUT, cfg.PATCH_SIZE, cfg.D_MODEL,
#             cfg.LAT_C, cfg.LON_C
#         )
#         nH, nW           = self.patch_embed.nH, self.patch_embed.nW
#         self.nH, self.nW = nH, nW
#         self.N            = nH * nW

#         # Shared spatial encoder
#         self.spatial_pe = SinCos2DPE(cfg.D_MODEL, nH, nW)
#         self.s_blocks   = nn.ModuleList([
#             SpatialBlock(cfg.D_MODEL, cfg.N_HEADS, cfg.DROPOUT)
#             for _ in range(cfg.N_LAYERS)
#         ])
#         self.norm_s = nn.LayerNorm(cfg.D_MODEL)
#         oce_patch = cfg.PATCH_SIZE * 4          # e.g. 4*4=16 → ~31×75 tokens
#         import zarr as _zarr
#         _mur = _zarr.open(cfg.MUR_ZARR, 'r')
#         _H_fine, _W_fine = _mur.shape[1], _mur.shape[2]
#         self.oce_patch_embed = PatchEmbed(
#             1, oce_patch, cfg.D_MODEL,
#             _H_fine, _W_fine
#         )
#         oce_nH, oce_nW      = self.oce_patch_embed.nH, self.oce_patch_embed.nW
#         self.oce_nH         = oce_nH
#         self.oce_nW         = oce_nW
#         self.N_oce          = oce_nH * oce_nW
#         self.oce_spatial_pe = SinCos2DPE(cfg.D_MODEL, oce_nH, oce_nW)
#         # Atmospheric temporal stream (short)
#         self.atm_pe     = TemporalPE(cfg.D_MODEL, T_atm)
#         self.atm_blocks = nn.ModuleList([
#             TemporalBlock(cfg.D_MODEL, cfg.N_HEADS, cfg.DROPOUT)
#             for _ in range(cfg.N_LAYERS // 2)
#         ])
#         self.norm_atm = nn.LayerNorm(cfg.D_MODEL)

#         # Ocean temporal stream (long)
#         self.oce_pe     = TemporalPE(cfg.D_MODEL, T_oce)
#         self.oce_blocks = nn.ModuleList([
#             TemporalBlock(cfg.D_MODEL, cfg.N_HEADS, cfg.DROPOUT)
#             for _ in range(cfg.N_LAYERS // 2)
#         ])
#         self.norm_oce = nn.LayerNorm(cfg.D_MODEL)

#         # Three-way fusion
#         self.fusion = nn.Sequential(
#             nn.Linear(cfg.D_MODEL * 3, cfg.D_MODEL * 2),
#             nn.GELU(),
#             nn.Linear(cfg.D_MODEL * 2, cfg.D_MODEL),
#         )

#     def forward(self, era5_atm, mur_oce):
#         """
#         era5_atm: [B, T_atm=28, C_era5, H_c, W_c]  — ERA5 coarse atmospheric
#         mur_oce:  [B, T_oce=60, 1,      H_f, W_f]  — MUR fine ocean history
#         returns:  [B, D, nH, nW]
#         """
#         B = era5_atm.shape[0]

#         # ── Atmospheric stream — ERA5 coarse spatial + temporal ───────────
#         x_atm = rearrange(era5_atm, 'b t c h w -> (b t) c h w')
#         s_atm = self.spatial_pe(self.patch_embed(x_atm))       # [B*T_atm, N, D]
#         s_atm = rearrange(s_atm, '(b t) n d -> b (t n) d', b=B, t=self.T_atm)

#         for blk in self.s_blocks:
#             s_atm = blk(s_atm, self.T_atm, self.N)
#         s_atm = self.norm_s(s_atm)

#         a = self.atm_pe(s_atm, self.T_atm, self.N)
#         for blk in self.atm_blocks:
#             a = blk(a, self.T_atm, self.N)
#         a = self.norm_atm(a)
#         a = rearrange(a, 'b (t n) d -> b t n d', t=self.T_atm, n=self.N)[:, -1]
#         # a: [B, N, D] — atmospheric summary

#         # ── Ocean stream — MUR fine spatial + temporal ────────────────────
#         x_oce = rearrange(mur_oce, 'b t c h w -> (b t) c h w')
#         s_oce = self.oce_spatial_pe(self.oce_patch_embed(x_oce))  # [B*T_oce, N_oce, D]
#         s_oce = rearrange(s_oce, '(b t) n d -> b (t n) d', b=B, t=self.T_oce)

#         o = self.oce_pe(s_oce, self.T_oce, self.N_oce)
#         for blk in self.oce_blocks:
#             o = blk(o, self.T_oce, self.N_oce)
#         o = self.norm_oce(o)
#         o = rearrange(o, 'b (t n) d -> b t n d', t=self.T_oce, n=self.N_oce)[:, -1]
#         # o: [B, N_oce, D] — pool to [B, N, D] to match atm token count

#         o = F.adaptive_avg_pool1d(
#             o.transpose(1, 2), self.N          # [B, D, N_oce] → [B, D, N]
#         ).transpose(1, 2)                       # [B, N, D]

#         # ── Spatial summary from last atmospheric frame ───────────────────
#         s = rearrange(s_atm, 'b (t n) d -> b t n d', t=self.T_atm, n=self.N)[:, -1]
#         # s: [B, N, D]

#         # ── Three-way fusion ──────────────────────────────────────────────
#         fused = self.fusion(torch.cat([s, a, o], dim=-1))   # [B, N, D]
#         return rearrange(fused, 'b (h w) d -> b d h w', h=self.nH, w=self.nW)
class DynamicPatchEmbed(nn.Module):
    """
    Patch embedding for variable-resolution inputs.
    Returns:
        tokens : [B, N, D]
        nH     : int
        nW     : int
    """
    def __init__(self, in_channels, patch_size, d_model):
        super().__init__()
        self.P = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        B, C, H, W = x.shape

        pad_h = (self.P - H % self.P) % self.P
        pad_w = (self.P - W % self.P) % self.P
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))   # left, right, top, bottom

        x = self.proj(x)                         # [B, D, nH, nW]
        nH, nW = x.shape[-2], x.shape[-1]
        x = rearrange(x, 'b d h w -> b (h w) d')  # [B, N, D]

        return x, nH, nW


class Stage4Encoder(nn.Module):
    def __init__(self, T_atm=cfg.T_ATM, T_oce=cfg.T_OCE):
        super().__init__()
        self.T_atm = T_atm
        self.T_oce = T_oce

        # Atmospheric branch unchanged
        self.patch_embed = PatchEmbed(
            cfg.N_INPUT, cfg.PATCH_SIZE, cfg.D_MODEL, cfg.LAT_C, cfg.LON_C
        )
        nH, nW = self.patch_embed.nH, self.patch_embed.nW
        self.nH, self.nW = nH, nW
        self.N = nH * nW

        self.spatial_pe = SinCos2DPE(cfg.D_MODEL, nH, nW)
        self.sblocks = nn.ModuleList(
            SpatialBlock(cfg.D_MODEL, cfg.N_HEADS, cfg.DROPOUT)
            for _ in range(cfg.N_LAYERS)
        )
        self.norms = nn.LayerNorm(cfg.D_MODEL)

        # Ocean branch changed
        oce_patch = cfg.PATCH_SIZE * 16
        self.oce_patch_embed = DynamicPatchEmbed(1, oce_patch, cfg.D_MODEL)
        self.oce_spatial_blocks = nn.ModuleList(
            OceanSpatialBlockRoPE(cfg.D_MODEL, cfg.N_HEADS, cfg.DROPOUT)
            for _ in range(cfg.N_LAYERS)
        )

        self.atm_pe = TemporalPE(cfg.D_MODEL, T_atm)
        self.atm_blocks = nn.ModuleList(
            TemporalBlock(cfg.D_MODEL, cfg.N_HEADS, cfg.DROPOUT)
            for _ in range(cfg.N_LAYERS // 2)
        )
        self.norm_atm = nn.LayerNorm(cfg.D_MODEL)

        self.oce_pe = TemporalPE(cfg.D_MODEL, T_oce)
        self.oce_blocks = nn.ModuleList(
            TemporalBlock(cfg.D_MODEL, cfg.N_HEADS, cfg.DROPOUT)
            for _ in range(cfg.N_LAYERS // 2)
        )
        self.norm_oce = nn.LayerNorm(cfg.D_MODEL)

        self.fusion = nn.Sequential(
            nn.Linear(cfg.D_MODEL * 3, cfg.D_MODEL * 2),
            nn.GELU(),
            nn.Linear(cfg.D_MODEL * 2, cfg.D_MODEL),
        )

    def forward(self, era5_atm, mur_oce):
        B = era5_atm.shape[0]

        # --- Atmospheric stream unchanged ---
        x_atm = rearrange(era5_atm, 'b t c h w -> (b t) c h w')
        s_atm = self.spatial_pe(self.patch_embed(x_atm))
        s_atm = rearrange(s_atm, '(b t) n d -> b (t n) d', b=B, t=self.T_atm)

        for blk in self.sblocks:
            s_atm = blk(s_atm, self.T_atm, self.N)
        s_atm = self.norms(s_atm)

        a = self.atm_pe(s_atm, self.T_atm, self.N)
        for blk in self.atm_blocks:
            a = blk(a, self.T_atm, self.N)
        a = self.norm_atm(a)
        a = rearrange(a, 'b (t n) d -> b t n d', t=self.T_atm, n=self.N)[:, -1]

        # --- Ocean stream with dynamic patching + coord PE + RoPE ---
        x_oce = rearrange(mur_oce, 'b t c h w -> (b t) c h w')
        tok_oce, nH_oce, nW_oce = self.oce_patch_embed(x_oce)
        N_oce = nH_oce * nW_oce

        
        s_oce = tok_oce 
        s_oce = rearrange(s_oce, '(b t) n d -> b (t n) d', b=B, t=self.T_oce)

        for blk in self.oce_spatial_blocks:
            s_oce = blk(s_oce, self.T_oce, N_oce, nH_oce, nW_oce)

        o = self.oce_pe(s_oce, self.T_oce, N_oce)
        for blk in self.oce_blocks:
            o = blk(o, self.T_oce, N_oce)
        o = self.norm_oce(o)
        o = rearrange(o, 'b (t n) d -> b t n d', t=self.T_oce, n=N_oce)[:, -1]

        # Pool ocean tokens to atmospheric token count
        o = F.adaptive_avg_pool1d(o.transpose(1, 2), self.N).transpose(1, 2)

        # Last atmospheric spatial summary
        s = rearrange(s_atm, 'b (t n) d -> b t n d', t=self.T_atm, n=self.N)[:, -1]

        fused = self.fusion(torch.cat([s, a, o], dim=-1))
        return rearrange(fused, 'b (h w) d -> b d h w', h=self.nH, w=self.nW)

# ═══════════════════════════════════════════════════════════════════════════════
# DECODER — DETERMINISTIC BASELINE  (shared across all stages)
# ═══════════════════════════════════════════════════════════════════════════════

class BaselineDecoder(nn.Module):
    """
    Deterministic upsampler: coarse encoder latent → fine SST baseline.
    All interpolations driven from bathy.shape[-2:] — never cfg constants —
    so the actual 501×1201 tensor size is always matched exactly.

    Input:  latent [B, D, nH, nW]        e.g. [B, 256, 5, 13]
            bathy  [B, 1, H_fine, W_fine] e.g. [B, 1, 501, 1201]
    Output: [B, 1, H_fine, W_fine]
    """
    def __init__(self):
        super().__init__()
        D = cfg.D_MODEL
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(D,   128),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(128, 64),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(64,  32),
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(32,  32),
        )
        # Bathy injected as a skip at fine resolution
        self.final_conv = ConvBlock(32 + 1, 32)
        self.out        = nn.Conv2d(32, 1, 1)

    def forward(self, latent, bathy):
        # bathy.shape[-2:] is the ground-truth fine resolution
        fine_size = bathy.shape[-2:]   # (501, 1201) from actual data

        x = self.up1(latent)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        # Final interpolation to exact fine size — never hardcoded
        x = F.interpolate(x, size=fine_size, mode='bilinear', align_corners=False)
        x = self.final_conv(torch.cat([x, bathy], dim=1))
        return self.out(x)   # [B, 1, H_fine, W_fine]


# ═══════════════════════════════════════════════════════════════════════════════
# DECODER — RESIDUAL DIFFUSION  (shared across all stages)
# ═══════════════════════════════════════════════════════════════════════════════

class ResidualDiffusion(nn.Module):
    """
    Conditioned UNet on the fine-resolution residual.
    Explicit channel counts at every layer to avoid skip connection mismatches.
    """
    def __init__(self):
        super().__init__()
        C = cfg.DIFF_CHANNELS   # 64
        D = cfg.D_MODEL         # 256

        self.time_mlp = nn.Sequential(
            nn.Linear(1, C * 4), nn.SiLU(), nn.Linear(C * 4, C)
        )

        # Upsample latent channels to diffusion channels — spatial done in forward
        self.cond_proj = nn.Conv2d(D, C, 1)

        # Input channels: noisy_r(1) + x_base(1) + cond(C) + bathy(1) = C+3
        in_ch = C + 3

        # Encoder
        self.enc1 = ConvBlock(in_ch,  C)        # out: C     = 64
        self.enc2 = ConvBlock(C,      C * 2)    # out: C*2   = 128
        self.enc3 = ConvBlock(C * 2,  C * 2)    # out: C*2   = 128
        self.mid  = ConvBlock(C * 2,  C * 2)    # out: C*2   = 128

        # Decoder — each block receives upsampled + skip concatenated
        # dec3: upsample(mid)=C*2  + skip(e3)=C*2  → cat=C*4  → out=C*2
        # dec2: upsample(dec3)=C*2 + skip(e2)=C*2  → cat=C*4  → out=C
        # dec1: upsample(dec2)=C   + skip(e1)=C    → cat=C*2  → out=C
        self.dec3 = ConvBlock(C * 4,  C * 2)
        self.dec2 = ConvBlock(C * 4,  C)        # ← was C*3, now C*4
        self.dec1 = ConvBlock(C * 2,  C)

        self.out  = nn.Conv2d(C, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, r_noisy, x_base, latent, bathy, t_norm):
        fine_size = bathy.shape[-2:]

        # Time embedding
        t_emb = self.time_mlp(t_norm.unsqueeze(-1).float())   # [B, C]
        t_emb = t_emb.view(-1, t_emb.shape[1], 1, 1)         # [B, C, 1, 1]

        # Condition: upsample latent to fine size then project
        cond = F.interpolate(latent, size=fine_size,
                             mode='bilinear', align_corners=False)
        cond = self.cond_proj(cond)    # [B, C, H_fine, W_fine]

        # Input
        x = torch.cat([r_noisy, x_base, cond, bathy], dim=1)  # [B, C+3, H, W]

        # Encode
        e1 = self.enc1(x) + t_emb                  # [B, C,   H,   W  ]
        e2 = self.enc2(self.pool(e1))               # [B, C*2, H/2, W/2]
        e3 = self.enc3(self.pool(e2))               # [B, C*2, H/4, W/4]
        m  = self.mid(self.pool(e3))                # [B, C*2, H/8, W/8]

        # Decode with nearest-neighbour upsample + skip concat
        d3 = self.dec3(torch.cat([
            F.interpolate(m,  e3.shape[-2:], mode='nearest'), e3
        ], dim=1))                                  # [B, C*2, H/4, W/4]

        d2 = self.dec2(torch.cat([
            F.interpolate(d3, e2.shape[-2:], mode='nearest'), e2
        ], dim=1))                                  # [B, C,   H/2, W/2]

        d1 = self.dec1(torch.cat([
            F.interpolate(d2, e1.shape[-2:], mode='nearest'), e1
        ], dim=1))                                  # [B, C,   H,   W  ]

        return self.out(d1)                         # [B, 1,   H,   W  ]


# ═══════════════════════════════════════════════════════════════════════════════
# FULL MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalDownscaler(nn.Module):
    """
    Wraps any encoder stage (1–4) with the shared baseline decoder
    and residual diffusion decoder.
    """
    def __init__(self, stage=1):
        super().__init__()
        assert stage in (1, 2, 3, 4), f'stage must be 1–4, got {stage}'
        self.stage = stage

        encoders = {
            1: Stage1Encoder,
            2: Stage2Encoder,
            3: Stage3Encoder,
            4: lambda: Stage4Encoder(T_atm=cfg.T_ATM, T_oce=cfg.T_OCE),
        }
        self.encoder   = encoders[stage]()
        self.baseline  = BaselineDecoder()
        self.diffusion = ResidualDiffusion()

        # Linear beta noise schedule
        betas     = torch.linspace(1e-4, 0.02, cfg.DIFF_STEPS)
        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_ab',   alpha_bar.sqrt())
        self.register_buffer('sqrt_1mab', (1 - alpha_bar).sqrt())

    def _encode(self, batch):
        era5 = batch['era5']   # [B, T, C, H, W]
        if self.stage in (1, 2, 3):
            return self.encoder(era5)
        else:
            era5_atm = batch['era5']       # [B, T_atm, C, H_c, W_c]
            mur_oce  = batch['mur_seq']    # [B, T_oce, 1, H_f, W_f]
            return self.encoder(era5_atm, mur_oce)

    def forward(self, batch):
        sst    = batch['sst']
        weight = batch['weight']
        bathy  = batch['bathy']
        B      = sst.shape[0]

        # Hard guard — should not be needed after dataset fix but belt-and-suspenders
        sst    = torch.nan_to_num(sst,   nan=0.0, posinf=0.0, neginf=0.0)
        weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)

        latent = self._encode(batch)
        x_base = self.baseline(latent, bathy)
        r_gt   = (sst - x_base).detach()

        t       = torch.randint(0, cfg.DIFF_STEPS, (B,), device=sst.device)
        noise   = torch.randn_like(r_gt)
        r_noisy = (
            self.sqrt_ab[t].view(B, 1, 1, 1)   * r_gt +
            self.sqrt_1mab[t].view(B, 1, 1, 1) * noise
        )

        t_norm     = t.float() / cfg.DIFF_STEPS
        noise_pred = self.diffusion(
            r_noisy, x_base.detach(), latent.detach(), bathy, t_norm
        )

        def wmean(x):
            return (x * weight).sum() / (weight.sum() + 1e-6)

        loss_base = wmean(F.l1_loss(x_base, sst, reduction='none'))
        loss_diff = wmean(F.mse_loss(noise_pred, noise, reduction='none'))

        pred_m    = torch.nan_to_num(x_base * weight, nan=0.0)
        targ_m    = torch.nan_to_num(sst    * weight, nan=0.0)
        loss_spec = self._spectral_loss(pred_m, targ_m)

        loss = loss_base + 10.0 * loss_diff + 0.1 * loss_spec
        return {
            'loss':      loss,
            'loss_base': loss_base.item(),
            'loss_diff': loss_diff.item(),
            'loss_spec': loss_spec.item(),
        }

    @staticmethod
    def _spectral_loss(pred, target):
        pf = torch.fft.rfft2(pred)
        tf = torch.fft.rfft2(target)
        ky = torch.fft.fftfreq(pred.shape[-2], device=pred.device).abs()
        kx = torch.fft.rfftfreq(pred.shape[-1], device=pred.device).abs()
        k2 = ky[:, None] ** 2 + kx[None, :] ** 2
        w  = (1.0 + 20.0 * k2).sqrt()
        return (w * (pf.abs() - tf.abs()).abs()).mean()

    @torch.no_grad()
    def sample(self, batch, n_steps=None):
        """DDIM inference. Returns full HR SST prediction."""
        n_steps   = n_steps or cfg.DIFF_SAMPLE_STEPS
        bathy     = batch['bathy']
        latent    = self._encode(batch)
        x_base    = self.baseline(latent, bathy)
        if 'sst' in batch and 'weight' in batch:
            r_approx = (batch['sst'] - x_base).detach()
            valid    = batch['weight'] > 0
            r_std    = r_approx[valid].std().clamp(min=1e-4)
        else:
            r_std = torch.tensor(0.06, device=bathy.device)

        r_norm    = torch.randn_like(x_base)           # std=1 in normalised space
        timesteps = torch.linspace(
            cfg.DIFF_STEPS - 1, 0, n_steps, dtype=torch.long
        )

        for i, t in enumerate(timesteps):
            t_b    = t.expand(r_norm.shape[0]).to(r_norm.device)
            t_norm = t_b.float() / cfg.DIFF_STEPS

            noise_pred = self.diffusion(r_norm, x_base, latent, bathy, t_norm)

            # x0-prediction form of DDIM — more stable than epsilon form
            alpha_t  = self.sqrt_ab[t] ** 2
            alpha_t_ = (self.sqrt_ab[timesteps[i + 1]] ** 2
                        if i < n_steps - 1
                        else torch.zeros(1, device=r_norm.device))

            x0_pred  = (r_norm - (1 - alpha_t).sqrt() * noise_pred) \
                    / alpha_t.sqrt().clamp(min=1e-5)
            x0_pred  = x0_pred.clamp(-5, 5)           # prevent explosion

            r_norm   = alpha_t_.sqrt() * x0_pred \
                    + (1 - alpha_t_).sqrt() * noise_pred

        # Denormalise back to residual space
        r = r_norm * r_std
        return (x_base + r).clamp(-4, 4)



# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 v2 — EDM-PRECONDITIONED RESIDUAL DIFFUSION
# Same Stage4Encoder, but replaces ResidualDiffusion with EDMDiffusion
# and adopts Karras et al. 2022 preconditioning + domain-shift conditioning
# ═══════════════════════════════════════════════════════════════════════════════

class DomainNorm(nn.Module):
    """
    Injects per-sample domain statistics as a learned conditioning vector.
    At inference on unseen domains, pass the domain's sst_mean/sst_std
    so the diffusion head can adapt its scale without retraining.

    Input:  sst_mean [B], sst_std [B]  (scalar stats of the target domain)
    Output: [B, D] conditioning embedding
    """
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, sst_mean, sst_std):
        stats = torch.stack([sst_mean, sst_std], dim=-1).float()  # [B, 2]
        return self.mlp(stats)                                      # [B, D]


class EDMConvBlock(nn.Module):
    """ConvBlock with AdaGN conditioning from time + domain embedding."""
    def __init__(self, in_ch, out_ch, d_cond, groups=8):
        super().__init__()
        g = min(groups, out_ch)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(g, out_ch)
        self.norm2 = nn.GroupNorm(g, out_ch)
        self.act   = nn.SiLU()
        # AdaGN: scale+shift from conditioning
        self.cond_proj = nn.Linear(d_cond, out_ch * 2)

    def forward(self, x, cond_emb):
        # cond_emb: [B, d_cond]
        scale, shift = self.cond_proj(cond_emb).chunk(2, dim=-1)
        scale = scale.view(-1, scale.shape[1], 1, 1) + 1.0
        shift = shift.view(-1, shift.shape[1], 1, 1)

        x = self.act(self.norm1(self.conv1(x)) * scale + shift)
        x = self.act(self.norm2(self.conv2(x)) * scale + shift)
        return x


class EDMDiffusion(nn.Module):
    """
    EDM-style (Karras et al. 2022) diffusion UNet with:
      - c_in / c_skip / c_out preconditioning for training stability
      - Log-normal noise during training for better level coverage
      - Second-order Heun sampler at inference (18-35 steps)
      - AdaGN conditioning from time + domain stats (for zero-shot transfer)

    sigma_data ≈ std of the normalised residual in GSL training data.
    Set cfg.SIGMA_DATA ≈ 0.06 (tune from your actual residual std).
    """
    def __init__(self):
        super().__init__()
        C      = cfg.DIFF_CHANNELS   # 64
        D      = cfg.D_MODEL         # 256
        D_cond = C * 2               # 128

        # Time embedding: log(sigma) → d_cond
        self.time_mlp = nn.Sequential(
            nn.Linear(1, D_cond * 2), nn.SiLU(),
            nn.Linear(D_cond * 2, D_cond),
        )

        # Domain stats conditioning
        self.domain_norm = DomainNorm(D_cond)

        # Merge time + domain into single conditioning vector
        self.cond_merge = nn.Sequential(
            nn.Linear(D_cond * 2, D_cond), nn.SiLU(),
        )

        # Upsample latent to fine resolution
        self.cond_proj = nn.Conv2d(D, C, 1)

        # Input: c_in * x_noisy(1) + x_base(1) + cond(C) + bathy(1) = C+3
        in_ch = C + 3

        self.enc1 = EDMConvBlock(in_ch,  C,     D_cond)
        self.enc2 = EDMConvBlock(C,      C * 2, D_cond)
        self.enc3 = EDMConvBlock(C * 2,  C * 2, D_cond)
        self.mid  = EDMConvBlock(C * 2,  C * 2, D_cond)
        self.dec3 = EDMConvBlock(C * 4,  C * 2, D_cond)
        self.dec2 = EDMConvBlock(C * 4,  C,     D_cond)
        self.dec1 = EDMConvBlock(C * 2,  C,     D_cond)
        self.out  = nn.Conv2d(C, 1, 1)
        self.pool = nn.MaxPool2d(2)

        # EDM sigma bounds
        self.sigma_min  = 0.002
        self.sigma_max  = 80.0
        self.sigma_data = getattr(cfg, 'SIGMA_DATA', 0.06)
        self.P_mean     = -1.2    # log-normal P(sigma) mean
        self.P_std      = 1.2     # log-normal P(sigma) std

    # ── EDM preconditioning scalars ───────────────────────────────────────────
    def _c_skip(self, sigma):
        sd2 = self.sigma_data ** 2
        return sd2 / (sigma ** 2 + sd2)

    def _c_out(self, sigma):
        sd  = self.sigma_data
        return sigma * sd / (sigma ** 2 + sd ** 2).sqrt()

    def _c_in(self, sigma):
        return 1.0 / (sigma ** 2 + self.sigma_data ** 2).sqrt()

    def _c_noise(self, sigma):
        return sigma.log() / 4.0

    # ── Raw UNet forward (takes preconditioned inputs) ────────────────────────
    def _unet(self, x_in, x_base, latent, bathy, cond_emb):
        fine_size = bathy.shape[-2:]
        cond_map  = F.interpolate(latent, size=fine_size,
                                  mode='bilinear', align_corners=False)
        cond_map  = self.cond_proj(cond_map)
        x         = torch.cat([x_in, x_base, cond_map, bathy], dim=1)

        e1 = self.enc1(x,            cond_emb)
        e2 = self.enc2(self.pool(e1), cond_emb)
        e3 = self.enc3(self.pool(e2), cond_emb)
        m  = self.mid (self.pool(e3), cond_emb)

        d3 = self.dec3(torch.cat([
            F.interpolate(m,  e3.shape[-2:], mode='nearest'), e3], dim=1), cond_emb)
        d2 = self.dec2(torch.cat([
            F.interpolate(d3, e2.shape[-2:], mode='nearest'), e2], dim=1), cond_emb)
        d1 = self.dec1(torch.cat([
            F.interpolate(d2, e1.shape[-2:], mode='nearest'), e1], dim=1), cond_emb)
        return self.out(d1)

    # ── D_theta: preconditioned denoiser ─────────────────────────────────────
    def D_theta(self, x_noisy, sigma, x_base, latent, bathy,
                domain_mean=None, domain_std=None):
        """
        Returns denoised x0 estimate using EDM preconditioning.
        sigma: [B] noise level
        """
        B = x_noisy.shape[0]
        s = sigma.view(B, 1, 1, 1)

        c_skip  = self._c_skip(s)
        c_out   = self._c_out(s)
        c_in    = self._c_in(s)
        c_noise = self._c_noise(sigma)                 # [B]

        # Time conditioning
        t_emb = self.time_mlp(c_noise.unsqueeze(-1))  # [B, D_cond]

        # Domain conditioning (zero if not provided — GSL distribution)
        if domain_mean is None:
            domain_mean = torch.zeros(B, device=x_noisy.device)
        if domain_std is None:
            domain_std = torch.ones(B, device=x_noisy.device)
        d_emb = self.domain_norm(domain_mean, domain_std)  # [B, D_cond]

        cond_emb = self.cond_merge(torch.cat([t_emb, d_emb], dim=-1))  # [B, D_cond]

        F_x = self._unet(c_in * x_noisy, x_base, latent, bathy, cond_emb)

        # EDM output formula: D(x) = c_skip * x + c_out * F(x)
        return c_skip * x_noisy + c_out * F_x

    def forward_train(self, r_gt, x_base, latent, bathy,
                      domain_mean=None, domain_std=None):
        """
        EDM training loss with log-normal sigma distribution.
        r_gt: [B, 1, H, W] — ground truth residual (sst - x_base)
        """
        B = r_gt.shape[0]

        # Sample sigma from log-normal (Karras et al. Table 1)
        ln_sigma = torch.randn(B, device=r_gt.device) * self.P_std + self.P_mean
        sigma    = ln_sigma.exp().clamp(self.sigma_min, self.sigma_max)
        s        = sigma.view(B, 1, 1, 1)

        # Noise
        noise   = torch.randn_like(r_gt)
        r_noisy = r_gt + s * noise

        # Denoised prediction
        D_x = self.D_theta(r_noisy, sigma, x_base, latent, bathy,
                           domain_mean, domain_std)

        # EDM loss weight: lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
        lam  = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        lam  = lam.view(B, 1, 1, 1)

        loss = (lam * (D_x - r_gt) ** 2).mean()
        return loss

    @torch.no_grad()
    def edm_sample(self, x_base, latent, bathy, n_steps=20,
                   domain_mean=None, domain_std=None):
        """
        Second-order Heun sampler (Algorithm 1, Karras et al. 2022).
        n_steps=20 is typically sufficient; 35 for publication quality.
        """
        B         = x_base.shape[0]
        fine_size = bathy.shape[-2:]

        # Sigma schedule: geometric sequence from sigma_max to sigma_min
        rho       = 7.0   # EDM default
        steps     = torch.arange(n_steps + 1, device=x_base.device).float()
        sigma_max_rho = self.sigma_max ** (1 / rho)
        sigma_min_rho = self.sigma_min ** (1 / rho)
        sigmas    = (sigma_max_rho + steps / n_steps *
                     (sigma_min_rho - sigma_max_rho)) ** rho
        sigmas    = torch.cat([sigmas, sigmas.new_zeros(1)])  # append 0

        # Initial sample: x ~ N(0, sigma_max^2)
        x = torch.randn(B, 1, *fine_size, device=x_base.device) * sigmas[0]

        for i in range(n_steps):
            sigma_i   = sigmas[i].expand(B)
            sigma_i1  = sigmas[i + 1].expand(B)

            # First-order step
            D0 = self.D_theta(x, sigma_i, x_base, latent, bathy,
                              domain_mean, domain_std)
            d0 = (x - D0) / sigmas[i]
            x2 = x + d0 * (sigmas[i + 1] - sigmas[i])

            # Second-order correction (skip on last step where sigma → 0)
            if sigmas[i + 1] > 0:
                D1   = self.D_theta(x2, sigma_i1, x_base, latent, bathy,
                                    domain_mean, domain_std)
                d1   = (x2 - D1) / sigmas[i + 1]
                d_av = 0.5 * (d0 + d1)
                x    = x + d_av * (sigmas[i + 1] - sigmas[i])
            else:
                x = x2

        return x   # refined residual in normalised space


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — LATENT DIFFUSION  (scalable multi-domain deployment)
# Adds a lightweight VAE that compresses the fine SST residual
# into a small latent, then runs EDMDiffusion in that space.
# This makes inference feasible on large domains (Med, Lab, GoM).
# ═══════════════════════════════════════════════════════════════════════════════

class ResidualVAE(nn.Module):
    """
    Lightweight VAE: encodes fine-resolution SST residual to a compressed
    latent map, decodes back. Trained first (freeze before Stage 5 diffusion).

    Latent scale factor F=8: a 500×1200 field → 63×150 latent.
    This is the same compression ratio as Stable Diffusion.
    """
    def __init__(self, latent_ch=cfg.VAE_LATENT_CH):
        super().__init__()
        C  = cfg.DIFF_CHANNELS   # 64
        LC = latent_ch           # 4 (default)

        # Encoder: SST(1) + bathy(1) → latent [2*LC] (mean + logvar)
        self.encoder = nn.Sequential(
            ConvBlock(2, C),
            nn.MaxPool2d(2),
            ConvBlock(C, C * 2),
            nn.MaxPool2d(2),
            ConvBlock(C * 2, C * 2),
            nn.MaxPool2d(2),
            ConvBlock(C * 2, C * 2),
            nn.Conv2d(C * 2, LC * 2, 1),   # → [B, 2*LC, H/8, W/8]
        )

        # Decoder: latent [LC] + bathy(1) → residual [1]
        self.dec_in  = nn.Conv2d(LC + 1, C * 2, 1)
        self.decoder = nn.Sequential(
            ConvBlock(C * 2, C * 2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(C * 2, C * 2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(C * 2, C),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(C, C),
        )
        self.out = nn.Conv2d(C, 1, 1)

        self.kl_weight = getattr(cfg, 'VAE_KL_WEIGHT', 1e-4)

    def encode(self, r, bathy):
        """r, bathy: [B, 1, H, W] → mean, logvar: [B, LC, H/8, W/8]"""
        fine_size = r.shape[-2:]
        bathy_up  = F.interpolate(bathy, size=fine_size,
                                  mode='bilinear', align_corners=False)
        h        = self.encoder(torch.cat([r, bathy_up], dim=1))
        mean, logvar = h.chunk(2, dim=1)
        logvar   = logvar.clamp(-30, 20)
        return mean, logvar

    def decode(self, z, bathy):
        """z: [B, LC, H/8, W/8], bathy: [B, 1, H_f, W_f] → [B, 1, H_f, W_f]"""
        fine_size  = bathy.shape[-2:]
        bathy_lat  = F.adaptive_avg_pool2d(bathy, z.shape[-2:])
        x          = self.dec_in(torch.cat([z, bathy_lat], dim=1))
        x          = self.decoder(x)
        x          = F.interpolate(x, size=fine_size,
                                   mode='bilinear', align_corners=False)
        return self.out(x)

    def forward(self, r, bathy):
        """Returns reconstruction and KL loss term."""
        mean, logvar = self.encode(r, bathy)
        std          = (0.5 * logvar).exp()
        z            = mean + std * torch.randn_like(std)
        r_rec        = self.decode(z, bathy)
        kl           = -0.5 * (1 + logvar - mean ** 2 - logvar.exp()).mean()
        return r_rec, kl * self.kl_weight


class Stage5Encoder(nn.Module):
    """
    Stage 5 encoder = Stage 4 encoder (unchanged).
    Kept as a separate class so TemporalDownscaler can key on stage=5.
    """
    def __init__(self):
        super().__init__()
        self._enc = Stage4Encoder(T_atm=cfg.T_ATM, T_oce=cfg.T_OCE)

    @property
    def nH(self): return self._enc.nH
    @property
    def nW(self): return self._enc.nW

    def forward(self, era5_atm, mur_oce):
        return self._enc(era5_atm, mur_oce)


# class Stage5LatentDiffusion(nn.Module):
#     """
#     Runs EDMDiffusion in the VAE latent space (H/8 × W/8).
#     The UNet is much smaller and faster than pixel-space diffusion.

#     Training protocol:
#       Phase A: train ResidualVAE alone (freeze encoder + baseline).
#       Phase B: freeze VAE encoder, train latent EDMDiffusion.
#     """
#     def __init__(self):
#         super().__init__()
#         LC = getattr(cfg, 'VAE_LATENT_CH', 4)
#         C  = cfg.DIFF_CHANNELS
#         D  = cfg.D_MODEL
#         D_cond = C * 2

#         # VAE (pre-trained, then frozen in Phase B)
#         self.vae = ResidualVAE(latent_ch=LC)

#         # Time + domain conditioning (same as EDMDiffusion)
#         self.time_mlp    = nn.Sequential(
#             nn.Linear(1, D_cond * 2), nn.SiLU(),
#             nn.Linear(D_cond * 2, D_cond),
#         )
#         self.domain_norm = DomainNorm(D_cond)
#         self.cond_merge  = nn.Sequential(
#             nn.Linear(D_cond * 2, D_cond), nn.SiLU(),
#         )

#         # Project encoder latent (D) + x_base to latent spatial size
#         self.cond_proj  = nn.Conv2d(D + 1, C, 1)

#         # UNet operates on LC-channel latent
#         in_ch = LC + C   # VAE latent + spatial conditioning
#         self.enc1 = EDMConvBlock(in_ch,  C,     D_cond)
#         self.enc2 = EDMConvBlock(C,      C * 2, D_cond)
#         self.mid  = EDMConvBlock(C * 2,  C * 2, D_cond)
#         self.dec2 = EDMConvBlock(C * 4,  C,     D_cond)
#         self.dec1 = EDMConvBlock(C * 2,  C,     D_cond)
#         self.out  = nn.Conv2d(C, LC, 1)   # predict in latent space
#         self.pool = nn.MaxPool2d(2)

#         # EDM sigma bounds
#         self.sigma_min  = 0.002
#         self.sigma_max  = 80.0
#         self.sigma_data = getattr(cfg, 'SIGMA_DATA', 0.06)
#         self.P_mean     = -1.2
#         self.P_std      = 1.2

#     def _cond_emb(self, sigma, B, device,
#                   domain_mean=None, domain_std=None):
#         c_noise = sigma.log() / 4.0
#         t_emb   = self.time_mlp(c_noise.unsqueeze(-1))
#         if domain_mean is None:
#             domain_mean = torch.zeros(B, device=device)
#         if domain_std is None:
#             domain_std  = torch.ones(B, device=device)
#         d_emb   = self.domain_norm(domain_mean, domain_std)
#         return self.cond_merge(torch.cat([t_emb, d_emb], dim=-1))

#     def _c_skip(self, s): return self.sigma_data**2 / (s**2 + self.sigma_data**2)
#     def _c_out (self, s): return s * self.sigma_data / (s**2 + self.sigma_data**2).sqrt()
#     def _c_in  (self, s): return 1.0 / (s**2 + self.sigma_data**2).sqrt()

#     def _unet(self, z_in, lat_cond, cond_emb):
#         x  = torch.cat([z_in, lat_cond], dim=1)
#         e1 = self.enc1(x,             cond_emb)
#         e2 = self.enc2(self.pool(e1), cond_emb)
#         m  = self.mid (self.pool(e2), cond_emb)
#         d2 = self.dec2(torch.cat([
#             F.interpolate(m,  e2.shape[-2:], mode='nearest'), e2], dim=1), cond_emb)
#         d1 = self.dec1(torch.cat([
#             F.interpolate(d2, e1.shape[-2:], mode='nearest'), e1], dim=1), cond_emb)
#         return self.out(d1)   # [B, LC, H_lat, W_lat]

#     def D_theta_latent(self, z_noisy, sigma, lat_cond,
#                        domain_mean=None, domain_std=None):
#         B  = z_noisy.shape[0]
#         s  = sigma.view(B, 1, 1, 1)
#         ce = self._cond_emb(sigma, B, z_noisy.device, domain_mean, domain_std)
#         Fz = self._unet(self._c_in(s) * z_noisy, lat_cond, ce)
#         return self._c_skip(s) * z_noisy + self._c_out(s) * Fz

#     def _spatial_cond(self, enc_latent, x_base, lat_hw):
#         """Project encoder latent + x_base to VAE latent spatial size."""
#         cond = F.adaptive_avg_pool2d(
#             torch.cat([
#                 enc_latent,
#                 F.adaptive_avg_pool2d(x_base,
#                     (enc_latent.shape[-2], enc_latent.shape[-1]))
#             ], dim=1),
#             lat_hw
#         )
#         return self.cond_proj(cond)   # [B, C, H_lat, W_lat]

#     def forward_train(self, r_gt, x_base, enc_latent, bathy,
#                       domain_mean=None, domain_std=None):
#         """Phase B training: VAE encoder frozen, train latent diffusion."""
#         B = r_gt.shape[0]

#         # Encode residual to latent (frozen VAE encoder)
#         with torch.no_grad():
#             z_mean, z_logvar = self.vae.encode(r_gt, bathy)
#         z_std = (0.5 * z_logvar).exp()
#         z0    = z_mean + z_std * torch.randn_like(z_std)  # [B, LC, H_lat, W_lat]

#         lat_hw   = z0.shape[-2:]
#         lat_cond = self._spatial_cond(enc_latent, x_base, lat_hw)

#         # EDM log-normal sigma
#         ln_sigma = torch.randn(B, device=r_gt.device) * self.P_std + self.P_mean
#         sigma    = ln_sigma.exp().clamp(self.sigma_min, self.sigma_max)
#         s        = sigma.view(B, 1, 1, 1)
#         z_noisy  = z0 + s * torch.randn_like(z0)

#         D_z = self.D_theta_latent(z_noisy, sigma, lat_cond,
#                                   domain_mean, domain_std)

#         lam  = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data)**2
#         loss = (lam.view(B, 1, 1, 1) * (D_z - z0)**2).mean()
#         return loss

#     @torch.no_grad()
#     def sample(self, x_base, enc_latent, bathy, n_steps=20,
#                domain_mean=None, domain_std=None):
#         """Heun sampler in latent space, then decode with VAE."""
#         B         = x_base.shape[0]
#         # derive latent spatial size from VAE compression factor (8×)
#         lat_hw    = (bathy.shape[-2] // 8, bathy.shape[-1] // 8)
#         lat_cond  = self._spatial_cond(enc_latent, x_base, lat_hw)

#         rho       = 7.0
#         steps     = torch.arange(n_steps + 1, device=x_base.device).float()
#         smax_rho  = self.sigma_max ** (1 / rho)
#         smin_rho  = self.sigma_min ** (1 / rho)
#         sigmas    = (smax_rho + steps / n_steps * (smin_rho - smax_rho)) ** rho
#         sigmas    = torch.cat([sigmas, sigmas.new_zeros(1)])

#         LC = self.vae.encoder[-1].weight.shape[0] // 2  # recover LC
#         z  = torch.randn(B, LC, *lat_hw, device=x_base.device) * sigmas[0]

#         for i in range(n_steps):
#             si  = sigmas[i].expand(B)
#             si1 = sigmas[i + 1].expand(B)

#             D0 = self.D_theta_latent(z, si, lat_cond, domain_mean, domain_std)
#             d0 = (z - D0) / sigmas[i]
#             z2 = z + d0 * (sigmas[i + 1] - sigmas[i])

#             if sigmas[i + 1] > 0:
#                 D1   = self.D_theta_latent(z2, si1, lat_cond, domain_mean, domain_std)
#                 d1   = (z2 - D1) / sigmas[i + 1]
#                 z    = z + 0.5 * (d0 + d1) * (sigmas[i + 1] - sigmas[i])
#             else:
#                 z = z2

#         # Decode latent → pixel-space residual
#         r = self.vae.decode(z, bathy)
#         return (x_base + r).clamp(-4, 4)


# # ─────────────────────────────────────────────────────────────────────────────
# # Use existing TemporalDownscaler(stage=1..4) unchanged for ablation stages.
# #
# # Stage 4 v2 and Stage 5 are separate top-level model classes.
# # In train.py:
# #   if args.stage == '4v2': model = TemporalDownscalerV2()
# #   if args.stage == '5':   model = TemporalDownscalerV5()
# # ─────────────────────────────────────────────────────────────────────────────


# class TemporalDownscalerV2(nn.Module):
#     """
#     Stage 4 v2: Stage4Encoder + BaselineDecoder + EDMDiffusion.
#     Replaces only the diffusion component with EDM preconditioning.
#     TemporalDownscaler(stage=1..4) is completely unchanged.
#     """
#     def __init__(self):
#         super().__init__()
#         self.encoder   = Stage4Encoder(T_atm=cfg.T_ATM, T_oce=cfg.T_OCE)
#         self.baseline  = BaselineDecoder()
#         self.diffusion = EDMDiffusion()

#     def _encode(self, batch):
#         return self.encoder(
#             batch['era5'],
#             batch['mur_seq'],
#             batch['mur_lat'],
#             batch['mur_lon'],
#         )

#     def _domain_cond(self, batch, B, device):
#         dm = batch.get('domain_sst_mean', torch.zeros(B, device=device))
#         ds = batch.get('domain_sst_std',  torch.ones(B,  device=device))
#         return dm, ds

#     def forward(self, batch):
#         sst    = torch.nan_to_num(batch['sst'],    nan=0.0, posinf=0.0, neginf=0.0)
#         weight = torch.nan_to_num(batch['weight'], nan=0.0, posinf=0.0, neginf=0.0)
#         bathy  = batch['bathy']
#         B      = sst.shape[0]

#         latent = self._encode(batch)
#         x_base = self.baseline(latent, bathy)
#         r_gt   = (sst - x_base).detach()

#         def wmean(x):
#             return (x * weight).sum() / (weight.sum() + 1e-6)

#         loss_base = wmean(F.l1_loss(x_base, sst, reduction='none'))

#         dm, ds    = self._domain_cond(batch, B, sst.device)
#         loss_diff = self.diffusion.forward_train(
#             r_gt, x_base.detach(), latent.detach(), bathy, dm, ds)

#         pred_m    = torch.nan_to_num(x_base * weight, nan=0.0)
#         targ_m    = torch.nan_to_num(sst    * weight, nan=0.0)
#         loss_spec = TemporalDownscaler._spectral_loss(pred_m, targ_m)

#         loss = loss_base + 10.0 * loss_diff + 0.1 * loss_spec
#         return {
#             'loss':      loss,
#             'loss_base': loss_base.item(),
#             'loss_diff': loss_diff.item(),
#             'loss_spec': loss_spec.item(),
#         }

#     @torch.no_grad()
#     def sample(self, batch, n_steps=None):
#         n_steps = n_steps or cfg.DIFF_SAMPLE_STEPS
#         bathy   = batch['bathy']
#         latent  = self._encode(batch)
#         x_base  = self.baseline(latent, bathy)
#         B       = x_base.shape[0]
#         dm, ds  = self._domain_cond(batch, B, x_base.device)
#         r = self.diffusion.edm_sample(x_base, latent, bathy, n_steps, dm, ds)
#         return (x_base + r).clamp(-4, 4)


# class TemporalDownscalerV5(nn.Module):
#     """
#     Stage 5: Stage4Encoder + BaselineDecoder + Stage5LatentDiffusion.
#     Completely separate from TemporalDownscaler.
#     Train in two phases — pass phase='a' or phase='b' to forward().
#     """
#     def __init__(self):
#         super().__init__()
#         self.encoder   = Stage4Encoder(T_atm=cfg.T_ATM, T_oce=cfg.T_OCE)
#         self.baseline  = BaselineDecoder()
#         self.diffusion = Stage5LatentDiffusion()

#     def _encode(self, batch):
#         return self.encoder(
#             batch['era5'],
#             batch['mur_seq'],
#             batch['mur_lat'],
#             batch['mur_lon'],
#         )

#     def _domain_cond(self, batch, B, device):
#         dm = batch.get('domain_sst_mean', torch.zeros(B, device=device))
#         ds = batch.get('domain_sst_std',  torch.ones(B,  device=device))
#         return dm, ds

#     def forward(self, batch, phase='b'):
#         """
#         phase='a' — train VAE only (freeze encoder + baseline)
#         phase='b' — train latent diffusion only (freeze vae.encoder)
#         """
#         assert phase in ('a', 'b'), "phase must be 'a' or 'b'"

#         sst    = torch.nan_to_num(batch['sst'],    nan=0.0, posinf=0.0, neginf=0.0)
#         weight = torch.nan_to_num(batch['weight'], nan=0.0, posinf=0.0, neginf=0.0)
#         bathy  = batch['bathy']
#         B      = sst.shape[0]

#         def wmean(x):
#             return (x * weight).sum() / (weight.sum() + 1e-6)

#         if phase == 'a':
#             # Freeze encoder + baseline, train VAE only
#             with torch.no_grad():
#                 latent = self._encode(batch)
#                 x_base = self.baseline(latent, bathy)
#             r_gt    = (sst - x_base).detach()
#             r_rec, kl_loss = self.diffusion.vae(r_gt, bathy)
#             loss_vae  = wmean(F.l1_loss(r_rec, r_gt, reduction='none')) + kl_loss
#             return {
#                 'loss':      loss_vae,
#                 'loss_vae':  loss_vae.item(),
#                 'loss_diff': 0.0,
#                 'loss_base': 0.0,
#                 'loss_spec': 0.0,
#             }

#         # phase == 'b': freeze vae encoder, train latent diffusion
#         latent = self._encode(batch)
#         x_base = self.baseline(latent, bathy)
#         r_gt   = (sst - x_base).detach()

#         loss_base = wmean(F.l1_loss(x_base, sst, reduction='none'))

#         dm, ds    = self._domain_cond(batch, B, sst.device)
#         loss_diff = self.diffusion.forward_train(
#             r_gt, x_base.detach(), latent.detach(), bathy, dm, ds)

#         pred_m    = torch.nan_to_num(x_base * weight, nan=0.0)
#         targ_m    = torch.nan_to_num(sst    * weight, nan=0.0)
#         loss_spec = TemporalDownscaler._spectral_loss(pred_m, targ_m)

#         loss = loss_base + 10.0 * loss_diff + 0.1 * loss_spec
#         return {
#             'loss':      loss,
#             'loss_base': loss_base.item(),
#             'loss_diff': loss_diff.item(),
#             'loss_spec': loss_spec.item(),
#             'loss_vae':  0.0,
#         }

#     @torch.no_grad()
#     def sample(self, batch, n_steps=None):
#         n_steps = n_steps or cfg.DIFF_SAMPLE_STEPS
#         bathy   = batch['bathy']
#         latent  = self._encode(batch)
#         x_base  = self.baseline(latent, bathy)
#         B       = x_base.shape[0]
#         dm, ds  = self._domain_cond(batch, B, x_base.device)
#         return self.diffusion.sample(x_base, latent, bathy, n_steps, dm, ds)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 v2 — EDM-PRECONDITIONED RESIDUAL DIFFUSION
# Same Stage4Encoder, but replaces ResidualDiffusion with EDMDiffusion
# and adopts Karras et al. 2022 preconditioning + domain-shift conditioning
# ═══════════════════════════════════════════════════════════════════════════════

class DomainNorm(nn.Module):
    """
    Injects per-sample domain statistics as a learned conditioning vector.
    At inference on unseen domains, pass the domain's sst_mean/sst_std
    so the diffusion head can adapt its scale without retraining.

    Input:  sst_mean [B], sst_std [B]  (scalar stats of the target domain)
    Output: [B, D] conditioning embedding
    """
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, sst_mean, sst_std):
        stats = torch.stack([sst_mean, sst_std], dim=-1).float()  # [B, 2]
        return self.mlp(stats)                                      # [B, D]


class EDMConvBlock(nn.Module):
    """ConvBlock with AdaGN conditioning from time + domain embedding."""
    def __init__(self, in_ch, out_ch, d_cond, groups=8):
        super().__init__()
        g = min(groups, out_ch)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(g, out_ch)
        self.norm2 = nn.GroupNorm(g, out_ch)
        self.act   = nn.SiLU()
        # AdaGN: scale+shift from conditioning
        self.cond_proj = nn.Linear(d_cond, out_ch * 2)

    def forward(self, x, cond_emb):
        # cond_emb: [B, d_cond]
        scale, shift = self.cond_proj(cond_emb).chunk(2, dim=-1)
        scale = scale.view(-1, scale.shape[1], 1, 1) + 1.0
        shift = shift.view(-1, shift.shape[1], 1, 1)

        x = self.act(self.norm1(self.conv1(x)) * scale + shift)
        x = self.act(self.norm2(self.conv2(x)) * scale + shift)
        return x


class EDMDiffusion(nn.Module):
    """
    EDM-style (Karras et al. 2022) diffusion UNet with:
      - c_in / c_skip / c_out preconditioning for training stability
      - Log-normal noise during training for better level coverage
      - Second-order Heun sampler at inference (18-35 steps)
      - AdaGN conditioning from time + domain stats (for zero-shot transfer)

    sigma_data ≈ std of the normalised residual in GSL training data.
    Set cfg.SIGMA_DATA ≈ 0.06 (tune from your actual residual std).
    """
    def __init__(self):
        super().__init__()
        C      = cfg.DIFF_CHANNELS   # 64
        D      = cfg.D_MODEL         # 256
        D_cond = C * 2               # 128

        # Time embedding: log(sigma) → d_cond
        self.time_mlp = nn.Sequential(
            nn.Linear(1, D_cond * 2), nn.SiLU(),
            nn.Linear(D_cond * 2, D_cond),
        )

        # Domain stats conditioning
        self.domain_norm = DomainNorm(D_cond)

        # Merge time + domain into single conditioning vector
        self.cond_merge = nn.Sequential(
            nn.Linear(D_cond * 2, D_cond), nn.SiLU(),
        )

        # Upsample latent to fine resolution
        self.cond_proj = nn.Conv2d(D, C, 1)

        # Input: c_in * x_noisy(1) + x_base(1) + cond(C) + bathy(1) = C+3
        in_ch = C + 3

        self.enc1 = EDMConvBlock(in_ch,  C,     D_cond)
        self.enc2 = EDMConvBlock(C,      C * 2, D_cond)
        self.enc3 = EDMConvBlock(C * 2,  C * 2, D_cond)
        self.mid  = EDMConvBlock(C * 2,  C * 2, D_cond)
        self.dec3 = EDMConvBlock(C * 4,  C * 2, D_cond)
        self.dec2 = EDMConvBlock(C * 4,  C,     D_cond)
        self.dec1 = EDMConvBlock(C * 2,  C,     D_cond)
        self.out  = nn.Conv2d(C, 1, 1)
        self.pool = nn.MaxPool2d(2)

        # EDM sigma bounds
        self.sigma_min  = 0.002
        self.sigma_max  = 80.0
        self.sigma_data = getattr(cfg, 'SIGMA_DATA', 0.06)
        self.P_mean     = -1.2    # log-normal P(sigma) mean
        self.P_std      = 1.2     # log-normal P(sigma) std

    # ── EDM preconditioning scalars ───────────────────────────────────────────
    def _c_skip(self, sigma):
        sd2 = self.sigma_data ** 2
        return sd2 / (sigma ** 2 + sd2)

    def _c_out(self, sigma):
        sd  = self.sigma_data
        return sigma * sd / (sigma ** 2 + sd ** 2).sqrt()

    def _c_in(self, sigma):
        return 1.0 / (sigma ** 2 + self.sigma_data ** 2).sqrt()

    def _c_noise(self, sigma):
        return sigma.log() / 4.0

    # ── Raw UNet forward (takes preconditioned inputs) ────────────────────────
    def _unet(self, x_in, x_base, latent, bathy, cond_emb):
        fine_size = bathy.shape[-2:]
        cond_map  = F.interpolate(latent, size=fine_size,
                                  mode='bilinear', align_corners=False)
        cond_map  = self.cond_proj(cond_map)
        x         = torch.cat([x_in, x_base, cond_map, bathy], dim=1)

        e1 = self.enc1(x,            cond_emb)
        e2 = self.enc2(self.pool(e1), cond_emb)
        e3 = self.enc3(self.pool(e2), cond_emb)
        m  = self.mid (self.pool(e3), cond_emb)

        d3 = self.dec3(torch.cat([
            F.interpolate(m,  e3.shape[-2:], mode='nearest'), e3], dim=1), cond_emb)
        d2 = self.dec2(torch.cat([
            F.interpolate(d3, e2.shape[-2:], mode='nearest'), e2], dim=1), cond_emb)
        d1 = self.dec1(torch.cat([
            F.interpolate(d2, e1.shape[-2:], mode='nearest'), e1], dim=1), cond_emb)
        return self.out(d1)

    # ── D_theta: preconditioned denoiser ─────────────────────────────────────
    def D_theta(self, x_noisy, sigma, x_base, latent, bathy,
                domain_mean=None, domain_std=None):
        """
        Returns denoised x0 estimate using EDM preconditioning.
        sigma: [B] noise level
        """
        B = x_noisy.shape[0]
        s = sigma.view(B, 1, 1, 1)

        c_skip  = self._c_skip(s)
        c_out   = self._c_out(s)
        c_in    = self._c_in(s)
        c_noise = self._c_noise(sigma)                 # [B]

        # Time conditioning
        t_emb = self.time_mlp(c_noise.unsqueeze(-1))  # [B, D_cond]

        # Domain conditioning (zero if not provided — GSL distribution)
        if domain_mean is None:
            domain_mean = torch.zeros(B, device=x_noisy.device)
        if domain_std is None:
            domain_std = torch.ones(B, device=x_noisy.device)
        d_emb = self.domain_norm(domain_mean, domain_std)  # [B, D_cond]

        cond_emb = self.cond_merge(torch.cat([t_emb, d_emb], dim=-1))  # [B, D_cond]

        F_x = self._unet(c_in * x_noisy, x_base, latent, bathy, cond_emb)

        # EDM output formula: D(x) = c_skip * x + c_out * F(x)
        return c_skip * x_noisy + c_out * F_x

    def forward_train(self, r_gt, x_base, latent, bathy,
                      domain_mean=None, domain_std=None):
        """
        EDM training loss with log-normal sigma distribution.
        r_gt: [B, 1, H, W] — ground truth residual (sst - x_base)
        """
        B = r_gt.shape[0]

        # Sample sigma from log-normal (Karras et al. Table 1)
        ln_sigma = torch.randn(B, device=r_gt.device) * self.P_std + self.P_mean
        sigma    = ln_sigma.exp().clamp(self.sigma_min, self.sigma_max)
        s        = sigma.view(B, 1, 1, 1)

        # Noise
        noise   = torch.randn_like(r_gt)
        r_noisy = r_gt + s * noise

        # Denoised prediction
        D_x = self.D_theta(r_noisy, sigma, x_base, latent, bathy,
                           domain_mean, domain_std)

        # EDM loss weight: lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
        lam  = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        lam  = lam.view(B, 1, 1, 1)

        loss = (lam * (D_x - r_gt) ** 2).mean()
        return loss

    @torch.no_grad()
    def edm_sample(self, x_base, latent, bathy, n_steps=20,
                   domain_mean=None, domain_std=None):
        """
        Second-order Heun sampler (Algorithm 1, Karras et al. 2022).
        n_steps=20 is typically sufficient; 35 for publication quality.
        """
        B         = x_base.shape[0]
        fine_size = bathy.shape[-2:]

        # Sigma schedule: geometric sequence from sigma_max to sigma_min
        rho       = 7.0   # EDM default
        steps     = torch.arange(n_steps + 1, device=x_base.device).float()
        sigma_max_rho = self.sigma_max ** (1 / rho)
        sigma_min_rho = self.sigma_min ** (1 / rho)
        sigmas    = (sigma_max_rho + steps / n_steps *
                     (sigma_min_rho - sigma_max_rho)) ** rho
        sigmas    = torch.cat([sigmas, sigmas.new_zeros(1)])  # append 0

        # Initial sample: x ~ N(0, sigma_max^2)
        x = torch.randn(B, 1, *fine_size, device=x_base.device) * sigmas[0]

        for i in range(n_steps):
            sigma_i   = sigmas[i].expand(B)
            sigma_i1  = sigmas[i + 1].expand(B)

            # First-order step
            D0 = self.D_theta(x, sigma_i, x_base, latent, bathy,
                              domain_mean, domain_std)
            d0 = (x - D0) / sigmas[i]
            x2 = x + d0 * (sigmas[i + 1] - sigmas[i])

            # Second-order correction (skip on last step where sigma → 0)
            if sigmas[i + 1] > 0:
                D1   = self.D_theta(x2, sigma_i1, x_base, latent, bathy,
                                    domain_mean, domain_std)
                d1   = (x2 - D1) / sigmas[i + 1]
                d_av = 0.5 * (d0 + d1)
                x    = x + d_av * (sigmas[i + 1] - sigmas[i])
            else:
                x = x2

        return x   # refined residual in normalised space


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — LATENT DIFFUSION  (scalable multi-domain deployment)
# Adds a lightweight VAE that compresses the fine SST residual
# into a small latent, then runs EDMDiffusion in that space.
# This makes inference feasible on large domains (Med, Lab, GoM).
# ═══════════════════════════════════════════════════════════════════════════════

class ResidualVAE(nn.Module):
    """
    Lightweight VAE: encodes fine-resolution SST residual to a compressed
    latent map, decodes back. Trained first (freeze before Stage 5 diffusion).

    Latent scale factor F=8: a 500×1200 field → 63×150 latent.
    This is the same compression ratio as Stable Diffusion.
    """
    def __init__(self, latent_ch=cfg.VAE_LATENT_CH):
        super().__init__()
        C  = cfg.DIFF_CHANNELS   # 64
        LC = latent_ch           # 4 (default)

        # Encoder: SST(1) + bathy(1) → latent [2*LC] (mean + logvar)
        self.encoder = nn.Sequential(
            ConvBlock(2, C),
            nn.MaxPool2d(2),
            ConvBlock(C, C * 2),
            nn.MaxPool2d(2),
            ConvBlock(C * 2, C * 2),
            nn.MaxPool2d(2),
            ConvBlock(C * 2, C * 2),
            nn.Conv2d(C * 2, LC * 2, 1),   # → [B, 2*LC, H/8, W/8]
        )

        # Decoder: latent [LC] + bathy(1) → residual [1]
        self.dec_in  = nn.Conv2d(LC + 1, C * 2, 1)
        self.decoder = nn.Sequential(
            ConvBlock(C * 2, C * 2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(C * 2, C * 2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(C * 2, C),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(C, C),
        )
        self.out = nn.Conv2d(C, 1, 1)

        self.kl_weight = getattr(cfg, 'VAE_KL_WEIGHT', 1e-4)

    def encode(self, r, bathy):
        """r, bathy: [B, 1, H, W] → mean, logvar: [B, LC, H/8, W/8]"""
        fine_size = r.shape[-2:]
        bathy_up  = F.interpolate(bathy, size=fine_size,
                                  mode='bilinear', align_corners=False)
        h        = self.encoder(torch.cat([r, bathy_up], dim=1))
        mean, logvar = h.chunk(2, dim=1)
        logvar   = logvar.clamp(-30, 20)
        return mean, logvar

    def decode(self, z, bathy):
        """z: [B, LC, H/8, W/8], bathy: [B, 1, H_f, W_f] → [B, 1, H_f, W_f]"""
        fine_size  = bathy.shape[-2:]
        bathy_lat  = F.adaptive_avg_pool2d(bathy, z.shape[-2:])
        x          = self.dec_in(torch.cat([z, bathy_lat], dim=1))
        x          = self.decoder(x)
        x          = F.interpolate(x, size=fine_size,
                                   mode='bilinear', align_corners=False)
        return self.out(x)

    def forward(self, r, bathy):
        """Returns reconstruction and KL loss term."""
        mean, logvar = self.encode(r, bathy)
        std          = (0.5 * logvar).exp()
        z            = mean + std * torch.randn_like(std)
        r_rec        = self.decode(z, bathy)
        kl           = -0.5 * (1 + logvar - mean ** 2 - logvar.exp()).mean()
        return r_rec, kl * self.kl_weight


class Stage5Encoder(nn.Module):
    """
    Stage 5 encoder = Stage 4 encoder (unchanged).
    Kept as a separate class so TemporalDownscaler can key on stage=5.
    """
    def __init__(self):
        super().__init__()
        self._enc = Stage4Encoder(T_atm=cfg.T_ATM, T_oce=cfg.T_OCE)

    @property
    def nH(self): return self._enc.nH
    @property
    def nW(self): return self._enc.nW

    def forward(self, era5_atm, mur_oce):
        return self._enc(era5_atm, mur_oce)


class Stage5LatentDiffusion(nn.Module):
    """
    Runs EDMDiffusion in the VAE latent space (H/8 × W/8).
    The UNet is much smaller and faster than pixel-space diffusion.

    Training protocol:
      Phase A: train ResidualVAE alone (freeze encoder + baseline).
      Phase B: freeze VAE encoder, train latent EDMDiffusion.
    """
    def __init__(self):
        super().__init__()
        LC = getattr(cfg, 'VAE_LATENT_CH', 4)
        C  = cfg.DIFF_CHANNELS
        D  = cfg.D_MODEL
        D_cond = C * 2

        # VAE (pre-trained, then frozen in Phase B)
        self.vae = ResidualVAE(latent_ch=LC)

        # Time + domain conditioning (same as EDMDiffusion)
        self.time_mlp    = nn.Sequential(
            nn.Linear(1, D_cond * 2), nn.SiLU(),
            nn.Linear(D_cond * 2, D_cond),
        )
        self.domain_norm = DomainNorm(D_cond)
        self.cond_merge  = nn.Sequential(
            nn.Linear(D_cond * 2, D_cond), nn.SiLU(),
        )

        # Project encoder latent (D) + x_base to latent spatial size
        self.cond_proj  = nn.Conv2d(D + 1, C, 1)

        # UNet operates on LC-channel latent
        in_ch = LC + C   # VAE latent + spatial conditioning
        self.enc1 = EDMConvBlock(in_ch,  C,     D_cond)
        self.enc2 = EDMConvBlock(C,      C * 2, D_cond)
        self.mid  = EDMConvBlock(C * 2,  C * 2, D_cond)
        self.dec2 = EDMConvBlock(C * 4,  C,     D_cond)
        self.dec1 = EDMConvBlock(C * 2,  C,     D_cond)
        self.out  = nn.Conv2d(C, LC, 1)   # predict in latent space
        self.pool = nn.MaxPool2d(2)

        # EDM sigma bounds
        self.sigma_min  = 0.002
        self.sigma_max  = 80.0
        self.sigma_data = getattr(cfg, 'SIGMA_DATA', 0.06)
        self.P_mean     = -1.2
        self.P_std      = 1.2

    def _cond_emb(self, sigma, B, device,
                  domain_mean=None, domain_std=None):
        c_noise = sigma.log() / 4.0
        t_emb   = self.time_mlp(c_noise.unsqueeze(-1))
        if domain_mean is None:
            domain_mean = torch.zeros(B, device=device)
        if domain_std is None:
            domain_std  = torch.ones(B, device=device)
        d_emb   = self.domain_norm(domain_mean, domain_std)
        return self.cond_merge(torch.cat([t_emb, d_emb], dim=-1))

    def _c_skip(self, s): return self.sigma_data**2 / (s**2 + self.sigma_data**2)
    def _c_out (self, s): return s * self.sigma_data / (s**2 + self.sigma_data**2).sqrt()
    def _c_in  (self, s): return 1.0 / (s**2 + self.sigma_data**2).sqrt()

    def _unet(self, z_in, lat_cond, cond_emb):
        x  = torch.cat([z_in, lat_cond], dim=1)
        e1 = self.enc1(x,             cond_emb)
        e2 = self.enc2(self.pool(e1), cond_emb)
        m  = self.mid (self.pool(e2), cond_emb)
        d2 = self.dec2(torch.cat([
            F.interpolate(m,  e2.shape[-2:], mode='nearest'), e2], dim=1), cond_emb)
        d1 = self.dec1(torch.cat([
            F.interpolate(d2, e1.shape[-2:], mode='nearest'), e1], dim=1), cond_emb)
        return self.out(d1)   # [B, LC, H_lat, W_lat]

    def D_theta_latent(self, z_noisy, sigma, lat_cond,
                       domain_mean=None, domain_std=None):
        B  = z_noisy.shape[0]
        s  = sigma.view(B, 1, 1, 1)
        ce = self._cond_emb(sigma, B, z_noisy.device, domain_mean, domain_std)
        Fz = self._unet(self._c_in(s) * z_noisy, lat_cond, ce)
        return self._c_skip(s) * z_noisy + self._c_out(s) * Fz

    def _spatial_cond(self, enc_latent, x_base, lat_hw):
        """Project encoder latent + x_base to VAE latent spatial size."""
        cond = F.adaptive_avg_pool2d(
            torch.cat([
                enc_latent,
                F.adaptive_avg_pool2d(x_base,
                    (enc_latent.shape[-2], enc_latent.shape[-1]))
            ], dim=1),
            lat_hw
        )
        return self.cond_proj(cond)   # [B, C, H_lat, W_lat]

    def forward_train(self, r_gt, x_base, enc_latent, bathy,
                      domain_mean=None, domain_std=None):
        """Phase B training: VAE encoder frozen, train latent diffusion."""
        B = r_gt.shape[0]

        # Encode residual to latent (frozen VAE encoder)
        with torch.no_grad():
            z_mean, z_logvar = self.vae.encode(r_gt, bathy)
        z_std = (0.5 * z_logvar).exp()
        z0    = z_mean + z_std * torch.randn_like(z_std)  # [B, LC, H_lat, W_lat]

        lat_hw   = z0.shape[-2:]
        lat_cond = self._spatial_cond(enc_latent, x_base, lat_hw)

        # EDM log-normal sigma
        ln_sigma = torch.randn(B, device=r_gt.device) * self.P_std + self.P_mean
        sigma    = ln_sigma.exp().clamp(self.sigma_min, self.sigma_max)
        s        = sigma.view(B, 1, 1, 1)
        z_noisy  = z0 + s * torch.randn_like(z0)

        D_z = self.D_theta_latent(z_noisy, sigma, lat_cond,
                                  domain_mean, domain_std)

        lam  = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data)**2
        loss = (lam.view(B, 1, 1, 1) * (D_z - z0)**2).mean()
        return loss

    @torch.no_grad()
    def sample(self, x_base, enc_latent, bathy, n_steps=20,
               domain_mean=None, domain_std=None):
        """Heun sampler in latent space, then decode with VAE."""
        B         = x_base.shape[0]
        # derive latent spatial size from VAE compression factor (8×)
        lat_hw    = (bathy.shape[-2] // 8, bathy.shape[-1] // 8)
        lat_cond  = self._spatial_cond(enc_latent, x_base, lat_hw)

        rho       = 7.0
        steps     = torch.arange(n_steps + 1, device=x_base.device).float()
        smax_rho  = self.sigma_max ** (1 / rho)
        smin_rho  = self.sigma_min ** (1 / rho)
        sigmas    = (smax_rho + steps / n_steps * (smin_rho - smax_rho)) ** rho
        sigmas    = torch.cat([sigmas, sigmas.new_zeros(1)])

        LC = self.vae.encoder[-1].weight.shape[0] // 2  # recover LC
        z  = torch.randn(B, LC, *lat_hw, device=x_base.device) * sigmas[0]

        for i in range(n_steps):
            si  = sigmas[i].expand(B)
            si1 = sigmas[i + 1].expand(B)

            D0 = self.D_theta_latent(z, si, lat_cond, domain_mean, domain_std)
            d0 = (z - D0) / sigmas[i]
            z2 = z + d0 * (sigmas[i + 1] - sigmas[i])

            if sigmas[i + 1] > 0:
                D1   = self.D_theta_latent(z2, si1, lat_cond, domain_mean, domain_std)
                d1   = (z2 - D1) / sigmas[i + 1]
                z    = z + 0.5 * (d0 + d1) * (sigmas[i + 1] - sigmas[i])
            else:
                z = z2

        # Decode latent → pixel-space residual
        r = self.vae.decode(z, bathy)
        return (x_base + r).clamp(-4, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Use existing TemporalDownscaler(stage=1..4) unchanged for ablation stages.
#
# Stage 4 v2 and Stage 5 are separate top-level model classes.
# In train.py:
#   if args.stage == '4v2': model = TemporalDownscalerV2()
#   if args.stage == '5':   model = TemporalDownscalerV5()
# ─────────────────────────────────────────────────────────────────────────────


class TemporalDownscalerV2(nn.Module):
    """
    Stage 4 v2: Stage4Encoder + BaselineDecoder + EDMDiffusion.
    Replaces only the diffusion component with EDM preconditioning.
    TemporalDownscaler(stage=1..4) is completely unchanged.
    """
    def __init__(self):
        super().__init__()
        self.encoder   = Stage4Encoder(T_atm=cfg.T_ATM, T_oce=cfg.T_OCE)
        self.baseline  = BaselineDecoder()
        self.diffusion = EDMDiffusion()

    def _encode(self, batch):
        return self.encoder(batch['era5'], batch['mur_seq'])

    def _domain_cond(self, batch, B, device):
        dm = batch.get('domain_sst_mean', torch.zeros(B, device=device))
        ds = batch.get('domain_sst_std',  torch.ones(B,  device=device))
        return dm, ds

    def forward(self, batch):
        sst    = torch.nan_to_num(batch['sst'],    nan=0.0, posinf=0.0, neginf=0.0)
        weight = torch.nan_to_num(batch['weight'], nan=0.0, posinf=0.0, neginf=0.0)
        bathy  = batch['bathy']
        B      = sst.shape[0]

        latent = self._encode(batch)
        x_base = self.baseline(latent, bathy)
        r_gt   = (sst - x_base).detach()

        def wmean(x):
            return (x * weight).sum() / (weight.sum() + 1e-6)

        loss_base = wmean(F.l1_loss(x_base, sst, reduction='none'))

        dm, ds    = self._domain_cond(batch, B, sst.device)
        loss_diff = self.diffusion.forward_train(
            r_gt, x_base.detach(), latent.detach(), bathy, dm, ds)

        pred_m    = torch.nan_to_num(x_base * weight, nan=0.0)
        targ_m    = torch.nan_to_num(sst    * weight, nan=0.0)
        loss_spec = TemporalDownscaler._spectral_loss(pred_m, targ_m)

        loss = loss_base + 10.0 * loss_diff + 0.1 * loss_spec
        return {
            'loss':      loss,
            'loss_base': loss_base.item(),
            'loss_diff': loss_diff.item(),
            'loss_spec': loss_spec.item(),
        }

    @torch.no_grad()
    def sample(self, batch, n_steps=None):
        n_steps = n_steps or cfg.DIFF_SAMPLE_STEPS
        bathy   = batch['bathy']
        latent  = self._encode(batch)
        x_base  = self.baseline(latent, bathy)
        B       = x_base.shape[0]
        dm, ds  = self._domain_cond(batch, B, x_base.device)
        r = self.diffusion.edm_sample(x_base, latent, bathy, n_steps, dm, ds)
        return (x_base + r).clamp(-4, 4)


class TemporalDownscalerV5(nn.Module):
    """
    Stage 5: Stage4Encoder + BaselineDecoder + Stage5LatentDiffusion.
    Completely separate from TemporalDownscaler.
    Train in two phases — pass phase='a' or phase='b' to forward().
    """
    def __init__(self):
        super().__init__()
        self.encoder   = Stage4Encoder(T_atm=cfg.T_ATM, T_oce=cfg.T_OCE)
        self.baseline  = BaselineDecoder()
        self.diffusion = Stage5LatentDiffusion()

    def _encode(self, batch):
        return self.encoder(batch['era5'], batch['mur_seq'])

    def _domain_cond(self, batch, B, device):
        dm = batch.get('domain_sst_mean', torch.zeros(B, device=device))
        ds = batch.get('domain_sst_std',  torch.ones(B,  device=device))
        return dm, ds

    def forward(self, batch, phase='b'):
        """
        phase='a' — train VAE only (freeze encoder + baseline)
        phase='b' — train latent diffusion only (freeze vae.encoder)
        """
        assert phase in ('a', 'b'), "phase must be 'a' or 'b'"

        sst    = torch.nan_to_num(batch['sst'],    nan=0.0, posinf=0.0, neginf=0.0)
        weight = torch.nan_to_num(batch['weight'], nan=0.0, posinf=0.0, neginf=0.0)
        bathy  = batch['bathy']
        B      = sst.shape[0]

        def wmean(x):
            return (x * weight).sum() / (weight.sum() + 1e-6)

        if phase == 'a':
            # Freeze encoder + baseline, train VAE only
            with torch.no_grad():
                latent = self._encode(batch)
                x_base = self.baseline(latent, bathy)
            r_gt    = (sst - x_base).detach()
            r_rec, kl_loss = self.diffusion.vae(r_gt, bathy)
            loss_vae  = wmean(F.l1_loss(r_rec, r_gt, reduction='none')) + kl_loss
            return {
                'loss':      loss_vae,
                'loss_vae':  loss_vae.item(),
                'loss_diff': 0.0,
                'loss_base': 0.0,
                'loss_spec': 0.0,
            }

        # phase == 'b': freeze vae encoder, train latent diffusion
        latent = self._encode(batch)
        x_base = self.baseline(latent, bathy)
        r_gt   = (sst - x_base).detach()

        loss_base = wmean(F.l1_loss(x_base, sst, reduction='none'))

        dm, ds    = self._domain_cond(batch, B, sst.device)
        loss_diff = self.diffusion.forward_train(
            r_gt, x_base.detach(), latent.detach(), bathy, dm, ds)

        pred_m    = torch.nan_to_num(x_base * weight, nan=0.0)
        targ_m    = torch.nan_to_num(sst    * weight, nan=0.0)
        loss_spec = TemporalDownscaler._spectral_loss(pred_m, targ_m)

        loss = loss_base + 10.0 * loss_diff + 0.1 * loss_spec
        return {
            'loss':      loss,
            'loss_base': loss_base.item(),
            'loss_diff': loss_diff.item(),
            'loss_spec': loss_spec.item(),
            'loss_vae':  0.0,
        }

    @torch.no_grad()
    def sample(self, batch, n_steps=None):
        n_steps = n_steps or cfg.DIFF_SAMPLE_STEPS
        bathy   = batch['bathy']
        latent  = self._encode(batch)
        x_base  = self.baseline(latent, bathy)
        B       = x_base.shape[0]
        dm, ds  = self._domain_cond(batch, B, x_base.device)
        return self.diffusion.sample(x_base, latent, bathy, n_steps, dm, ds)
class OceanCoordPE(nn.Module):
    """
    Sinusoidal encoding of normalized lat/lon at patch centers.
    No learned positional table; optional linear projection at the end.
    """
    def __init__(self, d_model, n_freqs=16, use_proj=True):
        super().__init__()
        self.n_freqs = n_freqs
        self.out_dim = 4 * n_freqs   # sin/cos for lat and lon
        self.use_proj = use_proj
        if use_proj:
            self.proj = nn.Linear(self.out_dim, d_model)

    def forward(self, lat_patch, lon_patch):
        """
        lat_patch, lon_patch: [B, N] normalized to [-1, 1]
        """
        device = lat_patch.device
        freqs = 2.0 ** torch.arange(self.n_freqs, device=device).float() * torch.pi

        lat = lat_patch.unsqueeze(-1) * freqs   # [B, N, F]
        lon = lon_patch.unsqueeze(-1) * freqs

        feat = torch.cat([
            torch.sin(lat), torch.cos(lat),
            torch.sin(lon), torch.cos(lon),
        ], dim=-1)                              # [B, N, 4F]

        if self.use_proj:
            feat = self.proj(feat)              # [B, N, D]
        return feat

def patch_center_coords(lat, lon, nH, nW):
    """
    lat, lon: [B, H, W] or [B, 1, H, W]
    Returns patch-center coords [B, N] for the runtime patch grid.
    Uses adaptive average pooling to map full-res coordinates to patch grid.
    """
    if lat.dim() == 4:
        lat = lat[:, 0]
    if lon.dim() == 4:
        lon = lon[:, 0]

    lat_p = F.adaptive_avg_pool2d(lat.unsqueeze(1), (nH, nW)).squeeze(1)
    lon_p = F.adaptive_avg_pool2d(lon.unsqueeze(1), (nH, nW)).squeeze(1)

    lat_p = rearrange(lat_p, 'b h w -> b (h w)')
    lon_p = rearrange(lon_p, 'b h w -> b (h w)')

    return lat_p, lon_p

class OceanSpatialBlockRoPE(nn.Module):
    """
    Spatial attention over ocean tokens within each frame,
    with 2D rotary positional embedding applied to q/k.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        assert (d_model // n_heads) % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def _rope_2d(self, q, k, nH, nW):
        """
        q, k: [BT, heads, N, head_dim]
        Applies separable 2D RoPE using y/x token indices.
        """
        BT, Hh, N, Dh = q.shape
        device = q.device
        assert N == nH * nW

        y, x = torch.meshgrid(
            torch.arange(nH, device=device),
            torch.arange(nW, device=device),
            indexing='ij'
        )
        y = y.reshape(-1).float()   # [N]
        x = x.reshape(-1).float()   # [N]

        quarter = Dh // 4
        freq = torch.exp(
            -torch.arange(quarter, device=device).float() * (np.log(10000.0) / quarter)
        )  # [quarter]

        y_theta = y[:, None] * freq[None, :]    # [N, quarter]
        x_theta = x[:, None] * freq[None, :]

        cos_y = torch.cos(y_theta)[None, None, :, :]
        sin_y = torch.sin(y_theta)[None, None, :, :]
        cos_x = torch.cos(x_theta)[None, None, :, :]
        sin_x = torch.sin(x_theta)[None, None, :, :]

        qy, qx = q[..., :2*quarter], q[..., 2*quarter:]
        ky, kx = k[..., :2*quarter], k[..., 2*quarter:]

        def rotate_half(z):
            z1, z2 = z[..., ::2], z[..., 1::2]
            return torch.stack([-z2, z1], dim=-1).flatten(-2)

        qy = qy * cos_y.repeat_interleave(2, dim=-1) + rotate_half(qy) * sin_y.repeat_interleave(2, dim=-1)
        ky = ky * cos_y.repeat_interleave(2, dim=-1) + rotate_half(ky) * sin_y.repeat_interleave(2, dim=-1)
        qx = qx * cos_x.repeat_interleave(2, dim=-1) + rotate_half(qx) * sin_x.repeat_interleave(2, dim=-1)
        kx = kx * cos_x.repeat_interleave(2, dim=-1) + rotate_half(kx) * sin_x.repeat_interleave(2, dim=-1)

        q = torch.cat([qy, qx], dim=-1)
        k = torch.cat([ky, kx], dim=-1)
        return q, k

    def forward(self, x, T, N, nH, nW):
        """
        x: [B, T*N, D]
        """
        B = x.shape[0]
        xs = rearrange(x, 'b (t n) d -> (b t) n d', t=T, n=N)

        h = self.norm1(xs)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)

        q = rearrange(q, 'bt n (h d) -> bt h n d', h=self.n_heads)
        k = rearrange(k, 'bt n (h d) -> bt h n d', h=self.n_heads)
        v = rearrange(v, 'bt n (h d) -> bt h n d', h=self.n_heads)

        q, k = self._rope_2d(q, k, nH, nW)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(out, 'bt h n d -> bt n (h d)')
        out = self.out_proj(out)
        xs = xs + self.dropout(out)
        xs = xs + self.ffn(self.norm2(xs))

        return rearrange(xs, '(b t) n d -> b (t n) d', b=B, t=T)