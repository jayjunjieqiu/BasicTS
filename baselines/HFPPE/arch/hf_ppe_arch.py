from typing import Optional, Tuple, Dict

import math
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from baselines.PatchTST.arch.revin import RevIN


def _huber(x: Tensor, delta: float = 1.0, reduction: str = "mean") -> Tensor:
    """Huber loss on x with configurable delta.

    Args:
        x (Tensor): input tensor
        delta (float): huber delta
        reduction (str): 'mean' or 'sum' or 'none'
    """
    abs_x = x.abs()
    quad = torch.clamp(abs_x, max=delta)
    lin = abs_x - quad
    loss = 0.5 * quad.pow(2) + delta * lin
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def hf_ppe_loss(
    prediction: Tensor,
    target: Tensor,
    null_val: float = float("nan"),
    reg_unit: Optional[Tensor] = None,
    reg_phase: Optional[Tensor] = None,
    reg_cycle: Optional[Tensor] = None,
    lambda_unit: float = 0.0,
    lambda_phase: float = 0.0,
    lambda_cycle: float = 0.0,
    huber_delta: float = 1.0,
    lambda_fft: float = 0.0,
) -> Tensor:
    """Composite loss: masked MAE + regularizers.

    The runner will pass only matched keyword args, so it's safe to expose
    additional regularizer terms here; when the model doesn't return them,
    they default to zero contribution via None checks.
    """
    # Masked MAE
    if math.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).to(target.device).expand_as(target), atol=eps, rtol=0.0)
    mask = mask.float()
    mask = torch.nan_to_num(mask)
    mask = mask / (mask.mean() + 1e-8)
    mae = torch.abs(prediction - target)
    mae = torch.nan_to_num(mae * mask).mean()

    loss = mae
    if reg_unit is not None:
        loss = loss + lambda_unit * reg_unit
    if reg_phase is not None:
        # apply huber to phase regularization if it is a vector
        # but reg_phase here is usually pre-aggregated scalar; still keep behavior
        loss = loss + lambda_phase * reg_phase
    if reg_cycle is not None:
        loss = loss + lambda_cycle * reg_cycle

    # FFT spectrum alignment (optional)
    if lambda_fft > 0.0:
        # compute along time dim (L)
        # shapes: [B, L, N, 1]
        pred = prediction.squeeze(-1)
        tgt = target.squeeze(-1)
        # move to [B*N, L]
        B, L, N = pred.shape
        pred = pred.permute(0, 2, 1).reshape(B * N, L)
        tgt = tgt.permute(0, 2, 1).reshape(B * N, L)
        # rfft
        P = torch.fft.rfft(pred, dim=-1)
        T = torch.fft.rfft(tgt, dim=-1)
        # power spectra
        Pm = (P.real.pow(2) + P.imag.pow(2)).sqrt()
        Tm = (T.real.pow(2) + T.imag.pow(2)).sqrt()
        # normalize to avoid scale mismatch
        Pm = Pm / (Pm.sum(dim=-1, keepdim=True) + 1e-8)
        Tm = Tm / (Tm.sum(dim=-1, keepdim=True) + 1e-8)
        fft_mse = F.mse_loss(Pm, Tm)
        loss = loss + lambda_fft * fft_mse
    return loss


class PPEHyperNet(nn.Module):
    """Hyper-network h_phi that predicts per-harmonic parameters from a short context window.

    Given last `ctx_len` points per-node, predicts for each harmonic j=1..m:
        - amplitude a_j >= 0
        - angular frequency omega_j >= 0
        - phase phi_j in [-pi, pi]

    The same MLP is shared across nodes (applied independently).
    """

    def __init__(self, ctx_len: int, m: int, hidden: int = 128,
                 a_max: float = 1.5,
                 omega_min: float = 2 * math.pi / 336.0,
                 omega_max: float = 2 * math.pi / 12.0,
                 phi_mode: str = 'angle') -> None:
        super().__init__()
        self.ctx_len = ctx_len
        self.m = m
        self.a_max = float(a_max)
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.phi_mode = phi_mode
        out_dim = (3 if self.phi_mode == 'angle' else 4) * m
        self.net = nn.Sequential(
            nn.Linear(ctx_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, ctx: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Predict (a, omega, phi) from context.

        Args:
            ctx: [B, N, ctx_len] last ctx_len values of the target (or selected feature)

        Returns:
            a: [B, N, m]
            omega: [B, N, m]
            phi: [B, N, m]
        """
        B, N, Lc = ctx.shape
        assert Lc == self.ctx_len, "Context length mismatch for PPEHyperNet."
        x = ctx.reshape(B * N, Lc)
        raw = self.net(x)  # [B*N, out_dim]
        if self.phi_mode == 'angle':
            raw = raw.view(B, N, 3, self.m)
            a_raw, omega_raw, phi_raw = raw[:, :, 0, :], raw[:, :, 1, :], raw[:, :, 2, :]
        else:
            raw = raw.view(B, N, 4, self.m)
            a_raw, omega_raw, sin_raw, cos_raw = raw[:, :, 0, :], raw[:, :, 1, :], raw[:, :, 2, :], raw[:, :, 3, :]

        # Constrain amplitude and omega to sensible ranges for stability
        a = self.a_max * self.sigmoid(a_raw)
        omega = self.omega_min + (self.omega_max - self.omega_min) * self.sigmoid(omega_raw)
        if self.phi_mode == 'angle':
            phi = math.pi * torch.tanh(phi_raw)  # [-pi, pi]
            return a, omega, (phi, None, None)
        # predict sinφ, cosφ on unit circle
        # normalize to unit norm to avoid drift
        denom = torch.clamp((sin_raw.pow(2) + cos_raw.pow(2)).sqrt(), min=1e-6)
        sin_phi = sin_raw / denom
        cos_phi = cos_raw / denom
        return a, omega, (None, sin_phi, cos_phi)


class PPEHyperNetLSTM(nn.Module):
    """Stronger hyper-network using a tiny LSTM over context.

    Input per node sequence [ctx_len] -> LSTM -> heads for a, ω, and φ (angle or sin/cos).
    """

    def __init__(self, ctx_len: int, m: int, hidden: int = 128,
                 a_max: float = 1.5,
                 omega_min: float = 2 * math.pi / 336.0,
                 omega_max: float = 2 * math.pi / 12.0,
                 phi_mode: str = 'angle', lstm_hidden: int = 64, lstm_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.ctx_len = ctx_len
        self.m = m
        self.a_max = float(a_max)
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.phi_mode = phi_mode
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        out_dim = (3 if self.phi_mode == 'angle' else 4) * m
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, ctx: Tensor) -> Tuple[Tensor, Tensor, Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]]:
        B, N, Lc = ctx.shape
        assert Lc == self.ctx_len
        x = ctx.reshape(B * N, Lc, 1)
        h, _ = self.lstm(x)  # [B*N, Lc, H]
        feat = h[:, -1, :]  # [B*N, H]
        raw = self.head(feat)
        if self.phi_mode == 'angle':
            raw = raw.view(B, N, 3, self.m)
            a_raw, omega_raw, phi_raw = raw[:, :, 0, :], raw[:, :, 1, :], raw[:, :, 2, :]
        else:
            raw = raw.view(B, N, 4, self.m)
            a_raw, omega_raw, sin_raw, cos_raw = raw[:, :, 0, :], raw[:, :, 1, :], raw[:, :, 2, :], raw[:, :, 3, :]

        a = self.a_max * self.sigmoid(a_raw)
        omega = self.omega_min + (self.omega_max - self.omega_min) * self.sigmoid(omega_raw)
        if self.phi_mode == 'angle':
            phi = math.pi * torch.tanh(phi_raw)
            return a, omega, (phi, None, None)
        denom = torch.clamp((sin_raw.pow(2) + cos_raw.pow(2)).sqrt(), min=1e-6)
        sin_phi = sin_raw / denom
        cos_phi = cos_raw / denom
        return a, omega, (None, sin_phi, cos_phi)


class HFPPE(nn.Module):
    """HF-PPE: Attention-based encoder-decoder with Periodic Positional Encoder (PPE).

    Key ideas:
    - A hyper-network h_phi() predicts per-harmonic parameters (omega, phi, amplitude)
      from a short context window of the history.
    - A periodic code p_t is built by stacking [a_j cos(omega_j r_t + phi_j), a_j sin(omega_j r_t + phi_j)] for j=1..m.
    - An attention-based encoder consumes [p_t, x_t, (optional y_t)] over the history window.
    - A cross-attention decoder attends from PPE-derived queries at future steps to the encoder hidden states.

    The module supports arbitrary prediction lengths at inference time. Internal weights are independent of
    the horizon length.
    """

    def __init__(
        self,
        enc_in: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        m: int = 4,
        ctx_len: int = 24,
        use_y_in_encoder: bool = False,
        compute_phase_reg: bool = False,
        compute_cycle_reg: bool = False,
        reg_stride: int = 8,
        # RevIN toggle
        revin: bool = False,
        revin_affine: bool = False,
        revin_subtract_last: bool = False,
        # Fixed encoder projection input dim (avoid lazy init)
        hist_feat_dim: int = 1,
        # PPE normalization flag
        apply_ppe_ln: bool = False,
        # Hyper-net and phase options
        hnet_type: str = 'mlp',  # 'mlp' | 'lstm'
        phi_mode: str = 'sincos',  # 'angle' | 'sincos'
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.0,
        # Waveform head over [cosθ, sinθ]
        wave_mlp_hidden: int = 0,
        ppe_code_dim: int = 0,
        # Query fusion/gating and positional query features
        use_query_gate: bool = True,
        query_pos_dim: int = 0,
        a_max: float = 1.5,
        period_min: int = 12,
        period_max: int = 336,
        use_scaled_r: bool = False,
        pos_dim: int = 0,
        use_query_context: bool = True,
        # Debug options
        debug: bool = False,
        debug_every: int = 100,
        attn_debug: bool = False,
        # Simple mode: minimal, stable variant
        simple: bool = False,
    ) -> None:
        super().__init__()
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len  # not used for parameter shapes; kept for config clarity
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.m = m
        self.ctx_len = ctx_len
        self.use_y_in_encoder = use_y_in_encoder
        self.compute_phase_reg = compute_phase_reg
        self.compute_cycle_reg = compute_cycle_reg
        self.reg_stride = max(1, reg_stride)

        # PPE hyper-network (uses only target feature by default)
        self.use_scaled_r = bool(use_scaled_r)
        self.pos_dim = int(pos_dim)
        self.use_query_context = bool(use_query_context)
        # Debug
        self.debug = bool(debug)
        self.debug_every = int(debug_every)
        self.attn_debug = bool(attn_debug)
        # RevIN
        self.revin = bool(revin)
        if self.revin:
            self.revin_layer = RevIN(enc_in, affine=bool(revin_affine), subtract_last=bool(revin_subtract_last))
        omega_min = 2 * math.pi / float(max(1, period_max))
        omega_max = 2 * math.pi / float(max(1, period_min))
        self.phi_mode = phi_mode
        self.hnet_type = hnet_type
        self.simple = bool(simple)

        # If simple, override complex knobs to a minimal stable setup.
        if self.simple:
            self.use_scaled_r = False
            self.pos_dim = 0
            self.query_pos_dim = 0
            self.use_query_context = True
            self.use_query_gate = False
            self.apply_ppe_ln = True
            self.phi_mode = 'sincos'
            self.hnet_type = 'mlp'

        if self.hnet_type == 'lstm':
            self.ppe_hnet = PPEHyperNetLSTM(ctx_len=ctx_len, m=m, hidden=max(64, d_model),
                                            a_max=a_max, omega_min=omega_min, omega_max=omega_max,
                                            phi_mode=phi_mode, lstm_hidden=lstm_hidden, lstm_layers=lstm_layers, dropout=lstm_dropout)
        else:
            self.ppe_hnet = PPEHyperNet(ctx_len=ctx_len, m=m, hidden=max(64, d_model),
                                        a_max=a_max, omega_min=omega_min, omega_max=omega_max, phi_mode=phi_mode)

        # Waveform MLP over PPE code (optional)
        code_in = 2 * m
        # In simple mode, keep raw PPE and skip waveform head
        if self.simple:
            self.ppe_code_dim = code_in
            self.wave_mlp = None
        else:
            self.ppe_code_dim = int(ppe_code_dim) if ppe_code_dim > 0 else code_in
            self.wave_mlp = None
        if (not self.simple) and wave_mlp_hidden and code_in > 0:
            self.wave_mlp = nn.Sequential(
                nn.Linear(code_in, wave_mlp_hidden), nn.GELU(), nn.Linear(wave_mlp_hidden, self.ppe_code_dim)
            )

        # Encoder projection with fixed known input dim to avoid lazy creation
        # input dim = PPE(code_dim) + history features (hist_feat_dim) + optional positional feats (pos_dim)
        self.hist_feat_dim = int(hist_feat_dim)
        enc_proj_in = (self.ppe_code_dim if m > 0 else 0) + self.hist_feat_dim + self.pos_dim
        self.enc_in_ln = nn.LayerNorm(enc_proj_in)
        self.enc_proj = nn.Linear(enc_proj_in, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # Decoder: cross-attention from PPE queries to encoder outputs
        self.q_proj = nn.Linear(self.ppe_code_dim if m > 0 else d_model, d_model) if m > 0 else None
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.q_ln = nn.LayerNorm(d_model)

        # PPE normalization to avoid scale domination (optional)
        self.apply_ppe_ln = bool(apply_ppe_ln)
        self.ppe_ln = nn.LayerNorm(2 * m) if self.apply_ppe_ln else nn.Identity()

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        # Query context projection and gating
        self.q_ctx = nn.Linear(d_model, d_model)
        self.use_query_gate = bool(use_query_gate)
        if self.use_query_gate:
            self.alpha = nn.Parameter(torch.zeros(()))  # scalar gate
        # Positional features for queries
        self.query_pos_dim = int(query_pos_dim)
        if self.query_pos_dim > 0:
            self.q_pos_proj = nn.Linear(self.query_pos_dim, d_model)

    @staticmethod
    def _positional_feats(L: int, dim: int, device: torch.device) -> Tensor:
        """Create simple sinusoidal positional features [L, dim]."""
        if dim <= 0:
            return torch.zeros(L, 0, device=device)
        pe = torch.zeros(L, dim, device=device)
        position = torch.arange(0, L, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / max(1, dim)))
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim % 2 == 1:
            cos_part = torch.cos(position * div_term)
            pe[:, 1::2] = cos_part[:, : (dim // 2)]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    @staticmethod
    def _build_r_indices(length: int, device: torch.device, start: int = 0) -> Tensor:
        """Build linear relative indices r_t = start, start+1, ..., start+length-1."""
        return torch.arange(start, start + length, device=device).float()

    @staticmethod
    def _ppe_from_params(r: Tensor, a: Tensor, omega: Tensor, phi: Tensor) -> Tensor:
        """Construct PPE code p_t for all harmonics.

        Args:
            r: [T] time indices
            a, omega, phi: [B, N, m]

        Returns:
            p: [B, T, N, 2m]
        """
        # theta: [B, T, N, m]
        theta = omega.unsqueeze(1) * r.view(1, -1, 1, 1) + phi.unsqueeze(1)
        cos_comp = a.unsqueeze(1) * torch.cos(theta)
        sin_comp = a.unsqueeze(1) * torch.sin(theta)
        p = torch.cat([cos_comp, sin_comp], dim=-1)  # [B, T, N, 2m]
        return p

    @staticmethod
    def _ppe_from_params_sincos(r: Tensor, a: Tensor, omega: Tensor, sin_phi: Tensor, cos_phi: Tensor) -> Tensor:
        """Construct PPE when predicting sinφ and cosφ directly.

        cos(ωr+φ) = cos(ωr)cosφ - sin(ωr)sinφ
        sin(ωr+φ) = sin(ωr)cosφ + cos(ωr)sinφ
        """
        theta = omega.unsqueeze(1) * r.view(1, -1, 1, 1)
        cos_w = torch.cos(theta)
        sin_w = torch.sin(theta)
        cos_comp = a.unsqueeze(1) * (cos_w * cos_phi.unsqueeze(1) - sin_w * sin_phi.unsqueeze(1))
        sin_comp = a.unsqueeze(1) * (sin_w * cos_phi.unsqueeze(1) + cos_w * sin_phi.unsqueeze(1))
        return torch.cat([cos_comp, sin_comp], dim=-1)

    def _ensure_enc_proj(self, feat_dim: int) -> None:
        # No-op: enc_proj is created in __init__ with a fixed input dim.
        return

    def _compute_regularizers(
        self,
        r_hist: Tensor,
        a: Tensor,
        omega: Tensor,
        phi: Tensor,
        huber_delta: float = 1.0,
    ) -> Dict[str, Tensor]:
        """Compute unit-circle, phase-smoothness, and cycle-consistency regularizers.

        Returns a dict with scalar tensors for each regularizer.
        """
        B, N, M = a.shape

        # Unit-circle regularizer: encourage amplitude near 1
        reg_unit = (a - 1.0).pow(2).mean()

        # Phase-smoothness: Huber on (Δθ - ω * Δr)
        # Use downsampled r_hist to reduce memory; optionally disable.
        if self.compute_phase_reg and r_hist.numel() >= 2:
            r_ds = r_hist[:: self.reg_stride]
            if r_ds.numel() >= 2:
                dr = (r_ds[1:] - r_ds[:-1]).view(1, -1, 1, 1)  # [1, T-1, 1, 1]
                theta_hist = omega.unsqueeze(1) * r_ds.view(1, -1, 1, 1) + phi.unsqueeze(1)
                dtheta = theta_hist[:, 1:] - theta_hist[:, :-1]  # [B, T-1, N, M]
                phase_err = dtheta - omega.unsqueeze(1) * dr  # ideally zero
                reg_phase = _huber(phase_err, delta=huber_delta, reduction="mean")
            else:
                reg_phase = torch.zeros((), device=a.device)
        else:
            reg_phase = torch.zeros((), device=a.device)

        # Cycle-consistency: p_{t+T} ≈ p_t with T ~= 2π / mean(omega)
        # Use a single median shift across (B,N) for vectorized computation to save memory.
        if self.compute_cycle_reg:
            mean_omega = omega.mean(dim=-1)  # [B, N]
            T_est = (2 * math.pi) / torch.clamp(mean_omega, min=1e-6)
            T_int = torch.clamp(T_est.round().long(), min=1)  # [B, N]
            shift = int(T_int.median().item())
            L_hist = int(r_hist.shape[0])
            if shift < L_hist and shift > 0:
                p_hist_bn = self._ppe_from_params(r_hist, a, omega, phi)  # [B, L, N, 2m]
                p_hist_bn = p_hist_bn  # keep dims
                p1 = p_hist_bn[:, :-shift, :, :]
                p2 = p_hist_bn[:, shift:, :, :]
                reg_cycle = F.mse_loss(p1, p2)
            else:
                reg_cycle = torch.zeros((), device=a.device)
        else:
            reg_cycle = torch.zeros((), device=a.device)

        return {"reg_unit": reg_unit, "reg_phase": reg_phase, "reg_cycle": reg_cycle}

    def forward(
        self,
        history_data: Tensor,
        future_data: Tensor,
        batch_seen: int,
        epoch: Optional[int],
        train: bool,
        huber_delta: float = 1.0,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Forward pass.

        Args:
            history_data: [B, Lh, N, C_h]
            future_data: [B, Lf, N, C_f]
            huber_delta: delta for phase-smoothness Huber regularizer

        Returns:
            Dict with keys: 'prediction' [B, Lf, N, 1], 'inputs', 'target' (added by runner),
            and regularizer scalars: 'reg_unit', 'reg_phase', 'reg_cycle'.
        """
        B, Lh, N, Ch = history_data.shape
        _, Lf, _, _ = future_data.shape
        device = history_data.device

        # Optionally apply RevIN on target channel (inputs only)
        hist_for_model = history_data
        target_hist = history_data[..., 0]  # [B, Lh, N]
        if self.revin:
            norm_target = self.revin_layer(target_hist, mode='norm')
            hist_for_model = history_data.clone()
            hist_for_model[..., 0] = norm_target

        # Use the last ctx_len targets for hyper-net; fall back if shorter
        ctx_len = min(self.ctx_len, Lh)
        # Assume first channel is the target feature
        ctx = hist_for_model[:, -ctx_len:, :, 0].transpose(1, 2)  # [B, N, ctx]
        a, omega, phi_tuple = self.ppe_hnet(ctx)

        # Build r indices (raw or scaled)
        if self.use_scaled_r:
            if Lh > 1:
                r_hist = torch.linspace(-1.0, 1.0, steps=Lh, device=device)
                step = 2.0 / (Lh - 1)
            else:
                r_hist = torch.tensor([0.0], device=device)
                step = 1.0
            r_fut = 1.0 + step * torch.arange(1, Lf + 1, device=device)
        else:
            r_hist = self._build_r_indices(Lh, device=device, start=0)
            r_fut = self._build_r_indices(Lf, device=device, start=Lh)

        # PPE codes
        if self.m > 0:
            if self.phi_mode == 'angle':
                phi, _, _ = phi_tuple
                p_hist = self._ppe_from_params(r_hist, a, omega, phi)
                p_fut = self._ppe_from_params(r_fut, a, omega, phi)
            else:
                _, sin_phi, cos_phi = phi_tuple
                p_hist = self._ppe_from_params_sincos(r_hist, a, omega, sin_phi, cos_phi)
                p_fut = self._ppe_from_params_sincos(r_fut, a, omega, sin_phi, cos_phi)
        else:
            # PPE disabled
            p_hist = history_data.new_zeros((B, Lh, N, 0))
            p_fut = history_data.new_zeros((B, Lf, N, 0))
        # Normalize PPE code
        if self.apply_ppe_ln and self.m > 0:
            p_hist = self.ppe_ln(p_hist)
            p_fut = self.ppe_ln(p_fut)
        # Neural waveform head
        if self.wave_mlp is not None and self.m > 0:
            B1, L1, N1, C1 = p_hist.shape
            p_hist = self.wave_mlp(p_hist.view(B1 * L1 * N1, C1)).view(B1, L1, N1, -1)
            B2, L2, N2, C2 = p_fut.shape
            p_fut = self.wave_mlp(p_fut.view(B2 * L2 * N2, C2)).view(B2, L2, N2, -1)

        # Encoder inputs: concat [p_t, x_t, (optional y_t)]
        feats_hist = [p_hist, hist_for_model]
        if self.pos_dim > 0:
            pos = self._positional_feats(Lh, self.pos_dim, device=device).view(1, Lh, 1, self.pos_dim).expand(B, Lh, N, self.pos_dim)
            feats_hist.append(pos)
        # y_t optional if provided explicitly in kwargs under 'y_hist'; otherwise skip
        if self.use_y_in_encoder and history_data.shape[-1] >= 1:
            # already included in history_data; keep flag for compatibility
            pass
        enc_in = torch.cat(feats_hist, dim=-1)  # [B, Lh, N, code + Ch + pos]
        
        # Prepare for transformer: merge batch and nodes, time as sequence
        x_enc = enc_in.reshape(B * N, Lh, enc_in.shape[-1])
        x_enc = self.enc_in_ln(x_enc)
        x_enc = self.enc_proj(x_enc)  # [B*N, Lh, d_model]
        # guard against NaNs/Infs
        x_enc = torch.nan_to_num(x_enc, nan=0.0, posinf=1e6, neginf=-1e6)
        h_enc = self.encoder(x_enc)  # [B*N, Lh, d_model]
        h_enc = torch.nan_to_num(h_enc, nan=0.0, posinf=1e6, neginf=-1e6)

        # Decoder: queries from p_fut
        if self.m > 0 and self.q_proj is not None:
            q = self.q_proj(p_fut.reshape(B * N, Lf, -1))  # [B*N, Lf, d_model]
        else:
            # PPE disabled: start from context
            q = self.q_ctx(h_enc.mean(dim=1)).unsqueeze(1).expand(B * N, Lf, -1)
        if self.use_query_context:
            ctx = self.q_ctx(h_enc.mean(dim=1))  # [B*N, d_model]
            if self.use_query_gate:
                gate = torch.sigmoid(self.alpha)
                q = gate * q + (1.0 - gate) * ctx.unsqueeze(1)
            else:
                q = q + ctx.unsqueeze(1)
        # Optional positional features for queries
        if self.query_pos_dim > 0:
            pos_f = self._positional_feats(Lf, self.query_pos_dim, device=device)  # [Lf, D]
            pos_f = self.q_pos_proj(pos_f).unsqueeze(0).expand(B * N, Lf, -1)
            q = q + pos_f
        # normalize queries and guard
        q = self.q_ln(q)
        q = torch.nan_to_num(q, nan=0.0, posinf=1e6, neginf=-1e6)
        need_w = self.attn_debug and (not train or (batch_seen is not None and self.debug and (batch_seen % max(1, self.debug_every) == 0)))
        z, attn_w = self.cross_attn(query=q, key=h_enc, value=h_enc, need_weights=need_w, average_attn_weights=True)  # [B*N, Lf, d_model]
        z = torch.nan_to_num(z, nan=0.0, posinf=1e6, neginf=-1e6)

        y = self.head(z)  # [B*N, Lf, 1]
        y = y.view(B, N, Lf, 1).transpose(1, 2).contiguous()  # [B, Lf, N, 1]
        # Residual anchor with last observed value for stability
        seq_last = history_data[:, -1:, :, 0:1]
        y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6) + seq_last

        # Regularizers use an explicit angle; reconstruct if using sin/cos
        if self.phi_mode == 'angle':
            phi_for_reg = phi_tuple[0]
        else:
            sin_phi, cos_phi = phi_tuple[1], phi_tuple[2]
            phi_for_reg = torch.atan2(sin_phi, cos_phi)
        regs = self._compute_regularizers(r_hist=r_hist, a=a, omega=omega, phi=phi_for_reg, huber_delta=huber_delta)

        # Debug prints (lightweight)
        if self.debug and (batch_seen is not None) and (batch_seen % max(1, self.debug_every) == 0):
            with torch.no_grad():
                a_mean = a.mean().item(); a_std = a.std().item()
                w_mean = omega.mean().item(); w_std = omega.std().item()
                q_norm = q.norm(p=2, dim=-1).mean().item()
                h_norm = h_enc.norm(p=2, dim=-1).mean().item()
                attn_entropy = None
                if need_w and attn_w is not None:
                    w = attn_w.clamp_min(1e-8)
                    ent = -(w * (w.log())).sum(dim=-1) / math.log(max(2, w.shape[-1]))
                    attn_entropy = ent.mean().item()
                print(f"[HFPPE][step {batch_seen}] a: {a_mean:.3f}±{a_std:.3f}  ω: {w_mean:.3f}±{w_std:.3f}  |q|: {q_norm:.3f}  |h|: {h_norm:.3f}  H(attn): {attn_entropy}")

        return {
            "prediction": y,
            **regs,
        }
