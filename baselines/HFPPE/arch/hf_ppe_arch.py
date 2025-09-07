from typing import Optional, Tuple, Dict

import math
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


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
    return loss


class PPEHyperNet(nn.Module):
    """Hyper-network h_phi that predicts per-harmonic parameters from a short context window.

    Given last `ctx_len` points per-node, predicts for each harmonic j=1..m:
        - amplitude a_j >= 0
        - angular frequency omega_j >= 0
        - phase phi_j in [-pi, pi]

    The same MLP is shared across nodes (applied independently).
    """

    def __init__(self, ctx_len: int, m: int, hidden: int = 128) -> None:
        super().__init__()
        self.ctx_len = ctx_len
        self.m = m
        out_dim = 3 * m
        self.net = nn.Sequential(
            nn.Linear(ctx_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.amp_act = nn.Softplus()  # ensure amplitude >= 0
        self.omega_act = nn.Softplus()  # ensure omega >= 0

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
        raw = self.net(x)  # [B*N, 3m]
        raw = raw.view(B, N, 3, self.m)
        a_raw, omega_raw, phi_raw = raw[:, :, 0, :], raw[:, :, 1, :], raw[:, :, 2, :]

        a = self.amp_act(a_raw) + 1e-6
        omega = self.omega_act(omega_raw) + 1e-6  # avoid zero
        # map to [-pi, pi]
        phi = math.pi * torch.tanh(phi_raw)
        return a, omega, phi


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

        # PPE hyper-network (uses only target feature by default)
        self.ppe_hnet = PPEHyperNet(ctx_len=ctx_len, m=m, hidden=max(64, d_model))

        # Encoder: input dim = PPE(2m) + x_t (features in history) + optional y_t (target)
        # We accept variable feature dimension at runtime; project via a small Linear
        # Hence we build the projection lazily based on provided input dim using parameter-free ops.
        # To stay compatible with torchscript & DDP, predefine a generic projection conditioned by dim.
        # Here, keep a single nn.Linear that will be re-created if dim changes (unlikely in this repo).
        self._enc_in_dim: Optional[int] = None
        self.enc_proj: Optional[nn.Linear] = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # Decoder: cross-attention from PPE queries to encoder outputs
        self.q_proj = nn.Linear(2 * m, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

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

    def _ensure_enc_proj(self, feat_dim: int) -> None:
        if self._enc_in_dim != feat_dim:
            self._enc_in_dim = feat_dim
            self.enc_proj = nn.Linear(feat_dim, self.d_model).to(next(self.parameters()).device)

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
        # Using θ_t = ω * r_t + φ with constant ω (per context). Δθ_t = ω * Δr_t; ideally Δr_t = 1.
        # We still compute with provided r_hist to keep it general.
        if r_hist.numel() >= 2:
            dr = (r_hist[1:] - r_hist[:-1]).view(1, -1, 1, 1)  # [1, T-1, 1, 1]
            theta_hist = omega.unsqueeze(1) * r_hist.view(1, -1, 1, 1) + phi.unsqueeze(1)
            dtheta = theta_hist[:, 1:] - theta_hist[:, :-1]  # [B, T-1, N, M]
            phase_err = dtheta - omega.unsqueeze(1) * dr  # ideally zero
            reg_phase = _huber(phase_err, delta=huber_delta, reduction="mean")
        else:
            reg_phase = torch.zeros((), device=a.device)

        # Cycle-consistency: p_{t+T} ≈ p_t with T ~= 2π / mean(omega)
        mean_omega = omega.mean(dim=-1)  # [B, N]
        # Avoid division by zero; clamp omega
        T_est = (2 * math.pi) / torch.clamp(mean_omega, min=1e-6)
        T_int = torch.clamp(T_est.round().long(), min=1)
        L_hist = int(r_hist.shape[0])
        # choose a single shift per (B,N) for simplicity; compute on available pairs
        reg_cycle_terms = []
        for b in range(B):
            for n in range(N):
                shift = int(T_int[b, n].item())
                if shift >= L_hist:
                    continue
                # build p over history only
                p_hist = self._ppe_from_params(r_hist, a[b : b + 1, n : n + 1, :], omega[b : b + 1, n : n + 1, :], phi[b : b + 1, n : n + 1, :])
                p_hist = p_hist.squeeze(2)  # [1, L, 2m]
                p1 = p_hist[:, :-shift, :]
                p2 = p_hist[:, shift:, :]
                reg_cycle_terms.append(F.mse_loss(p1, p2))
        if len(reg_cycle_terms) == 0:
            reg_cycle = torch.zeros((), device=a.device)
        else:
            reg_cycle = torch.stack(reg_cycle_terms).mean()

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
        _, Lf, _, Cf = future_data.shape
        device = history_data.device

        # Use the last ctx_len targets for hyper-net; fall back if shorter
        ctx_len = min(self.ctx_len, Lh)
        # Assume first channel is the target feature
        ctx = history_data[:, -ctx_len:, :, 0].transpose(1, 2)  # [B, N, ctx]
        a, omega, phi = self.ppe_hnet(ctx)

        # Build r indices
        r_hist = self._build_r_indices(Lh, device=device, start=0)
        r_fut = self._build_r_indices(Lf, device=device, start=Lh)

        # PPE codes
        p_hist = self._ppe_from_params(r_hist, a, omega, phi)  # [B, Lh, N, 2m]
        p_fut = self._ppe_from_params(r_fut, a, omega, phi)  # [B, Lf, N, 2m]

        # Encoder inputs: concat [p_t, x_t, (optional y_t)]
        feats_hist = [p_hist, history_data]
        # y_t optional if provided explicitly in kwargs under 'y_hist'; otherwise skip
        if self.use_y_in_encoder and history_data.shape[-1] >= 1:
            # already included in history_data; keep flag for compatibility
            pass
        enc_in = torch.cat(feats_hist, dim=-1)  # [B, Lh, N, 2m + Ch]
        feat_dim = enc_in.shape[-1]
        self._ensure_enc_proj(feat_dim)

        # Prepare for transformer: merge batch and nodes, time as sequence
        x_enc = enc_in.reshape(B * N, Lh, feat_dim)
        x_enc = self.enc_proj(x_enc)  # [B*N, Lh, d_model]
        h_enc = self.encoder(x_enc)  # [B*N, Lh, d_model]

        # Decoder: queries from p_fut
        q = self.q_proj(p_fut.reshape(B * N, Lf, -1))  # [B*N, Lf, d_model]
        k = self.k_proj(h_enc)  # [B*N, Lh, d_model]
        v = self.v_proj(h_enc)  # [B*N, Lh, d_model]
        z, _ = self.cross_attn(query=q, key=k, value=v)  # [B*N, Lf, d_model]

        y = self.head(z)  # [B*N, Lf, 1]
        y = y.view(B, N, Lf, 1).transpose(1, 2).contiguous()  # [B, Lf, N, 1]

        regs = self._compute_regularizers(r_hist=r_hist, a=a, omega=omega, phi=phi, huber_delta=huber_delta)
        return {
            "prediction": y,
            **regs,
        }

