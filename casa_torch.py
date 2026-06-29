"""
CASA — Climate-Adaptive Spatial Attention (PyTorch)
====================================================

A compact (265-parameter) PyTorch implementation with autograd, so gradients
flow when CASA is embedded inside the LSTM-GSTARX-CASA model.

Supports both single-sample (Y shape `(N,)`) and batched (Y shape `(B, N)`)
forward calls. `A_geo` is shared across the batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Forward-pass cache (intermediate tensors, exposed for inspection)
# ---------------------------------------------------------------------------

@dataclass
class CASAStateTorch:
    Y_mean:    torch.Tensor   # (B,)
    Y_std:     torch.Tensor   # (B,)
    x_c:       torch.Tensor   # (B, 4)
    z_c:       torch.Tensor   # (B, d_c)
    c_t:       torch.Tensor   # (B, d_c)
    pair_feat: torch.Tensor   # (B, N, N, 3+d_c)
    z_a:       torch.Tensor   # (B, N, N, d_a)
    h_a:       torch.Tensor   # (B, N, N, d_a)
    e_t:       torch.Tensor   # (B, N, N)
    A_learned: torch.Tensor   # (B, N, N)
    z_alpha:   torch.Tensor   # (B,)
    alpha_t:   torch.Tensor   # (B,)
    W_t:       torch.Tensor   # (B, N, N)
    F_star:    torch.Tensor   # (B, N)


# ---------------------------------------------------------------------------
# CASATorch
# ---------------------------------------------------------------------------

class CASATorch(nn.Module):
    """Climate-Adaptive Spatial Attention (PyTorch).

    Parameters
    ----------
    N      : number of provinces (default 38)
    d_c    : climate-context hidden dim (default 16)
    d_a    : attention hidden dim (default 8)
    gamma  : LeakyReLU negative slope (default 0.2)
    seed   : RNG seed for parameter initialisation (NumPy generator,
             so initialisation is reproducible across runs).
    dtype  : default torch.float32; use float64 for numeric cross-check.
    """

    #: ablation modes recognised by `forward()`
    #: full           — full CASA, no ablation
    #: no_geo         — drop geographic prior (force α_t = 0)
    #: no_climate     — drop climate conditioning (zero out niño/dmi in c_t input)
    #: static_alpha   — α as a single trained scalar (no dependence on c_t)
    #: neighbours_only — A_learned softmax restricted to neighbour pairs only
    ABLATIONS = ("full", "no_geo", "no_climate", "static_alpha", "neighbours_only")

    def __init__(
        self,
        N: int = 38,
        d_c: int = 16,
        d_a: int = 8,
        gamma: float = 0.2,
        seed: int = 42,
        dtype: torch.dtype = torch.float32,
        b_alpha_init: float = 0.0,
        ablation: str = "full",
        freeze_alpha: Optional[float] = None,
    ) -> None:
        if ablation not in self.ABLATIONS:
            raise ValueError(f"ablation must be one of {self.ABLATIONS}, got {ablation!r}")
        super().__init__()
        self.ablation = ablation
        # freeze_alpha: if set, override the gate with a CONSTANT alpha each
        # forward (ablation study — isolates whether gate adaptivity matters).
        self.freeze_alpha = freeze_alpha
        self.N, self.d_c, self.d_a, self.gamma = N, d_c, d_a, gamma

        # NumPy RNG with a fixed seed for reproducible initialisation.
        rng = np.random.default_rng(seed)

        def _np_to_param(arr: np.ndarray) -> nn.Parameter:
            return nn.Parameter(torch.tensor(arr, dtype=dtype))

        # Stage 1 — Climate context  (80 params for default dims)
        W_c = rng.standard_normal((d_c, 4)) * np.sqrt(1.0 / 4)
        b_c = np.zeros(d_c)
        self.W_c = _np_to_param(W_c)
        self.b_c = _np_to_param(b_c)

        # Stage 2a — Attention. Input dim = 3 + d_c.
        in_a = 3 + d_c
        W_a = rng.standard_normal((d_a, in_a)) * np.sqrt(1.0 / in_a)
        b_a = np.zeros(d_a)
        v_a = rng.standard_normal(d_a) * np.sqrt(1.0 / d_a)
        self.W_a = _np_to_param(W_a)
        self.b_a = _np_to_param(b_a)
        self.v_a = _np_to_param(v_a)

        # Stage 2b — Blending gate  (17 params)
        # b_alpha_init > 0 → warm-start gate near α_t = σ(b_alpha) ≈ geo-dominant
        # so gradient has room to drop α_t when an extreme regime is detected.
        w_alpha = rng.standard_normal(d_c) * np.sqrt(1.0 / d_c)
        b_alpha = np.array([b_alpha_init], dtype=np.float64)
        self.w_alpha = _np_to_param(w_alpha)
        self.b_alpha = _np_to_param(b_alpha)

    # -- introspection -------------------------------------------------------

    def num_params(self) -> dict:
        bd = {
            "W_c":      self.W_c.numel(),
            "b_c":      self.b_c.numel(),
            "W_a":      self.W_a.numel(),
            "b_a":      self.b_a.numel(),
            "v_a":      self.v_a.numel(),
            "w_alpha":  self.w_alpha.numel(),
            "b_alpha":  self.b_alpha.numel(),
        }
        bd["stage1"]  = bd["W_c"]    + bd["b_c"]
        bd["stage2a"] = bd["W_a"]    + bd["b_a"] + bd["v_a"]
        bd["stage2b"] = bd["w_alpha"] + bd["b_alpha"]
        bd["total"]   = bd["stage1"] + bd["stage2a"] + bd["stage2b"]
        return bd

    # -- ports from NumPy reference -----------------------------------------

    @classmethod
    def from_numpy_module(cls, np_module, dtype: torch.dtype = torch.float64) -> "CASATorch":
        """Copy parameters from a compatible reference module for a numeric cross-check.

        Parameters carry over by reference-copy (no shared memory); default
        dtype is float64 for maximum precision.
        """
        m = cls(
            N=np_module.N, d_c=np_module.d_c, d_a=np_module.d_a,
            gamma=np_module.gamma, seed=0, dtype=dtype,
        )
        with torch.no_grad():
            m.W_c.copy_(torch.tensor(np_module.W_c, dtype=dtype))
            m.b_c.copy_(torch.tensor(np_module.b_c, dtype=dtype))
            m.W_a.copy_(torch.tensor(np_module.W_a, dtype=dtype))
            m.b_a.copy_(torch.tensor(np_module.b_a, dtype=dtype))
            m.v_a.copy_(torch.tensor(np_module.v_a, dtype=dtype))
            m.w_alpha.copy_(torch.tensor(np_module.w_alpha, dtype=dtype))
            m.b_alpha.copy_(torch.tensor(np_module.b_alpha, dtype=dtype))
        return m

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        Y_prev: torch.Tensor,
        nino_t: torch.Tensor,
        dmi_t:  torch.Tensor,
        A_geo:  torch.Tensor,
        neighbor_mask: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        """One CASA forward pass; supports single sample or batch.

        Shapes
        ------
        Y_prev  : (N,)          OR (B, N)
        nino_t  : scalar        OR (B,)
        dmi_t   : scalar        OR (B,)
        A_geo   : (N, N)         (shared across batch)

        Returns
        -------
        F_star  : same leading shape as Y_prev   ((N,) or (B, N))
        W_t     : (N, N)        OR (B, N, N)
        alpha_t : scalar tensor OR (B,)
        c_t     : (d_c,)        OR (B, d_c)
        state   : CASAStateTorch (only if return_state=True; always batched)
        """
        # --- normalise inputs to batched form ---------------------------
        single = (Y_prev.dim() == 1)
        if single:
            Y_prev = Y_prev.unsqueeze(0)                         # (1, N)
        B, N = Y_prev.shape
        if N != self.N:
            raise ValueError(f"Y_prev last dim {N} != module.N {self.N}")
        if A_geo.shape != (self.N, self.N):
            raise ValueError(f"A_geo expected ({self.N},{self.N}), got {tuple(A_geo.shape)}")

        if not torch.is_tensor(nino_t):
            nino_t = torch.tensor(nino_t, dtype=Y_prev.dtype, device=Y_prev.device)
        if not torch.is_tensor(dmi_t):
            dmi_t = torch.tensor(dmi_t, dtype=Y_prev.dtype, device=Y_prev.device)
        if nino_t.dim() == 0:
            nino_t = nino_t.expand(B)
        if dmi_t.dim() == 0:
            dmi_t = dmi_t.expand(B)
        if nino_t.shape != (B,) or dmi_t.shape != (B,):
            raise ValueError(
                f"nino_t/dmi_t expected scalar or shape ({B},); got "
                f"{tuple(nino_t.shape)}/{tuple(dmi_t.shape)}"
            )

        # --- Stage 1 — Climate context ---------------------------------
        Y_mean = Y_prev.mean(dim=1)                              # (B,)
        Y_std  = Y_prev.std(dim=1, unbiased=False)               # (B,) — population std (ddof=0)
        if self.ablation == "no_climate":
            # Ablation (remove climate conditioning): zero out
            # nino/dmi channels of x_c so c_t depends only on rainfall stats.
            zero = torch.zeros_like(nino_t)
            x_c = torch.stack([Y_mean, Y_std, zero, zero], dim=1)
        else:
            x_c = torch.stack([Y_mean, Y_std, nino_t, dmi_t], dim=1)  # (B, 4)
        z_c    = x_c @ self.W_c.T + self.b_c                     # (B, d_c)
        c_t    = torch.tanh(z_c)                                 # (B, d_c)

        # --- Stage 2a — Pairwise attention ------------------------------
        Yi   = Y_prev.unsqueeze(2).expand(B, N, N)               # rows
        Yj   = Y_prev.unsqueeze(1).expand(B, N, N)               # cols
        diff = (Yi - Yj).abs()
        c_block = c_t.unsqueeze(1).unsqueeze(1).expand(B, N, N, self.d_c)
        pair_feat = torch.cat(
            [Yi.unsqueeze(-1), Yj.unsqueeze(-1), diff.unsqueeze(-1), c_block],
            dim=-1,
        )                                                        # (B, N, N, 3+d_c)

        z_a = pair_feat @ self.W_a.T + self.b_a                  # (B, N, N, d_a)
        h_a = F.leaky_relu(z_a, negative_slope=self.gamma)
        e_t = h_a @ self.v_a                                     # (B, N, N)

        if self.ablation == "neighbours_only":
            # Ablation #5: A_learned softmax restricted to neighbour pairs only.
            if neighbor_mask is None:
                raise ValueError(
                    "neighbor_mask must be provided when ablation='neighbours_only'"
                )
            # Set non-neighbour scores very negative; softmax then puts ≈ 0 there.
            e_t = e_t.masked_fill(~neighbor_mask, -1e9)
        A_learned = F.softmax(e_t, dim=-1)                       # row-stochastic

        # --- Stage 2b — Blending gate -----------------------------------
        if self.ablation == "static_alpha":
            # Ablation #4: drop coupling to c_t; α is a constant scalar
            # determined only by b_α (w_α is unused in this mode).
            z_alpha = self.b_alpha.expand(B)                     # (B,)
            alpha_t = torch.sigmoid(z_alpha)
        elif self.ablation == "no_geo":
            # Ablation #2: drop geographic prior — force α_t = 0,
            # so W_t = A_learned (pure learned attention).
            z_alpha = torch.zeros(B, device=Y_prev.device, dtype=Y_prev.dtype)
            alpha_t = torch.zeros(B, device=Y_prev.device, dtype=Y_prev.dtype)
        else:                                                    # full / no_climate
            z_alpha = c_t @ self.w_alpha + self.b_alpha[0]       # (B,)
            alpha_t = torch.sigmoid(z_alpha)

        # Ablation override: pin alpha to a constant (gate adaptivity off).
        if self.freeze_alpha is not None:
            alpha_t = torch.full_like(alpha_t, float(self.freeze_alpha))

        # --- Stage 3 — Final weight + spatial lag -----------------------
        a = alpha_t.view(B, 1, 1)
        W_t = a * A_geo.unsqueeze(0) + (1.0 - a) * A_learned     # (B, N, N)
        F_star = torch.bmm(W_t, Y_prev.unsqueeze(-1)).squeeze(-1)  # (B, N)

        if single:
            F_out, W_out, a_out, c_out = F_star[0], W_t[0], alpha_t[0], c_t[0]
        else:
            F_out, W_out, a_out, c_out = F_star, W_t, alpha_t, c_t

        if return_state:
            state = CASAStateTorch(
                Y_mean=Y_mean, Y_std=Y_std, x_c=x_c, z_c=z_c, c_t=c_t,
                pair_feat=pair_feat, z_a=z_a, h_a=h_a, e_t=e_t,
                A_learned=A_learned, z_alpha=z_alpha, alpha_t=alpha_t,
                W_t=W_t, F_star=F_star,
            )
            return F_out, W_out, a_out, c_out, state
        return F_out, W_out, a_out, c_out


# ---------------------------------------------------------------------------
# CLI sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    m = CASATorch()
    bd = m.num_params()
    print("CASATorch parameter breakdown:")
    for k in ("W_c", "b_c", "W_a", "b_a", "v_a", "w_alpha", "b_alpha"):
        print(f"  {k:8s} {bd[k]:>5d}")
    print(f"  stage1  {bd['stage1']:>5d}   (spec: 80)")
    print(f"  stage2a {bd['stage2a']:>5d}   (spec: 168)")
    print(f"  stage2b {bd['stage2b']:>5d}   (spec: 17)")
    print(f"  TOTAL   {bd['total']:>5d}    (spec: 265)")

    # Smoke run + autograd check
    torch.manual_seed(0)
    Y = torch.randn(38)
    A = torch.rand(38, 38); A.fill_diagonal_(0.0); A = A / A.sum(1, keepdim=True)
    F_, W, alpha, c = m(Y, nino_t=1.2, dmi_t=0.3, A_geo=A)
    print(f"\nForward OK | alpha_t={alpha.item():.4f} "
          f"row-sum max-err={(W.sum(1) - 1).abs().max().item():.2e} "
          f"F* shape={tuple(F_.shape)}")

    loss = (F_**2).sum()
    loss.backward()
    g = [p.grad for p in m.parameters() if p.grad is not None]
    print(f"Backward OK | non-zero grads: {len(g)}/{len(list(m.parameters()))}")
