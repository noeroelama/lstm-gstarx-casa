#!/usr/bin/env python3
"""
train_casa_n40.py -- LSTM-GSTARX-CASA at N=40 (38 IDN + 2 boundary).

Default base: --ablation neighbours_only.

Experiment switches:
  --b-alpha-init        : b_alpha warm-start value (default 2.0).
  --alpha-entropy-reg L : lambda * H(alpha) loss regularizer.
  --lookback K          : multi-step LSTM window. Default 1.
  --rich-context        : extend CASA c_t with 8 extra features.
  --coupled-fstar       : coupled spatial lag -- keep F* (=W.Y) inside the LSTM
                          recurrence => LSTM-GSTARX(K;1); needs K>1.
                          (output tag: _coupledF)
  --tag SUFFIX          : output filename suffix.

Combinations supported:
  - default: single-step LSTM + standard CASA
  - --rich-context: single-step LSTM + rich CASA              (A2)
  - --lookback K: K-step LSTM + standard CASA                 (A1)
  - --lookback K --rich-context: K-step LSTM + rich CASA      (A1+A2 combo, M1)
  - --lookback K --coupled-fstar: K-step LSTM, coupled spatial lag (A8)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from casa_torch import CASATorch

# ============= CONFIG =========================================
PANEL_NPZ    = 'data/panel_data.npz'
A_GEO_NPY    = 'data/a_geo.npy'
OUTPUT_DIR   = 'outputs'

N_TOTAL      = 40
N_IDN        = 38
N_EXOG       = 2
HIDDEN_SIZE  = 64
EPOCHS       = 200
LR           = 1e-3
CLIP_NORM    = 1.0
N_SPLITS     = 5
MIN_TRAIN    = 380
SEED         = 42
B_ALPHA_INIT = 2.0
DEFAULT_ABLATION = 'neighbours_only'
N_RICH_EXTRA = 8


# ============= DATA LOADING ===================================
def load_data(a_geo_path=A_GEO_NPY):
    panel = np.load(PANEL_NPZ, allow_pickle=True)
    Y, X = panel['Y_rainfall'], panel['X_climate']
    dates, region_ids, countries = panel['dates'], panel['region_ids'], panel['region_countries']
    A_geo = np.load(a_geo_path)
    idn_indices = np.where(countries == 'IDN')[0]
    print(f"Loaded panel: T={Y.shape[0]}, N={Y.shape[1]}, IDN targets={len(idn_indices)}")
    print(f"  Period: {pd.Timestamp(dates[0]):%Y-%m} .. {pd.Timestamp(dates[-1]):%Y-%m}")
    return Y, X, dates, region_ids, idn_indices, A_geo


# ============= CASA Rich Context subclass ====================
class CASATorchRich(CASATorch):
    """CASA with extended climate context (4 base + n_extra features)."""

    def __init__(self, N=N_TOTAL, d_c=16, d_a=8, gamma=0.2, seed=42,
                 dtype=torch.float32, b_alpha_init=B_ALPHA_INIT,
                 ablation='full', n_extra=N_RICH_EXTRA):
        super().__init__(N=N, d_c=d_c, d_a=d_a, gamma=gamma, seed=seed,
                         dtype=dtype, b_alpha_init=b_alpha_init,
                         ablation=ablation)
        self.n_extra = n_extra
        rng = np.random.default_rng(seed + 1000)
        in_c = 4 + n_extra
        W_c_new = rng.standard_normal((d_c, in_c)) * np.sqrt(1.0 / in_c)
        self.W_c = nn.Parameter(torch.tensor(W_c_new, dtype=dtype))

    def forward(self, Y_prev, nino_t, dmi_t, A_geo,
                neighbor_mask=None, return_state=False, extra_context=None):
        single = (Y_prev.dim() == 1)
        if single:
            Y_prev = Y_prev.unsqueeze(0)
        B, N = Y_prev.shape
        if not torch.is_tensor(nino_t):
            nino_t = torch.tensor(nino_t, dtype=Y_prev.dtype, device=Y_prev.device)
        if not torch.is_tensor(dmi_t):
            dmi_t = torch.tensor(dmi_t, dtype=Y_prev.dtype, device=Y_prev.device)
        if nino_t.dim() == 0:
            nino_t = nino_t.expand(B)
        if dmi_t.dim() == 0:
            dmi_t = dmi_t.expand(B)
        if extra_context is None or extra_context.shape[1] != self.n_extra:
            raise ValueError(f"extra_context shape (B, {self.n_extra}) required")

        Y_mean = Y_prev.mean(dim=1)
        Y_std  = Y_prev.std(dim=1, unbiased=False)
        if self.ablation == "no_climate":
            zero = torch.zeros_like(nino_t)
            x_c_base = torch.stack([Y_mean, Y_std, zero, zero], dim=1)
        else:
            x_c_base = torch.stack([Y_mean, Y_std, nino_t, dmi_t], dim=1)
        x_c = torch.cat([x_c_base, extra_context], dim=1)
        z_c = x_c @ self.W_c.T + self.b_c
        c_t = torch.tanh(z_c)

        Yi = Y_prev.unsqueeze(2).expand(B, N, N)
        Yj = Y_prev.unsqueeze(1).expand(B, N, N)
        diff = (Yi - Yj).abs()
        c_block = c_t.unsqueeze(1).unsqueeze(1).expand(B, N, N, self.d_c)
        pair_feat = torch.cat(
            [Yi.unsqueeze(-1), Yj.unsqueeze(-1), diff.unsqueeze(-1), c_block],
            dim=-1,
        )
        z_a = pair_feat @ self.W_a.T + self.b_a
        h_a = F.leaky_relu(z_a, negative_slope=self.gamma)
        e_t = h_a @ self.v_a

        if self.ablation == "neighbours_only":
            if neighbor_mask is None:
                raise ValueError("neighbor_mask required")
            e_t = e_t.masked_fill(~neighbor_mask, -1e9)
        A_learned = F.softmax(e_t, dim=-1)

        if self.ablation == "static_alpha":
            z_alpha = self.b_alpha.expand(B)
            alpha_t = torch.sigmoid(z_alpha)
        elif self.ablation == "no_geo":
            z_alpha = torch.zeros(B, device=Y_prev.device, dtype=Y_prev.dtype)
            alpha_t = torch.zeros(B, device=Y_prev.device, dtype=Y_prev.dtype)
        else:
            z_alpha = c_t @ self.w_alpha + self.b_alpha[0]
            alpha_t = torch.sigmoid(z_alpha)

        a = alpha_t.view(B, 1, 1)
        W_t = a * A_geo.unsqueeze(0) + (1.0 - a) * A_learned
        F_star = torch.bmm(W_t, Y_prev.unsqueeze(-1)).squeeze(-1)

        if single:
            return F_star[0], W_t[0], alpha_t[0], c_t[0]
        return F_star, W_t, alpha_t, c_t


# ============= MODELS ==========================================
class LSTMGSTARX_CASA(nn.Module):
    """Default: single-step LSTM + standard CASA."""

    def __init__(self, N=N_TOTAL, N_out=N_IDN, hidden=HIDDEN_SIZE,
                 n_exog=N_EXOG, ablation='full', b_alpha_init=B_ALPHA_INIT):
        super().__init__()
        self.casa = CASATorch(N=N, b_alpha_init=b_alpha_init, ablation=ablation)
        d_in = 2 * N + n_exog
        self.lstm   = nn.LSTM(input_size=d_in, hidden_size=hidden,
                              batch_first=True)
        self.linear = nn.Linear(hidden, N_out)

    def forward(self, Y_prev_norm, X_norm, X_raw, A_geo,
                neighbor_mask=None, return_alpha=False):
        nino_raw, dmi_raw = X_raw[:, 0], X_raw[:, 1]
        F_star, W_t, alpha_t, c_t = self.casa(
            Y_prev_norm, nino_t=nino_raw, dmi_t=dmi_raw,
            A_geo=A_geo, neighbor_mask=neighbor_mask,
        )
        x = torch.cat([Y_prev_norm, F_star, X_norm], dim=-1).unsqueeze(1)
        h, _ = self.lstm(x)
        h = h.squeeze(1)
        y_hat = self.linear(h)
        if return_alpha:
            return y_hat, alpha_t, W_t
        return y_hat


class LSTMGSTARX_CASA_Rich(nn.Module):
    """A2: single-step LSTM + rich-context CASA."""

    def __init__(self, N=N_TOTAL, N_out=N_IDN, hidden=HIDDEN_SIZE,
                 n_exog=N_EXOG, ablation='full', b_alpha_init=B_ALPHA_INIT,
                 n_extra=N_RICH_EXTRA):
        super().__init__()
        self.casa = CASATorchRich(N=N, b_alpha_init=b_alpha_init,
                                  ablation=ablation, n_extra=n_extra)
        d_in = 2 * N + n_exog
        self.lstm   = nn.LSTM(input_size=d_in, hidden_size=hidden,
                              batch_first=True)
        self.linear = nn.Linear(hidden, N_out)

    def forward(self, Y_prev_norm, X_norm, X_raw, A_geo, extra_context,
                neighbor_mask=None, return_alpha=False):
        nino_raw, dmi_raw = X_raw[:, 0], X_raw[:, 1]
        F_star, W_t, alpha_t, c_t = self.casa(
            Y_prev_norm, nino_t=nino_raw, dmi_t=dmi_raw, A_geo=A_geo,
            neighbor_mask=neighbor_mask, extra_context=extra_context,
        )
        x = torch.cat([Y_prev_norm, F_star, X_norm], dim=-1).unsqueeze(1)
        h, _ = self.lstm(x)
        h = h.squeeze(1)
        y_hat = self.linear(h)
        if return_alpha:
            return y_hat, alpha_t, W_t
        return y_hat


class LSTMGSTARX_CASA_Window(nn.Module):
    """A1: K-step LSTM + standard CASA."""

    def __init__(self, N=N_TOTAL, N_out=N_IDN, hidden=HIDDEN_SIZE,
                 lookback=12, n_exog=N_EXOG, ablation='full',
                 b_alpha_init=B_ALPHA_INIT):
        super().__init__()
        self.lookback = lookback
        self.casa = CASATorch(N=N, b_alpha_init=b_alpha_init, ablation=ablation)
        d_seq = N + n_exog
        self.lstm   = nn.LSTM(input_size=d_seq, hidden_size=hidden,
                              batch_first=True)
        self.linear = nn.Linear(hidden + N, N_out)

    def forward(self, Y_window, X_window, Y_last, X_raw, A_geo,
                neighbor_mask=None, return_alpha=False):
        nino_raw, dmi_raw = X_raw[:, 0], X_raw[:, 1]
        F_star, W_t, alpha_t, c_t = self.casa(
            Y_last, nino_t=nino_raw, dmi_t=dmi_raw,
            A_geo=A_geo, neighbor_mask=neighbor_mask,
        )
        x_seq = torch.cat([Y_window, X_window], dim=-1)
        h_seq, _ = self.lstm(x_seq)
        h_last = h_seq[:, -1, :]
        combined = torch.cat([h_last, F_star], dim=-1)
        y_hat = self.linear(combined)
        if return_alpha:
            return y_hat, alpha_t, W_t
        return y_hat


class LSTMGSTARX_CASA_WindowRich(nn.Module):
    """M1: K-step LSTM + rich-context CASA (A1 + A2 combined)."""

    def __init__(self, N=N_TOTAL, N_out=N_IDN, hidden=HIDDEN_SIZE,
                 lookback=12, n_exog=N_EXOG, ablation='full',
                 b_alpha_init=B_ALPHA_INIT, n_extra=N_RICH_EXTRA):
        super().__init__()
        self.lookback = lookback
        self.casa = CASATorchRich(N=N, b_alpha_init=b_alpha_init,
                                  ablation=ablation, n_extra=n_extra)
        d_seq = N + n_exog
        self.lstm   = nn.LSTM(input_size=d_seq, hidden_size=hidden,
                              batch_first=True)
        self.linear = nn.Linear(hidden + N, N_out)

    def forward(self, Y_window, X_window, Y_last, X_raw, A_geo, extra_context,
                neighbor_mask=None, return_alpha=False):
        nino_raw, dmi_raw = X_raw[:, 0], X_raw[:, 1]
        F_star, W_t, alpha_t, c_t = self.casa(
            Y_last, nino_t=nino_raw, dmi_t=dmi_raw, A_geo=A_geo,
            neighbor_mask=neighbor_mask, extra_context=extra_context,
        )
        x_seq = torch.cat([Y_window, X_window], dim=-1)
        h_seq, _ = self.lstm(x_seq)
        h_last = h_seq[:, -1, :]
        combined = torch.cat([h_last, F_star], dim=-1)
        y_hat = self.linear(combined)
        if return_alpha:
            return y_hat, alpha_t, W_t
        return y_hat


class LSTMGSTARX_CASA_WindowF(nn.Module):
    """A8: coupled-spatial-lag multi-step LSTM -- F_star kept INSIDE the recurrence.

    Terminology: F_star = W.Y is the GSTAR *spatial lag operator*. Keeping it
    inside the recurrence at every step makes this a genuine LSTM-GSTARX(K;1)
    (spatial order 1) -- the "coupled spatial lag" design. The class keeps the
    name 'WindowF' (its windowed forward pass); the CLI/output tag is '_coupledF'
    and the paper term is "coupled".

    Confound control for the single-step vs multi-step comparison. The decoupled
    multi-step model (LSTMGSTARX_CASA_Window) skip-connects F_star AFTER the
    LSTM, so going single-step -> multi-step bundles TWO changes at once:
    (a) the lookback K, and (b) relocating F_star out of the LSTM input. This
    variant keeps F_star in the per-timestep LSTM input -- [Y_{t-1}, F_star,
    X_t], 82-dim -- exactly as the single-step LSTMGSTARX_CASA does, so the
    spatial lag flows through the recurrence (consistent with the GSTAR-family
    design). The ONLY architectural difference from single-step LSTMGSTARX_CASA
    is the sequence length K; the parameter count is identical, which isolates
    lookback as the single explanatory variable.

    F_star at timestep tau is conditioned on the climate of month tau -- the
    same month already carried by that timestep's X channel -- so at K=1 this
    model reduces exactly to LSTMGSTARX_CASA.

    Efficiency: CASA is evaluated ONCE, batched over all months, then the
    per-month F_star vector is unfolded into sliding windows (no recompute).
    """

    def __init__(self, N=N_TOTAL, N_out=N_IDN, hidden=HIDDEN_SIZE,
                 lookback=12, n_exog=N_EXOG, ablation='full',
                 b_alpha_init=B_ALPHA_INIT, freeze_alpha=None):
        super().__init__()
        self.lookback = lookback
        self.casa = CASATorch(N=N, b_alpha_init=b_alpha_init, ablation=ablation,
                              freeze_alpha=freeze_alpha)
        d_in = 2 * N + n_exog          # 82 -- identical to single-step LSTMGSTARX_CASA
        self.lstm   = nn.LSTM(input_size=d_in, hidden_size=hidden,
                              batch_first=True)
        self.linear = nn.Linear(hidden, N_out)

    def forward(self, Y_window, X_window, Y_prev_full, X_raw_full, win_bounds,
                A_geo, neighbor_mask=None, return_alpha=False):
        """Coupled multi-step forward pass.

        Y_window    : (B, K, N)   normalized rainfall, months t-K..t-1
        X_window    : (B, K, 2)   normalized climate,  months t-K+1..t
        Y_prev_full : (T-1, N)    Y_norm[:-1] -- batched CASA rainfall input
        X_raw_full  : (T-1, 2)    X_raw[1:]   -- batched CASA raw climate
        win_bounds  : (lo, hi)    window-sample slice this batch maps to
        """
        K = self.lookback
        lo, hi = win_bounds
        # CASA once, batched over every month. F_all[k] is the spatial lag for
        # target month k+1 (built from Y_norm[k] and the climate of month k+1).
        F_all, W_all, alpha_all, _ = self.casa(
            Y_prev_full, nino_t=X_raw_full[:, 0], dmi_t=X_raw_full[:, 1],
            A_geo=A_geo, neighbor_mask=neighbor_mask,
        )
        # Unfold per-month F_star into sliding windows: F_win[i, j] = F_all[i+j],
        # aligned 1:1 with the Y_window / X_window timesteps.
        F_win = F_all.unfold(0, K, 1).permute(0, 2, 1)      # (T-K, K, N)
        F_window = F_win[lo:hi]                             # (B, K, N)
        # Per-timestep LSTM input [Y, F_star, X] = 82-dim; F_star is recurrent.
        x_seq = torch.cat([Y_window, F_window, X_window], dim=-1)
        h_seq, _ = self.lstm(x_seq)
        h_last = h_seq[:, -1, :]
        y_hat = self.linear(h_last)
        if return_alpha:
            # report CASA gate / weight at the target month t (last timestep)
            alpha_t = alpha_all[lo + K - 1:hi + K - 1]
            W_t     = W_all[lo + K - 1:hi + K - 1]
            return y_hat, alpha_t, W_t
        return y_hat


# ============= NORMALIZATION + HELPERS =======================
def normalize(Y, X, tr_end):
    Y_min = Y[:tr_end].min(axis=0); Y_max = Y[:tr_end].max(axis=0)
    Y_range = Y_max - Y_min + 1e-9
    Y_norm = (Y - Y_min) / Y_range
    X_min = X[:tr_end].min(axis=0); X_max = X[:tr_end].max(axis=0)
    X_range = X_max - X_min + 1e-9
    X_norm = (X - X_min) / X_range
    return Y_norm, X_norm, (Y_min, Y_max, Y_range)


def inverse_y_idn(y_hat_norm, idn_indices, Y_stats):
    Y_min, _, Y_range = Y_stats
    return y_hat_norm * Y_range[idn_indices] + Y_min[idn_indices]


def build_extras_at_time(X_raw, dates):
    """Build (T, 8) extras features for each time t. Edge t<3 uses padded lags."""
    T = X_raw.shape[0]
    nino, dmi = X_raw[:, 0], X_raw[:, 1]
    months = pd.to_datetime(dates).month.values
    feats = np.zeros((T, 8))
    for t in range(T):
        feats[t, 0] = nino[max(t - 1, 0)]
        feats[t, 1] = nino[max(t - 2, 0)]
        feats[t, 2] = nino[max(t - 3, 0)]
        feats[t, 3] = dmi[max(t - 1, 0)]
        feats[t, 4] = dmi[max(t - 2, 0)]
        feats[t, 5] = nino[t] - nino[max(t - 1, 0)]
        m = months[t]
        feats[t, 6] = np.sin(2 * np.pi * m / 12.0)
        feats[t, 7] = np.cos(2 * np.pi * m / 12.0)
    return feats


def normalize_extras(extras, tr_end_t):
    """MinMax normalize first 6 columns of extras using training-time stats."""
    out = extras.copy()
    train = out[:tr_end_t, :6]
    fmin = train.min(axis=0); fmax = train.max(axis=0)
    frange = fmax - fmin + 1e-9
    out[:, :6] = (out[:, :6] - fmin) / frange
    return out


def build_window(Y_norm, X_norm, X_raw, idn_indices, lookback):
    """Sliding-window arrays for multi-step LSTM training."""
    T, N = Y_norm.shape
    K = lookback
    n_samples = T - K
    Y_window = np.stack([Y_norm[i:i + K] for i in range(n_samples)], axis=0)
    X_window = np.stack([X_norm[i + 1:i + K + 1] for i in range(n_samples)], axis=0)
    Y_last     = Y_norm[K - 1:K - 1 + n_samples]
    X_raw_last = X_raw[K:K + n_samples]
    Y_tgt      = Y_norm[K:, idn_indices]
    return Y_window, X_window, Y_last, X_raw_last, Y_tgt


# ============= TRAINING =======================================
def train_fold(model, batch_inputs, Y_tgt, A_geo_t, epochs, lr, clip_norm,
               neighbor_mask=None, alpha_entropy_reg=0.0, mode='default'):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    eps = 1e-7
    use_alpha = alpha_entropy_reg > 0

    for _ in range(epochs):
        opt.zero_grad()
        if mode == 'default':
            Y_prev, X_norm, X_raw = batch_inputs
            kwargs = dict(neighbor_mask=neighbor_mask)
            args = (Y_prev, X_norm, X_raw, A_geo_t)
        elif mode == 'window':
            Y_window, X_window, Y_last, X_raw_last = batch_inputs
            kwargs = dict(neighbor_mask=neighbor_mask)
            args = (Y_window, X_window, Y_last, X_raw_last, A_geo_t)
        elif mode == 'rich':
            Y_prev, X_norm, X_raw, extras = batch_inputs
            kwargs = dict(neighbor_mask=neighbor_mask)
            args = (Y_prev, X_norm, X_raw, A_geo_t, extras)
        elif mode == 'window_rich':
            Y_window, X_window, Y_last, X_raw_last, extras = batch_inputs
            kwargs = dict(neighbor_mask=neighbor_mask)
            args = (Y_window, X_window, Y_last, X_raw_last, A_geo_t, extras)
        elif mode == 'window_f':
            Y_window, X_window, Y_prev_full, X_raw_full, win_bounds = batch_inputs
            kwargs = dict(neighbor_mask=neighbor_mask)
            args = (Y_window, X_window, Y_prev_full, X_raw_full, win_bounds, A_geo_t)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if use_alpha:
            y_hat, alpha_t, _ = model(*args, return_alpha=True, **kwargs)
            a = alpha_t.clamp(eps, 1.0 - eps)
            H = -(a * a.log() + (1.0 - a) * (1.0 - a).log())
            loss = F.mse_loss(y_hat, Y_tgt) * 0.5 - alpha_entropy_reg * H.mean()
        else:
            y_hat = model(*args, **kwargs)
            loss = F.mse_loss(y_hat, Y_tgt) * 0.5
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        opt.step()
        losses.append(loss.item())
    return losses


def walk_forward_cv(Y, X, dates, A_geo, idn_indices, n_splits, epochs, lr,
                    seed, min_train, ablation, b_alpha_init, alpha_entropy_reg,
                    lookback, rich_context, coupled_fstar, freeze_alpha=None):
    T, N = Y.shape
    fold_size = (T - min_train) // n_splits

    if lookback > 1 and coupled_fstar:
        mode = 'window_f'
    elif lookback > 1 and rich_context:
        mode = 'window_rich'
    elif lookback > 1:
        mode = 'window'
    elif rich_context:
        mode = 'rich'
    else:
        mode = 'default'

    print(f"\nWalk-forward CV: T={T}, min_train={min_train}, "
          f"fold_size={fold_size}, mode={mode}")
    print(f"  ablation={ablation}, b_alpha_init={b_alpha_init}, "
          f"alpha_entropy_reg={alpha_entropy_reg}, lookback={lookback}, "
          f"rich_context={rich_context}, coupled_fstar={coupled_fstar}")

    A_geo_t = torch.tensor(A_geo, dtype=torch.float32)
    neighbor_mask_t = (A_geo_t > 0) if ablation == 'neighbours_only' else None

    # Pre-compute extras (only used by rich/window_rich modes)
    extras_raw = build_extras_at_time(X, dates) if rich_context else None

    results, predictions = [], {}

    for fold in range(n_splits):
        train_end = min_train + fold * fold_size
        test_start, test_end = train_end, train_end + fold_size
        if test_end > T:
            break

        Y_norm, X_norm, Y_stats = normalize(Y, X, train_end)
        if rich_context:
            extras_norm = normalize_extras(extras_raw, train_end)

        torch.manual_seed(seed + fold)
        np.random.seed(seed + fold)

        if mode == 'window_rich':
            K = lookback
            Y_window, X_window, Y_last, X_raw_last, Y_tgt_arr = build_window(
                Y_norm, X_norm, X, idn_indices, K,
            )
            # Extras at target time t = i + K → extras_norm[i + K]
            extras_window = extras_norm[K:]
            tr_lo, tr_hi = 0, train_end - K
            te_lo, te_hi = train_end - K, test_end - K

            Y_window_t = torch.tensor(Y_window,  dtype=torch.float32)
            X_window_t = torch.tensor(X_window,  dtype=torch.float32)
            Y_last_t   = torch.tensor(Y_last,    dtype=torch.float32)
            X_raw_t    = torch.tensor(X_raw_last, dtype=torch.float32)
            Y_tgt_t    = torch.tensor(Y_tgt_arr, dtype=torch.float32)
            extras_t   = torch.tensor(extras_window, dtype=torch.float32)

            model = LSTMGSTARX_CASA_WindowRich(
                lookback=K, ablation=ablation, b_alpha_init=b_alpha_init,
            )
            batch_inputs_tr = (
                Y_window_t[tr_lo:tr_hi], X_window_t[tr_lo:tr_hi],
                Y_last_t[tr_lo:tr_hi],   X_raw_t[tr_lo:tr_hi],
                extras_t[tr_lo:tr_hi],
            )
            eval_args = (
                Y_window_t[te_lo:te_hi], X_window_t[te_lo:te_hi],
                Y_last_t[te_lo:te_hi],   X_raw_t[te_lo:te_hi],
                A_geo_t, extras_t[te_lo:te_hi],
            )

        elif mode == 'window':
            K = lookback
            Y_window, X_window, Y_last, X_raw_last, Y_tgt_arr = build_window(
                Y_norm, X_norm, X, idn_indices, K,
            )
            tr_lo, tr_hi = 0, train_end - K
            te_lo, te_hi = train_end - K, test_end - K
            Y_window_t = torch.tensor(Y_window,  dtype=torch.float32)
            X_window_t = torch.tensor(X_window,  dtype=torch.float32)
            Y_last_t   = torch.tensor(Y_last,    dtype=torch.float32)
            X_raw_t    = torch.tensor(X_raw_last, dtype=torch.float32)
            Y_tgt_t    = torch.tensor(Y_tgt_arr, dtype=torch.float32)
            model = LSTMGSTARX_CASA_Window(
                lookback=K, ablation=ablation, b_alpha_init=b_alpha_init,
            )
            batch_inputs_tr = (
                Y_window_t[tr_lo:tr_hi], X_window_t[tr_lo:tr_hi],
                Y_last_t[tr_lo:tr_hi],   X_raw_t[tr_lo:tr_hi],
            )
            eval_args = (
                Y_window_t[te_lo:te_hi], X_window_t[te_lo:te_hi],
                Y_last_t[te_lo:te_hi],   X_raw_t[te_lo:te_hi], A_geo_t,
            )

        elif mode == 'window_f':
            K = lookback
            Y_window, X_window, _, _, Y_tgt_arr = build_window(
                Y_norm, X_norm, X, idn_indices, K,
            )
            tr_lo, tr_hi = 0, train_end - K
            te_lo, te_hi = train_end - K, test_end - K
            Y_window_t  = torch.tensor(Y_window,  dtype=torch.float32)
            X_window_t  = torch.tensor(X_window,  dtype=torch.float32)
            Y_tgt_t     = torch.tensor(Y_tgt_arr, dtype=torch.float32)
            Y_prev_full = torch.tensor(Y_norm[:-1], dtype=torch.float32)
            X_raw_full  = torch.tensor(X[1:],       dtype=torch.float32)
            model = LSTMGSTARX_CASA_WindowF(
                lookback=K, ablation=ablation, b_alpha_init=b_alpha_init,
                freeze_alpha=freeze_alpha,
            )
            batch_inputs_tr = (
                Y_window_t[tr_lo:tr_hi], X_window_t[tr_lo:tr_hi],
                Y_prev_full, X_raw_full, (tr_lo, tr_hi),
            )
            eval_args = (
                Y_window_t[te_lo:te_hi], X_window_t[te_lo:te_hi],
                Y_prev_full, X_raw_full, (te_lo, te_hi), A_geo_t,
            )

        elif mode == 'rich':
            Y_prev_all = torch.tensor(Y_norm[:-1],             dtype=torch.float32)
            Y_tgt_all  = torch.tensor(Y_norm[1:, idn_indices], dtype=torch.float32)
            X_norm_all = torch.tensor(X_norm[1:],              dtype=torch.float32)
            X_raw_all  = torch.tensor(X[1:],                   dtype=torch.float32)
            # extras at target t = i+1 → extras_norm[i+1]
            extras_t   = torch.tensor(extras_norm[1:],         dtype=torch.float32)
            tr_lo, tr_hi = 0, train_end - 1
            te_lo, te_hi = train_end - 1, test_end - 1
            Y_tgt_t = Y_tgt_all
            model = LSTMGSTARX_CASA_Rich(
                ablation=ablation, b_alpha_init=b_alpha_init,
            )
            batch_inputs_tr = (
                Y_prev_all[tr_lo:tr_hi], X_norm_all[tr_lo:tr_hi],
                X_raw_all[tr_lo:tr_hi],  extras_t[tr_lo:tr_hi],
            )
            eval_args = (
                Y_prev_all[te_lo:te_hi], X_norm_all[te_lo:te_hi],
                X_raw_all[te_lo:te_hi],  A_geo_t, extras_t[te_lo:te_hi],
            )

        else:   # default
            Y_prev_all = torch.tensor(Y_norm[:-1],             dtype=torch.float32)
            Y_tgt_all  = torch.tensor(Y_norm[1:, idn_indices], dtype=torch.float32)
            X_norm_all = torch.tensor(X_norm[1:],              dtype=torch.float32)
            X_raw_all  = torch.tensor(X[1:],                   dtype=torch.float32)
            tr_lo, tr_hi = 0, train_end - 1
            te_lo, te_hi = train_end - 1, test_end - 1
            Y_tgt_t = Y_tgt_all
            model = LSTMGSTARX_CASA(
                ablation=ablation, b_alpha_init=b_alpha_init,
            )
            batch_inputs_tr = (
                Y_prev_all[tr_lo:tr_hi], X_norm_all[tr_lo:tr_hi],
                X_raw_all[tr_lo:tr_hi],
            )
            eval_args = (
                Y_prev_all[te_lo:te_hi], X_norm_all[te_lo:te_hi],
                X_raw_all[te_lo:te_hi], A_geo_t,
            )

        # Train
        t0 = time.time()
        losses = train_fold(
            model, batch_inputs_tr, Y_tgt_t[tr_lo:tr_hi], A_geo_t,
            epochs=epochs, lr=lr, clip_norm=CLIP_NORM,
            neighbor_mask=neighbor_mask_t,
            alpha_entropy_reg=alpha_entropy_reg, mode=mode,
        )
        train_time = time.time() - t0

        # Evaluate
        model.eval()
        with torch.no_grad():
            y_hat_norm, alpha_te, W_te = model(
                *eval_args, neighbor_mask=neighbor_mask_t, return_alpha=True,
            )
        y_hat_mm  = inverse_y_idn(y_hat_norm.numpy(), idn_indices, Y_stats)
        y_true_mm = Y[test_start:test_end][:, idn_indices]
        alpha_np  = alpha_te.numpy()
        test_dates = dates[test_start:test_end]

        rmse = float(np.sqrt(((y_hat_mm - y_true_mm) ** 2).mean()))
        mae  = float(np.abs(y_hat_mm - y_true_mm).mean())
        alpha_mean = float(alpha_te.mean())
        alpha_std  = float(alpha_te.std())

        print(f"  fold {fold+1:>2d}: loss={losses[-1]:.4f} "
              f"alpha=[{alpha_mean:.3f}+-{alpha_std:.3f}] "
              f"RMSE={rmse:.2f} MAE={mae:.2f} time={train_time:.1f}s")

        results.append({
            'fold': fold + 1, 'ablation': ablation, 'mode': mode,
            'b_alpha_init': b_alpha_init,
            'alpha_entropy_reg': alpha_entropy_reg,
            'lookback': lookback, 'rich_context': rich_context,
            'final_loss': losses[-1], 'alpha_mean': alpha_mean,
            'alpha_std': alpha_std, 'rmse': rmse, 'mae': mae,
            'train_time': train_time,
        })

        k = fold + 1
        predictions[f'y_hat_fold_{k}']  = y_hat_mm.astype(np.float64)
        predictions[f'y_true_fold_{k}'] = y_true_mm.astype(np.float64)
        predictions[f'alpha_fold_{k}']  = alpha_np.astype(np.float64)
        predictions[f'dates_fold_{k}']  = test_dates

    return results, predictions


# ============= MAIN ===========================================
def main(epochs, n_splits, seed, min_train, ablation, b_alpha_init,
         alpha_entropy_reg, lookback, rich_context, coupled_fstar, tag,
         a_geo_path=A_GEO_NPY, freeze_alpha=None):
    if coupled_fstar and lookback <= 1:
        raise SystemExit("--coupled-fstar requires --lookback K with K > 1")
    if coupled_fstar and rich_context:
        raise SystemExit("--coupled-fstar is not compatible with --rich-context")
    suffix = f"_{tag}" if tag else ""
    run_id = f"{ablation}{suffix}"
    print(f"=== CASA N=40 walk-forward CV (run_id={run_id}) ===")
    if a_geo_path != A_GEO_NPY:
        print(f"  A_geo override: {a_geo_path}")

    Y, X, dates, region_ids, idn_indices, A_geo = load_data(a_geo_path)

    results, predictions = walk_forward_cv(
        Y, X, dates, A_geo, idn_indices,
        n_splits=n_splits, epochs=epochs, lr=LR, seed=seed,
        min_train=min_train, ablation=ablation,
        b_alpha_init=b_alpha_init, alpha_entropy_reg=alpha_entropy_reg,
        lookback=lookback, rich_context=rich_context,
        coupled_fstar=coupled_fstar, freeze_alpha=freeze_alpha,
    )

    df = pd.DataFrame(results)
    print(f"\n=== SUMMARY (run_id={run_id}) ===")
    print(f"  Mean RMSE  : {df['rmse'].mean():.2f} +- {df['rmse'].std():.2f} mm")
    print(f"  Mean MAE   : {df['mae'].mean():.2f} +- {df['mae'].std():.2f} mm")
    print(f"  Mean alpha : {df['alpha_mean'].mean():.3f}")
    print(f"  Total time : {df['train_time'].sum():.1f} s")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    csv_out = f'{OUTPUT_DIR}/cv_results_casa_{run_id}.csv'
    npz_out = f'{OUTPUT_DIR}/predictions_casa_{run_id}.npz'
    df.to_csv(csv_out, index=False)
    np.savez_compressed(npz_out, **predictions)
    print(f"\nSaved {csv_out}\nSaved {npz_out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',    type=int, default=EPOCHS)
    p.add_argument('--folds',     type=int, default=N_SPLITS)
    p.add_argument('--seed',      type=int, default=SEED)
    p.add_argument('--min-train', type=int, default=MIN_TRAIN)
    p.add_argument('--ablation',  type=str, default=DEFAULT_ABLATION,
                   choices=['full', 'no_geo', 'no_climate',
                            'static_alpha', 'neighbours_only'])
    p.add_argument('--b-alpha-init', type=float, default=B_ALPHA_INIT)
    p.add_argument('--alpha-entropy-reg', type=float, default=0.0)
    p.add_argument('--lookback', type=int, default=1)
    p.add_argument('--rich-context', action='store_true')
    p.add_argument('--coupled-fstar', action='store_true',
                   dest='coupled_fstar',
                   help='coupled spatial lag: keep F* (=W.Y) inside the LSTM '
                        'recurrence => LSTM-GSTARX(K;1) (needs --lookback>1). '
                        'Output tag: _coupledF.')
    p.add_argument('--tag', type=str, default='')
    p.add_argument('--a-geo-path', type=str, default=A_GEO_NPY,
                   help='override A_geo .npy (e.g. data/a_geo_k3.npy for KNN sensitivity)')
    p.add_argument('--freeze-alpha', type=float, default=None,
                   help='pin the CASA blend gate to a constant alpha (ablation: '
                        '1.0 = pure static A_geo; e.g. 0.897 = constant-blend). '
                        'Isolates whether gate adaptivity helps.')
    args = p.parse_args()
    main(epochs=args.epochs, n_splits=args.folds, seed=args.seed,
         min_train=args.min_train, ablation=args.ablation,
         b_alpha_init=args.b_alpha_init,
         alpha_entropy_reg=args.alpha_entropy_reg,
         lookback=args.lookback, rich_context=args.rich_context,
         coupled_fstar=args.coupled_fstar,
         tag=args.tag, a_geo_path=args.a_geo_path, freeze_alpha=args.freeze_alpha)
