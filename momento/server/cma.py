"""Continuum-Momentum Aggregation (CMA).

The server keeps K momentum memories of the (gated) aggregate at geometrically spaced decay rates

    m^(k)_{t+1} = beta_k m^(k)_t + (1 - beta_k) tilde_delta_t,      beta_k = 1 - 2^{-k},  k = 1..K

(normalised EMA form, so each m^(k) is an estimate of the expected honest update over a window of
~2^k rounds). CMA is used as the *aggregator*; the server optimizer is then plain FedAvg with eta_s,
because the momentum lives inside CMA. For adaptive servers, `adaptive=True` additionally keeps a
second-moment estimate at two timescales and normalises by sqrt(max(v_fast, v_slow)) + tau, which
prevents the cold-coordinate amplification that the FedAdam echo exploits.

Two aggregate-only tests, both normalised by sigma_hat^2_t:
    innovation  D_t     = ||delta_t - m^(K)_t||^2 / sigma_hat^2_t        (incoming vs slow memory; on fire the
                          aggregate is centered-clipped around m^(K) to the calibrated radius sqrt(lam sigma^2))
    persistence D^mem_t = ||m^(1)_t - m^(K)_t||^2 / sigma_hat^2_t        (fast vs slow memory after the update; on
                          fire the fast memories are contracted toward the slow one along the surprise direction)
    per-level   D^(k)_t = ||m^(k)_t - m^(k+1)_t||^2 / sigma_hat^2_t     (localises the timescale that absorbed a burst)

where sigma_hat^2_t = OTA-noise term (known from sigma_z^2, gamma, |S|) + running benign residual
variance (EMA of ||delta_t - m^(K)_t||^2 over NON-firing rounds). Thresholds are *quantiles of the
benign warm-up distribution* (never tuned on attacks); the chi-square fit is logged for the theory
check (Prop. C) but is not what the decision uses by default (`threshold_mode='chi2'` switches).

Purge (Delta-Momentum, NL Eq. 49 with input := disagreement direction u):
    m^(k) <- m^(k) (I - kappa u u^T / ||u||^2)   for k <= k*   (k* = highest firing level)
Gated consolidation: levels k > k* receive only the component of delta orthogonal to u this round,
and are frozen if D_t exceeds the second threshold lambda'.

Update direction returned to the server: sum_k omega_k m^(k)_{t+1}, omega_k ∝ 2^{-k} by default.

Lineage: multi-timescale EMA change detection (GMA / CUSUM control charts) and NL's CMS / M3.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..channel.ota import ChannelState
from .defenses import Aggregator


class CMA(Aggregator):
    name = "cma"

    def __init__(self, d_tx, device, K: int = 4, kappa: float = 1.0, alpha_fa: float = 0.01, alpha_freeze: float = 0.001,
                 weights: str = "geometric", adaptive: bool = False, tau: float = 1e-3, beta2_fast: float = 0.9,
                 beta2_slow: float = 0.99, warmup_rounds: int = 20, resid_decay: float = 0.9, purge: bool = True,
                 orth_consolidate: bool = True, threshold_mode: str = "chi2", per_level: bool = False,
                 inner_aggregator: str = "mean", gain: float = 10.0, calibration: str = "sliding",
                 null_window: int = 60, min_null: int = 20, winsor: float = 4.0, **kw):
        super().__init__(d_tx, device)
        self.K = K
        self.betas = torch.tensor([1 - 2.0 ** (-k) for k in range(1, K + 1)], device=device)
        self.m = torch.zeros(K, d_tx, device=device)
        self.kappa = kappa
        self.alpha_fa, self.alpha_freeze = alpha_fa, alpha_freeze
        if weights == "geometric":
            w = torch.tensor([2.0 ** (-k) for k in range(1, K + 1)])
        elif weights == "uniform":
            w = torch.ones(K)
        elif weights == "slow":
            w = torch.zeros(K); w[-1] = 1.0
        else:
            raise ValueError(weights)
        self.omega = (w / w.sum()).to(device)
        self.adaptive, self.tau = adaptive, tau
        self.b2f, self.b2s = beta2_fast, beta2_slow
        self.v_fast = torch.zeros(d_tx, device=device)
        self.v_slow = torch.zeros(d_tx, device=device)
        self.warmup = warmup_rounds
        self.resid_decay = resid_decay
        self.purge_on, self.orth_on = purge, orth_consolidate
        self.threshold_mode = threshold_mode
        self.per_level = per_level
        self.inner = inner_aggregator
        # Reddi's FedAvgM (m <- beta m + delta) equals a normalised EMA times 1/(1-beta); gain=10 matches beta=0.9
        self.gain = gain
        self.t = 0
        self.calibration, self.null_window, self.min_null, self.winsor = calibration, null_window, min_null, winsor
        self.sigma_grad2: Optional[float] = None
        self._null_D: List[float] = []
        self._null_Dk: List[List[float]] = [[] for _ in range(K - 1)]
        self.lam = self.lam_freeze = self.lam_mem = None
        self._null_Dmem: List[float] = []
        self.lam_k: Optional[List[float]] = None
        self.chi2_fit: Dict[str, float] = {}

    # ------------------------------------------------------------------ helpers
    def _mhat_at(self, t: int) -> torch.Tensor:
        """Bias-corrected memories (Adam-style) after t updates, so that zero initialisation does not bias
        the statistics. For t <= 0 returns zeros (no information yet)."""
        if t <= 0:
            return torch.zeros_like(self.m)
        corr = (1 - self.betas ** t).clamp_min(1e-6)
        return self.m / corr[:, None]

    def _inner(self, state: ChannelState) -> torch.Tensor:
        G = torch.stack(state.group_sums)
        if G.shape[0] == 1 or self.inner == "mean":
            return G.mean(0)
        from .defenses import geometric_median
        return geometric_median(G)

    def _sigma2(self, state: ChannelState) -> float:
        n_part = int(state.participating.sum())
        ota = float(state.sigma_z2 / 2.0) / float((state.gamma.mean() * max(n_part, 1)) ** 2) * state.coord_idx.numel()
        return ota + (self.sigma_grad2 if self.sigma_grad2 is not None else 0.0)

    def _finalise_thresholds(self) -> None:
        D = np.asarray(self._null_D[-self.null_window:] if self.calibration == "sliding" else self._null_D)
        if len(D) < 5:
            return
        Dm = np.asarray(self._null_Dmem[-self.null_window:] if self.calibration == "sliding" else self._null_Dmem)

        def thr(x, alpha):
            x = np.asarray(x, dtype=np.float64)
            if self.threshold_mode == "quantile":
                return float(np.quantile(x, 1 - alpha))
            # ROBUST scaled chi-square calibration: x ~ c chi2_r with (c, r) matched to the median and the
            # MAD-based scale of the window (mean ~ c r, var ~ 2 c^2 r). Bursts occupying a minority of the
            # window do not move median/MAD appreciably, so no benign-only selection is needed.
            from scipy.stats import chi2
            med = float(np.median(x)); sd = 1.4826 * float(np.median(np.abs(x - med))) + 1e-12
            c = sd ** 2 / (2 * med + 1e-12); r = max(med / (c + 1e-12), 1.0)
            return float(c * chi2.ppf(1 - alpha, df=r))

        self.lam = thr(D, self.alpha_fa)
        self.lam_freeze = thr(D, self.alpha_freeze)
        if len(Dm) >= 5:
            self.lam_mem = thr(Dm, self.alpha_fa)
        self.lam_k = [thr(dk, self.alpha_fa) if len(dk) >= 5 else float("inf") for dk in self._null_Dk] if self.per_level else None
        mu, var = D.mean(), D.var() + 1e-12
        c = var / (2 * mu + 1e-12); r = mu / (c + 1e-12)
        self.chi2_fit = {"c": float(c), "r": float(r), "n": int(len(D))}

    # ------------------------------------------------------------------ main
    def aggregate(self, state: ChannelState, server_ctx: Dict) -> Tuple[torch.Tensor, Dict]:
        self.t += 1
        delta = self._inner(state)
        sig2 = self._sigma2(state)
        in_warmup = self.t <= self.warmup or self.lam is None
        # ---------------- (1) innovation test on arrival: incoming aggregate vs slowest memory
        mhat_prev = self._mhat_at(self.t - 1)
        mK = mhat_prev[-1]
        e = delta - mK
        D = float(e.pow(2).sum() / (sig2 + 1e-12))
        fired = (not in_warmup) and D > self.lam
        freeze = (not in_warmup) and (self.lam_freeze is not None) and (D > self.lam_freeze)
        tilde = delta
        if fired and self.orth_on and e.norm() > 0:
            # centered clipping around the slow memory to the calibrated radius: the surprise component is
            # shrunk to the benign scale, never zeroed (learning along the established direction continues)
            radius = float(np.sqrt(self.lam * sig2))
            tilde = mK + e * min(1.0, radius / float(e.norm()))
        # ---------------- memory updates with the gated input
        for k in range(self.K):
            if freeze and k == self.K - 1:
                continue                              # slowest level frozen on extreme surprise
            self.m[k] = self.betas[k] * self.m[k] + (1 - self.betas[k]) * tilde
        # ---------------- (2) persistence test on the memories: fast vs slow after the update
        mhat = self._mhat_at(self.t)
        u = mhat[0] - mhat[-1]
        Dk = [float((mhat[k] - mhat[k + 1]).pow(2).sum() / (sig2 + 1e-12)) for k in range(self.K - 1)]
        D_mem = float(u.pow(2).sum() / (sig2 + 1e-12))
        k_star = 0; purged = False
        if (not in_warmup) and self.lam_mem is not None and D_mem > self.lam_mem and self.purge_on and u.norm() > 0:
            un = u / u.norm()
            if self.per_level and self.lam_k is not None:
                firing = [k + 1 for k in range(self.K - 1) if Dk[k] > self.lam_k[k]]
                k_star = max(firing) if firing else self.K - 1
            else:
                k_star = self.K - 1
            corr = (1 - self.betas ** self.t).clamp_min(1e-6)
            mK_cur = mhat[-1]
            for k in range(k_star):                   # delta-rule: contract fast memories toward the slow one
                excess = (mhat[k] - mK_cur) @ un
                self.m[k] = self.m[k] - self.kappa * excess * un * corr[k]
            purged = True
        # ---------------- benign variance: winsorised EMA of the gated residual (never gets stuck)
        # from the UNCLIPPED innovation (so benign non-stationarity is tracked) but winsorised at
        # winsor x the current estimate (so a burst cannot inflate it in one round)
        r2 = float(e.pow(2).sum())
        if self.sigma_grad2 is None:
            self.sigma_grad2 = r2
        else:
            r2 = min(r2, self.winsor * self.sigma_grad2)
            self.sigma_grad2 = self.resid_decay * self.sigma_grad2 + (1 - self.resid_decay) * r2
        # ---------------- null-distribution bookkeeping (benign-only by construction; fired rounds winsorised)
        # all rounds enter the sliding buffers (no decision-dependent selection); robustness to bursts
        # comes from the median/MAD calibration above
        if self.t > 2 and (in_warmup or self.calibration == "sliding"):
            self._null_D.append(D)
            self._null_Dmem.append(D_mem)
            for k in range(self.K - 1):
                self._null_Dk[k].append(Dk[k])
            if self.calibration == "sliding":
                self._null_D = self._null_D[-self.null_window:]; self._null_Dmem = self._null_Dmem[-self.null_window:]
                self._null_Dk = [dk[-self.null_window:] for dk in self._null_Dk]
            if (self.t == self.warmup) or (self.calibration == "sliding" and self.t > self.warmup and len(self._null_D) >= self.min_null):
                self._finalise_thresholds()
        # update direction
        direction = self.gain * (self.omega[:, None] * self._mhat_at(self.t)).sum(0)
        if self.adaptive:
            d2 = tilde * tilde
            self.v_fast = self.b2f * self.v_fast + (1 - self.b2f) * d2
            self.v_slow = self.b2s * self.v_slow + (1 - self.b2s) * d2
            direction = direction / (torch.maximum(self.v_fast, self.v_slow).sqrt() + self.tau)
        diag = {"D": D, "Dk": Dk, "fired": fired, "k_star": k_star, "freeze": freeze, "lam": self.lam,
                "D_mem": D_mem, "lam_mem": self.lam_mem, "purged": purged, "sigma2": sig2, "warmup": in_warmup}
        return direction, diag

    def reset(self):
        self.m.zero_(); self.v_fast.zero_(); self.v_slow.zero_()

    def state_dict(self):
        return {"m": self.m, "v_fast": self.v_fast, "v_slow": self.v_slow}
