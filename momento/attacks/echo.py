"""The momentum-echo attack family (MEMENTO's attack contribution).

Threat model: M colluding clients, active only in rounds [t0, t0+tau). White-box knowledge of the
server optimizer and its hyper-parameters; sees every broadcast w_t; does NOT see other clients'
updates, channels, or any per-round server randomisation. Physics (own fading, CSI error, P_max)
is enforced by the channel.

Mechanism. The attacker maintains a *shadow* copy of the server optimizer and inverts consecutive
broadcasts to recover the aggregate trajectory {bar_delta_s}. From this it (i) estimates the honest
drift direction and magnitude (surrogate), (ii) identifies the Neurotoxin-style null-space of the
honest aggregate, and (iii) for FedAdam, reconstructs the second-moment estimate v_hat to find
*cold coordinates* (bottom-q quantile of v_hat).

Variants
  echo          FedAvgM server:  m <- beta m + delta, w <- w - eta m.
                A single aggregate injection a at round s produces displacement
                    eta * a * (1 - beta^{t-s+1}) / (1 - beta)                   (= eta a / (1-beta) as t->inf)
                i.e. the momentum *amplifies* the injection by 1/(1-beta) and delivers it over
                ~1/(1-beta) rounds. The attacker therefore needs only (1-beta) of the per-round
                energy a model-replacement attacker needs for the same displacement, and spends it
                front-loaded so that the displacement matures while it is still active to also
                pre-compensate the honest correction it measures locally (unlearning rate xi).
  echo_adam     FedAdam/FedYogi server: the injection is placed on cold coordinates with a magnitude
                chosen so that (1-beta2) a^2 >> v_hat: the normalised step saturates at ~eta*sign(a)
                per coordinate regardless of |a| (so the attacker gets a *full-size* step from a tiny
                injection), AND v on those coordinates is inflated for ~1/(1-beta2) rounds, which
                damps the benign corrective gradients by 1/sqrt(v): the preconditioner echo.
  echo_adaptive Attacker that knows CMA (decay rates and calibrated thresholds). It runs shadow CMA
                memories on the recovered aggregates and chooses the per-round magnitude so that the
                predicted disagreement statistic stays below the threshold (safety factor), spreading
                the injection over a longer window.

Pre-compensation. Locally the attacker simulates: start from w + displacement, run one clean epoch on
its own data, and measures the fraction xi of backdoor displacement undone per round. It then scales
the target displacement by c = 1 / max(1 - xi * H_eff, 0.2), H_eff = min(1/(1-beta), tau_rem).

Stealth: per-round aggregate contribution ||a_s|| <= rho * honest_norm_est. For TTI, echo clients
report indicators equal to the median of their own honest history (benign-looking reports).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .base import Attack, BackdoorTrainer, clip_norm, neurotoxin_mask
from ..server.optimizers import FedAvgM, FedAdam
from ..utils import vector_to_params


class MomentumEcho(Attack):
    def __init__(self, malicious, cfg, device, seed, trainer: BackdoorTrainer, server_opt_factory, cma_factory=None, **ctx):
        super().__init__(malicious, cfg, device, seed, **ctx)
        self.variant = cfg.get("variant", "echo")
        self.tr = trainer
        self.shadow = server_opt_factory()
        self.aggs: List[torch.Tensor] = []
        self.n_part_est = cfg.get("n_participants_est", 20)
        self.keep_frac = cfg.get("neurotoxin_keep", 0.1)
        self.cold_q = cfg.get("cold_quantile", 0.02)
        self.use_full_budget = cfg.get("use_full_budget", True)
        self.precomp = cfg.get("precompensate", True)
        self.front_load = cfg.get("front_load", True)
        self.safety = cfg.get("adaptive_safety", 0.5)
        self.shadow_cma = cma_factory() if (cma_factory is not None and self.variant == "echo_adaptive") else None
        self.v_hat: Optional[torch.Tensor] = None
        self._honest_ind: List[torch.Tensor] = []
        self.direction: Optional[torch.Tensor] = None
        self.schedule: Optional[List[float]] = None
        self.diag: Dict = {}
        self.own_mag: Optional[torch.Tensor] = None

    @property
    def name(self):
        return self.variant

    # ------------------------------------------------------------------ observation
    def observe_broadcast(self, t, w):
        if self.history:
            # fallback magnitude for the non-identifiable (saturated) Adam regime: the attacker's own
            # clean-update magnitudes, refreshed every few rounds (cheap: one local epoch)
            if isinstance(self.shadow, FedAdam) and (self.own_mag is None or t % 5 == 0):
                self.own_mag = self.tr.clean_delta(self.ctx.get("model"), w).abs() if self.ctx.get("model") is not None else None
            agg = self.shadow.invert(self.history[-1], w, fallback_mag=self.own_mag)
            self.aggs.append(agg.detach().clone())
            if isinstance(self.shadow, FedAdam):
                self.v_hat = self.shadow.v.clone()
            if self.shadow_cma is not None:
                self._shadow_cma_step(agg)
        super().observe_broadcast(t, w)

    def _shadow_cma_step(self, agg):
        c = self.shadow_cma
        for k in range(c.K):
            c.m[k] = c.betas[k] * c.m[k] + (1 - c.betas[k]) * agg

    # ------------------------------------------------------------------ pieces
    def _honest_drift(self) -> torch.Tensor:
        if not self.aggs:
            return torch.zeros(self.history[-1].numel(), device=self.device)
        k = min(5, len(self.aggs))
        return torch.stack(self.aggs[-k:]).mean(0)

    def _unlearning_rate(self, model, w, disp: torch.Tensor) -> float:
        """xi: fraction of the backdoor displacement undone by one honest-like round (measured locally)."""
        if not self.precomp:
            return 0.0
        w_bd = w - disp
        clean = self.tr.clean_delta(model, w_bd)                 # honest correction from the backdoored point
        # component of the correction along the displacement, relative to its size
        xi = float((clean @ disp) / (disp.norm() ** 2 + 1e-12))
        return float(np.clip(xi, 0.0, 0.95))

    def _adam_cold_direction(self, model, w, delta_bd, mask, budget):
        """Preconditioner echo: choose the cold-coordinate set by LOCAL simulation. Adam's normalised step
        on a cold coordinate saturates at ~eta*sign(a) whatever |a| is, so the attacker cannot control the
        per-coordinate step size -- it controls the *number* of coordinates. For each candidate set size
        it simulates the shadow server over the window (+ honest-drift surrogate) and keeps the largest set
        whose clean accuracy on its own data drops by at most `max_ma_drop` (default 3 pts)."""
        v = self.v_hat.clone()
        if mask is not None:
            v[mask == 0] = float("inf")
        kb = max(1, int(0.1 * v.numel()))                              # backdoor-relevant candidates
        cand = torch.zeros_like(v); cand[delta_bd.abs().topk(kb).indices] = 1.0
        v[cand == 0] = float("inf")
        order = v.argsort()                                             # coldest first
        fracs = self.cfg.get("cold_fracs", [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2])
        max_drop = self.cfg.get("max_ma_drop", 0.03)
        acc0 = self.tr.clean_acc(model, w)
        drift = self._honest_drift()
        best = None
        for f in fracs:
            k = max(1, int(f * v.numel()))
            cold = torch.zeros_like(v); cold[order[:k]] = 1.0
            raw = cold * torch.sign(delta_bd)
            a = raw / (raw.norm() + 1e-12) * budget
            sh = self.shadow.clone(); w_sim = w.clone()
            for j in range(self.tau):
                w_sim = sh.step(w_sim, drift + a)
            for j in range(5):                                          # a few honest rounds after the window
                w_sim = sh.step(w_sim, drift)
            acc = self.tr.clean_acc(model, w_sim); bd = self.tr.backdoor_acc(model, w_sim)
            self.diag.setdefault("cold_search", []).append({"frac": f, "k": k, "clean_acc": acc, "bd_acc": bd})
            if acc >= acc0 - max_drop and (best is None or bd >= best[3]):
                best = (f, k, raw, bd)
            if acc < acc0 - max_drop:
                break
        if best is None:
            f, k = fracs[0], max(1, int(fracs[0] * v.numel()))
            cold = torch.zeros_like(v); cold[order[:k]] = 1.0; raw = cold * torch.sign(delta_bd)
        else:
            f, k, raw, _ = best
        self.diag["cold_frac"] = f; self.diag["cold_coords"] = int(k); self.diag["clean_acc_before"] = acc0
        return raw / (raw.norm() + 1e-12)

    def _plan(self, t, w, model, honest_norm_est):
        """Compute the backdoor direction and the per-round magnitude schedule once at window start."""
        mask = neurotoxin_mask(self._honest_drift(), self.keep_frac) if self.aggs else None
        delta_bd = self.tr.backdoor_delta(model, w, grad_mask=mask)
        budget = self.rho * honest_norm_est
        if self.variant == "echo_adam" and self.v_hat is not None:
            direction = self._adam_cold_direction(model, w, delta_bd, mask, budget)
        else:
            direction = delta_bd / (delta_bd.norm() + 1e-12)
        self.direction = direction
        rem = self.t0 + self.tau - t
        if isinstance(self.shadow, FedAvgM):
            beta, eta = self.shadow.beta, self.shadow.eta_s
            amp = 1.0 / (1.0 - beta)                            # momentum amplification
        elif isinstance(self.shadow, FedAdam):
            beta, eta = self.shadow.b1, self.shadow.eta_s
            amp = 1.0                                          # step saturates; amplification is in the damping
        else:
            beta, eta, amp = 0.0, self.shadow.eta_s, 1.0
        # target displacement: what the plain backdoor update would move the model by, times pre-compensation
        target = delta_bd.norm() * self.cfg.get("target_scale", 1.0)
        H_eff = min(1.0 / max(1 - beta, 1e-3), rem)
        xi = self._unlearning_rate(model, w, eta * delta_bd) if self.precomp else 0.0
        comp = 1.0 / max(1.0 - xi * H_eff, 0.2)
        total_needed = target * comp / (eta * amp)             # total aggregate energy to inject over the window
        # schedule: front-loaded geometric (earlier injections mature sooner) or uniform
        if self.front_load:
            wts = np.array([(beta if beta > 0 else 0.5) ** j for j in range(rem)]); wts = wts / wts.sum()
        else:
            wts = np.ones(rem) / rem
        if self.use_full_budget:
            sched = [budget for _ in wts]                        # equal-footprint comparison with other attacks
        else:
            sched = [min(float(total_needed * wj), budget) for wj in wts]
        self.schedule = sched
        self.diag.update({"xi": xi, "comp": comp, "amp": amp, "total_needed": float(total_needed), "budget": budget,
                          "schedule": sched})

    # ------------------------------------------------------------------ action
    def act(self, t, w, model, honest_norm_est):
        if not self.is_active(t):
            self.schedule = None
            return {}
        if self.schedule is None or self.direction is None:
            self._plan(t, w, model, honest_norm_est)
        j = t - self.t0
        mag = self.schedule[min(j, len(self.schedule) - 1)]
        if self.variant == "echo_adaptive" and self.shadow_cma is not None and self.shadow_cma.lam is not None:
            mag = self._adaptive_magnitude(mag)
        if self.cfg.get("refresh_direction", False) and j > 0:
            d = self.tr.backdoor_delta(model, w)
            self.direction = d / (d.norm() + 1e-12)
        a = self.direction * mag
        v = a * self.n_part_est / len(self.A)
        return {cid: v for cid in self.A}

    def _adaptive_magnitude(self, mag):
        c = self.shadow_cma
        sig2 = (c.sigma_grad2 or 1.0)
        # predicted disagreement after injecting a along direction u with magnitude x:
        # m1 - mK moves by ((1-b1) - (1-bK)) x along the direction  =>  D ~ ||u + dx||^2/sig2
        u = c.m[0] - c.m[-1]
        g = (1 - c.betas[0]) - (1 - c.betas[-1])
        lo, hi = 0.0, mag
        for _ in range(20):
            x = 0.5 * (lo + hi)
            D_pred = float((u + g * x * self.direction).pow(2).sum() / sig2)
            if D_pred < self.safety * c.lam:
                lo = x
            else:
                hi = x
        return lo

    # ------------------------------------------------------------------ TTI spoofing
    def report_indicators(self, cid, honest_fn):
        ind = honest_fn()
        if not self.is_active(len(self.history) - 1):
            self._honest_ind.append(ind)
            return ind
        if self._honest_ind:
            return torch.stack(self._honest_ind[-10:]).median(0).values
        return ind
