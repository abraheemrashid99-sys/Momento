"""Attack interface and the classical backdoor baselines.

An Attack controls the malicious client set A. Each round the simulator calls

    attack.act(t, w_t, model, honest_norm_est) -> {cid: delivery vector v (d,)} or {} (no override)

A delivery vector is what the attacker wants to *arrive at the server* in 'one honest client's update'
units; the channel enforces the attacker's own fading and power cap. Outside its window the attacker
behaves honestly (its clients run the honest update) so that reputation-based defenses see a normal
client; the simulator handles that by calling `attack.is_active(t)`.

Common machinery (BackdoorTrainer):
  * poisoned local training: mixture of clean and triggered samples relabelled to the target
  * stealth: ||delivered||_2 <= rho * honest_norm_est   (rho is the stealth budget; honest_norm_est is
    the median norm of the attacker's OWN clean updates — the attacker has no access to others)
  * honest-gradient null-space mask (Neurotoxin): coordinates with the smallest magnitude in the last
    recovered aggregate

Baselines implemented here:
  scale        model replacement / constrain-and-scale (Bagdasaryan et al. 2020): clip to rho then boost
  neurotoxin   Zhang et al. 2022: train the backdoor only on bottom-(1-q) coordinates of the benign aggregate
  a3fl_lite    Zhang et al. 2023: additionally adapt the trigger against a simulated benign continuation
  chameleon_lite Dai & Li 2023: supervised-contrastive facilitator loss on logits (lite)

The momentum-echo family is in echo.py.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..client import local_sgd
from ..data.triggers import Trigger
from ..utils import params_to_vector, vector_to_params


class Attack:
    name = "none"

    def __init__(self, malicious: List[int], cfg: dict, device: torch.device, seed: int, **ctx):
        self.A = list(malicious)
        self.cfg = cfg
        self.device = device
        self.seed = seed
        self.ctx = ctx
        self.t0 = cfg.get("start", 50)
        self.tau = cfg.get("window", 5)
        self.rho = cfg.get("rho", 1.0)
        self.history: List[torch.Tensor] = []     # broadcasts seen
        self.honest_norms: List[float] = []

    def is_active(self, t: int) -> bool:
        return self.t0 <= t < self.t0 + self.tau

    def observe_broadcast(self, t: int, w: torch.Tensor) -> None:
        self.history.append(w.detach().clone())

    def act(self, t: int, w: torch.Tensor, model: torch.nn.Module, honest_norm_est: float) -> Dict[int, torch.Tensor]:
        return {}

    def report_indicators(self, cid: int, honest_fn) -> torch.Tensor:
        """What this client reports to TTI. Default: the truth about its *clean* update."""
        return honest_fn()


class NoAttack(Attack):
    def is_active(self, t):
        return False


class BackdoorTrainer:
    """Shared poisoned-training machinery."""

    def __init__(self, trigger: Trigger, x: torch.Tensor, y: torch.Tensor, cfg: dict, device: torch.device, seed: int):
        self.trigger, self.x, self.y, self.cfg, self.device = trigger, x, y, cfg, device
        self.gen = torch.Generator().manual_seed(seed * 7 + 99)
        self.poison_frac = cfg.get("poison_frac", 0.5)

    def poisoned_batch_fn(self):
        trig, frac, tgt = self.trigger, self.poison_frac, self.trigger.target

        def hook(model, xb, yb, out):
            n = xb.shape[0]; k = int(frac * n)
            if k == 0:
                return F.cross_entropy(out, yb)
            xt = trig.apply(xb[:k]); yt = torch.full((k,), tgt, device=xb.device, dtype=yb.dtype)
            out_t = model(xt)
            return 0.5 * F.cross_entropy(out, yb) + 0.5 * F.cross_entropy(out_t, yt)
        return hook

    def backdoor_delta(self, model, w, grad_mask: Optional[torch.Tensor] = None, epochs: Optional[int] = None,
                       extra_loss=None) -> torch.Tensor:
        c = self.cfg
        hook = self.poisoned_batch_fn()
        if extra_loss is not None:
            base = hook
            def hook2(model, xb, yb, out):
                return base(model, xb, yb, out) + extra_loss(model, xb, yb, out)
            hook = hook2
        delta, _ = local_sgd(model, w, self.x, self.y, epochs or c.get("epochs", 2), c.get("batch", 64),
                             c.get("lr", 0.05), self.device, self.gen, loss_hook=hook, grad_mask=grad_mask)
        return delta

    def clean_delta(self, model, w) -> torch.Tensor:
        c = self.cfg
        delta, _ = local_sgd(model, w, self.x, self.y, c.get("clean_epochs", 1), c.get("batch", 64),
                             c.get("clean_lr", c.get("lr", 0.05)), self.device, self.gen)
        return delta

    @torch.no_grad()
    def clean_acc(self, model, w, n: int = 512) -> float:
        vector_to_params(model, w); model.eval()
        xb, yb = self.x[:n].to(self.device), self.y[:n].to(self.device)
        return float((model(xb).argmax(1) == yb).float().mean())

    @torch.no_grad()
    def backdoor_acc(self, model, w, n: int = 512) -> float:
        vector_to_params(model, w); model.eval()
        xb, yb = self.x[:n].to(self.device), self.y[:n].to(self.device)
        keep = yb != self.trigger.target
        if not keep.any():
            return 0.0
        return float((model(self.trigger.apply(xb[keep])).argmax(1) == self.trigger.target).float().mean())

    def backdoor_loss(self, model, w, n: int = 256) -> float:
        vector_to_params(model, w); model.eval()
        with torch.no_grad():
            xb = self.trigger.apply(self.x[:n].to(self.device))
            yb = torch.full((xb.shape[0],), self.trigger.target, device=self.device, dtype=torch.long)
            return float(F.cross_entropy(model(xb), yb))


def clip_norm(v: torch.Tensor, max_norm: float) -> torch.Tensor:
    n = v.norm()
    return v * min(1.0, float(max_norm / (n + 1e-12)))


def neurotoxin_mask(benign_agg: torch.Tensor, keep_frac: float) -> torch.Tensor:
    """1 on the (1-keep_frac) coordinates with the smallest |benign_agg| (rarely updated), 0 elsewhere."""
    d = benign_agg.numel(); k = int((1 - keep_frac) * d)
    mask = torch.zeros(d, device=benign_agg.device)
    idx = benign_agg.abs().topk(k, largest=False).indices
    mask[idx] = 1.0
    return mask


class ClassicBackdoor(Attack):
    """scale / neurotoxin / a3fl_lite / chameleon_lite share this class; `variant` selects."""

    def __init__(self, malicious, cfg, device, seed, trainer: BackdoorTrainer, server_opt_factory, **ctx):
        super().__init__(malicious, cfg, device, seed, **ctx)
        self.variant = cfg.get("variant", "scale")
        self.tr = trainer
        self.boost = cfg.get("boost", 1.0)            # model-replacement boost (bounded by rho anyway)
        self.keep_frac = cfg.get("neurotoxin_keep", 0.1)
        self.shadow = server_opt_factory()            # attacker's copy of the server optimizer (white-box)
        self.last_agg: Optional[torch.Tensor] = None
        self.n_part_est = cfg.get("n_participants_est", 20)

    @property
    def name(self):
        return self.variant

    def observe_broadcast(self, t, w):
        if self.history:
            fb = self.ctx.get("own_mag")
            self.last_agg = self.shadow.invert(self.history[-1], w, fallback_mag=fb)
        super().observe_broadcast(t, w)

    def _trigger_adapt(self, model, w):
        """A3FL-lite: a few gradient steps on the trigger pattern to maximise backdoor loss *after* a
        simulated benign continuation (one clean epoch on the attacker's data)."""
        params = self.tr.trigger.learnable_parameters()
        if not params:
            return
        w_c = w - self.tr.clean_delta(model, w)       # simulated benign continuation
        vector_to_params(model, w_c); model.eval()
        opt = torch.optim.Adam(params, lr=self.cfg.get("trigger_lr", 0.01))
        xb = self.tr.x[:128].to(self.device)
        yb = torch.full((xb.shape[0],), self.tr.trigger.target, device=self.device, dtype=torch.long)
        for _ in range(self.cfg.get("trigger_steps", 10)):
            opt.zero_grad()
            loss = F.cross_entropy(model(self.tr.trigger.apply(xb)), yb)
            loss.backward(); opt.step()
            with torch.no_grad():
                self.tr.trigger.pattern.clamp_(0, 1)
        self.tr.trigger.pattern = self.tr.trigger.pattern.detach()

    def _chameleon_loss(self):
        tgt = self.tr.trigger.target; trig = self.tr.trigger
        def extra(model, xb, yb, out):
            k = xb.shape[0] // 2
            if k == 0:
                return out.sum() * 0
            z_t = F.normalize(model(trig.apply(xb[:k])), dim=1)
            z_c = F.normalize(out, dim=1)
            pos = z_c[yb == tgt]
            if pos.shape[0] == 0:
                return out.sum() * 0
            sim_pos = (z_t @ pos.T).mean()
            sim_neg = (z_t @ z_c[yb != tgt].T).mean() if (yb != tgt).any() else 0.0
            return self.cfg.get("contrastive_w", 0.5) * (sim_neg - sim_pos)
        return extra

    def act(self, t, w, model, honest_norm_est):
        if not self.is_active(t):
            return {}
        mask = None; extra = None
        if self.variant in ("neurotoxin", "a3fl_lite") and self.last_agg is not None:
            mask = neurotoxin_mask(self.last_agg, self.keep_frac)
        if self.variant == "a3fl_lite":
            self._trigger_adapt(model, w)
        if self.variant == "chameleon_lite":
            extra = self._chameleon_loss()
        delta_bd = self.tr.backdoor_delta(model, w, grad_mask=mask, extra_loss=extra)
        budget = self.rho * honest_norm_est
        a = clip_norm(delta_bd * self.boost, budget)        # desired contribution to the *aggregate*
        v = a * self.n_part_est / len(self.A)              # per-attacker delivery so that the sum gives a
        return {cid: v for cid in self.A}

    def report_indicators(self, cid, honest_fn):
        # a classic attacker reports the truth about what it sent (it does not model TTI)
        return honest_fn()


def build_attack(name: str, malicious, cfg, device, seed, **ctx) -> Attack:
    if name == "none" or not malicious:
        return NoAttack(malicious, cfg, device, seed)
    if name in ("scale", "neurotoxin", "a3fl_lite", "chameleon_lite"):
        cfg = dict(cfg); cfg["variant"] = name
        return ClassicBackdoor(malicious, cfg, device, seed, **ctx)
    if name in ("echo", "echo_adam", "echo_adaptive"):
        from .echo import MomentumEcho
        cfg = dict(cfg); cfg["variant"] = name
        return MomentumEcho(malicious, cfg, device, seed, **ctx)
    raise KeyError(name)
