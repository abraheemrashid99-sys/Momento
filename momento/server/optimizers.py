"""Server optimizers with fully inspectable state (Reddi et al., ICLR 2021 conventions).

All operate on a flat vector. `step(delta)` consumes the *aggregate update* delta (in the same
units as a single client's local update, i.e. w_t - w_local) and returns the new global vector.

    FedAvg  : w <- w - eta_s * delta
    FedAvgM : m <- beta m + delta ;             w <- w - eta_s m          (Hsu et al. 2019 form, beta=0.9)
    FedAdam : m <- b1 m + (1-b1) delta ; v <- b2 v + (1-b2) delta^2 ; w <- w - eta_s m / (sqrt(v) + tau)
    FedYogi : as FedAdam but v <- v - (1-b2) delta^2 sign(v - delta^2)

Note on FedAvgM form: Reddi et al. use the un-normalised heavy-ball m <- beta m + delta. The
"(1-beta)" normalised form used in the proposal text is equivalent up to eta_s scaling; we keep
Reddi's form so that hyper-parameters are directly comparable with the literature, and expose
`effective_kernel()` which returns beta^k for the analysis scripts.

The optimizer exposes `state_dict()` / `load_state_dict()` and `reset()` (for the reset ablation),
and `invert(w_prev, w_new)` which recovers delta from two consecutive broadcasts when the
optimizer state is known: this is what the attacker uses to reconstruct the aggregate trajectory.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch


class ServerOptimizer:
    name = "fedavg"

    def __init__(self, d: int, device: torch.device, eta_s: float = 1.0, **kw):
        self.d, self.device, self.eta_s = d, device, eta_s

    def step(self, w: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return w - self.eta_s * delta

    def reset(self) -> None:
        pass

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self, s: Dict[str, torch.Tensor]) -> None:
        pass

    def clone(self) -> "ServerOptimizer":
        o = type(self)(self.d, self.device, **self.hparams())
        o.load_state_dict({k: v.clone() for k, v in self.state_dict().items()})
        return o

    def hparams(self) -> Dict[str, float]:
        return {"eta_s": self.eta_s}

    def invert(self, w_prev: torch.Tensor, w_new: torch.Tensor, fallback_mag=None) -> torch.Tensor:
        """Recover the aggregate delta that produced w_new from w_prev (attacker-side, state known)."""
        return (w_prev - w_new) / self.eta_s


class FedAvgM(ServerOptimizer):
    name = "fedavgm"

    def __init__(self, d, device, eta_s=1.0, beta=0.9, **kw):
        super().__init__(d, device, eta_s)
        self.beta = beta
        self.m = torch.zeros(d, device=device)

    def step(self, w, delta):
        self.m = self.beta * self.m + delta
        return w - self.eta_s * self.m

    def reset(self):
        self.m.zero_()

    def state_dict(self):
        return {"m": self.m}

    def load_state_dict(self, s):
        self.m = s["m"].to(self.device)

    def hparams(self):
        return {"eta_s": self.eta_s, "beta": self.beta}

    def invert(self, w_prev, w_new, fallback_mag=None):
        m_new = (w_prev - w_new) / self.eta_s
        delta = m_new - self.beta * self.m
        self.m = m_new
        return delta

    def effective_kernel(self, k: int) -> float:
        return self.beta ** k


class FedAdam(ServerOptimizer):
    name = "fedadam"

    def __init__(self, d, device, eta_s=0.01, beta1=0.9, beta2=0.99, tau=1e-3, yogi=False, **kw):
        super().__init__(d, device, eta_s)
        self.b1, self.b2, self.tau, self.yogi = beta1, beta2, tau, yogi
        self.m = torch.zeros(d, device=device)
        self.v = torch.zeros(d, device=device)
        self.t = 0

    def step(self, w, delta):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * delta
        d2 = delta * delta
        if self.yogi:
            self.v = self.v - (1 - self.b2) * d2 * torch.sign(self.v - d2)
        else:
            self.v = self.b2 * self.v + (1 - self.b2) * d2
        return w - self.eta_s * self.m / (self.v.sqrt() + self.tau)

    def reset(self):
        self.m.zero_(); self.v.zero_(); self.t = 0

    def state_dict(self):
        return {"m": self.m, "v": self.v, "t": torch.tensor(self.t)}

    def load_state_dict(self, s):
        self.m = s["m"].to(self.device); self.v = s["v"].to(self.device); self.t = int(s["t"])

    def hparams(self):
        return {"eta_s": self.eta_s, "beta1": self.b1, "beta2": self.b2, "tau": self.tau, "yogi": self.yogi}

    def invert(self, w_prev, w_new, fallback_mag=None):
        """Exact inversion is a scalar quadratic per coordinate for Adam (v depends on delta^2).
        We solve it in closed form per coordinate: step s = (w_prev-w_new)/eta = m'/(sqrt(v')+tau)
        with m' = b1 m + (1-b1) delta, v' = b2 v + (1-b2) delta^2. Solve for delta by Newton (3 iters
        from the FedAvgM-style initial guess)."""
        s = (w_prev - w_new) / self.eta_s
        b1, b2, tau = self.b1, self.b2, self.tau
        m, v = self.m, self.v

        def g(delta):
            m_ = b1 * m + (1 - b1) * delta
            v_ = b2 * v + (1 - b2) * delta * delta
            return m_ / (v_.sqrt() + tau)

        # g is bounded: |g| -> (1-b1)/sqrt(1-b2) as |delta| -> inf. Bracket and bisect (vectorised).
        scale = torch.maximum(torch.maximum(m.abs() / (1 - b1), (b2 * v).sqrt() / (1 - b2) ** 0.5), torch.full_like(m, tau / (1 - b2) ** 0.5))
        L = 50.0 * scale + 1e-6
        lo, hi = -L, L
        glo, ghi = g(lo), g(hi)
        saturated = (s >= ghi) | (s <= glo)
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            gm = g(mid)
            go_right = gm < s
            lo = torch.where(go_right, mid, lo)
            hi = torch.where(go_right, hi, mid)
        delta = 0.5 * (lo + hi)
        # Saturated coordinates: |delta| is not identifiable (only sign). The caller supplies a
        # fallback magnitude (the attacker uses its OWN clean-update magnitudes); recovered values are
        # capped at 5x the fallback so that one bad round cannot blow up the shadow state.
        if fallback_mag is None:
            fallback_mag = torch.full_like(s, float(scale.median()))
        fb = torch.as_tensor(fallback_mag, device=s.device, dtype=s.dtype).expand_as(s)
        delta = torch.where(saturated, torch.sign(s) * fb, delta)
        delta = delta.clamp(-5 * fb.abs() - 1e-8, 5 * fb.abs() + 1e-8)
        # NOTE: this non-identifiability in the cold regime is the mechanism echo_adam exploits.
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * delta
        self.v = self.b2 * self.v + (1 - self.b2) * delta * delta
        return delta

    def effective_kernel(self, k: int) -> float:
        return self.b1 ** k


class FedYogi(FedAdam):
    name = "fedyogi"

    def __init__(self, d, device, **kw):
        kw["yogi"] = True
        super().__init__(d, device, **kw)


def build_server_optimizer(name: str, d: int, device: torch.device, **kw) -> ServerOptimizer:
    return {"fedavg": ServerOptimizer, "fedavgm": FedAvgM, "fedadam": FedAdam, "fedyogi": FedYogi}[name](d, device, **kw)
