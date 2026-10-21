"""OTA-compatible aggregators / defenses, reimplemented from the papers (no official code exists for
most of them; our reimplementations are part of the release).

Every defense receives the ChannelState (the list of per-slot group sums, each already in
'average update' units) and returns a single aggregate vector of length d' (transmitted coords)
plus a dict of diagnostics. None of them sees per-client vectors.

  none         : mean of group sums (plain OTA-FedAvg; with n_slots=1 this is the raw aggregate)
  bev          : BEV-SGD (Fan et al., IoT-J 2022). The defense is the *power-control policy*
                 (set channel.power_control='bev'); the aggregator is the plain mean.
  rotaf        : ROTAF (Sifaou & Li 2022): geometric median across group sums (n_slots = G > 1),
                 smoothed Weiszfeld.
  aircomp_gm   : AirComp Weiszfeld geometric median (Huang et al. 2021). In the single-slot setting
                 it degenerates to the mean; with slots it coincides with ROTAF's statistic. We expose
                 it as ROTAF with Weiszfeld iterations = 1 (one extra channel use) for honesty.
  group_cc     : centered clipping (Karimireddy et al. 2021) applied to group sums around the server's
                 running estimate (an OTA-compatible adaptation; the paper's own CC needs per-worker vectors).
  group_tm     : coordinate-wise trimmed mean across group sums.
  fedsac       : FedSAC (Yao et al., JSAC 2025) *as reimplementable from the aggregate*: adaptive
                 cluster weights by distance to the robust centre (P-CCP-style weighting) plus a
                 historical cluster reputation. Device-level reputation is not available to a strict
                 OTA server; we implement the cluster-level variant and document this.
  tti          : TTI (Ma et al. 2026) requires client-reported indicators and individual uploads of
                 'suspicious' clients. It is implemented in tti.py because it breaks the aggregate-only
                 interface (it needs the client dictionary); the simulator calls it through the same
                 Aggregator API but passes extra information and charges extra uplink bytes.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from ..channel.ota import ChannelState


def geometric_median(xs: torch.Tensor, iters: int = 20, eps: float = 1e-6) -> torch.Tensor:
    """Smoothed Weiszfeld on rows of xs (G, d)."""
    z = xs.mean(0)
    for _ in range(iters):
        dist = (xs - z).norm(dim=1).clamp_min(eps)
        wgt = 1.0 / dist
        z = (wgt[:, None] * xs).sum(0) / wgt.sum()
    return z


class Aggregator:
    name = "none"
    extra_uplink_bytes = 0.0

    def __init__(self, d_tx: int, device: torch.device, **kw):
        self.device = device

    def aggregate(self, state: ChannelState, server_ctx: Dict) -> Tuple[torch.Tensor, Dict]:
        G = torch.stack(state.group_sums)
        return G.mean(0), {}

    def calibrate(self, benign_aggregates: List[torch.Tensor]) -> None:
        """Optional: calibrate thresholds from benign-only warm-up aggregates."""
        pass


class BEV(Aggregator):
    name = "bev"
    # the defense is the power control; aggregate is the mean (channel must be configured with 'bev')


class ROTAF(Aggregator):
    name = "rotaf"

    def __init__(self, d_tx, device, weiszfeld_iters: int = 20, **kw):
        super().__init__(d_tx, device)
        self.iters = weiszfeld_iters

    def aggregate(self, state, server_ctx):
        G = torch.stack(state.group_sums)
        if G.shape[0] == 1:
            return G[0], {"note": "single slot: GM = mean"}
        return geometric_median(G, self.iters), {}


class AirCompGM(ROTAF):
    name = "aircomp_gm"

    def __init__(self, d_tx, device, **kw):
        super().__init__(d_tx, device, weiszfeld_iters=kw.get("weiszfeld_iters", 3))
        # each Weiszfeld iteration is one extra AirComp channel use; charge it
        self.extra_channel_uses = self.iters


class GroupCC(Aggregator):
    name = "group_cc"

    def __init__(self, d_tx, device, radius: float = 1.0, iters: int = 3, **kw):
        super().__init__(d_tx, device)
        self.radius, self.iters = radius, iters
        self.center: Optional[torch.Tensor] = None

    def aggregate(self, state, server_ctx):
        G = torch.stack(state.group_sums)
        v = self.center if self.center is not None else G.mean(0)
        for _ in range(self.iters):
            diff = G - v
            scale = torch.clamp(self.radius / (diff.norm(dim=1) + 1e-12), max=1.0)
            v = v + (scale[:, None] * diff).mean(0)
        self.center = v.detach().clone()
        return v, {}


class GroupTM(Aggregator):
    name = "group_tm"

    def __init__(self, d_tx, device, trim_frac: float = 0.2, **kw):
        super().__init__(d_tx, device)
        self.trim = trim_frac

    def aggregate(self, state, server_ctx):
        G = torch.stack(state.group_sums)
        n = G.shape[0]; k = int(self.trim * n)
        if n - 2 * k <= 0:
            return G.mean(0), {}
        s, _ = G.sort(dim=0)
        return s[k:n - k].mean(0), {}


class FedSAC(Aggregator):
    """Cluster-level reimplementation. Weights w_g ∝ rep_g * exp(-||G_g - z||^2 / (2 s^2)) with z the
    geometric median of group sums; reputation rep_g is an EMA of the normalised weight. With random
    slot assignment per round, cluster reputation has no persistent identity; we therefore run FedSAC
    with slot_assignment='fixed' (the ZTA-reputation setting of the paper)."""
    name = "fedsac"

    def __init__(self, d_tx, device, temp: float = 1.0, rep_decay: float = 0.9, n_slots: int = 4, **kw):
        super().__init__(d_tx, device)
        self.temp, self.rep_decay = temp, rep_decay
        self.rep = torch.ones(n_slots, device=device)

    def aggregate(self, state, server_ctx):
        G = torch.stack(state.group_sums)
        if G.shape[0] == 1:
            return G[0], {}
        z = geometric_median(G)
        d2 = (G - z).pow(2).sum(1)
        s2 = d2.median() + 1e-12
        w = self.rep[: G.shape[0]] * torch.exp(-d2 / (2 * self.temp * s2))
        w = w / w.sum()
        self.rep[: G.shape[0]] = self.rep_decay * self.rep[: G.shape[0]] + (1 - self.rep_decay) * w * G.shape[0]
        return (w[:, None] * G).sum(0), {"weights": w.tolist()}


def build_aggregator(name: str, d_tx: int, device: torch.device, **kw) -> Aggregator:
    table = {"none": Aggregator, "bev": BEV, "rotaf": ROTAF, "aircomp_gm": AirCompGM,
             "group_cc": GroupCC, "group_tm": GroupTM, "fedsac": FedSAC}
    if name == "tti":
        from .tti import TTI
        return TTI(d_tx, device, **kw)
    if name == "cma":
        from .cma import CMA
        return CMA(d_tx, device, **kw)
    return table[name](d_tx, device, **kw)
